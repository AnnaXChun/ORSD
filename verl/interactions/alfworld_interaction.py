# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Official ALFWorld TextWorld interaction for VERL multi-turn rollouts."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .agent_environment_utils import (
    extract_action,
    latest_assistant_text,
    match_admissible_action,
    normalize_rubric,
    render_environment_observation,
)
from .base import BaseInteraction

logger = logging.getLogger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


# TextWorld/TatSu keeps parser stacks in process-global state. VERL creates a
# distinct ``AlfworldInteraction`` for every concurrent ToolAgentLoop, so an
# instance-local asyncio.Lock does not protect concurrent parser use. Both
# reset/load and later environment steps can enter those parsers; serialize all
# TextWorld calls within one worker process. Model generation remains fully
# concurrent and dominates runtime.
_TEXTWORLD_ENVIRONMENT_LOCK = threading.RLock()
_ALFWORLD_WRAPPER_CACHE: dict[tuple[Any, ...], Any] = {}
_ALFWORLD_ENVIRONMENT_POOL: dict[tuple[Any, ...], list[Any]] = {}


def _alfworld_wrapper_cache_key(factory: Any, config: dict[str, Any], split: str) -> tuple[Any, ...]:
    dataset = config.get("dataset", {})
    env_config = config.get("env", {})
    return (
        factory,
        split,
        os.path.expandvars(str(dataset.get("data_path", ""))),
        os.path.expandvars(str(dataset.get("eval_id_data_path", ""))),
        os.path.expandvars(str(dataset.get("eval_ood_data_path", ""))),
        tuple(env_config.get("task_types", ())),
        dataset.get("num_train_games", 0),
        dataset.get("num_eval_games", 0),
    )


def _create_alfworld_wrapper(factory: Any, config: dict[str, Any], split: str) -> Any:
    """Create a wrapper while reusing ALFWorld's expensive game-file index."""
    key = _alfworld_wrapper_cache_key(factory, config, split)
    template = _ALFWORLD_WRAPPER_CACHE.get(key)
    if template is None:
        wrapper = factory(config, train_eval=split)
        # The template contains only Python configuration and the collected
        # game-file list; native TextWorld environments are created later.
        _ALFWORLD_WRAPPER_CACHE[key] = copy.copy(wrapper)
        return wrapper

    wrapper = copy.copy(template)
    wrapper.config = config
    wrapper.train_eval = split
    wrapper.game_files = list(template.game_files)
    wrapper.num_games = len(wrapper.game_files)
    return wrapper


def _alfworld_environment_pool_key(
    factory: Any, config: dict[str, Any], split: str, max_steps: int
) -> tuple[Any, ...]:
    return (*_alfworld_wrapper_cache_key(factory, config, split), int(max_steps))


def _acquire_alfworld_environment(wrapper: Any, pool_key: tuple[Any, ...]) -> Any:
    """Checkout a reusable native environment or create one for a new slot."""
    if _as_bool(os.environ.get("ALFWORLD_ENV_POOL_ENABLE", "True")):
        available = _ALFWORLD_ENVIRONMENT_POOL.get(pool_key)
        if available:
            env = available.pop()
            env._verl_environment_in_pool = False
            return env

    env = wrapper.init_env(batch_size=1)
    env._verl_environment_in_pool = False
    return env


def _return_alfworld_environment(env: Any, pool_key: tuple[Any, ...]) -> None:
    """Return an environment while keeping its Fast Downward library mapped."""
    if not _as_bool(os.environ.get("ALFWORLD_ENV_POOL_ENABLE", "True")):
        _release_alfworld_environment(env)
        return
    if getattr(env, "_verl_environment_in_pool", False):
        return
    env._verl_environment_in_pool = True
    _ALFWORLD_ENVIRONMENT_POOL.setdefault(pool_key, []).append(env)


def _release_alfworld_environment(env: Any) -> None:
    """Close one episode without unloading Fast Downward's native library.

    Fast Downward registers C++ plugins through process-global static
    initializers. Unloading and loading the shared object again in the same Ray
    worker can re-run that registration against stale native state and segfault
    inside ``RawRegistry``. Keep the library mapped for the worker lifetime;
    the operating system releases it when the worker exits.
    """
    if getattr(env, "_verl_environment_released", False):
        return
    env._verl_environment_released = True

    try:
        env.close()
    except Exception:
        logger.exception("ALFWorld wrapper close failed")


class AlfworldInteraction(BaseInteraction):
    """Run one fixed official ALFWorld game per interaction instance.

    The benchmark metric remains the official binary ``infos['won']`` signal.
    An optional R2 training reward scales success and penalizes invalid actions;
    rubrics remain privileged decision context rather than reward labels.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.config_path = os.path.expandvars(os.path.expanduser(config["config_path"]))
        self.data_dir = os.path.expandvars(os.path.expanduser(config.get("data_dir", "")))
        self.default_split = config.get("split", "train")
        self.seed = int(config.get("seed", 1234))
        self.r2_reward_enabled = _as_bool(config.get("r2_reward_enabled", False))
        self.r2_success_scale = float(config.get("r2_success_scale", 10.0))
        self.r2_invalid_action_coef = float(config.get("r2_invalid_action_coef", 0.1))
        if self.r2_success_scale <= 0:
            raise ValueError("ALFWorld R2 success scale must be positive")
        if self.r2_invalid_action_coef < 0:
            raise ValueError("ALFWorld R2 invalid-action coefficient must be non-negative")

    def _start_sync(self, instance_id: str, game_file: str, split: str, rubric: str, max_steps: int) -> None:
        try:
            import yaml
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise RuntimeError(
                "ALFWorld is not installed. Install the official package and data before using AlfworldInteraction."
            ) from exc

        if self.data_dir:
            os.environ["ALFWORLD_DATA"] = self.data_dir
        config = yaml.safe_load(Path(self.config_path).read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["general"]["use_cuda"] = False
        # ALFWorld's ``dagger`` train environment attaches ``AlfredExpert`` and
        # recomputes ``extra.expert_plan`` after every environment action. VERL
        # never consumes that field: it needs only observations, admissible
        # commands, and the official ``won`` signal. More importantly, the
        # upstream hand-coded expert can call ``random.choice`` on an empty
        # candidate list for otherwise valid states, aborting a long training
        # run from inside ``env.step``. The ``dqn`` branch registers the same
        # official game without this unused expert wrapper.
        config["general"]["training_method"] = "dqn"
        config.setdefault("rl", {}).setdefault("training", {})[
            "max_nb_steps_per_episode"
        ] = max_steps

        environment_factory = get_environment("AlfredTWEnv")
        wrapper = _create_alfworld_wrapper(environment_factory, config, split)
        resolved_game = str(Path(os.path.expandvars(os.path.expanduser(game_file))).resolve())
        available = {str(Path(path).resolve()) for path in wrapper.game_files}
        if resolved_game not in available:
            raise ValueError(f"ALFWorld game is not part of split {split!r}: {resolved_game}")
        wrapper.game_files = [resolved_game]
        wrapper.num_games = 1
        pool_key = _alfworld_environment_pool_key(environment_factory, config, split, max_steps)
        env = _acquire_alfworld_environment(wrapper, pool_key)
        # A pooled TextworldBatchGymEnv keeps its SyncBatchEnv and PddlEnv.
        # Re-seeding rebuilds the game iterator for the newly requested file;
        # reset() then loads the new problem into the existing native library.
        env.gamefiles = [resolved_game]
        env.seed(self.seed)
        try:
            observations, infos = env.reset()
        except BaseException:
            _release_alfworld_environment(env)
            raise
        actual_game = str(Path(infos["extra.gamefile"][0]).resolve())
        if actual_game != resolved_game:
            _release_alfworld_environment(env)
            raise RuntimeError(f"ALFWorld pairing mismatch: expected {resolved_game}, got {actual_game}")

        self._instance_dict[instance_id] = {
            "wrapper": wrapper,
            "env": env,
            "environment_pool_key": pool_key,
            "observation": observations[0],
            "admissible_actions": list(infos["admissible_commands"][0]),
            "rubric": rubric,
            "won": bool(infos["won"][0]),
            "done": False,
            "environment_score": 0.0,
            "steps": 0,
            "max_steps": max_steps,
            "invalid_actions": 0,
        }

    def _start_sync_serialized(
        self, instance_id: str, game_file: str, split: str, rubric: str, max_steps: int
    ) -> None:
        with _TEXTWORLD_ENVIRONMENT_LOCK:
            self._start_sync(instance_id, game_file, split, rubric, max_steps)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        game_file: Optional[str] = None,
        split: Optional[str] = None,
        rubric: Any = None,
        inject_rubric: bool = False,
        max_steps: int = 50,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        if not game_file:
            raise ValueError("AlfworldInteraction requires interaction_kwargs.game_file")
        startup_task = asyncio.create_task(
            asyncio.to_thread(
                self._start_sync_serialized,
                instance_id,
                game_file,
                split or self.default_split,
                normalize_rubric(rubric) if inject_rubric else "",
                int(max_steps),
            )
        )
        try:
            await asyncio.shield(startup_task)
        except BaseException:
            # A cancelled coroutine cannot stop the startup thread. Wait for it
            # to finish and release any environment it managed to publish.
            try:
                await asyncio.shield(startup_task)
            except BaseException:
                pass
            state = self._instance_dict.pop(instance_id, None)
            if state is not None:
                await asyncio.to_thread(
                    self._close_sync_serialized,
                    state["env"],
                    state.get("environment_pool_key"),
                )
            raise
        return instance_id

    async def get_initial_observation(self, instance_id: str, **kwargs) -> str:
        state = self._instance_dict[instance_id]
        return render_environment_observation(
            benchmark="ALFWorld",
            observation=state["observation"],
            admissible_actions=state["admissible_actions"],
            rubric=state["rubric"],
        )

    def _step_sync(self, instance_id: str, action: str) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._instance_dict[instance_id]
        # Preserve compatibility with manually constructed states and older
        # serialized utilities created before the shaped reward was added.
        state.setdefault("invalid_actions", 0)
        canonical_action = match_admissible_action(action, state["admissible_actions"])
        invalid = canonical_action is None
        if invalid:
            state["invalid_actions"] += 1
            observation = f"Invalid action: {action or '(missing)'}. Choose one action from the current list."
        else:
            with _TEXTWORLD_ENVIRONMENT_LOCK:
                observations, scores, dones, infos = state["env"].step([canonical_action])
            observation = observations[0]
            state["observation"] = observation
            state["admissible_actions"] = list(infos["admissible_commands"][0])
            state["environment_score"] = float(scores[0])
            state["won"] = bool(infos["won"][0])
            state["done"] = bool(dones[0])

        state["steps"] += 1
        timed_out = state["steps"] >= state["max_steps"]
        should_terminate = state["won"] or state["done"] or timed_out
        response = render_environment_observation(
            benchmark="ALFWorld",
            observation=observation,
            admissible_actions=state["admissible_actions"],
            # The initial rubric remains in the conversation history; repeating
            # it every turn would waste long-horizon context quadratically.
            rubric="",
        )
        metrics = {
            "done_reason": "terminal" if state["won"] or state["done"] else ("max_steps" if timed_out else ""),
            "step_reward": float(state["won"]),
            "invalid_action": float(invalid),
            "won": float(state["won"]),
            "environment_score": state["environment_score"],
            "steps": state["steps"],
            "invalid_action_count": float(state["invalid_actions"]),
        }
        # Sparse official reward is returned at episode finalization, not each turn.
        return should_terminate, response, 0.0, metrics

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        action = extract_action(latest_assistant_text(messages))
        return await asyncio.to_thread(self._step_sync, instance_id, action)

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return self._reward_details(self._instance_dict[instance_id])["training_reward"]

    def _reward_details(self, state: dict[str, Any]) -> dict[str, float]:
        official_reward = float(state["won"])
        invalid_action_count = float(state.get("invalid_actions", 0))
        episode_steps = max(int(state.get("steps", 0)), 1)
        penalty = 0.0
        training_reward = official_reward
        if self.r2_reward_enabled:
            penalty = self.r2_invalid_action_coef * invalid_action_count
            training_reward = self.r2_success_scale * official_reward - penalty
        return {
            "training_reward": training_reward,
            "official_reward": official_reward,
            "success": official_reward,
            "r2_reward_enabled": float(self.r2_reward_enabled),
            "r2_success_scale": self.r2_success_scale,
            "r2_penalty": penalty,
            "invalid_action_count": invalid_action_count,
            "invalid_action_rate": invalid_action_count / episode_steps,
            "episode_steps": float(state.get("steps", 0)),
        }

    async def get_reward_details(self, instance_id: str, **kwargs) -> dict[str, float]:
        return self._reward_details(self._instance_dict[instance_id])

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        state = self._instance_dict.pop(instance_id, None)
        if state is not None:
            await asyncio.to_thread(
                self._close_sync_serialized,
                state["env"],
                state.get("environment_pool_key"),
            )

    @staticmethod
    def _close_sync_serialized(env: Any, pool_key: Optional[tuple[Any, ...]]) -> None:
        with _TEXTWORLD_ENVIRONMENT_LOCK:
            if pool_key is None:
                _release_alfworld_environment(env)
            else:
                _return_alfworld_environment(env, pool_key)
