# WebShopInteraction
# Full implementation generated for VERL WebShop rollout sanity.

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import sys
import threading
from typing import Any, Optional
from uuid import uuid4

from .agent_environment_utils import (
    extract_action,
    latest_assistant_text,
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


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(normalized)


def _normalize_reward_mode(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "official": "official",
        "native": "official",
        "benchmark": "official",
        "r2": "r2",
        "hybrid": "hybrid",
        "success_hybrid": "hybrid",
        "hybrid_reward": "hybrid",
        "gigpo": "gigpo",
        "gigpo_reward": "gigpo",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "WebShop reward_mode must be one of: official, r2, hybrid, gigpo; "
            f"got {value!r}"
        ) from exc


class WebshopInteraction(BaseInteraction):

    _SERVER_CACHE = {}
    _SERVER_LOCK = threading.Lock()

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict = {}
        self.webshop_root = os.path.expandvars(os.path.expanduser(config["webshop_root"]))
        self.file_path = os.path.expandvars(os.path.expanduser(config.get("file_path", ""))) or None
        self.observation_mode = config.get("observation_mode", "text_rich")
        self.seed = int(config.get("seed", 1234))
        self.use_small = _as_bool(config.get("use_small", False))
        self.human_goals = _as_bool(config.get("human_goals", not self.use_small))
        self.num_products = _as_optional_int(config.get("num_products", 1000 if self.use_small else None))
        legacy_r2_enabled = _as_bool(config.get("r2_reward_enabled", False))
        configured_reward_mode = str(config.get("reward_mode", "")).strip()
        self.reward_mode = _normalize_reward_mode(
            configured_reward_mode or ("r2" if legacy_r2_enabled else "official")
        )
        self.r2_reward_enabled = self.reward_mode == "r2"
        self.r2_invalid_action_coef = float(config.get("r2_invalid_action_coef", 0.05))
        self.r2_repeat_action_coef = float(config.get("r2_repeat_action_coef", 0.025))
        self.gigpo_success_reward = float(config.get("gigpo_success_reward", 10.0))
        self.gigpo_invalid_action_coef = float(config.get("gigpo_invalid_action_coef", 0.1))
        if self.r2_invalid_action_coef < 0 or self.r2_repeat_action_coef < 0:
            raise ValueError("WebShop R2 reward coefficients must be non-negative")
        if self.gigpo_success_reward <= 0 or self.gigpo_invalid_action_coef < 0:
            raise ValueError(
                "WebShop GiGPO success reward must be positive and invalid-action coefficient non-negative"
            )

    def _load_environment_class(self):
        if self.webshop_root not in sys.path:
            sys.path.insert(0, self.webshop_root)
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        return WebAgentTextEnv

    def _configure_small_catalog_defaults(self) -> None:
        """Pin upstream WebShop globals to the 1k synthetic-goal assets."""
        import web_agent_site.utils as webshop_utils
        import web_agent_site.engine.engine as webshop_engine

        small_products = os.path.join(self.webshop_root, "data", "items_shuffle_1000.json")
        small_attributes = os.path.join(self.webshop_root, "data", "items_ins_v2_1000.json")
        webshop_utils.DEFAULT_FILE_PATH = small_products
        webshop_utils.DEFAULT_ATTR_PATH = small_attributes
        # engine.py imports these constants by value, so update its module
        # globals as well when a checkout was previously configured for full.
        webshop_engine.DEFAULT_FILE_PATH = small_products
        webshop_engine.DEFAULT_ATTR_PATH = small_attributes

    def _available_actions(self, env):
        available = env.get_available_actions()
        clicks = [
            f"click[{x}]"
            for x in available.get("clickables", [])
            if str(x).lower() != "search"
        ]
        return bool(available.get("has_search_bar", False)), clicks

    def _resolve_catalog(
        self,
        *,
        use_small: Any = None,
        file_path: Optional[str] = None,
        human_goals: Any = None,
        num_products: Any = None,
    ) -> tuple[Optional[str], bool, Optional[int]]:
        effective_use_small = self.use_small if use_small is None else _as_bool(use_small)

        if effective_use_small:
            # WebShop-small is a coupled benchmark configuration: synthetic
            # goals, the 1k product file, and indexes_1k must be used together.
            # Force those values so stale full-catalog environment variables
            # cannot silently create a mismatched task/index pair.
            small_file = os.path.join(self.webshop_root, "data", "items_shuffle_1000.json")
            return (
                os.path.expandvars(os.path.expanduser(file_path or small_file)),
                False,
                1000,
            )

        resolved_file = file_path or self.file_path
        if resolved_file:
            resolved_file = os.path.expandvars(os.path.expanduser(resolved_file))
        resolved_human_goals = self.human_goals if human_goals is None else _as_bool(human_goals)
        resolved_num_products = self.num_products if num_products is None else _as_optional_int(num_products)
        return resolved_file, resolved_human_goals, resolved_num_products

    def _start_sync(
        self,
        instance_id,
        goal_idx,
        rubric,
        max_steps,
        use_small=None,
        file_path=None,
        human_goals=None,
        num_products=None,
    ):
        resolved_file, resolved_human_goals, resolved_num_products = self._resolve_catalog(
            use_small=use_small,
            file_path=file_path,
            human_goals=human_goals,
            num_products=num_products,
        )
        env_cls = self._load_environment_class()
        if (
            resolved_human_goals is False
            and resolved_num_products == 1000
            and env_cls.__module__.startswith("web_agent_site.")
        ):
            self._configure_small_catalog_defaults()

        kwargs = {
            "observation_mode": self.observation_mode,
            "human_goals": resolved_human_goals,
            "filter_goals": None,
            "limit_goals": -1,
            "session_prefix": f"{instance_id}-",
        }

        if resolved_file:
            kwargs["file_path"] = resolved_file
        if resolved_num_products is not None:
            kwargs["num_products"] = resolved_num_products

        key = (
            self.webshop_root,
            resolved_file,
            resolved_human_goals,
            resolved_num_products,
            self.seed,
        )

        with self._SERVER_LOCK:
            server = self._SERVER_CACHE.get(key)
            if server is None:
                old = random.getstate()
                random.seed(self.seed)
                try:
                    env = env_cls(**kwargs)
                finally:
                    random.setstate(old)
                self._SERVER_CACHE[key] = env.server
            else:
                env = env_cls(server=server, **kwargs)

        obs, _ = env.reset(int(goal_idx))
        search_available, clicks = self._available_actions(env)

        self._instance_dict[instance_id] = {
            "env": env,
            "goal_idx": int(goal_idx),
            "observation": obs,
            "search_available": search_available,
            "click_actions": clicks,
            "rubric": rubric,
            "reward": 0.0,
            "done": False,
            "steps": 0,
            "max_steps": max_steps,
            "invalid_actions": 0,
            "repeat_no_progress_actions": 0,
            "seen_state_actions": set(),
        }

        logger.info("[WebShop] reset goal_idx=%s", goal_idx)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        goal_idx: Optional[int] = None,
        rubric: Any = None,
        inject_rubric: bool = False,
        max_steps: int = 100,
        use_small: Any = None,
        file_path: Optional[str] = None,
        human_goals: Any = None,
        num_products: Any = None,
        **kwargs,
    ):
        if instance_id is None:
            instance_id = str(uuid4())

        if goal_idx is None:
            raise ValueError("missing goal_idx")

        await asyncio.to_thread(
            self._start_sync,
            instance_id,
            int(goal_idx),
            normalize_rubric(rubric) if inject_rubric else "",
            int(max_steps),
            use_small,
            file_path,
            human_goals,
            num_products,
        )
        return instance_id

    async def get_initial_observation(self, instance_id, **kwargs):
        s = self._instance_dict[instance_id]
        return render_environment_observation(
            benchmark="WebShop",
            observation=s["observation"],
            admissible_actions=s["click_actions"],
            rubric=s["rubric"],
            search_available=s["search_available"],
        )

    @staticmethod
    def _match_action(action, search_available, click_actions):
        action = action.strip()

        if search_available and re.fullmatch(
            r"search\[[^\[\]]+\]",
            action,
            re.I,
        ):
            return action

        for x in click_actions:
            if action.lower() == x.lower():
                return x

        return None

    def _step_sync(self, instance_id, action):
        s = self._instance_dict[instance_id]
        # Keep compatibility with manually constructed interaction states in
        # tests and downstream utilities created before R2 was introduced.
        s.setdefault("invalid_actions", 0)
        s.setdefault("repeat_no_progress_actions", 0)
        s.setdefault("seen_state_actions", set())
        s.setdefault("reward", float(s.get("official_reward", 0.0)))

        canonical = self._match_action(
            action,
            s["search_available"],
            s["click_actions"],
        )

        invalid = canonical is None
        repeated_no_progress = False

        if invalid:
            s["invalid_actions"] += 1
            obs = f"Invalid action: {action}"
            done = False
        else:
            # WebShop is deterministic for a fixed page state. Executing the
            # same action after returning to the same observation repeats an
            # already visited transition and is counted as a no-progress loop.
            state_action = (
                str(s["observation"]),
                bool(s["search_available"]),
                tuple(s["click_actions"]),
                canonical.casefold(),
            )
            repeated_no_progress = state_action in s["seen_state_actions"]
            s["seen_state_actions"].add(state_action)
            if repeated_no_progress:
                s["repeat_no_progress_actions"] += 1
            obs, reward, done, _ = s["env"].step(canonical)
            if done:
                s["reward"] = float(reward)

            s["observation"] = obs
            s["done"] = bool(done)
            s["search_available"], s["click_actions"] = self._available_actions(s["env"])

        s["steps"] += 1

        return (
            s["done"] or s["steps"] >= s["max_steps"],
            render_environment_observation(
                benchmark="WebShop",
                observation=obs,
                admissible_actions=s["click_actions"],
                rubric="",
                search_available=s["search_available"],
            ),
            0.0,
            {
                "done_reason": "terminal" if s["done"] else ("max_steps" if s["steps"] >= s["max_steps"] else ""),
                "step_reward": s["reward"] if s["done"] else 0.0,
                "invalid_action": float(invalid),
                "repeat_no_progress_action": float(repeated_no_progress),
                "official_reward": s["reward"],
                "steps": s["steps"],
                "goal_idx": s["goal_idx"],
            },
        )

    async def generate_response(self, instance_id, messages, **kwargs):
        action = extract_action(latest_assistant_text(messages))
        return await asyncio.to_thread(self._step_sync, instance_id, action)

    async def calculate_score(self, instance_id, **kwargs):
        details = self._reward_details(self._instance_dict[instance_id])
        return details["training_reward"]

    def _reward_details(self, state: dict[str, Any]) -> dict[str, float]:
        official_reward = float(state.get("reward", state.get("official_reward", 0.0)))
        steps = max(int(state["steps"]), 1)
        invalid_rate = float(state.get("invalid_actions", 0)) / steps
        repeat_rate = float(state.get("repeat_no_progress_actions", 0)) / steps
        invalid_action_count = float(state.get("invalid_actions", 0))
        success = float(official_reward >= 1.0 - 1e-8)
        r2_penalty = 0.0
        hybrid_penalty = 0.0
        gigpo_penalty = 0.0
        if self.reward_mode == "r2":
            r2_penalty = (
                self.r2_invalid_action_coef * invalid_rate
                + self.r2_repeat_action_coef * repeat_rate
            )
            training_reward = official_reward - r2_penalty
        elif self.reward_mode == "hybrid":
            hybrid_penalty = (
                self.r2_invalid_action_coef * invalid_rate
                + self.r2_repeat_action_coef * repeat_rate
            )
            training_reward = 0.5 * official_reward + 0.5 * success - hybrid_penalty
        elif self.reward_mode == "gigpo":
            gigpo_penalty = self.gigpo_invalid_action_coef * invalid_action_count
            training_reward = self.gigpo_success_reward * success - gigpo_penalty
        else:
            training_reward = official_reward
        return {
            "training_reward": training_reward,
            "official_reward": official_reward,
            "success": success,
            "reward_mode_official": float(self.reward_mode == "official"),
            "reward_mode_r2": float(self.reward_mode == "r2"),
            "reward_mode_hybrid": float(self.reward_mode == "hybrid"),
            "reward_mode_gigpo": float(self.reward_mode == "gigpo"),
            "r2_reward_enabled": float(self.r2_reward_enabled),
            "r2_penalty": r2_penalty,
            "hybrid_penalty": hybrid_penalty,
            "hybrid_official_component": 0.5 * official_reward,
            "hybrid_success_component": 0.5 * success,
            "gigpo_reward_enabled": float(self.reward_mode == "gigpo"),
            "gigpo_success_reward": self.gigpo_success_reward,
            "gigpo_invalid_action_penalty": gigpo_penalty,
            "reward_penalty": r2_penalty + hybrid_penalty + gigpo_penalty,
            "invalid_action_count": invalid_action_count,
            "invalid_action_rate": invalid_rate,
            "repeat_no_progress_count": float(state.get("repeat_no_progress_actions", 0)),
            "repeat_no_progress_rate": repeat_rate,
            "episode_steps": float(state["steps"]),
        }

    async def get_reward_details(self, instance_id, **kwargs):
        return self._reward_details(self._instance_dict[instance_id])

    async def finalize_interaction(self, instance_id, **kwargs):
        state = self._instance_dict.pop(instance_id, None)
        if state:
            await asyncio.to_thread(state["env"].close)
