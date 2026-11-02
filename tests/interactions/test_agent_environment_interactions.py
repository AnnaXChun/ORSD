from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from verl.interactions.agent_environment_utils import extract_action, match_admissible_action, normalize_rubric
from verl.interactions.alfworld_interaction import (
    AlfworldInteraction,
    _ALFWORLD_ENVIRONMENT_POOL,
    _ALFWORLD_WRAPPER_CACHE,
    _acquire_alfworld_environment,
    _create_alfworld_wrapper,
    _release_alfworld_environment,
    _return_alfworld_environment,
)
from verl.interactions.webshop_interaction import WebshopInteraction


class FakeAlfworldEnv:
    def __init__(self):
        self.closed = False

    def step(self, actions):
        assert actions == ["take apple 1 from table 1"]
        return (
            ["You pick up the apple."],
            [1.0],
            [True],
            {
                "admissible_commands": [["look"]],
                "won": [True],
            },
        )

    def close(self):
        self.closed = True


class FakeWebshopEnv:
    def __init__(self):
        self.closed = False

    def step(self, action):
        assert action == "click[buy now]"
        return "Your score is 0.75", 0.75, True, None

    def get_available_actions(self):
        return {"has_search_bar": False, "clickables": ["< prev"]}

    def close(self):
        self.closed = True


class FakeWebshopLoopEnv:
    def __init__(self):
        self.closed = False
        self.calls = 0

    def step(self, action):
        assert action == "click[repeat]"
        self.calls += 1
        return "Same page", float(self.calls == 2), self.calls == 2, None

    def get_available_actions(self):
        return {"has_search_bar": False, "clickables": ["repeat"]}

    def close(self):
        self.closed = True


class FakeWebshopServer:
    pass


class FakeWebshopInitEnv:
    instances = []

    def __init__(self, server=None, **kwargs):
        self.server = server or FakeWebshopServer()
        self.kwargs = kwargs
        self.instances.append(self)

    def reset(self, goal_idx):
        return f"Goal {goal_idx}", None

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": []}

    def close(self):
        pass


def test_action_parsing_and_exact_matching():
    assert extract_action("<think>private</think>\nACTION: Open fridge 1") == "Open fridge 1"
    assert extract_action("look") == "look"
    assert extract_action("reasoning\nlook") == ""
    assert match_admissible_action(" OPEN   FRIDGE 1 ", ["open fridge 1", "look"]) == "open fridge 1"


def test_rubric_normalization():
    value = '[{"criterion": "Check the visible price."}, {"text": "Select every required option."}]'
    assert normalize_rubric(value) == "- Check the visible price.\n- Select every required option."


def test_alfworld_uses_official_won_reward():
    interaction = AlfworldInteraction({"config_path": "/unused", "data_dir": ""})
    env = FakeAlfworldEnv()
    interaction._instance_dict["episode"] = {
        "env": env,
        "observation": "An apple is on the table.",
        "admissible_actions": ["take apple 1 from table 1"],
        "rubric": "Acquire the requested object before delivery.",
        "won": False,
        "done": False,
        "environment_score": 0.0,
        "steps": 0,
        "max_steps": 5,
    }
    terminate, _, turn_reward, metrics = interaction._step_sync("episode", "take apple 1 from table 1")
    assert terminate is True
    assert turn_reward == 0.0
    assert metrics["won"] == 1.0
    assert asyncio.run(interaction.calculate_score("episode")) == 1.0
    asyncio.run(interaction.finalize_interaction("episode"))
    assert env.closed is True


def test_alfworld_r2_reward_scales_success_and_penalizes_invalid_actions():
    interaction = AlfworldInteraction(
        {
            "config_path": "/unused",
            "data_dir": "",
            "r2_reward_enabled": "True",
            "r2_success_scale": 10.0,
            "r2_invalid_action_coef": 0.1,
        }
    )
    env = FakeAlfworldEnv()
    interaction._instance_dict["episode"] = {
        "env": env,
        "observation": "An apple is on the table.",
        "admissible_actions": ["take apple 1 from table 1"],
        "rubric": "",
        "won": False,
        "done": False,
        "environment_score": 0.0,
        "steps": 0,
        "max_steps": 5,
        "invalid_actions": 0,
    }

    terminate, _, _, _ = interaction._step_sync("episode", "dance")
    assert terminate is False
    terminate, _, _, _ = interaction._step_sync("episode", "take apple 1 from table 1")
    assert terminate is True

    details = asyncio.run(interaction.get_reward_details("episode"))
    assert details["official_reward"] == 1.0
    assert details["success"] == 1.0
    assert details["invalid_action_count"] == 1.0
    assert details["invalid_action_rate"] == 0.5
    assert details["r2_penalty"] == 0.1
    assert asyncio.run(interaction.calculate_score("episode")) == 9.9


def test_alfworld_serializes_textworld_environment_startup():
    # AgentLoopWorker creates a separate ToolAgentLoop (and therefore a
    # separate interaction instance) per trajectory. Serialization must work
    # across instances, not only across calls on one instance.
    interactions = [AlfworldInteraction({"config_path": "/unused", "data_dir": ""}) for _ in range(4)]
    active_starts = 0
    max_active_starts = 0
    state_lock = threading.Lock()

    def fake_start(instance_id, game_file, split, rubric, max_steps):
        nonlocal active_starts, max_active_starts
        with state_lock:
            active_starts += 1
            max_active_starts = max(max_active_starts, active_starts)
        time.sleep(0.02)
        with state_lock:
            active_starts -= 1

    for interaction in interactions:
        interaction._start_sync = fake_start

    async def start_all():
        await asyncio.gather(
            *(
                interaction.start_interaction(instance_id=f"episode-{index}", game_file=f"game-{index}")
                for index, interaction in enumerate(interactions)
            )
        )

    asyncio.run(start_all())
    assert max_active_starts == 1


def test_alfworld_reuses_collected_game_file_index():
    _ALFWORLD_WRAPPER_CACHE.clear()

    class FakeWrapper:
        init_calls = 0

        def __init__(self, config, train_eval):
            type(self).init_calls += 1
            self.config = config
            self.train_eval = train_eval
            self.game_files = ["game-a", "game-b"]
            self.num_games = 2

    config = {
        "dataset": {"data_path": "/data", "num_train_games": 0, "num_eval_games": 0},
        "env": {"task_types": [1, 2]},
    }
    first = _create_alfworld_wrapper(FakeWrapper, config, "train")
    updated_config = {**config, "runtime": {"max_steps": 300}}
    second = _create_alfworld_wrapper(FakeWrapper, updated_config, "train")

    assert FakeWrapper.init_calls == 1
    assert first is not second
    assert second.config is updated_config
    assert second.game_files == ["game-a", "game-b"]
    second.game_files.pop()
    assert first.game_files == ["game-a", "game-b"]
    _ALFWORLD_WRAPPER_CACHE.clear()


def test_alfworld_closes_wrapper_without_unloading_fast_downward():
    events = []
    library = object()

    class FakePddlEnv:
        def __init__(self):
            self.downward_lib = library
            self._pddl_state = object()
            self.state = object()
            self.prev_state = object()

    class FakeWrappedEnv:
        def __init__(self, base):
            self.unwrapped = base

    class FakeBatchEnv:
        def __init__(self, base):
            self.envs = [FakeWrappedEnv(base)]

    class FakeGymEnv:
        def __init__(self, base):
            self.batch_env = FakeBatchEnv(base)

        def close(self):
            events.append("wrapper_close")
            assert base._pddl_state is not None

    base = FakePddlEnv()
    env = FakeGymEnv(base)

    _release_alfworld_environment(env)
    _release_alfworld_environment(env)

    assert events == ["wrapper_close"]
    assert base.downward_lib is library
    assert base._pddl_state is not None


def test_alfworld_reuses_environment_from_bounded_pool(monkeypatch):
    _ALFWORLD_ENVIRONMENT_POOL.clear()
    monkeypatch.setenv("ALFWORLD_ENV_POOL_ENABLE", "True")

    class FakeWrapper:
        def __init__(self):
            self.created = []

        def init_env(self, batch_size):
            assert batch_size == 1
            env = SimpleNamespace()
            self.created.append(env)
            return env

    wrapper = FakeWrapper()
    pool_key = ("train", 300)
    first = _acquire_alfworld_environment(wrapper, pool_key)
    _return_alfworld_environment(first, pool_key)
    _return_alfworld_environment(first, pool_key)
    second = _acquire_alfworld_environment(wrapper, pool_key)

    assert first is second
    assert wrapper.created == [first]
    assert _ALFWORLD_ENVIRONMENT_POOL[pool_key] == []
    _ALFWORLD_ENVIRONMENT_POOL.clear()


def test_alfworld_environment_pool_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ALFWORLD_ENV_POOL_ENABLE", "False")

    class FakeWrapper:
        def init_env(self, batch_size):
            assert batch_size == 1
            return FakeAlfworldEnv()

    wrapper = FakeWrapper()
    first = _acquire_alfworld_environment(wrapper, ("train", 300))
    _return_alfworld_environment(first, ("train", 300))
    second = _acquire_alfworld_environment(wrapper, ("train", 300))

    assert first.closed is True
    assert second is not first


def test_alfworld_serializes_startup_against_episode_step():
    startup = AlfworldInteraction({"config_path": "/unused", "data_dir": ""})
    episode = AlfworldInteraction({"config_path": "/unused", "data_dir": ""})
    startup_entered = threading.Event()
    release_startup = threading.Event()
    step_entered = threading.Event()

    def fake_start(instance_id, game_file, split, rubric, max_steps):
        startup_entered.set()
        assert release_startup.wait(timeout=1.0)

    class BlockingProbeEnv(FakeAlfworldEnv):
        def step(self, actions):
            step_entered.set()
            return super().step(actions)

    startup._start_sync = fake_start
    episode._instance_dict["episode"] = {
        "env": BlockingProbeEnv(),
        "observation": "An apple is on the table.",
        "admissible_actions": ["take apple 1 from table 1"],
        "rubric": "",
        "won": False,
        "done": False,
        "environment_score": 0.0,
        "steps": 0,
        "max_steps": 5,
    }

    start_thread = threading.Thread(
        target=startup._start_sync_serialized,
        args=("startup", "game", "train", "", 5),
    )
    step_thread = threading.Thread(
        target=episode._step_sync,
        args=("episode", "take apple 1 from table 1"),
    )
    start_thread.start()
    assert startup_entered.wait(timeout=1.0)
    step_thread.start()
    assert not step_entered.wait(timeout=0.05)
    release_startup.set()
    start_thread.join(timeout=1.0)
    step_thread.join(timeout=1.0)
    assert step_entered.is_set()


def test_webshop_uses_unscaled_official_reward():
    interaction = WebshopInteraction({"webshop_root": "/unused"})
    env = FakeWebshopEnv()
    interaction._instance_dict["episode"] = {
        "env": env,
        "observation": "Ready to purchase.",
        "search_available": False,
        "click_actions": ["click[buy now]"],
        "rubric": "Buy only after all visible constraints are checked.",
        "official_reward": 0.0,
        "done": False,
        "steps": 0,
        "max_steps": 5,
    }
    terminate, _, turn_reward, metrics = interaction._step_sync("episode", "click[buy now]")
    assert terminate is True
    assert turn_reward == 0.0
    assert metrics["official_reward"] == 0.75
    assert asyncio.run(interaction.calculate_score("episode")) == 0.75
    asyncio.run(interaction.finalize_interaction("episode"))
    assert env.closed is True


def test_webshop_r2_reward_penalizes_invalid_and_repeated_state_actions():
    interaction = WebshopInteraction(
        {
            "webshop_root": "/unused",
            "r2_reward_enabled": "True",
            "r2_invalid_action_coef": 0.05,
            "r2_repeat_action_coef": 0.025,
        }
    )
    env = FakeWebshopLoopEnv()
    interaction._instance_dict["episode"] = {
        "env": env,
        "goal_idx": 0,
        "observation": "Same page",
        "search_available": False,
        "click_actions": ["click[repeat]"],
        "rubric": "",
        "reward": 0.0,
        "done": False,
        "steps": 0,
        "max_steps": 5,
        "invalid_actions": 0,
        "repeat_no_progress_actions": 0,
        "seen_state_actions": set(),
    }

    terminate, _, _, _ = interaction._step_sync("episode", "click[missing]")
    assert terminate is False
    terminate, _, _, _ = interaction._step_sync("episode", "click[repeat]")
    assert terminate is False
    terminate, _, _, metrics = interaction._step_sync("episode", "click[repeat]")
    assert terminate is True
    assert metrics["repeat_no_progress_action"] == 1.0

    details = asyncio.run(interaction.get_reward_details("episode"))
    assert details["official_reward"] == 1.0
    assert details["success"] == 1.0
    assert details["invalid_action_rate"] == 1 / 3
    assert details["repeat_no_progress_rate"] == 1 / 3
    assert details["r2_penalty"] == (0.05 + 0.025) / 3
    assert asyncio.run(interaction.calculate_score("episode")) == 0.975


def test_webshop_gigpo_reward_uses_strict_success_and_per_invalid_action_penalty():
    interaction = WebshopInteraction(
        {
            "webshop_root": "/unused",
            "reward_mode": "gigpo",
            "gigpo_success_reward": 10.0,
            "gigpo_invalid_action_coef": 0.1,
        }
    )
    success_details = interaction._reward_details(
        {
            "reward": 1.0,
            "steps": 3,
            "invalid_actions": 1,
            "repeat_no_progress_actions": 2,
        }
    )
    assert success_details["success"] == 1.0
    assert success_details["training_reward"] == 9.9
    assert success_details["gigpo_invalid_action_penalty"] == 0.1
    assert success_details["reward_mode_gigpo"] == 1.0

    failure_details = interaction._reward_details(
        {
            "reward": 0.75,
            "steps": 4,
            "invalid_actions": 2,
            "repeat_no_progress_actions": 3,
        }
    )
    assert failure_details["success"] == 0.0
    assert failure_details["training_reward"] == -0.2
    assert failure_details["gigpo_invalid_action_penalty"] == 0.2
    # GiGPO does not use the native partial score or the custom repetition penalty.
    assert failure_details["r2_penalty"] == 0.0


def test_webshop_hybrid_reward_combines_partial_score_and_strict_success():
    interaction = WebshopInteraction(
        {
            "webshop_root": "/unused",
            "reward_mode": "hybrid",
            "r2_invalid_action_coef": 0.05,
            "r2_repeat_action_coef": 0.025,
        }
    )
    failure_details = interaction._reward_details(
        {
            "reward": 0.9,
            "steps": 10,
            "invalid_actions": 1,
            "repeat_no_progress_actions": 2,
        }
    )
    assert failure_details["success"] == 0.0
    assert failure_details["hybrid_penalty"] == 0.01
    assert failure_details["training_reward"] == 0.44
    assert failure_details["reward_mode_hybrid"] == 1.0

    success_details = interaction._reward_details(
        {
            "reward": 1.0,
            "steps": 10,
            "invalid_actions": 1,
            "repeat_no_progress_actions": 2,
        }
    )
    assert success_details["success"] == 1.0
    assert success_details["training_reward"] == 0.99


def test_webshop_reward_mode_rejects_unknown_value():
    try:
        WebshopInteraction({"webshop_root": "/unused", "reward_mode": "mystery"})
    except ValueError as exc:
        assert "reward_mode" in str(exc)
    else:
        raise AssertionError("Unknown WebShop reward mode should fail fast")


def test_webshop_parallel_rollouts_use_distinct_sessions(monkeypatch):
    interaction = WebshopInteraction({"webshop_root": "/unused"})
    interaction._SERVER_CACHE.clear()
    FakeWebshopInitEnv.instances.clear()
    monkeypatch.setattr(interaction, "_load_environment_class", lambda: FakeWebshopInitEnv)

    interaction._start_sync("episode-a", 7, "", 5)
    interaction._start_sync("episode-b", 7, "", 5)

    first, second = FakeWebshopInitEnv.instances[-2:]
    assert first.server is second.server
    assert first.kwargs["session_prefix"] == "episode-a-"
    assert second.kwargs["session_prefix"] == "episode-b-"


def test_webshop_small_forces_coupled_catalog_configuration(monkeypatch):
    interaction = WebshopInteraction(
        {
            "webshop_root": "/benchmark/WebShop",
            "file_path": "/stale/full/items_shuffle.json",
            "use_small": "True",
            "human_goals": "True",
        }
    )
    interaction._SERVER_CACHE.clear()
    FakeWebshopInitEnv.instances.clear()
    monkeypatch.setattr(interaction, "_load_environment_class", lambda: FakeWebshopInitEnv)

    interaction._start_sync("small-episode", 499, "", 15)

    env = FakeWebshopInitEnv.instances[-1]
    assert env.kwargs["file_path"] == "/benchmark/WebShop/data/items_shuffle_1000.json"
    assert env.kwargs["human_goals"] is False
    assert env.kwargs["num_products"] == 1000


def test_webshop_small_row_override_isolated_from_full_server_cache(monkeypatch):
    interaction = WebshopInteraction(
        {
            "webshop_root": "/benchmark/WebShop",
            "file_path": "/benchmark/WebShop/data/items_shuffle.json",
            "human_goals": "True",
        }
    )
    interaction._SERVER_CACHE.clear()
    FakeWebshopInitEnv.instances.clear()
    monkeypatch.setattr(interaction, "_load_environment_class", lambda: FakeWebshopInitEnv)

    interaction._start_sync("full-episode", 0, "", 15)
    interaction._start_sync("small-episode", 0, "", 15, use_small=True)

    full_env, small_env = FakeWebshopInitEnv.instances[-2:]
    assert full_env.server is not small_env.server
    assert full_env.kwargs["human_goals"] is True
    assert "num_products" not in full_env.kwargs
    assert small_env.kwargs["human_goals"] is False
    assert small_env.kwargs["num_products"] == 1000
