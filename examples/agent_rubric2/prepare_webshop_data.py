import json
import os
import random
import sys
from pathlib import Path
import pandas as pd


ROOT = Path(os.environ.get("VERL_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
WEBSHOP_ROOT = Path(os.environ.get("WEBSHOP_ROOT", ROOT / "WebShop")).expanduser().resolve()
MAX_STEPS = int(os.environ.get("WEBSHOP_MAX_STEPS", "15"))
SPLIT_PROTOCOL = os.environ.get("WEBSHOP_SPLIT_PROTOCOL", "public_repo").strip().lower()
CATALOG_MODE = os.environ.get("WEBSHOP_CATALOG_MODE", "full").strip().lower()

CATALOG_MODE_ALIASES = {
    "full": "full",
    "human": "full",
    "small": "small",
    "webshop_small": "small",
    "1k": "small",
}
if CATALOG_MODE not in CATALOG_MODE_ALIASES:
    raise ValueError(
        "WEBSHOP_CATALOG_MODE must be one of: full, small, webshop_small, 1k; "
        f"got {CATALOG_MODE!r}"
    )
CATALOG_MODE = CATALOG_MODE_ALIASES[CATALOG_MODE]

SPLIT_PROTOCOL_ALIASES = {
    "canonical": "canonical",
    "original": "canonical",
    "public_repo": "public_repo",
    "repo_matched": "public_repo",
    "webshop3": "public_repo",
}
if SPLIT_PROTOCOL not in SPLIT_PROTOCOL_ALIASES:
    raise ValueError(
        "WEBSHOP_SPLIT_PROTOCOL must be one of: canonical, public_repo, "
        f"repo_matched, webshop3; got {SPLIT_PROTOCOL!r}"
    )
SPLIT_PROTOCOL = SPLIT_PROTOCOL_ALIASES[SPLIT_PROTOCOL]

GOAL_FILE = (
    Path(os.environ["WEBSHOP_GOAL_FILE"]).expanduser().resolve()
    if "WEBSHOP_GOAL_FILE" in os.environ
    else WEBSHOP_ROOT / "baseline_models/data/human_goals.json"
)


OUT_DIR = (
    Path(os.environ["WEBSHOP_DATA_DIR"]).expanduser().resolve()
    if "WEBSHOP_DATA_DIR" in os.environ
    else ROOT / "Data/agent_rubric" / (
        "webshop_small"
        if CATALOG_MODE == "small"
        else ("webshop3" if SPLIT_PROTOCOL == "public_repo" else "webshop2")
    )
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if CATALOG_MODE == "small":
    small_products = WEBSHOP_ROOT / "data/items_shuffle_1000.json"
    small_attributes = WEBSHOP_ROOT / "data/items_ins_v2_1000.json"
    small_index = WEBSHOP_ROOT / "search_engine/indexes_1k"
    missing_small_assets = [
        str(path)
        for path in (small_products, small_attributes, small_index)
        if not path.exists()
    ]
    if missing_small_assets:
        raise FileNotFoundError(
            "WebShop-small assets are incomplete: "
            + ", ".join(missing_small_assets)
            + ". Run: bash examples/agent_rubric2/setup_webshop_small.sh"
        )

    # Generate the exact synthetic-goal ordering used by the runtime
    # interaction. The product-price sampling seed and SimServer's fixed goal
    # shuffle must match, otherwise a parquet goal_idx would address a
    # different instruction during training.
    if str(WEBSHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(WEBSHOP_ROOT))
    import web_agent_site.utils as webshop_utils

    webshop_utils.DEFAULT_FILE_PATH = str(small_products)
    webshop_utils.DEFAULT_ATTR_PATH = str(small_attributes)
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    old_random_state = random.getstate()
    random.seed(int(os.environ.get("WEBSHOP_SEED", "1234")))
    try:
        small_env = WebAgentTextEnv(
            observation_mode="text_rich",
            file_path=str(small_products),
            human_goals=False,
            num_products=1000,
            filter_goals=None,
            limit_goals=-1,
        )
        goals = [goal["instruction_text"] for goal in small_env.server.goals]
        small_env.close()
    finally:
        random.setstate(old_random_state)
else:
    if not GOAL_FILE.is_file():
        raise FileNotFoundError(
            f"WebShop goal file does not exist: {GOAL_FILE}. "
            "Set WEBSHOP_ROOT or WEBSHOP_GOAL_FILE to the official checkout."
        )

    with open(GOAL_FILE, encoding="utf-8") as f:
        goals = json.load(f)


print("Total goals:", len(goals))
print("Split protocol:", SPLIT_PROTOCOL)
print("Catalog mode:", CATALOG_MODE)


LEGACY_WEBSHOP_SYSTEM_PROMPT = """
You are a WebShop shopping agent.

Interact with the environment by selecting valid actions.

Your output must contain only:

ACTION: <action>

Examples:

ACTION: search[item name]

ACTION: click[item]
""".strip()


# Adapted from Trinity-RFT's WebShop_SYSTEM_PROMPT.  The environment contract,
# search/click rules, one-action constraint, and 15-step budget are preserved.
# Only the response wrapper is kept compatible with this repository's existing
# ACTION parser instead of changing it to Trinity's <action>...</action> tag.
TRINITY_WEBSHOP_SYSTEM_PROMPT = """
You are an agent interacting with a virtual text-based web shopping environment to test out your ability. Your job is to follow the Instruction provided and mimic the steps to buy the item that are closest to the Instruction provided.

## Action Format:
You should give both the action_name and action_arg like the format `action_name[action_arg]`. You can execute two types of actions, search and click.
- When the button `[button] Search [button_]` is available in the current observation, you can execute the action `search[xxx]` (you should type the query you want to search in the square brackets here).
- You can click buttons `[button] xxx [button_]` that is available in the current observation, by execute the action `click[xxx]`.

Below are some examples of action formats.
- ACTION: search[white shoes]
- ACTION: click[Buy Now]

## Notes:
At each step, you should first think then perform action to fulfill the instruction. After thinking, output the action on exactly one line in the format `ACTION: action_name[action_arg]`.
You should ALWAYS take one action each step.
You should finish the task and buy the item within 15 steps.
DO NOT try to interact with the user at anytime. Finish the task and buy the item by yourself.
""".strip()


WEBSHOP_PROMPT_VERSION = os.environ.get("WEBSHOP_PROMPT_VERSION", "trinity").strip().lower()
WEBSHOP_SYSTEM_PROMPTS = {
    "legacy": LEGACY_WEBSHOP_SYSTEM_PROMPT,
    "trinity": TRINITY_WEBSHOP_SYSTEM_PROMPT,
}
if WEBSHOP_PROMPT_VERSION not in WEBSHOP_SYSTEM_PROMPTS:
    raise ValueError(
        "WEBSHOP_PROMPT_VERSION must be 'trinity' or 'legacy', "
        f"got {WEBSHOP_PROMPT_VERSION!r}"
    )
WEBSHOP_SYSTEM_PROMPT = WEBSHOP_SYSTEM_PROMPTS[WEBSHOP_PROMPT_VERSION]


def build_rows(indices, split):

    rows = []

    for row_index, idx in enumerate(indices):

        instruction = goals[idx]
        task_id = f"webshop/{idx}"

        interaction_kwargs = {
            "name": "webshop",
            "goal_idx": idx,
            "max_steps": MAX_STEPS,
            "rubric": "",
            "inject_rubric": False,
            "use_interaction_reward": True,
        }
        if CATALOG_MODE == "small":
            interaction_kwargs.update(
                {
                    "use_small": True,
                    "human_goals": False,
                    "num_products": 1000,
                }
            )

        rows.append(
            {
                # These fields are consumed by VERL's AgentLoopWorker and
                # NaiveRewardManager.  Keep them even when the official
                # interaction reward is enabled so validation and fallback
                # scoring use the same dataset schema.
                "data_source": "webshop",
                "agent_name": "tool_agent",
                "goal_idx": idx,

                "instruction": instruction,

                "prompt": [
                    {
                        "role": "system",
                        "content": WEBSHOP_SYSTEM_PROMPT,
                    },

                    {
                        "role": "user",
                        "content":
                        instruction
                    }
                ],

                "ability": "long_horizon_agent",
                "reward_model": {
                    "style": "official_environment",
                    "ground_truth": "1.0",
                },
                "task_id": task_id,
                "task": instruction,
                "reference_answer_json": "{}",
                "rollout_trajectories_json": "[]",
                "rubric": "",
                "rubric_valid": False,


                "extra_info":
                {
                    "index": row_index,
                    "split": split,
                    "task_id": task_id,
                    "interaction_kwargs": interaction_kwargs,
                }
            }
        )

    return rows


if SPLIT_PROTOCOL == "public_repo":
    # Match the split used by public WebShop agent-training repositories:
    # goals 0--499 are held out for evaluation and every remaining goal is
    # available for training.  Evaluation deliberately reuses the test file;
    # no duplicate validation parquet is emitted.
    if len(goals) <= 500:
        raise ValueError(f"public_repo split requires more than 500 goals, got {len(goals)}")
    splits = {
        "test": range(0, 500),
        "train": range(500, len(goals)),
    }
else:
    # Preserve the earlier local protocol for exact reproduction of webshop2:
    # test=0--499, val=500--1499, train=1500--end.
    if len(goals) <= 1500:
        raise ValueError(f"canonical split requires more than 1500 goals, got {len(goals)}")
    splits = {
        "test": range(0, 500),
        "val": range(500, 1500),
        "train": range(1500, len(goals)),
    }

manifest = {
    "catalog_mode": CATALOG_MODE,
    "split_protocol": SPLIT_PROTOCOL,
    "source_goal_file": str(GOAL_FILE) if CATALOG_MODE == "full" else None,
    "source_products_file": str(small_products) if CATALOG_MODE == "small" else None,
    "source_attributes_file": str(small_attributes) if CATALOG_MODE == "small" else None,
    "search_index": str(small_index) if CATALOG_MODE == "small" else None,
    "total_goals": len(goals),
    "max_steps": MAX_STEPS,
    "prompt_version": WEBSHOP_PROMPT_VERSION,
    "splits": {
        name: {
            "start": indices.start,
            "stop": indices.stop,
            "count": len(indices),
        }
        for name, indices in splits.items()
    },
}
(OUT_DIR / "split_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

for name, indices in splits.items():

    rows = build_rows(indices, name)

    df = pd.DataFrame(rows)

    path = (
        OUT_DIR /
        f"webshop_{name}.parquet"
    )

    df.to_parquet(
        path,
        index=False
    )

    print(
        name,
        len(df),
        path
    )


print("Done")
print("WebShop prompt version:", WEBSHOP_PROMPT_VERSION)
print("Split manifest:", OUT_DIR / "split_manifest.json")
