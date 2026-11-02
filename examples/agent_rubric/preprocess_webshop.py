#!/usr/bin/env python3
"""Build fixed-goal VERL parquet files from an official WebShop checkout."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SPLIT_RANGES = {"test": (0, 500), "eval": (500, 1500), "train": (1500, None)}


def safe_goal_reference(goal: dict[str, Any]) -> dict[str, Any]:
    """Retain validator-only constraints without exposing them in the policy prompt."""
    return {
        "instruction_text": goal.get("instruction_text", ""),
        "query": goal.get("query", ""),
        "attributes": goal.get("attributes", []),
        "options": goal.get("goal_options", {}),
        "price_upper": goal.get("price_upper"),
        "target_asin": goal.get("asin"),
        "category": goal.get("category"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", required=True, help="Official princeton-nlp/WebShop checkout")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--file-path", default="", help="Optional WebShop products JSON path")
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_RANGES))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    webshop_root = str(Path(args.webshop_root).expanduser().resolve())
    sys.path.insert(0, webshop_root)
    try:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    except ImportError as exc:
        raise RuntimeError(f"Cannot import official WebShop from {webshop_root}") from exc

    kwargs: dict[str, Any] = {
        "observation_mode": "text_rich",
        "filter_goals": None,
        "limit_goals": -1,
    }
    if args.file_path:
        kwargs["file_path"] = str(Path(args.file_path).expanduser().resolve())
    # WebShop samples per-goal price ceilings while constructing the server.
    # Use the same seed as the online interaction so a fixed goal_idx denotes
    # exactly the same instruction and reward constraints in both phases.
    random.seed(args.seed)
    env = WebAgentTextEnv(**kwargs)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        total_goals = len(env.server.goals)
        for split in args.splits:
            if split not in SPLIT_RANGES:
                parser.error(f"Unsupported split {split!r}; choose from {sorted(SPLIT_RANGES)}")
            start, stop = SPLIT_RANGES[split]
            stop = total_goals if stop is None else min(stop, total_goals)
            goal_indices = list(range(start, stop))
            if args.limit > 0:
                goal_indices = goal_indices[: args.limit]
            rows = []
            for row_index, goal_idx in enumerate(goal_indices):
                env.reset(goal_idx)
                session = env.server.user_sessions[env.session]
                goal = session["goal"]
                instruction = env.instruction_text
                reference = safe_goal_reference(goal)
                task_id = f"webshop/{goal_idx}"
                rows.append(
                    {
                        "data_source": "webshop",
                        "agent_name": "tool_agent",
                        "prompt": [
                            {
                                "role": "system",
                                "content": (
                                    "You are acting in WebShop. Search, inspect products, select required options, "
                                    "and buy the best item. Reply exactly `ACTION: search[...]` or "
                                    "`ACTION: click[...]`."
                                ),
                            },
                            {"role": "user", "content": f"Shopping goal: {instruction}"},
                        ],
                        "ability": "long_horizon_agent",
                        "reward_model": {"style": "official_environment", "ground_truth": "1.0"},
                        "task_id": task_id,
                        "task": instruction,
                        "reference_answer_json": json.dumps(reference, ensure_ascii=False),
                        "rollout_trajectories_json": "[]",
                        "rubric": "",
                        "rubric_valid": False,
                        "extra_info": {
                            "index": row_index,
                            "split": split,
                            "task_id": task_id,
                            "interaction_kwargs": {
                                "name": "webshop",
                                "goal_idx": goal_idx,
                                "max_steps": args.max_steps,
                                "rubric": "",
                                "inject_rubric": False,
                                "use_interaction_reward": True,
                            },
                        },
                    }
                )
            if not rows:
                raise RuntimeError(f"No WebShop goals found for split {split!r}")
            output_path = output_dir / f"webshop_{split}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), output_path)
            print(f"Saved {len(rows)} {split} goals to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
