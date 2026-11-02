#!/usr/bin/env python3
"""Build VERL parquet files from an existing official ALFWorld data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SPLIT_TO_ENV = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def read_task(traj_path: Path, data_root: Path, max_steps: int) -> dict[str, Any]:
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    annotations = data.get("turk_annotations", {}).get("anns", [])
    if not annotations:
        raise ValueError(f"No task annotation in {traj_path}")
    annotation = annotations[0]
    goal = " ".join(annotation["task_desc"].split())
    split = traj_path.relative_to(data_root / "json_2.1.1").parts[0]
    game_file = traj_path.with_name("game.tw-pddl")
    if not game_file.is_file():
        raise FileNotFoundError(game_file)

    expert_steps = []
    for index, step in enumerate(data.get("plan", {}).get("high_pddl", [])):
        discrete = step.get("discrete_action", {})
        expert_steps.append(
            {
                "step": index,
                "action_type": discrete.get("action", ""),
                "arguments": discrete.get("args", []),
                "description": annotation.get("high_descs", [""] * (index + 1))[index]
                if index < len(annotation.get("high_descs", []))
                else "",
            }
        )
    task_id = str(traj_path.parent.relative_to(data_root / "json_2.1.1"))
    task_type = data.get("task_type", "")
    answer = {
        "task_type": task_type,
        "goal": goal,
        "pddl_params": data.get("pddl_params", {}),
        "successful_milestones": expert_steps,
    }
    return {
        "data_source": "alfworld",
        "agent_name": "tool_agent",
        "prompt": [
            {
                "role": "system",
                "content": (
                    "You are acting in ALFWorld. Observe the environment and its valid actions, maintain task state, "
                    "and reply with exactly `ACTION: <one currently valid action>`."
                ),
            },
            {"role": "user", "content": f"Goal: {goal}"},
        ],
        "ability": "long_horizon_agent",
        "reward_model": {"style": "official_environment", "ground_truth": "1.0"},
        "task_id": task_id,
        "task": goal,
        "reference_answer_json": json.dumps(answer, ensure_ascii=False),
        "rollout_trajectories_json": "[]",
        "rubric": "",
        "rubric_valid": False,
        "extra_info": {
            "index": 0,
            "split": split,
            "evaluation_split": (
                "seen" if split == "valid_seen" else "unseen" if split == "valid_unseen" else split
            ),
            "task_type": task_type,
            "task_id": task_id,
            "interaction_kwargs": {
                "name": "alfworld",
                "game_file": str(game_file.resolve()),
                "split": SPLIT_TO_ENV[split],
                "max_steps": max_steps,
                "rubric": "",
                "inject_rubric": False,
                "use_interaction_reward": True,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing json_2.1.1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_TO_ENV))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    data_root = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if split not in SPLIT_TO_ENV:
            parser.error(f"Unsupported split {split!r}; choose from {sorted(SPLIT_TO_ENV)}")
        discovered = sorted((data_root / "json_2.1.1" / split).rglob("traj_data.json"))
        paths = [path for path in discovered if path.with_name("game.tw-pddl").is_file()]
        skipped = len(discovered) - len(paths)
        if skipped:
            print(f"Skipped {skipped} incomplete {split} trials without game.tw-pddl")
        if args.limit > 0:
            paths = paths[: args.limit]
        rows = [read_task(path, data_root, args.max_steps) for path in paths]
        for index, row in enumerate(rows):
            row["extra_info"]["index"] = index
        if not rows:
            raise RuntimeError(f"No ALFWorld tasks found for split {split!r} under {data_root}")
        output_path = output_dir / f"alfworld_{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), output_path)
        print(f"Saved {len(rows)} {split} tasks to {output_path}")


if __name__ == "__main__":
    main()
