#!/usr/bin/env python3
"""Export compact, thinking-free agent rubrics from a parquet file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.trainer.ppo.rubric_token_credit import strip_thinking  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def split_items(rubric: str) -> list[str]:
    return [
        re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        for line in rubric.splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_parquet).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    rows = pq.read_table(input_path).to_pylist()

    exported = []
    leaks = []
    for row in rows:
        rubric = strip_thinking(str(row.get("rubric") or ""))
        contains_think = bool(re.search(r"</?think\b", rubric, flags=re.IGNORECASE))
        if contains_think:
            leaks.append(str(row.get("task_id") or row.get("task") or len(exported)))
        items = split_items(rubric)
        exported.append(
            {
                "task_id": row.get("task_id"),
                "task": row.get("task"),
                "rubric_valid": bool(row.get("rubric_valid", bool(items))),
                "rubric_item_count": len(items),
                "contains_think_after_cleaning": contains_think,
                "rubrics": items,
            }
        )

    if leaks:
        raise RuntimeError(f"Thinking tags remain after cleaning for {len(leaks)} tasks: {leaks[:5]}")

    payload = {
        "source_parquet": str(input_path),
        "num_tasks": len(exported),
        "valid_tasks": sum(item["rubric_valid"] for item in exported),
        "all_thinking_removed": True,
        "rubrics": exported,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Thinking-free rubric JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
