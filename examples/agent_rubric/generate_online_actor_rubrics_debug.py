#!/usr/bin/env python3
"""Debug mixed-outcome actor rubrics with the exact online generation/filter path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

# This module imports VERL (and therefore PyTorch) before constructing vLLM.
# Forking a CUDA-initialized parent makes TP workers fail with
# "Cannot re-initialize CUDA in forked subprocess".
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.trainer.ppo.rubric_token_credit import (  # noqa: E402
    ALFWORLD_RUBRIC_PROMPT_VERSION,
    RUBRIC_FILTER_VERSION,
    build_agent_rubric_generation_messages,
    build_agent_rubric_judge_messages,
    compact_online_trajectory,
    filter_rubrics_by_outcome_correlation,
    format_rubric_items,
    parse_rubric_judge_response,
    parse_rubric_response,
    rubric_text_to_items,
    strip_thinking,
    validate_alfworld_rubric_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--rubrics-per-prompt", type=int, default=4)
    parser.add_argument("--min-rubrics", type=int, default=2)
    parser.add_argument("--min-reliable-rubrics", type=int, default=1)
    parser.add_argument("--reliability-threshold", type=float, default=0.5)
    parser.add_argument("--correctness-threshold", type=float, default=0.5)
    parser.add_argument("--max-candidate-rubrics", type=int, default=24)
    parser.add_argument("--max-trajectories", type=int, default=1)
    parser.add_argument("--max-trajectory-chars", type=int, default=6000)
    parser.add_argument("--max-rubric-chars", type=int, default=4000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_trajectories(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value or "[]")
    return list(value or [])


def trajectory_transcript(trajectory: dict[str, Any]) -> str:
    lines = [f"INITIAL OBSERVATION:\n{trajectory.get('initial_observation', '')}"]
    for step_index, step in enumerate(trajectory.get("steps", []), start=1):
        lines.append(f"STEP {step_index}")
        lines.append(f"CHOSEN ACTION: {step.get('action', '')}")
        lines.append(f"NEXT OBSERVATION: {step.get('environment_response', '')}")
    return "\n".join(lines)


def select_evidence(
    row: dict[str, Any], args: argparse.Namespace, trajectory_index: int
) -> list[dict[str, Any]]:
    trajectories = load_trajectories(row.get("rollout_trajectories_json"))
    trajectory = trajectories[trajectory_index]
    return [
        {
            "evidence_id": f"rollout_{trajectory_index + 1}",
            "transcript": compact_online_trajectory(
                trajectory_transcript(trajectory), max_chars=args.max_trajectory_chars
            ),
        }
    ]


def render(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> None:
    args = parse_args()
    if args.min_rubrics < 1 or args.rubrics_per_prompt < args.min_rubrics:
        raise ValueError("Require rubrics-per-prompt >= min-rubrics >= 1")
    if args.max_trajectories != 1:
        raise ValueError("Single-trajectory rubric generation requires --max-trajectories 1")
    dataset = load_dataset("parquet", data_files=args.input_parquet, split="train")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    rows = [dict(row) for row in dataset]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    states: list[dict[str, Any]] = []
    generation_prompts: list[str] = []
    generation_keys: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        trajectories = load_trajectories(row.get("rollout_trajectories_json"))
        outcomes = [float(item.get("official_reward", 0.0)) for item in trajectories]
        labels = [value > args.correctness_threshold for value in outcomes]
        outcome_group = "mixed" if any(labels) and not all(labels) else ("all_correct" if all(labels) else "all_wrong")
        state = {
            "outcomes": outcomes,
            "outcome_group": outcome_group,
            "candidates": [],
            "generations": [],
            "parsed_item_count": 0,
            "grounded_item_count": 0,
            "grounding_rejection_reasons": {},
            "judge_vectors": [],
        }
        states.append(state)
        if outcome_group != "mixed":
            continue
        for trajectory_index in range(len(trajectories)):
            evidence = select_evidence(row, args, trajectory_index)
            generation_prompts.append(
                render(
                    tokenizer,
                    build_agent_rubric_generation_messages(
                        str(row["task"]),
                        evidence,
                        args.rubrics_per_prompt,
                        benchmark="alfworld",
                    ),
                    args.enable_thinking,
                )
            )
            generation_keys.append((row_index, trajectory_index))

    for start in range(0, len(generation_prompts), args.batch_size):
        prompts = generation_prompts[start : start + args.batch_size]
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed + start,
            ),
            use_tqdm=False,
        )
        keys = generation_keys[start : start + args.batch_size]
        for (row_index, trajectory_index), prompt, output in zip(keys, prompts, outputs, strict=True):
            row = rows[row_index]
            state = states[row_index]
            raw_text = output.outputs[0].text
            parsed = parse_rubric_response(raw_text, args.rubrics_per_prompt, args.max_rubric_chars)
            evidence = select_evidence(row, args, trajectory_index)
            grounded, grounding = validate_alfworld_rubric_text(
                parsed,
                evidence,
                max_items=args.rubrics_per_prompt,
            )
            state["parsed_item_count"] += len(rubric_text_to_items(parsed))
            grounded_items = rubric_text_to_items(grounded)
            state["grounded_item_count"] += len(grounded_items)
            for reason, count in grounding["rejection_reasons"].items():
                reasons = state["grounding_rejection_reasons"]
                reasons[reason] = reasons.get(reason, 0) + int(count)
            existing_keys = {
                item.rsplit("; SOURCE", 1)[0].strip().lower() for item in state["candidates"]
            }
            for item in grounded_items:
                semantic_key = item.rsplit("; SOURCE", 1)[0].strip().lower()
                if semantic_key not in existing_keys and len(state["candidates"]) < args.max_candidate_rubrics:
                    state["candidates"].append(item)
                    existing_keys.add(semantic_key)
            state["generations"].append(
                {
                    "evidence_id": evidence[0]["evidence_id"],
                    "generation_without_thinking": strip_thinking(raw_text),
                    "grounding": grounding,
                    "prompt_chars": len(prompt),
                }
            )

    judge_prompts: list[str] = []
    judge_keys: list[tuple[int, int]] = []
    for row_index, (row, state) in enumerate(zip(rows, states, strict=True)):
        if state["outcome_group"] != "mixed" or len(state["candidates"]) < args.min_rubrics:
            continue
        trajectories = load_trajectories(row.get("rollout_trajectories_json"))
        for trajectory_index in range(len(trajectories)):
            transcript = select_evidence(row, args, trajectory_index)[0]["transcript"]
            judge_prompts.append(
                render(
                    tokenizer,
                    build_agent_rubric_judge_messages(
                        str(row["task"]), transcript, state["candidates"]
                    ),
                    args.enable_thinking,
                )
            )
            judge_keys.append((row_index, trajectory_index))

    judge_parse_successes = 0
    for start in range(0, len(judge_prompts), args.batch_size):
        prompts = judge_prompts[start : start + args.batch_size]
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=args.judge_temperature,
                top_p=1.0,
                max_tokens=args.judge_max_tokens,
                seed=args.seed + 100000 + start,
            ),
            use_tqdm=False,
        )
        keys = judge_keys[start : start + args.batch_size]
        for (row_index, _), output in zip(keys, outputs, strict=True):
            vector = parse_rubric_judge_response(
                output.outputs[0].text,
                len(states[row_index]["candidates"]),
            )
            if vector is not None:
                states[row_index]["judge_vectors"].append(vector)
                judge_parse_successes += 1

    audit = []
    for row, state in zip(rows, states, strict=True):
        trajectories = load_trajectories(row.get("rollout_trajectories_json"))
        selected: list[str] = []
        score_matrix: list[list[float]] = []
        filter_stats: dict[str, Any] = {
            "eligible_group": False,
            "outcome_group": state["outcome_group"],
            "candidate_items": len(state["candidates"]),
            "reliable_items": 0,
            "reliability_scores": [],
            "pearson_scores": [],
            "selected_indices": [],
            "constant_items": 0,
        }
        if (
            state["outcome_group"] == "mixed"
            and len(state["candidates"]) >= args.min_rubrics
            and len(state["judge_vectors"]) == len(trajectories)
        ):
            score_matrix = [list(values) for values in zip(*state["judge_vectors"], strict=True)]
            selected, filter_stats = filter_rubrics_by_outcome_correlation(
                state["candidates"],
                score_matrix,
                state["outcomes"],
                reliability_threshold=args.reliability_threshold,
                correctness_threshold=args.correctness_threshold,
            )
        valid = len(selected) >= args.min_reliable_rubrics
        rubric = format_rubric_items(selected, max_chars=args.max_rubric_chars) if valid else ""
        row["rubric"] = rubric
        row["rubric_valid"] = valid
        row["rubric_selected_count"] = len(selected) if valid else 0
        extra_info = dict(row.get("extra_info") or {})
        kwargs = dict(extra_info.get("interaction_kwargs") or {})
        kwargs["rubric"] = rubric
        kwargs["inject_rubric"] = valid
        extra_info["interaction_kwargs"] = kwargs
        row["extra_info"] = extra_info
        audit.append(
            {
                "task_id": row.get("task_id"),
                "task": row.get("task"),
                "outcomes": state["outcomes"],
                "outcome_group": state["outcome_group"],
                "valid": valid,
                "candidate_item_count": len(state["candidates"]),
                "item_count": len(selected),
                "parsed_item_count": state["parsed_item_count"],
                "grounded_item_count": state["grounded_item_count"],
                "grounding_rejection_reasons": state["grounding_rejection_reasons"],
                "candidate_rubrics": state["candidates"],
                "score_matrix": score_matrix,
                "filter": filter_stats,
                "rubric": rubric,
                "generations": state["generations"],
            }
        )

    output = Path(args.output_parquet).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))
    valid_count = sum(item["valid"] for item in audit)
    rejection_reason_totals: dict[str, int] = {}
    for item in audit:
        for reason, count in item["grounding_rejection_reasons"].items():
            rejection_reason_totals[reason] = rejection_reason_totals.get(reason, 0) + int(count)
    mixed_count = sum(item["outcome_group"] == "mixed" for item in audit)
    summary = {
        "model": args.model,
        "num_tasks": len(rows),
        "valid_tasks": valid_count,
        "selection_success_rate": valid_count / max(len(rows), 1),
        "mixed_outcome_tasks": mixed_count,
        "mixed_outcome_task_frac": mixed_count / max(len(rows), 1),
        "items_mean": sum(item["item_count"] for item in audit) / max(len(audit), 1),
        "candidate_items_mean": sum(item["candidate_item_count"] for item in audit) / max(len(audit), 1),
        "parsed_items_mean": sum(item["parsed_item_count"] for item in audit) / max(len(audit), 1),
        "grounded_item_acceptance_rate": sum(item["grounded_item_count"] for item in audit)
        / max(sum(item["parsed_item_count"] for item in audit), 1),
        "judge_parse_success_rate": judge_parse_successes / max(len(judge_prompts), 1),
        "reliability_threshold": args.reliability_threshold,
        "min_reliable_rubrics": args.min_reliable_rubrics,
        "grounding_rejection_reason_totals": rejection_reason_totals,
        "same_prompt_as_online_training": True,
        "rubric_prompt_version": ALFWORLD_RUBRIC_PROMPT_VERSION,
        "rubric_filter_version": RUBRIC_FILTER_VERSION,
        "generator_enable_thinking": args.enable_thinking,
        "rubric_text_strips_thinking": True,
        "actor_self_judge_used": True,
        "judge_receives_outcome_labels": False,
        "items": audit,
    }
    summary_path = output.with_suffix(".prompt-audit.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))
    print(f"Rubric parquet: {output}")
    print(f"Prompt audit: {summary_path}")


if __name__ == "__main__":
    main()
