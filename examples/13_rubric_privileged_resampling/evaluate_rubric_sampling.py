#!/usr/bin/env python3
"""Compare base-model sampling with and without precomputed rubric privilege."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from functools import cache
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

VERL_ROOT = Path(__file__).resolve().parents[2]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from examples.remote_chat_generation import (  # noqa: E402
    RemoteChatGenerator,
    completion_token_count,
)

RUBRIC_SUFFIX = """\

Selected solution rubric (privileged information):
{rubric}

Use the rubric as guidance for the critical reasoning steps. Solve the problem
yourself, show a coherent derivation, and put the final answer in \\boxed{{}}.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--inference-backend",
        choices=("vllm", "dashscope"),
        default="vllm",
    )
    parser.add_argument(
        "--remote-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--remote-api-key-env", default="RUBRIC_API_KEY")
    parser.add_argument("--remote-concurrency", type=int, default=8)
    parser.add_argument("--remote-timeout", type=float, default=180.0)
    parser.add_argument("--remote-max-retries", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--progress-file")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--rubric-column", default="rubric")
    parser.add_argument("--valid-column", default="rubric_valid")
    parser.add_argument("--default-data-source", default="math_dapo")
    parser.add_argument(
        "--selection-mode",
        choices=("opsd_filtered", "preselected"),
        default="opsd_filtered",
        help="Run OPSD reliability filtering, or trust that the parquet rubric is already selected",
    )
    parser.add_argument("--opsd-root", default="/workspace/verl_wsm/OPSD")
    parser.add_argument("--judge-api-key-env", default="RUBRIC_API_KEY")
    parser.add_argument(
        "--judge-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--judge-model", default="qwen3.5-plus")
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-concurrency", type=int, default=8)
    parser.add_argument("--selection-prompt-concurrency", type=int, default=2)
    parser.add_argument("--selection-num-samples", type=int, default=4)
    parser.add_argument("--reliability-threshold", type=float, default=0.5)
    parser.add_argument("--max-selected-rubrics", type=int, default=8)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-completions", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--http-client-log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_http_client_logging(level_name: str) -> None:
    """Suppress per-request OpenAI/httpx INFO lines while retaining failures."""
    level = getattr(logging, level_name)
    for logger_name in (
        "httpx",
        "httpcore",
        "openai",
        "openai._base_client",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(level)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def normalize_rubric(value: Any) -> str:
    """Normalize common parquet representations of a selected rubric."""
    if value is None:
        return ""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list, tuple)):
        value = value.tolist()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("selected_rubrics", "rubrics", "items"):
            if key in value:
                value = value[key]
                break
        else:
            return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))
    return str(value).strip()


def normalize_messages(raw_prompt: Any) -> list[dict[str, str]]:
    """Normalize VERL parquet prompts without importing the VERL package."""
    if hasattr(raw_prompt, "tolist") and not isinstance(
        raw_prompt, (str, bytes, dict, list, tuple)
    ):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]
    if isinstance(raw_prompt, dict):
        if "messages" in raw_prompt:
            raw_prompt = raw_prompt["messages"]
        elif "role" in raw_prompt and "content" in raw_prompt:
            raw_prompt = [raw_prompt]
        else:
            return [
                {
                    "role": "user",
                    "content": json.dumps(raw_prompt, ensure_ascii=False),
                }
            ]
    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)
    if not isinstance(raw_prompt, list):
        return [{"role": "user", "content": str(raw_prompt)}]

    messages = []
    for message in raw_prompt:
        if isinstance(message, dict):
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(part for part in text_parts if part)
            messages.append({"role": role, "content": str(content)})
        else:
            messages.append({"role": "user", "content": str(message)})
    return messages or [{"role": "user", "content": ""}]


def extract_problem_text(raw_prompt: Any) -> str:
    messages = normalize_messages(raw_prompt)
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"].strip()
    return messages[-1]["content"].strip()


def split_candidate_rubrics(rubric_text: str) -> list[str]:
    """Split the numbered block emitted by precompute_rubrics.py."""
    items = []
    for line in rubric_text.splitlines():
        item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if item and item.lower() not in {"rubric", "rubrics", "selected rubrics"}:
            items.append(item)
    if not items and rubric_text.strip():
        items.append(rubric_text.strip())
    return list(dict.fromkeys(items))


def load_opsd_pipeline(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load the original OPSD selector without copying or approximating it."""
    opsd_root = Path(args.opsd_root).expanduser().resolve()
    pipeline_file = opsd_root / "rubric_pipeline.py"
    if not pipeline_file.is_file():
        raise FileNotFoundError(
            f"OPSD rubric selector does not exist: {pipeline_file}. "
            "Set --opsd-root to the server OPSD repository."
        )
    api_key = os.environ.get(args.judge_api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{args.judge_api_key_env} must be exported for --selection-mode=opsd_filtered"
        )
    sys.path.insert(0, str(opsd_root))
    from rubric_pipeline import RubricItem, RubricPipeline

    configure_http_client_logging(args.http_client_log_level)
    pipeline = RubricPipeline(
        judge_api_key=api_key,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        reliability_threshold=args.reliability_threshold,
        max_concurrent_judge=args.judge_max_concurrency,
        judge_timeout=args.judge_timeout,
        judge_temperature=0.0,
        problem_based=True,
        experiment_mode="filtered",
    )
    return pipeline, RubricItem


async def select_rubrics_for_batch(
    pipeline: Any,
    rubric_item_cls: Any,
    batch_rows: list[dict[str, Any]],
    selection_texts: list[list[str]],
    selection_correct: list[list[bool]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run the original OPSD judge, reliability score, and leakage filter."""
    import numpy as np

    semaphore = asyncio.Semaphore(
        max(1, min(args.selection_prompt_concurrency, len(batch_rows)))
    )

    async def select_one(
        row: dict[str, Any],
        trajectories: list[str],
        correctness: list[bool],
    ) -> dict[str, Any]:
        candidate_descriptions = split_candidate_rubrics(row["rubric"])
        candidate_items = [
            rubric_item_cls(
                name=f"Criterion {index}",
                description=description,
                source_trajectory_idx=-1,
            )
            for index, description in enumerate(candidate_descriptions, start=1)
        ]
        y = np.asarray(correctness, dtype=np.float32)
        async with semaphore:
            r_matrix = await pipeline.judge_all_trajectories(
                row["problem"], candidate_items, trajectories
            )
        scored = pipeline.compute_reliability_scores(candidate_items, r_matrix, y)
        selected = pipeline.filter_reliable_rubrics(
            scored,
            row["ground_truth"],
            y_all_wrong=bool(y.sum() == 0),
        )[: args.max_selected_rubrics]
        selected_text = "\n".join(
            f"{index}. {rubric.description}"
            for index, rubric in enumerate(selected, start=1)
        )
        return {
            "mode": "opsd_filtered",
            "eligible": bool(selected),
            "outcome_group": (
                "all_wrong"
                if y.sum() == 0
                else "all_correct"
                if y.sum() == len(y)
                else "mixed"
            ),
            "rollout_correct": correctness,
            "candidate_count": len(candidate_items),
            "selected_count": len(selected),
            "selected_rubric": selected_text,
            "selected_scores": [float(rubric.score) for rubric in selected],
            "selected_satisfaction": [rubric.R_i for rubric in selected],
        }

    tasks = [
        select_one(row, trajectories, correctness)
        for row, trajectories, correctness in zip(
            batch_rows, selection_texts, selection_correct, strict=True
        )
    ]
    return list(await asyncio.gather(*tasks))


def extract_ground_truth(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    for key in ("ground_truth", "answer", "target"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("No ground truth found in reward_model.ground_truth, ground_truth, answer, or target")


def append_rubric(messages: list[dict[str, str]], rubric: str) -> list[dict[str, str]]:
    conditioned = copy.deepcopy(messages)
    suffix = RUBRIC_SUFFIX.format(rubric=rubric)
    for message in reversed(conditioned):
        if message["role"] == "user":
            message["content"] = f"{message['content']}{suffix}"
            break
    else:
        conditioned.append({"role": "user", "content": suffix.strip()})
    return conditioned


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


@cache
def load_standalone_reward_module(module_name: str) -> Any:
    """Load a reward file directly, bypassing verl/__init__.py and tensordict."""
    reward_path = (
        Path(__file__).resolve().parents[2]
        / "verl"
        / "utils"
        / "reward_score"
        / f"{module_name}.py"
    )
    if not reward_path.is_file():
        raise FileNotFoundError(f"Reward implementation does not exist: {reward_path}")
    spec = importlib.util.spec_from_file_location(
        f"_rubric_sampling_{module_name}", reward_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reward implementation: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_completion(
    text: str,
    *,
    data_source: str,
    ground_truth: str,
    extra_info: Any,
) -> bool:
    """Use VERL's math graders without importing the VERL runtime package."""
    del extra_info
    math_reward_sources = {
        "lighteval/MATH",
        "DigitalLearningGmbH/MATH-lighteval",
        "MATH500",
        "AMC23",
        "AIME24",
        "AIME25",
        "aime24",
        "aime25",
        "aime_2024",
        "aime_2025",
    }
    if data_source in math_reward_sources:
        result = load_standalone_reward_module("math_reward").compute_score(
            text, ground_truth
        )
    elif data_source in {"math_dapo", "math", "math_dapo_reasoning"} or data_source.startswith(
        "aime"
    ):
        result = load_standalone_reward_module("math_dapo").compute_score(
            text, ground_truth
        )
    else:
        raise NotImplementedError(
            f"Experiment 13 only supports VERL math reward sources, got {data_source=!r}"
        )
    if isinstance(result, dict):
        if "acc" in result:
            return bool(result["acc"])
        result = result.get("score", 0.0)
    return float(result) > 0.0


def extract_vote_answer(text: str) -> str:
    math_reward = load_standalone_reward_module("math_reward")

    try:
        boxed = math_reward.last_boxed_only_string(text)
        if boxed is None:
            return "[INVALID]"
        return math_reward.strip_string(math_reward.remove_boxed(boxed))
    except Exception:
        return "[INVALID]"


def majority_vote_correct(answers: list[str], ground_truth: str) -> bool:
    math_reward = load_standalone_reward_module("math_reward")

    valid = [answer for answer in answers if answer != "[INVALID]"]
    if not valid:
        return False
    majority_answer = Counter(valid).most_common(1)[0][0]
    return bool(math_reward.is_equiv(majority_answer, ground_truth))


def stable_problem_id(problem: str, ground_truth: str, rubric: str) -> str:
    payload = "\0".join((problem, ground_truth, rubric))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_config_signature(args: argparse.Namespace, input_path: Path) -> tuple[str, dict[str, Any]]:
    stat = input_path.stat()
    config = {
        "evaluation_schema_version": 2,
        "input_parquet": str(input_path),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "model": args.model,
        "inference_backend": args.inference_backend,
        "remote_base_url": args.remote_base_url,
        "remote_concurrency": args.remote_concurrency,
        "remote_timeout": args.remote_timeout,
        "remote_max_retries": args.remote_max_retries,
        "prompt_column": args.prompt_column,
        "rubric_column": args.rubric_column,
        "valid_column": args.valid_column,
        "default_data_source": args.default_data_source,
        "selection_mode": args.selection_mode,
        "opsd_root": args.opsd_root,
        "judge_base_url": args.judge_base_url,
        "judge_model": args.judge_model,
        "judge_timeout": args.judge_timeout,
        "judge_max_concurrency": args.judge_max_concurrency,
        "selection_prompt_concurrency": args.selection_prompt_concurrency,
        "selection_num_samples": args.selection_num_samples,
        "reliability_threshold": args.reliability_threshold,
        "max_selected_rubrics": args.max_selected_rubrics,
        "limit": args.limit,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "save_completions": args.save_completions,
    }
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), config


def load_progress(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        try:
            metadata = json.loads(first)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid progress metadata: {path}") from exc
        if metadata.get("type") != "metadata" or metadata.get("signature") != signature:
            raise ValueError(
                f"Progress file was created with different settings: {path}. "
                "Use --overwrite or choose another --progress-file."
            )
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[str(record["problem_id"])] = record
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid progress record at {path}:{line_number}") from exc
    return records


def initialize_progress(path: Path, signature: str, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"type": "metadata", "signature": signature, "config": config},
                ensure_ascii=False,
            )
            + "\n"
        )


def append_progress(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def bootstrap_delta_ci(
    baseline_rates: list[float],
    rubric_rates: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if not baseline_rates or samples <= 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    deltas = []
    count = len(baseline_rates)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        delta = mean(rubric_rates[index] - baseline_rates[index] for index in indices)
        deltas.append(delta * 100.0)
    deltas.sort()
    low = deltas[int(0.025 * (samples - 1))]
    high = deltas[int(0.975 * (samples - 1))]
    return [low, high]


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    eligible_records = [record for record in records if record["selection"]["eligible"]]
    problem_count = len(records)
    selection_groups = Counter(
        record["selection"].get("outcome_group", "preselected") for record in records
    )
    selection_summary = {
        "attempted_problems": problem_count,
        "eligible_problems": len(eligible_records),
        "selection_success_rate_pct": 100.0 * len(eligible_records) / problem_count,
        "outcome_groups": dict(selection_groups),
        "candidate_rubrics_mean": mean(
            record["selection"]["candidate_count"] for record in records
        ),
        "selected_rubrics_mean_over_attempted": mean(
            record["selection"]["selected_count"] for record in records
        ),
        "selected_rubrics_mean_over_eligible": (
            mean(record["selection"]["selected_count"] for record in eligible_records)
            if eligible_records
            else 0.0
        ),
    }

    def condition_summary(
        selected_records: list[dict[str, Any]], name: str
    ) -> dict[str, Any] | None:
        if not selected_records:
            return None
        selected_problem_count = len(selected_records)
        solution_count = selected_problem_count * args.num_samples
        correct = sum(sum(record[name]["correct"]) for record in selected_records)
        passes = sum(any(record[name]["correct"]) for record in selected_records)
        majority = sum(record[name]["majority_vote_correct"] for record in selected_records)
        formatted = sum(
            answer != "[INVALID]"
            for record in selected_records
            for answer in record[name]["answers"]
        )
        token_counts = [
            count
            for record in selected_records
            for count in record[name]["completion_tokens"]
        ]
        return {
            "correct_solutions": correct,
            "total_solutions": solution_count,
            "accuracy_pct": 100.0 * correct / solution_count,
            "pass_at_n": passes,
            "pass_at_n_pct": 100.0 * passes / selected_problem_count,
            "majority_vote_at_n": majority,
            "majority_vote_at_n_pct": 100.0 * majority / selected_problem_count,
            "formatted_count": formatted,
            "format_rate_pct": 100.0 * formatted / solution_count,
            "completion_tokens_mean": mean(token_counts),
        }

    def comparison_summary(
        selected_records: list[dict[str, Any]], *, bootstrap_seed: int
    ) -> dict[str, Any] | None:
        if not selected_records:
            return None
        baseline = condition_summary(selected_records, "baseline")
        rubric = condition_summary(selected_records, "rubric")
        assert baseline is not None and rubric is not None
        baseline_rates = [
            mean(record["baseline"]["correct"]) for record in selected_records
        ]
        rubric_rates = [mean(record["rubric"]["correct"]) for record in selected_records]
        wins = sum(
            rubric_rate > baseline_rate
            for baseline_rate, rubric_rate in zip(
                baseline_rates, rubric_rates, strict=True
            )
        )
        losses = sum(
            rubric_rate < baseline_rate
            for baseline_rate, rubric_rate in zip(
                baseline_rates, rubric_rates, strict=True
            )
        )
        baseline_only = 0
        rubric_only = 0
        for record in selected_records:
            for baseline_correct, rubric_correct in zip(
                record["baseline"]["correct"], record["rubric"]["correct"], strict=True
            ):
                baseline_only += baseline_correct and not rubric_correct
                rubric_only += rubric_correct and not baseline_correct
        return {
            "accuracy_delta_pp": rubric["accuracy_pct"] - baseline["accuracy_pct"],
            "pass_at_n_delta_pp": rubric["pass_at_n_pct"] - baseline["pass_at_n_pct"],
            "majority_vote_delta_pp": (
                rubric["majority_vote_at_n_pct"] - baseline["majority_vote_at_n_pct"]
            ),
            "problem_wins_ties_losses": {
                "wins": wins,
                "ties": len(selected_records) - wins - losses,
                "losses": losses,
            },
            "paired_sample_flips": {
                "baseline_correct_rubric_wrong": baseline_only,
                "baseline_wrong_rubric_correct": rubric_only,
            },
            "accuracy_delta_bootstrap_95pct_ci_pp": bootstrap_delta_ci(
                baseline_rates,
                rubric_rates,
                samples=args.bootstrap_samples,
                seed=bootstrap_seed,
            ),
        }

    baseline = condition_summary(records, "baseline")
    rubric = condition_summary(records, "rubric")
    comparison = comparison_summary(records, bootstrap_seed=args.seed + 100003)
    eligible_baseline = condition_summary(eligible_records, "baseline")
    eligible_rubric = condition_summary(eligible_records, "rubric")
    eligible_comparison = comparison_summary(
        eligible_records, bootstrap_seed=args.seed + 200003
    )

    return {
        "experiment": "13_rubric_privileged_resampling",
        "selection": selection_summary,
        "num_problems": problem_count,
        "num_samples_per_condition": args.num_samples,
        "baseline": baseline,
        "rubric_privileged": rubric,
        "comparison": comparison,
        "eligible_subset": {
            "num_problems": len(eligible_records),
            "baseline": eligible_baseline,
            "rubric_privileged": eligible_rubric,
            "comparison": eligible_comparison,
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    selection = summary["selection"]
    print("\n" + "=" * 78)
    print(
        "Rubric selection: "
        f"{selection['eligible_problems']}/{selection['attempted_problems']} eligible "
        f"({selection['selection_success_rate_pct']:.2f}%)"
    )
    print(f"Selection outcome groups: {selection['outcome_groups']}")
    baseline = summary["baseline"]
    rubric = summary["rubric_privileged"]
    comparison = summary["comparison"]
    n = summary["num_samples_per_condition"]
    print(
        "Full-set rubric/fallback evaluation: "
        f"{summary['num_problems']} problems, N={n}"
    )
    print("-" * 78)
    print(
        f"{'condition':<20} {'Accuracy':>10} {f'Pass@{n}':>10} "
        f"{f'Majority@{n}':>12} {'Format':>10}"
    )
    print(
        f"{'baseline':<20} {baseline['accuracy_pct']:>9.2f}% "
        f"{baseline['pass_at_n_pct']:>9.2f}% "
        f"{baseline['majority_vote_at_n_pct']:>11.2f}% "
        f"{baseline['format_rate_pct']:>9.2f}%"
    )
    print(
        f"{'rubric / fallback':<20} {rubric['accuracy_pct']:>9.2f}% "
        f"{rubric['pass_at_n_pct']:>9.2f}% "
        f"{rubric['majority_vote_at_n_pct']:>11.2f}% "
        f"{rubric['format_rate_pct']:>9.2f}%"
    )
    ci = comparison["accuracy_delta_bootstrap_95pct_ci_pp"]
    print("-" * 78)
    print(
        f"Full-set accuracy delta: {comparison['accuracy_delta_pp']:+.2f} pp "
        f"(paired problem bootstrap 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] pp)"
    )
    print(f"Problem wins/ties/losses: {comparison['problem_wins_ties_losses']}")
    eligible_subset = summary["eligible_subset"]
    if eligible_subset["comparison"] is not None:
        eligible_comparison = eligible_subset["comparison"]
        eligible_ci = eligible_comparison["accuracy_delta_bootstrap_95pct_ci_pp"]
        print(
            "Eligible-subset diagnostic: "
            f"{eligible_subset['num_problems']} problems, "
            f"delta {eligible_comparison['accuracy_delta_pp']:+.2f} pp "
            f"(95% CI [{eligible_ci[0]:+.2f}, {eligible_ci[1]:+.2f}] pp)"
        )
    else:
        print("Eligible-subset diagnostic: no rubric passed selection.")
    print("=" * 78)


def main() -> None:
    args = parse_args()
    configure_http_client_logging(args.http_client_log_level)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.selection_num_samples <= 0:
        raise ValueError("--selection-num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.selection_prompt_concurrency <= 0:
        raise ValueError("--selection-prompt-concurrency must be positive")
    if args.max_selected_rubrics <= 0:
        raise ValueError("--max-selected-rubrics must be positive")
    if args.remote_concurrency <= 0:
        raise ValueError("--remote-concurrency must be positive")
    if args.remote_timeout <= 0:
        raise ValueError("--remote-timeout must be positive")
    if args.remote_max_retries < 0:
        raise ValueError("--remote-max-retries cannot be negative")
    if args.limit == 0:
        raise ValueError("--limit must be positive or -1 for all eligible rows")
    if args.inference_backend == "vllm" and args.max_model_len <= args.max_tokens:
        raise ValueError("--max-model-len must be larger than --max-tokens")

    input_path = Path(args.input_parquet).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    progress_path = (
        Path(args.progress_file).expanduser().resolve()
        if args.progress_file
        else output_path.with_suffix(".progress.jsonl")
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_path}")
    if args.overwrite:
        progress_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(input_path), split="train")
    required_columns = {args.prompt_column, args.rubric_column}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise KeyError(
            f"Missing required parquet columns {sorted(missing_columns)}. "
            f"Available columns: {dataset.column_names}"
        )
    if args.valid_column and args.valid_column not in dataset.column_names:
        raise KeyError(
            f"Validity column {args.valid_column!r} is absent. "
            "Set --valid-column '' only if every non-empty rubric has already been selected."
        )

    rows = []
    seen_problems: set[str] = set()
    invalid_count = 0
    duplicate_count = 0
    for row_index, row in enumerate(dataset):
        rubric = normalize_rubric(row[args.rubric_column])
        is_valid = _as_bool(row[args.valid_column]) if args.valid_column else bool(rubric)
        if not is_valid or not rubric:
            invalid_count += 1
            continue
        problem = extract_problem_text(row[args.prompt_column])
        problem_hash = hashlib.sha256(problem.encode("utf-8")).hexdigest()
        if problem_hash in seen_problems:
            duplicate_count += 1
            continue
        seen_problems.add(problem_hash)
        ground_truth = extract_ground_truth(row)
        rows.append(
            {
                "row_index": row_index,
                "problem": problem,
                "ground_truth": ground_truth,
                "rubric": rubric,
                "messages": normalize_messages(row[args.prompt_column]),
                "data_source": str(row.get("data_source") or args.default_data_source),
                "extra_info": row.get("extra_info"),
                "problem_id": stable_problem_id(problem, ground_truth, rubric),
            }
        )
        if args.limit > 0 and len(rows) >= args.limit:
            break
    if not rows:
        raise RuntimeError("No valid, non-empty rubrics were selected from the input parquet")

    signature, signature_config = build_config_signature(args, input_path)
    completed = load_progress(progress_path, signature)
    initialize_progress(progress_path, signature, signature_config)
    pending = [row for row in rows if row["problem_id"] not in completed]

    print(f"Input rows: {len(dataset)}")
    print(f"Selected unique problems: {len(rows)}")
    print(f"Skipped invalid/empty rubrics before limit: {invalid_count}")
    print(f"Skipped duplicate problems before limit: {duplicate_count}")
    print(f"Resumed problems: {len(rows) - len(pending)}")
    print(f"Pending problems: {len(pending)}")

    if pending:
        llm = None
        tokenizer = None
        sampling_params_cls = None
        remote_generator = None
        if args.inference_backend == "vllm":
            from vllm import LLM, SamplingParams

            # Construct vLLM before importing the old OPSD module. rubric_pipeline.py
            # imports PyTorch; importing it first can initialize CUDA in this parent
            # process and make vLLM's tensor-parallel workers fail under multiprocessing.
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                dtype="bfloat16",
                trust_remote_code=True,
            )
            tokenizer = llm.get_tokenizer()
            sampling_params_cls = SamplingParams
        else:
            remote_api_key = os.environ.get(args.remote_api_key_env, "").strip()
            if not remote_api_key:
                raise RuntimeError(
                    f"Export {args.remote_api_key_env} for DashScope generation."
                )
            remote_generator = RemoteChatGenerator(
                model=args.model,
                api_key=remote_api_key,
                base_url=args.remote_base_url,
                concurrency=args.remote_concurrency,
                timeout=args.remote_timeout,
                max_retries=args.remote_max_retries,
            )

        def generate_messages(
            messages_batch: list[list[dict[str, str]]],
            sampling_kwargs: dict[str, Any],
            *,
            stage: str,
        ) -> list[Any]:
            total_generations = len(messages_batch) * sampling_kwargs["n"]
            started_at = time.monotonic()
            print(
                f"[{stage}] starting {len(messages_batch)} prompts × "
                f"{sampling_kwargs['n']} = {total_generations} generations "
                f"via {args.inference_backend}",
                flush=True,
            )
            if remote_generator is not None:
                outputs = remote_generator.generate(
                    messages_batch,
                    temperature=sampling_kwargs["temperature"],
                    top_p=sampling_kwargs["top_p"],
                    top_k=sampling_kwargs["top_k"],
                    max_tokens=sampling_kwargs["max_tokens"],
                    n=sampling_kwargs["n"],
                    seed=sampling_kwargs["seed"],
                    enable_thinking=args.enable_thinking,
                )
            else:
                assert llm is not None
                assert tokenizer is not None
                assert sampling_params_cls is not None
                prompts = [
                    render_prompt(tokenizer, messages, args.enable_thinking)
                    for messages in messages_batch
                ]
                outputs = llm.generate(
                    prompts,
                    sampling_params_cls(**sampling_kwargs),
                    use_tqdm=False,
                )
            print(
                f"[{stage}] finished {total_generations} generations in "
                f"{time.monotonic() - started_at:.1f}s",
                flush=True,
            )
            return outputs

        pipeline = None
        rubric_item_cls = None
        if args.selection_mode == "opsd_filtered":
            pipeline, rubric_item_cls = load_opsd_pipeline(args)

        for start in range(0, len(pending), args.batch_size):
            batch_rows = pending[start : start + args.batch_size]
            original_messages = [row["messages"] for row in batch_rows]
            print(
                f"[Batch] problems {start + 1}-"
                f"{start + len(batch_rows)} of {len(pending)}",
                flush=True,
            )

            selection_completions: list[list[str]] = [[] for _ in batch_rows]
            if args.selection_mode == "opsd_filtered":
                selection_outputs = generate_messages(
                    original_messages,
                    {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_tokens": args.max_tokens,
                        "n": args.selection_num_samples,
                        "seed": args.seed + 1_000_000 + start,
                    },
                    stage="selection rollout",
                )
                selection_completions = [
                    [output.text for output in request_output.outputs]
                    for request_output in selection_outputs
                ]
                selection_correct = [
                    [
                        score_completion(
                            text,
                            data_source=row["data_source"],
                            ground_truth=row["ground_truth"],
                            extra_info=row["extra_info"],
                        )
                        for text in texts
                    ]
                    for row, texts in zip(batch_rows, selection_completions, strict=True)
                ]
                print(
                    f"[rubric filter] judging {len(batch_rows)} problems",
                    flush=True,
                )
                selections = asyncio.run(
                    select_rubrics_for_batch(
                        pipeline,
                        rubric_item_cls,
                        batch_rows,
                        selection_completions,
                        selection_correct,
                        args,
                    )
                )
                print(
                    "[rubric filter] eligible "
                    f"{sum(selection['eligible'] for selection in selections)}/"
                    f"{len(selections)} problems",
                    flush=True,
                )
            else:
                selections = []
                for row in batch_rows:
                    candidate_count = len(split_candidate_rubrics(row["rubric"]))
                    selections.append(
                        {
                            "mode": "preselected",
                            "eligible": True,
                            "outcome_group": "preselected",
                            "candidate_count": candidate_count,
                            "selected_count": candidate_count,
                            "selected_rubric": row["rubric"],
                            "selected_scores": [],
                            "selected_satisfaction": [],
                        }
                    )

            eligible_indices = [
                index for index, selection in enumerate(selections) if selection["eligible"]
            ]
            eligible_rows = [batch_rows[index] for index in eligible_indices]
            eligible_selections = [selections[index] for index in eligible_indices]
            ineligible_rows = [
                row
                for row, selection in zip(batch_rows, selections, strict=True)
                if not selection["eligible"]
            ]
            baseline_outputs_by_id: dict[str, Any] = {}
            rubric_outputs_by_id: dict[str, Any] = {}
            sampling_kwargs = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
                "n": args.num_samples,
                "seed": args.seed + start,
            }
            if eligible_rows:
                eligible_baseline_messages = [
                    row["messages"] for row in eligible_rows
                ]
                rubric_messages = [
                    append_rubric(row["messages"], selection["selected_rubric"])
                    for row, selection in zip(
                        eligible_rows, eligible_selections, strict=True
                    )
                ]
                baseline_outputs = generate_messages(
                    eligible_baseline_messages,
                    sampling_kwargs,
                    stage="eligible baseline",
                )
                rubric_outputs = generate_messages(
                    rubric_messages,
                    sampling_kwargs,
                    stage="rubric privilege",
                )
                baseline_outputs_by_id.update(
                    {
                        row["problem_id"]: baseline_output
                        for row, baseline_output in zip(
                            eligible_rows, baseline_outputs, strict=True
                        )
                    }
                )
                rubric_outputs_by_id.update(
                    {
                        row["problem_id"]: rubric_output
                        for row, rubric_output in zip(
                            eligible_rows, rubric_outputs, strict=True
                        )
                    }
                )
            if ineligible_rows:
                ineligible_baseline_messages = [
                    row["messages"] for row in ineligible_rows
                ]
                ineligible_sampling_kwargs = {
                    **sampling_kwargs,
                    "seed": args.seed + 500_000 + start,
                }
                ineligible_baseline_outputs = generate_messages(
                    ineligible_baseline_messages,
                    ineligible_sampling_kwargs,
                    stage="fallback baseline",
                )
                baseline_outputs_by_id.update(
                    {
                        row["problem_id"]: baseline_output
                        for row, baseline_output in zip(
                            ineligible_rows, ineligible_baseline_outputs, strict=True
                        )
                    }
                )

            new_records = []
            for row, selection, selector_texts in zip(
                batch_rows, selections, selection_completions, strict=True
            ):
                record: dict[str, Any] = {
                    "problem_id": row["problem_id"],
                    "row_index": row["row_index"],
                    "data_source": row["data_source"],
                    "selection": selection,
                }
                condition_outputs = [
                    ("baseline", baseline_outputs_by_id[row["problem_id"]])
                ]
                if selection["eligible"]:
                    condition_outputs.append(
                        ("rubric", rubric_outputs_by_id[row["problem_id"]])
                    )
                for condition, request_output in condition_outputs:
                    texts = [output.text for output in request_output.outputs]
                    correct = [
                        score_completion(
                            text,
                            data_source=row["data_source"],
                            ground_truth=row["ground_truth"],
                            extra_info=row["extra_info"],
                        )
                        for text in texts
                    ]
                    answers = [extract_vote_answer(text) for text in texts]
                    condition_record: dict[str, Any] = {
                        "correct": correct,
                        "answers": answers,
                        "majority_vote_correct": majority_vote_correct(
                            answers, row["ground_truth"]
                        ),
                        "completion_tokens": [
                            completion_token_count(output)
                            for output in request_output.outputs
                        ],
                        "fallback_to_baseline": False,
                    }
                    if args.save_completions:
                        condition_record["completions"] = texts
                    record[condition] = condition_record
                if not selection["eligible"]:
                    record["rubric"] = copy.deepcopy(record["baseline"])
                    record["rubric"]["fallback_to_baseline"] = True
                if args.save_completions:
                    record["problem"] = row["problem"]
                    record["ground_truth"] = row["ground_truth"]
                    record["candidate_rubric_text"] = row["rubric"]
                    record["selection_completions"] = selector_texts
                new_records.append(record)
                completed[row["problem_id"]] = record
            append_progress(progress_path, new_records)
            print(
                f"Completed {min(start + len(batch_rows), len(pending))}/"
                f"{len(pending)} pending problems",
                flush=True,
            )

    ordered_records = [completed[row["problem_id"]] for row in rows]
    summary = summarize(ordered_records, args)
    summary["config"] = signature_config
    summary["progress_file"] = str(progress_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print_summary(summary)
    print(f"Summary saved to: {output_path}")
    print(f"Resumable paired records: {progress_path}")


if __name__ == "__main__":
    main()
