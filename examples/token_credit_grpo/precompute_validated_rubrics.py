#!/usr/bin/env python3
"""Generate and answer-aware validate problem rubrics, then write a training parquet.

The rubric generator sees only the problem. A separate judge sees the problem,
ground truth, and candidate criteria, and validates every criterion for
mathematical correctness, relevance, actionability, and answer leakage. Failed
problems are regenerated for several rounds. Progress is resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALIDATION_VERSION = 1
MIN_REQUIRED_RUBRICS = 4
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

VERL_ROOT = Path(__file__).resolve().parents[2]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))


JUDGE_SYSTEM_PROMPT = """\
You are a strict validator of mathematical reasoning rubrics.

The rubric generator was not shown the answer. You are given the problem, the
ground truth, and numbered candidate criteria. Evaluate every candidate
independently. A criterion passes only if all of the following are true:
1. It is mathematically correct for this exact problem.
2. It is relevant and problem-specific, not generic advice.
3. It gives an actionable reasoning step or decision check.
4. It does not reveal, encode, or strongly hint at the final answer.
5. It does not assume the conclusion and is understandable without hidden context.

Do not rewrite or repair a failed criterion. Return JSON only:
{"items": [{"index": 1, "correct": true, "relevant": true,
"actionable": true, "no_answer_leak": true, "self_contained": true,
"score": 0.95, "reason": "short reason"}, ...]}.
Include exactly one result for every candidate index. Score is from 0 to 1.
"""

JUDGE_USER_PROMPT = """\
Problem:
{problem}

Ground truth (validation only; never copy it into a rubric):
{ground_truth}

Candidate criteria:
{candidates}
"""


@dataclass
class ProblemState:
    key: str
    problem: str
    ground_truth: str
    existing_rubric: str = ""
    accepted: dict[str, tuple[str, float]] = field(default_factory=dict)
    attempts: int = 0
    candidate_count: int = 0
    last_reasons: list[str] = field(default_factory=list)

    def merge(self, accepted_items: list[tuple[str, float]]) -> None:
        for text, score in accepted_items:
            normalized = normalize_item(text)
            previous = self.accepted.get(normalized)
            if previous is None or score > previous[1]:
                self.accepted[normalized] = (text, score)

    def selected(self, max_selected: int) -> list[tuple[str, float]]:
        return sorted(self.accepted.values(), key=lambda item: item[1], reverse=True)[:max_selected]

    def is_valid(self, min_selected: int) -> bool:
        return len(self.accepted) >= min_selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--generation-backend", choices=("vllm", "dashscope"), default="vllm")
    parser.add_argument("--remote-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--generator-api-key-env", default="RUBRIC_API_KEY")
    parser.add_argument("--judge-api-key-env", default="RUBRIC_API_KEY")
    parser.add_argument("--judge-model", default="qwen3.5-plus")
    parser.add_argument("--remote-concurrency", type=int, default=16)
    parser.add_argument("--remote-timeout", type=float, default=180.0)
    parser.add_argument("--remote-max-retries", type=int, default=3)
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--rubric-column", default="rubric")
    parser.add_argument("--valid-column", default="rubric_valid")
    parser.add_argument("--progress-file")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--min-selected", type=int, default=MIN_REQUIRED_RUBRICS)
    parser.add_argument("--max-selected", type=int, default=8)
    parser.add_argument("--validation-threshold", type=float, default=0.75)
    parser.add_argument("--max-validation-rounds", type=int, default=4)
    parser.add_argument("--target-valid-rate", type=float, default=0.70)
    parser.add_argument("--generator-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-rubric-chars", type=int, default=4000)
    parser.add_argument("--generator-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reuse-existing-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-invalid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-below-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--http-client-log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name)
    for logger_name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(logger_name).setLevel(level)


def normalize_messages(raw_prompt: Any) -> list[dict[str, str]]:
    if hasattr(raw_prompt, "tolist") and not isinstance(raw_prompt, (str, bytes, dict, list, tuple)):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]
    if isinstance(raw_prompt, dict):
        if "messages" in raw_prompt:
            raw_prompt = raw_prompt["messages"]
        elif "role" in raw_prompt and "content" in raw_prompt:
            raw_prompt = [raw_prompt]
        else:
            return [{"role": "user", "content": json.dumps(raw_prompt, ensure_ascii=False)}]
    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)
    if not isinstance(raw_prompt, list):
        return [{"role": "user", "content": str(raw_prompt)}]
    messages = []
    for message in raw_prompt:
        if not isinstance(message, dict):
            messages.append({"role": "user", "content": str(message)})
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        messages.append({"role": str(message.get("role", "user")), "content": str(content)})
    return messages or [{"role": "user", "content": ""}]


def extract_problem_text(raw_prompt: Any) -> str:
    messages = normalize_messages(raw_prompt)
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"].strip()
    return messages[-1]["content"].strip()


def extract_ground_truth(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    for key in ("ground_truth", "answer", "target"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("No ground truth found in reward_model.ground_truth, ground_truth, answer, or target")


def normalize_item(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def split_rubric_items(rubric_text: str) -> list[str]:
    items = []
    for line in str(rubric_text or "").splitlines():
        item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if len(item) >= 8 and normalize_item(item) not in {normalize_item(value) for value in items}:
            items.append(item)
    return items


def strip_to_json(raw_text: str) -> Any:
    cleaned = _THINK_RE.sub("", raw_text or "").strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    candidates = [cleaned]
    for left, right in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(left), cleaned.rfind(right)
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError("Judge response did not contain valid JSON")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def parse_judge_response(
    raw_text: str,
    candidates: list[str],
    threshold: float,
) -> tuple[list[tuple[str, float]], list[str]]:
    parsed = strip_to_json(raw_text)
    items = parsed.get("items", []) if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise ValueError("Judge JSON must contain an items list")

    by_index: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= index <= len(candidates) and index not in by_index:
            by_index[index] = item

    accepted: list[tuple[str, float]] = []
    reasons = []
    required_flags = ("correct", "relevant", "actionable", "no_answer_leak", "self_contained")
    for index, candidate in enumerate(candidates, start=1):
        result = by_index.get(index, {})
        try:
            score = float(result.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        passed = all(_as_bool(result.get(flag, False)) for flag in required_flags)
        passed = passed and score >= threshold
        if passed:
            accepted.append((candidate, min(max(score, 0.0), 1.0)))
        else:
            reason = str(result.get("reason", "missing or malformed validation result")).strip()
            reasons.append(f"{index}: {reason}")
    return accepted, reasons


def problem_key(problem: str) -> str:
    return hashlib.sha256(problem.encode("utf-8")).hexdigest()


def config_signature(args: argparse.Namespace, input_path: Path) -> str:
    payload = {
        "version": VALIDATION_VERSION,
        "input": str(input_path),
        "input_size": input_path.stat().st_size,
        "input_mtime_ns": input_path.stat().st_mtime_ns,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "candidate_count": args.candidate_count,
        "min_selected": args.min_selected,
        "max_selected": args.max_selected,
        "validation_threshold": args.validation_threshold,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_progress(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid progress JSON at {path}:{line_number}") from exc
            if record.get("signature") != signature:
                raise ValueError(
                    f"Progress signature mismatch at {path}:{line_number}; use --overwrite for a new run"
                )
            records[str(record["key"])] = record
    return records


def append_progress(path: Path, signature: str, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"signature": signature, **record}, ensure_ascii=False) + "\n")
        handle.flush()


def state_from_record(state: ProblemState, record: dict[str, Any]) -> None:
    state.attempts = int(record.get("attempts", 0))
    state.candidate_count = int(record.get("candidate_count", 0))
    state.last_reasons = [str(value) for value in record.get("last_reasons", [])]
    accepted_items = record.get("accepted_items", [])
    for item in accepted_items:
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            text = str(item["text"]).strip()
            state.accepted[normalize_item(text)] = (text, float(item.get("score", 0.0)))


def state_record(state: ProblemState, args: argparse.Namespace) -> dict[str, Any]:
    selected = state.selected(args.max_selected)
    rubric = "\n".join(f"{index}. {text}" for index, (text, _) in enumerate(selected, start=1))
    scores = [score for _, score in selected]
    return {
        "key": state.key,
        "rubric": rubric,
        "rubric_valid": state.is_valid(args.min_selected),
        "validation_score": sum(scores) / len(scores) if scores else 0.0,
        "selected_count": len(selected),
        "candidate_count": state.candidate_count,
        "attempts": state.attempts,
        "accepted_items": [{"text": text, "score": score} for text, score in selected],
        "last_reasons": state.last_reasons[-20:],
        "validation_mode": "answer_aware_semantic_judge",
    }


def build_judge_messages(state: ProblemState, candidates: list[str]) -> list[dict[str, str]]:
    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(candidates, start=1))
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_PROMPT.format(
                problem=state.problem,
                ground_truth=state.ground_truth,
                candidates=numbered,
            ),
        },
    ]


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.remote_concurrency <= 0:
        raise ValueError("--batch-size and --remote-concurrency must be positive")
    if not MIN_REQUIRED_RUBRICS <= args.min_selected <= args.max_selected <= args.candidate_count:
        raise ValueError(
            f"Require {MIN_REQUIRED_RUBRICS} <= min-selected <= max-selected <= candidate-count"
        )
    if not 0.0 < args.validation_threshold <= 1.0:
        raise ValueError("--validation-threshold must be in (0, 1]")
    if not 0.0 <= args.target_valid_rate <= 1.0:
        raise ValueError("--target-valid-rate must be in [0, 1]")
    if args.max_validation_rounds <= 0:
        raise ValueError("--max-validation-rounds must be positive")
    if args.limit == 0:
        raise ValueError("--limit must be positive or -1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_logging(args.http_client_log_level)

    input_path = Path(args.input_parquet).expanduser().resolve()
    output_path = Path(args.output_parquet).expanduser().resolve()
    progress_path = (
        Path(args.progress_file).expanduser().resolve()
        if args.progress_file
        else output_path.with_suffix(".validation.progress.jsonl")
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_path}")
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    from datasets import load_dataset
    from examples.remote_chat_generation import RemoteChatGenerator
    from examples.token_credit_grpo.precompute_rubrics import (
        build_rubric_generation_messages,
        parse_rubric_response,
    )

    dataset = load_dataset("parquet", data_files=str(input_path), split="train")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.prompt_column not in dataset.column_names:
        raise KeyError(f"Prompt column {args.prompt_column!r} is absent: {dataset.column_names}")

    states: list[ProblemState] = []
    state_by_key: dict[str, ProblemState] = {}
    row_keys = []
    has_existing_rubric = args.rubric_column in dataset.column_names
    for row in dataset:
        problem = extract_problem_text(row[args.prompt_column])
        key = problem_key(problem)
        row_keys.append(key)
        if key in state_by_key:
            continue
        existing_rubric = str(row.get(args.rubric_column) or "").strip() if has_existing_rubric else ""
        state = ProblemState(
            key=key,
            problem=problem,
            ground_truth=extract_ground_truth(row),
            existing_rubric=existing_rubric,
        )
        states.append(state)
        state_by_key[key] = state

    signature = config_signature(args, input_path)
    completed = load_progress(progress_path, signature)
    pending = []
    for state in states:
        record = completed.get(state.key)
        if record is not None:
            state_from_record(state, record)
        if record is None or (args.retry_invalid and not state.is_valid(args.min_selected)):
            pending.append(state)

    print(f"Rows: {len(dataset)}")
    print(f"Unique problems: {len(states)}")
    print(f"Resumed valid problems: {sum(state.is_valid(args.min_selected) for state in states)}")
    print(f"Pending validation: {len(pending)}")

    generator_api_key = os.environ.get(args.generator_api_key_env, "").strip()
    judge_api_key = os.environ.get(args.judge_api_key_env, "").strip()
    judge = None
    if pending:
        if not judge_api_key:
            raise RuntimeError(f"Export {args.judge_api_key_env} for rubric validation")
        if args.generation_backend == "dashscope" and not generator_api_key:
            raise RuntimeError(f"Export {args.generator_api_key_env} for remote rubric generation")
        judge = RemoteChatGenerator(
            model=args.judge_model,
            api_key=judge_api_key,
            base_url=args.remote_base_url,
            concurrency=args.remote_concurrency,
            timeout=args.remote_timeout,
            max_retries=args.remote_max_retries,
        )

    llm = None
    tokenizer = None
    sampling_params_cls = None
    remote_generator = None
    if pending and args.generation_backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.generator_model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()
        sampling_params_cls = SamplingParams
    elif pending:
        remote_generator = RemoteChatGenerator(
            model=args.generator_model,
            api_key=generator_api_key,
            base_url=args.remote_base_url,
            concurrency=args.remote_concurrency,
            timeout=args.remote_timeout,
            max_retries=args.remote_max_retries,
        )

    def generate_candidates(batch_states: list[ProblemState], round_index: int) -> list[str]:
        messages = [
            build_rubric_generation_messages(state.problem, args.candidate_count)
            for state in batch_states
        ]
        if round_index > 0:
            for message_list in messages:
                message_list.append(
                    {
                        "role": "user",
                        "content": (
                            "Previous candidates did not provide enough independently valid criteria. "
                            "Generate a new, non-overlapping set and follow the JSON format exactly."
                        ),
                    }
                )
        if remote_generator is not None:
            outputs = remote_generator.generate(
                messages,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=-1,
                max_tokens=args.generator_max_tokens,
                n=1,
                seed=args.seed + round_index * 1_000_000,
                enable_thinking=args.generator_enable_thinking,
            )
        else:
            assert llm is not None and tokenizer is not None and sampling_params_cls is not None
            prompts = []
            for message_list in messages:
                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": args.generator_enable_thinking,
                }
                try:
                    prompts.append(tokenizer.apply_chat_template(message_list, **kwargs))
                except TypeError:
                    kwargs.pop("enable_thinking")
                    prompts.append(tokenizer.apply_chat_template(message_list, **kwargs))
            outputs = llm.generate(
                prompts,
                sampling_params_cls(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.generator_max_tokens,
                    n=1,
                    seed=args.seed + round_index * 1_000_000,
                ),
                use_tqdm=False,
            )
        return [
            parse_rubric_response(
                output.outputs[0].text,
                max_items=args.candidate_count,
                max_chars=args.max_rubric_chars,
            )
            for output in outputs
        ]

    for start in range(0, len(pending), args.batch_size):
        batch_states = pending[start : start + args.batch_size]
        unresolved = list(batch_states)
        print(
            f"[Batch] problems {start + 1}-{start + len(batch_states)} of {len(pending)}",
            flush=True,
        )
        for round_index in range(args.max_validation_rounds):
            unresolved = [state for state in unresolved if not state.is_valid(args.min_selected)]
            if not unresolved:
                break

            candidate_blocks: list[str] = []
            need_generation = []
            need_generation_indices = []
            for index, state in enumerate(unresolved):
                use_existing = (
                    args.reuse_existing_candidates
                    and state.attempts == 0
                    and bool(state.existing_rubric)
                )
                if use_existing:
                    candidate_blocks.append(state.existing_rubric)
                else:
                    candidate_blocks.append("")
                    need_generation.append(state)
                    need_generation_indices.append(index)
            if need_generation:
                generated = generate_candidates(need_generation, round_index)
                for index, rubric in zip(need_generation_indices, generated, strict=True):
                    candidate_blocks[index] = rubric

            candidate_lists = [split_rubric_items(block) for block in candidate_blocks]
            judge_messages = [
                build_judge_messages(state, candidates)
                for state, candidates in zip(unresolved, candidate_lists, strict=True)
            ]
            assert judge is not None
            judge_outputs = judge.generate(
                judge_messages,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=args.judge_max_tokens,
                n=1,
                seed=args.seed + 10_000_000 + round_index * 1_000_000 + start,
                enable_thinking=args.judge_enable_thinking,
            )

            for state, candidates, output in zip(
                unresolved, candidate_lists, judge_outputs, strict=True
            ):
                state.attempts += 1
                state.candidate_count += len(candidates)
                try:
                    accepted, reasons = parse_judge_response(
                        output.outputs[0].text,
                        candidates,
                        args.validation_threshold,
                    )
                except ValueError as exc:
                    accepted, reasons = [], [str(exc)]
                state.merge(accepted)
                state.last_reasons = reasons

            valid_so_far = sum(state.is_valid(args.min_selected) for state in batch_states)
            print(
                f"[Validation round {round_index + 1}] "
                f"valid {valid_so_far}/{len(batch_states)}",
                flush=True,
            )

        records = [state_record(state, args) for state in batch_states]
        append_progress(progress_path, signature, records)
        completed.update({record["key"]: {"signature": signature, **record} for record in records})

    final_records = {}
    for state in states:
        record = state_record(state, args)
        final_records[state.key] = record

    rubrics = [final_records[key]["rubric"] for key in row_keys]
    valid = [bool(final_records[key]["rubric_valid"]) for key in row_keys]
    validation_scores = [float(final_records[key]["validation_score"]) for key in row_keys]
    selected_counts = [int(final_records[key]["selected_count"]) for key in row_keys]
    candidate_counts = [int(final_records[key]["candidate_count"]) for key in row_keys]
    attempts = [int(final_records[key]["attempts"]) for key in row_keys]
    modes = [str(final_records[key]["validation_mode"]) for key in row_keys]

    output_columns = {
        args.rubric_column: rubrics,
        args.valid_column: valid,
        "rubric_validation_score": validation_scores,
        "rubric_selected_count": selected_counts,
        "rubric_candidate_count": candidate_counts,
        "rubric_validation_attempts": attempts,
        "rubric_validation_mode": modes,
    }
    for column in output_columns:
        if column in dataset.column_names:
            dataset = dataset.remove_columns(column)
    for column, values in output_columns.items():
        dataset = dataset.add_column(column, values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))

    unique_valid = sum(final_records[state.key]["rubric_valid"] for state in states)
    unique_rate = unique_valid / max(len(states), 1)
    row_rate = sum(valid) / max(len(valid), 1)
    summary_path = output_path.with_suffix(".validation.summary.json")
    summary = {
        "validation_version": VALIDATION_VERSION,
        "validation_mode": "answer_aware_semantic_judge",
        "input_parquet": str(input_path),
        "output_parquet": str(output_path),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "validation_threshold": args.validation_threshold,
        "min_selected": args.min_selected,
        "max_selected": args.max_selected,
        "target_valid_rate": args.target_valid_rate,
        "unique_problems": len(states),
        "valid_unique_problems": unique_valid,
        "unique_problem_valid_rate": unique_rate,
        "rows": len(valid),
        "valid_rows": sum(valid),
        "row_valid_rate": row_rate,
        "mean_selected_rubrics": (
            sum(final_records[state.key]["selected_count"] for state in states) / max(len(states), 1)
        ),
        "mean_validation_attempts": (
            sum(final_records[state.key]["attempts"] for state in states) / max(len(states), 1)
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"Validated unique problems: {unique_valid}/{len(states)} ({100 * unique_rate:.2f}%)")
    print(f"Validated rows: {sum(valid)}/{len(valid)} ({100 * row_rate:.2f}%)")
    print(f"Target unique-problem rate: {100 * args.target_valid_rate:.2f}%")
    print(f"Saved dataset: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Progress file: {progress_path}")
    print("=" * 78)

    if args.fail_below_target and unique_rate < args.target_valid_rate:
        raise SystemExit(
            f"Validated rubric rate {100 * unique_rate:.2f}% is below target "
            f"{100 * args.target_valid_rate:.2f}%. Rerun to retry invalid problems or increase "
            "MAX_VALIDATION_ROUNDS/CANDIDATE_COUNT."
        )


if __name__ == "__main__":
    main()
