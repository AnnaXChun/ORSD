#!/usr/bin/env python3
"""Generate trajectory-aware agent rubrics and independently validate them."""

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

from datasets import Dataset, load_dataset

VERL_ROOT = Path(__file__).resolve().parents[2]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from examples.remote_chat_generation import RemoteChatGenerator  # noqa: E402
from examples.token_credit_grpo.precompute_rubrics import parse_rubric_response  # noqa: E402

VERSION = 3
MIN_RUBRICS = 4
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

GENERATOR_SYSTEM = """\
You design decision rubrics for a long-horizon interactive agent. You receive
the task and COMPLETE self-generated rollout trajectories containing observations,
actions, transitions, and official environment rewards. You do not receive an
oracle plan, hidden answer, target identifier, or validator-only reference.

Generate exactly {count} concise, non-overlapping criteria. Each criterion must:
1. Describe one state-conditioned decision rule, not generic advice.
2. Use only information observable to the acting agent at that point.
3. State the preferred action/stage check and expected evidence of progress.
4. Agree with the task and successful rollouts while identifying
   concrete failure patterns in unsuccessful rollouts.
5. Generalize beyond one verbatim trajectory; do not copy a full action sequence.
6. Never reveal hidden identifiers, the target ASIN, a hidden game file, or an
   answer that was not already visible in the task/observations.
7. Never replace the official environment reward or claim that a rubric itself
   proves task completion.

ALFWorld criteria should respect the search -> acquire -> optional transform ->
deliver/examine milestone order and current admissible actions. WebShop criteria
should verify query relevance, required attributes, options, price, and purchase
readiness using visible page evidence.

Use evidence in this order: successful official outcomes; higher versus lower
partial rewards; concrete transitions that explain a failed outcome. If every
actor rollout fails, extract only prevention rules supported by observed failed
transitions; never invent a successful action, hidden location, or oracle plan,
and never pretend the failed action sequence is a successful demonstration.

Write every criterion in this compact shape:
"IF <observable state>; CHECK <visible evidence>; THEN <action rule>; EXPECT
<next-state evidence>; AVOID <trajectory-grounded failure>."

Return JSON only: {{"rubrics": ["criterion 1", ..., "criterion {count}"]}}.
"""

GENERATOR_USER = """\
Benchmark: {benchmark}

Task shown to the agent:
{task}

Complete officially scored rollout trajectories:
{trajectories}

Generate {count} criteria now.
"""

JUDGE_SYSTEM = """\
You are an independent, strict validator of long-horizon agent rubrics. You are
given the task, validator-only reference constraints, complete trajectories, the
official outcome of every trajectory, and numbered candidate criteria.

Evaluate each candidate independently. It passes only if ALL are true:
- answer_consistent: it agrees with the task and validator-only constraints;
- trajectory_grounded: an observed transition, demonstrated milestone, or
  concrete failed transition supports it;
- outcome_consistent: it promotes decisions evidenced by higher-reward behavior
  or prevents a failure evidenced by lower-reward behavior;
- actionable: it changes a state-conditioned action choice;
- observable: the acting policy can check its precondition/evidence;
- no_hidden_answer_leak: it reveals no hidden ASIN, internal game path, unseen
  object location, or copied oracle sequence;
- non_redundant: it contributes a distinct decision check.

Use the same evidence hierarchy as the generator: official success first,
higher partial reward second, successful reference milestones third, and
specific failed transitions only as negative evidence. Equal-reward failed
rollouts cannot by themselves prove that an action is correct.

Do not repair failed items. The official environment reward is authoritative.
Return JSON only:
{{"items": [{{"index": 1, "answer_consistent": true,
"trajectory_grounded": true, "outcome_consistent": true,
"actionable": true, "observable": true, "no_hidden_answer_leak": true,
"non_redundant": true, "score": 0.95, "reason": "short evidence"}}, ...]}}.
Include exactly one result for every candidate index.
"""

JUDGE_USER = """\
Benchmark: {benchmark}

Task:
{task}

Validator-only reference constraints:
{reference}

Complete officially scored trajectories:
{trajectories}

Candidate criteria:
{candidates}
"""


@dataclass
class TaskState:
    key: str
    benchmark: str
    task: str
    reference: Any
    trajectories: list[dict[str, Any]]
    forbidden_terms: list[str]
    accepted: dict[str, tuple[str, float, str]] = field(default_factory=dict)
    attempts: int = 0
    candidate_count: int = 0
    last_reasons: list[str] = field(default_factory=list)

    def merge(self, items: list[tuple[str, float, str]]) -> None:
        for text, score, reason in items:
            key = normalize(text)
            previous = self.accepted.get(key)
            if previous is None or score > previous[1]:
                self.accepted[key] = (text, score, reason)

    def selected(self, maximum: int) -> list[tuple[str, float, str]]:
        return sorted(self.accepted.values(), key=lambda item: item[1], reverse=True)[:maximum]

    def valid(self, minimum: int) -> bool:
        return len(self.accepted) >= minimum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--generation-backend", choices=("vllm", "dashscope"), default="vllm")
    parser.add_argument("--judge-backend", choices=("dashscope", "vllm"), default="dashscope")
    parser.add_argument("--judge-model", default="qwen3.5-plus")
    parser.add_argument("--remote-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--generator-api-key-env", default="RUBRIC_API_KEY")
    parser.add_argument("--judge-api-key-env", default="RUBRIC_API_KEY")
    parser.add_argument("--remote-concurrency", type=int, default=16)
    parser.add_argument("--remote-timeout", type=float, default=180.0)
    parser.add_argument("--remote-max-retries", type=int, default=3)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--min-trajectories", type=int, default=4)
    parser.add_argument("--min-selected", type=int, default=MIN_RUBRICS)
    parser.add_argument("--max-selected", type=int, default=8)
    parser.add_argument("--validation-threshold", type=float, default=0.80)
    parser.add_argument("--max-validation-rounds", type=int, default=5)
    parser.add_argument("--target-valid-rate", type=float, default=0.60)
    parser.add_argument("--generator-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-rubric-chars", type=int, default=6000)
    parser.add_argument("--progress-file", default="")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inject-rubric", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-invalid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-below-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def load_json_field(value: Any, name: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON") from exc
    return value


def task_key(row: dict[str, Any]) -> str:
    if row.get("task_id"):
        return str(row["task_id"])
    material = f"{row.get('data_source', '')}\n{row.get('task', '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def trajectory_reward(trajectory: dict[str, Any]) -> float:
    for key in ("official_reward", "environment_reward", "reward", "score"):
        if trajectory.get(key) is not None:
            return float(trajectory[key])
    raise ValueError("Every trajectory must contain official_reward (or environment_reward/reward/score)")


def forbidden_terms(reference: Any) -> list[str]:
    if not isinstance(reference, dict):
        return []
    values = []
    for key in ("target_asin", "game_file", "hidden_answer", "oracle_id"):
        value = reference.get(key)
        if value is not None and len(str(value).strip()) >= 4:
            values.append(str(value).strip().lower())
    return values


def compact_trajectory_for_prompt(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Keep every environment transition while dropping private model reasoning text."""
    compact = {
        key: value
        for key, value in trajectory.items()
        if key not in {"steps", "model_response", "messages", "raw_messages"}
    }
    compact_steps = []
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            compact_steps.append(step)
            continue
        compact_steps.append(
            {
                key: value
                for key, value in step.items()
                if key not in {"model_response", "reasoning", "thinking", "logprobs"}
            }
        )
    compact["steps"] = compact_steps
    return compact


def redact_hidden_reference(value: Any) -> Any:
    """Remove identifiers that the generator must never see or reproduce."""
    if isinstance(value, dict):
        hidden_keys = {"target_asin", "game_file", "hidden_answer", "oracle_id"}
        return {
            key: redact_hidden_reference(item) for key, item in value.items() if str(key).lower() not in hidden_keys
        }
    if isinstance(value, list):
        return [redact_hidden_reference(item) for item in value]
    return value


def serialize_evidence(state: TaskState, *, include_hidden_reference: bool) -> tuple[str, str]:
    reference_value = state.reference if include_hidden_reference else redact_hidden_reference(state.reference)
    reference = json.dumps(reference_value, ensure_ascii=False, indent=2, sort_keys=True)
    trajectories = json.dumps(
        [compact_trajectory_for_prompt(trajectory) for trajectory in state.trajectories],
        ensure_ascii=False,
        indent=2,
    )
    return reference, trajectories


def generation_messages(state: TaskState, args: argparse.Namespace, round_index: int) -> list[dict[str, str]]:
    _, trajectories = serialize_evidence(state, include_hidden_reference=False)
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM.format(count=args.candidate_count)},
        {
            "role": "user",
            "content": GENERATOR_USER.format(
                benchmark=state.benchmark,
                task=state.task,
                trajectories=trajectories,
                count=args.candidate_count,
            ),
        },
    ]
    if round_index > 0:
        accepted = "\n".join(f"- {item[0]}" for item in state.selected(args.max_selected)) or "(none)"
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous round left fewer than the required independently valid criteria.\n"
                    f"Already accepted (do not duplicate):\n{accepted}\n"
                    "The validator's private reference and free-text rejection reasons are withheld. "
                    "Generate a new non-overlapping JSON set grounded only in the same task and "
                    "self-generated complete trajectories."
                ),
            }
        )
    return messages


def judge_messages(state: TaskState, candidates: list[str]) -> list[dict[str, str]]:
    reference, trajectories = serialize_evidence(state, include_hidden_reference=True)
    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(candidates, start=1))
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                benchmark=state.benchmark,
                task=state.task,
                reference=reference,
                trajectories=trajectories,
                candidates=numbered,
            ),
        },
    ]


def strip_json(text: str) -> Any:
    cleaned = _THINK_RE.sub("", text or "").strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    variants = [cleaned]
    for left, right in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(left), cleaned.rfind(right)
        if 0 <= start < end:
            variants.append(cleaned[start : end + 1])
    for value in variants:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
    raise ValueError("response does not contain valid JSON")


def as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes", "pass"}


def split_items(text: str) -> list[str]:
    items = []
    for line in str(text).splitlines():
        candidate = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if len(candidate) >= 12 and normalize(candidate) not in {normalize(item) for item in items}:
            items.append(candidate)
    return items


def validate_response(
    raw_text: str, state: TaskState, candidates: list[str], threshold: float
) -> tuple[list[tuple[str, float, str]], list[str]]:
    parsed = strip_json(raw_text)
    results = parsed.get("items", []) if isinstance(parsed, dict) else parsed
    if not isinstance(results, list):
        raise ValueError("judge JSON must contain an items list")
    indexed = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            index = int(result.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= index <= len(candidates) and index not in indexed:
            indexed[index] = result

    flags = (
        "answer_consistent",
        "trajectory_grounded",
        "outcome_consistent",
        "actionable",
        "observable",
        "no_hidden_answer_leak",
        "non_redundant",
    )
    accepted = []
    reasons = []
    for index, candidate in enumerate(candidates, start=1):
        result = indexed.get(index, {})
        leaked = any(term in candidate.lower() for term in state.forbidden_terms)
        try:
            score = min(1.0, max(0.0, float(result.get("score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
        passed = not leaked and score >= threshold and all(as_bool(result.get(flag, False)) for flag in flags)
        reason = str(result.get("reason", "missing validation result")).strip()
        if passed:
            accepted.append((candidate, score, reason))
        else:
            if leaked:
                reason = "deterministic hidden-identifier leakage check failed"
            reasons.append(f"{index}: {reason}")
    return accepted, reasons


class Generator:
    def __init__(self, backend: str, model: str, args: argparse.Namespace, api_key_env: str):
        self.backend = backend
        self.model = model
        self.args = args
        self.remote = None
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        if backend == "dashscope":
            api_key = os.environ.get(api_key_env, "").strip()
            if not api_key:
                raise RuntimeError(f"Export {api_key_env} for {model}")
            self.remote = RemoteChatGenerator(
                model=model,
                api_key=api_key,
                base_url=args.remote_base_url,
                concurrency=args.remote_concurrency,
                timeout=args.remote_timeout,
                max_retries=args.remote_max_retries,
            )
        else:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=model,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                dtype="bfloat16",
                trust_remote_code=True,
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.sampling_params = SamplingParams

    def generate(
        self,
        messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        enable_thinking: bool,
    ) -> list[str]:
        if self.remote is not None:
            outputs = self.remote.generate(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=-1,
                max_tokens=max_tokens,
                n=1,
                seed=seed,
                enable_thinking=enable_thinking,
            )
            return [output.outputs[0].text for output in outputs]
        prompts = []
        for message_list in messages:
            kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking}
            try:
                prompt = self.tokenizer.apply_chat_template(message_list, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking")
                prompt = self.tokenizer.apply_chat_template(message_list, **kwargs)
            prompts.append(prompt)
        lengths = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        too_long = [length for length in lengths if length + max_tokens > self.args.max_model_len]
        if too_long:
            raise ValueError(
                "Complete trajectory evidence exceeds --max-model-len; no evidence was silently truncated. "
                f"Largest request needs {max(too_long) + max_tokens} tokens, configured "
                f"max is {self.args.max_model_len}. Use a longer-context model or collect fewer/shorter rollouts."
            )
        outputs = self.llm.generate(
            prompts,
            self.sampling_params(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=1,
                seed=seed,
            ),
            use_tqdm=False,
        )
        return [output.outputs[0].text for output in outputs]


def signature(args: argparse.Namespace, input_path: Path) -> str:
    payload = {
        "version": VERSION,
        "input": str(input_path),
        "size": input_path.stat().st_size,
        "mtime": input_path.stat().st_mtime_ns,
        "generator": args.generator_model,
        "judge": args.judge_model,
        "candidate_count": args.candidate_count,
        "threshold": args.validation_threshold,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def append_progress(path: Path, run_signature: str, states: list[TaskState], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for state in states:
            handle.write(
                json.dumps({"signature": run_signature, **state_record(state, args)}, ensure_ascii=False) + "\n"
            )


def load_progress(path: Path, run_signature: str) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("signature") != run_signature:
                raise ValueError(f"Progress signature mismatch at {path}:{line_number}; use --overwrite")
            records[record["key"]] = record
    return records


def restore_state(state: TaskState, record: dict[str, Any]) -> None:
    state.attempts = int(record.get("attempts", 0))
    state.candidate_count = int(record.get("candidate_count", 0))
    state.last_reasons = [str(value) for value in record.get("last_reasons", [])]
    for item in record.get("accepted_items", []):
        text = str(item.get("text", "")).strip()
        if text:
            state.accepted[normalize(text)] = (
                text,
                float(item.get("score", 0.0)),
                str(item.get("reason", "")),
            )


def state_record(state: TaskState, args: argparse.Namespace) -> dict[str, Any]:
    selected = state.selected(args.max_selected)
    return {
        "key": state.key,
        "rubric": "\n".join(f"{index}. {item[0]}" for index, item in enumerate(selected, start=1)),
        "valid": state.valid(args.min_selected),
        "selected_count": len(selected),
        "candidate_count": state.candidate_count,
        "attempts": state.attempts,
        "validation_score": sum(item[1] for item in selected) / max(len(selected), 1),
        "accepted_items": [{"text": text, "score": score, "reason": reason} for text, score, reason in selected],
        "last_reasons": state.last_reasons[-20:],
    }


def validate_args(args: argparse.Namespace) -> None:
    if not MIN_RUBRICS <= args.min_selected <= args.max_selected <= args.candidate_count:
        raise ValueError(f"Require {MIN_RUBRICS} <= min-selected <= max-selected <= candidate-count")
    if not 0 < args.validation_threshold <= 1 or not 0 <= args.target_valid_rate <= 1:
        raise ValueError("validation-threshold must be in (0,1] and target-valid-rate in [0,1]")
    if args.max_validation_rounds < 1 or args.batch_size < 1 or args.min_trajectories < 1:
        raise ValueError("max-validation-rounds, batch-size, and min-trajectories must be positive")
    if args.generation_backend == args.judge_backend == "vllm" and args.generator_model != args.judge_model:
        raise ValueError(
            "Two different local vLLM models cannot share this process; use the same model or remote judge"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    input_path = Path(args.input_parquet).expanduser().resolve()
    output_path = Path(args.output_parquet).expanduser().resolve()
    progress_path = (
        Path(args.progress_file).expanduser().resolve()
        if args.progress_file
        else output_path.with_suffix(".rubric-progress.jsonl")
    )
    if args.overwrite:
        progress_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    dataset = load_dataset("parquet", data_files=str(input_path), split="train")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    required = {"data_source", "task", "reference_answer_json", "rollout_trajectories_json", "extra_info"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    states = []
    row_keys = []
    state_by_key = {}
    for row in dataset:
        key = task_key(row)
        row_keys.append(key)
        if key in state_by_key:
            continue
        reference = load_json_field(row["reference_answer_json"], "reference_answer_json")
        trajectories = load_json_field(row["rollout_trajectories_json"], "rollout_trajectories_json")
        if not isinstance(trajectories, list) or not trajectories:
            raise ValueError(f"Task {key} has no rollout trajectory; run collect_rollout_trajectories.py first")
        if len(trajectories) < args.min_trajectories:
            raise ValueError(
                f"Task {key} has only {len(trajectories)} trajectories; at least {args.min_trajectories} are required"
            )
        for trajectory in trajectories:
            trajectory["official_reward"] = trajectory_reward(trajectory)
        state = TaskState(
            key=key,
            benchmark=str(row["data_source"]),
            task=str(row["task"]),
            reference=reference,
            trajectories=trajectories,
            forbidden_terms=forbidden_terms(reference),
        )
        states.append(state)
        state_by_key[key] = state

    run_signature = signature(args, input_path)
    completed = load_progress(progress_path, run_signature)
    pending = []
    for state in states:
        if state.key in completed:
            restore_state(state, completed[state.key])
        if not state.valid(args.min_selected):
            pending.append(state)

    print(f"Rows={len(dataset)} unique_tasks={len(states)} resumed_valid={len(states) - len(pending)}")
    generator = None
    judge = None
    if pending:
        generator = Generator(args.generation_backend, args.generator_model, args, args.generator_api_key_env)
        if args.judge_backend == "vllm" and args.generator_model == args.judge_model:
            judge = generator
        else:
            judge = Generator(args.judge_backend, args.judge_model, args, args.judge_api_key_env)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        for round_index in range(args.max_validation_rounds):
            unresolved = [state for state in batch if not state.valid(args.min_selected)]
            if not unresolved:
                break
            assert generator is not None and judge is not None
            generated = generator.generate(
                [generation_messages(state, args, round_index) for state in unresolved],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.generator_max_tokens,
                seed=args.seed + start * 10_000 + round_index * 1_000_000,
                enable_thinking=args.generator_enable_thinking,
            )
            candidate_lists = [
                split_items(parse_rubric_response(text, args.candidate_count, args.max_rubric_chars))
                for text in generated
            ]
            judged = judge.generate(
                [
                    judge_messages(state, candidates)
                    for state, candidates in zip(unresolved, candidate_lists, strict=True)
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=args.judge_max_tokens,
                seed=args.seed + 10_000_000 + start * 10_000 + round_index * 1_000_000,
                enable_thinking=args.judge_enable_thinking,
            )
            for state, candidates, raw_judgment in zip(unresolved, candidate_lists, judged, strict=True):
                state.attempts += 1
                state.candidate_count += len(candidates)
                try:
                    accepted, reasons = validate_response(raw_judgment, state, candidates, args.validation_threshold)
                except ValueError as exc:
                    accepted, reasons = [], [str(exc)]
                state.merge(accepted)
                state.last_reasons = reasons
            valid_count = sum(state.valid(args.min_selected) for state in batch)
            print(
                f"[validation] batch={start // args.batch_size + 1} round={round_index + 1} "
                f"valid={valid_count}/{len(batch)}",
                flush=True,
            )
        append_progress(progress_path, run_signature, batch, args)

    records = {state.key: state_record(state, args) for state in states}
    valid_states = sum(record["valid"] for record in records.values())
    valid_rate = valid_states / max(len(states), 1)
    summary_path = output_path.with_suffix(".rubric-summary.json")
    summary = {
        "version": VERSION,
        "input_parquet": str(input_path),
        "output_parquet": str(output_path),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "unique_tasks": len(states),
        "valid_tasks": valid_states,
        "task_coverage": valid_rate,
        "target_valid_rate": args.target_valid_rate,
        "minimum_rubrics_per_task": args.min_selected,
        "minimum_trajectories_per_task": args.min_trajectories,
        "emitted_rows": 0,
        "reward_source": "official_environment_only",
        "rubric_role": "current_actor_teacher_scoring_context_not_reward",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.fail_below_target and valid_rate < args.target_valid_rate:
        raise SystemExit(
            f"Rubric task coverage {valid_rate:.2%} is below the required {args.target_valid_rate:.2%}; "
            f"no training parquet was emitted. Audit summary: {summary_path}. "
            "Retry failed tasks or strengthen the judge/generator."
        )

    output_rows = []
    for row, key in zip(dataset, row_keys, strict=True):
        record = records[key]
        if args.drop_invalid and not record["valid"]:
            continue
        output_row = dict(row)
        # Invalid tasks stay in the training set as pure-GRPO fallback. Do not
        # expose partially accepted rubric fragments to the scoring context.
        output_row["rubric"] = record["rubric"] if record["valid"] else ""
        output_row["rubric_valid"] = bool(record["valid"])
        output_row["rubric_selected_count"] = int(record["selected_count"])
        output_row["rubric_validation_score"] = float(record["validation_score"])
        output_row["rubric_validation_json"] = json.dumps(record["accepted_items"], ensure_ascii=False)
        extra_info = dict(output_row.get("extra_info") or {})
        kwargs = dict(extra_info.get("interaction_kwargs") or {})
        kwargs["rubric"] = record["rubric"] if record["valid"] else ""
        kwargs["inject_rubric"] = bool(args.inject_rubric and record["valid"])
        kwargs["use_interaction_reward"] = True
        extra_info["interaction_kwargs"] = kwargs
        output_row["extra_info"] = extra_info
        output_rows.append(output_row)

    if not output_rows:
        raise RuntimeError("No rows satisfy the rubric validation policy; refusing to emit an empty parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(output_rows).to_parquet(str(output_path))
    summary["emitted_rows"] = len(output_rows)
    summary["all_valid_rows_have_minimum"] = all(
        records[key]["selected_count"] >= args.min_selected for key in row_keys if records[key]["valid"]
    )
    summary["invalid_rows_use_grpo_fallback"] = not args.drop_invalid
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"Validated task coverage: {valid_states}/{len(states)} ({valid_rate:.2%})")
    print(f"Each emitted valid task has >= {args.min_selected} independently accepted rubrics")
    print(f"Training parquet: {output_path}")
    print(f"Summary: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
