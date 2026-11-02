#!/usr/bin/env python3
"""Distill task-level privileged hints from baseline ALFWorld rollout groups.

For each task, all selected baseline rollouts are shown to the same model once.
The model summarizes 3-5 short task-specific hints for a fresh actor.

There is intentionally NO:
- mixed-outcome-only gating
- per-rollout rubric generation
- SOURCE/evidence-id requirement in actor-visible text
- rubric judge
- score matrix
- reward-correlation filtering

A non-empty parsed hint list is considered valid.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

HINT_SYSTEM_PROMPT = """\
You are a task coach for an interactive environment.

You will see one task and several previous attempts at the same task.
Use them to write useful hints for a NEW agent that has never seen those attempts.

Write 3 to {k} short hints.

Useful hints can describe:
- important subgoals needed to complete the task,
- useful ordering of subgoals,
- mistakes or no-progress behavior to avoid,
- what must be true before the task is finished.

Rules:
- Use only information supported by the task or the attempts.
- Do not mention attempts, trajectories, rollouts, rewards, sources, evidence, or IDs.
- Do not copy a complete action sequence.
- Do not invent object locations or hidden state.
- Keep each hint short, concrete, and useful by itself.
- Do not explain your reasoning.

Output only numbered hints:
1. ...
2. ...
3. ...
"""

HINT_USER_PROMPT = """\
Task:
{task}

Previous attempts:
{attempts}

Write useful hints for a new agent solving this same task.
"""

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|text)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--rubrics-per-prompt", type=int, default=5)
    parser.add_argument("--max-trajectories", type=int, default=4)
    parser.add_argument("--max-trajectory-chars", type=int, default=6000)
    parser.add_argument("--max-rubric-chars", type=int, default=1600)
    parser.add_argument("--max-item-chars", type=int, default=280)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Thinking is disabled by default for small-model hint distillation.",
    )
    return parser.parse_args()


def load_trajectories(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value or "[]")
    return list(value or [])


def strip_thinking(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", str(text or ""))
    closes = list(_THINK_CLOSE_RE.finditer(cleaned))
    if closes:
        cleaned = cleaned[closes[-1].end():]
    open_match = _THINK_OPEN_RE.search(cleaned)
    if open_match:
        cleaned = cleaned[:open_match.start()]
    return _THINK_CLOSE_RE.sub("", cleaned).strip()


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{text[:head]}\n...[middle omitted]...\n{text[-tail:]}"


def trajectory_transcript(trajectory: dict[str, Any]) -> str:
    lines = [f"INITIAL OBSERVATION:\n{trajectory.get('initial_observation', '')}"]
    for step_index, step in enumerate(trajectory.get("steps", []), start=1):
        lines.append(f"STEP {step_index}")
        lines.append(f"ACTION: {str(step.get('action') or '').strip()}")
        lines.append(f"OBSERVATION: {str(step.get('environment_response') or '').strip()}")
    return "\n".join(lines)


def build_attempts(
    trajectories: list[dict[str, Any]],
    *,
    max_trajectories: int,
    max_trajectory_chars: int,
) -> str:
    selected = trajectories[:max_trajectories]
    chunks: list[str] = []
    for index, trajectory in enumerate(selected, start=1):
        reward = float(trajectory.get("official_reward", 0.0))
        result = "success" if reward >= 1.0 - 1e-8 else "failure"
        transcript = compact_text(
            trajectory_transcript(trajectory),
            max_chars=max_trajectory_chars,
        )
        chunks.append(f"Attempt {index}:\n{transcript}\nResult: {result}")
    return "\n\n".join(chunks)


def build_messages(task: str, attempts: str, max_hints: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": HINT_SYSTEM_PROMPT.format(k=max_hints)},
        {"role": "user", "content": HINT_USER_PROMPT.format(task=task, attempts=attempts)},
    ]


def render(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
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


def _extract_json_items(cleaned: str) -> list[str]:
    candidates = [cleaned]
    for left, right in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(left), cleaned.rfind(right)
        if 0 <= start < end:
            candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(parsed, dict):
            values = parsed.get("rubrics", parsed.get("hints", parsed.get("criteria", [])))
        elif isinstance(parsed, list):
            values = parsed
        else:
            values = []

        if isinstance(values, list):
            result: list[str] = []
            for item in values:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    result.append(
                        str(item.get("text") or item.get("description") or item.get("criterion") or "")
                    )
            if result:
                return result
    return []


def parse_hint_response(
    raw_text: str,
    *,
    max_items: int,
    max_item_chars: int,
    max_rubric_chars: int,
) -> tuple[str, list[str]]:
    cleaned = strip_thinking(raw_text)
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    cleaned = re.sub(r"(?<!^)(?<!\n)\s+(?=\d+\s*[.)]\s+)", "\n", cleaned)

    raw_items = _extract_json_items(cleaned)
    if not raw_items:
        raw_items = cleaned.splitlines()

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = str(raw_item).strip()
        item = re.sub(r"^\s*(?:[-*•]|\d+\s*[.)])\s*", "", item).strip()
        item = re.sub(
            r"^<(?:MILESTONE|AVOID|VERIFY|SUGGEST|PITFALL|RECOVER|BONUS)>\s*",
            "",
            item,
            flags=re.IGNORECASE,
        )
        item = re.sub(
            r"\s*;\s*SOURCE\s*:?\s*[\w.:-]+\s*$",
            "",
            item,
            flags=re.IGNORECASE,
        )
        item = re.sub(r"^(?:PASS|FAIL)\s+WHEN\s+", "", item, flags=re.IGNORECASE)
        item = re.sub(r"\s+", " ", item).strip(" -\t")
        if len(item) < 8:
            continue
        item = item[:max_item_chars].rstrip()
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= max_items:
            break

    rubric = "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))
    return rubric[:max_rubric_chars].strip(), items


def main() -> None:
    args = parse_args()
    if args.rubrics_per_prompt < 1:
        raise ValueError("--rubrics-per-prompt must be >= 1")
    if args.max_trajectories < 1:
        raise ValueError("--max-trajectories must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

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

    prompts: list[str] = []
    task_meta: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        trajectories = load_trajectories(row.get("rollout_trajectories_json"))
        attempts = build_attempts(
            trajectories,
            max_trajectories=args.max_trajectories,
            max_trajectory_chars=args.max_trajectory_chars,
        )
        messages = build_messages(
            task=str(row.get("task") or ""),
            attempts=attempts,
            max_hints=args.rubrics_per_prompt,
        )
        prompts.append(render(tokenizer, messages, args.enable_thinking))
        task_meta.append(
            {
                "row_index": row_index,
                "outcomes": [float(t.get("official_reward", 0.0)) for t in trajectories],
                "num_trajectories_seen": min(len(trajectories), args.max_trajectories),
            }
        )

    audits: list[dict[str, Any] | None] = [None] * len(rows)

    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start:start + args.batch_size]
        outputs = llm.generate(
            batch_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed + start,
            ),
            use_tqdm=False,
        )

        for offset, (prompt, output) in enumerate(zip(batch_prompts, outputs, strict=True)):
            row_index = start + offset
            row = rows[row_index]
            meta = task_meta[row_index]
            raw_text = output.outputs[0].text
            rubric, items = parse_hint_response(
                raw_text,
                max_items=args.rubrics_per_prompt,
                max_item_chars=args.max_item_chars,
                max_rubric_chars=args.max_rubric_chars,
            )
            valid = bool(items)

            row["rubric"] = rubric
            row["rubric_valid"] = valid
            row["rubric_selected_count"] = len(items)

            extra_info = dict(row.get("extra_info") or {})
            interaction_kwargs = dict(extra_info.get("interaction_kwargs") or {})
            interaction_kwargs["rubric"] = rubric
            interaction_kwargs["inject_rubric"] = valid
            extra_info["interaction_kwargs"] = interaction_kwargs
            row["extra_info"] = extra_info

            audits[row_index] = {
                "task_id": row.get("task_id"),
                "task": row.get("task"),
                "outcomes": meta["outcomes"],
                "num_trajectories_seen": meta["num_trajectories_seen"],
                "valid": valid,
                "item_count": len(items),
                "rubric": rubric,
                "generation_without_thinking": strip_thinking(raw_text),
                "prompt_chars": len(prompt),
            }

    audit_items = [item for item in audits if item is not None]

    output = Path(args.output_parquet).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))

    valid_count = sum(bool(item["valid"]) for item in audit_items)
    summary = {
        "model": args.model,
        "num_tasks": len(rows),
        "valid_tasks": valid_count,
        "selection_success_rate": valid_count / max(len(rows), 1),
        "items_mean": sum(int(item["item_count"]) for item in audit_items) / max(len(audit_items), 1),
        "max_trajectories_per_task": args.max_trajectories,
        "rubrics_per_prompt": args.rubrics_per_prompt,
        "generator_enable_thinking": args.enable_thinking,
        "pipeline": "task_group_privileged_hints_v1",
        "judge_used": False,
        "correlation_filter_used": False,
        "items": audit_items,
    }

    summary_path = output.with_suffix(".prompt-audit.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False, indent=2))
    print(f"Rubric parquet: {output}")
    print(f"Prompt audit: {summary_path}")


if __name__ == "__main__":
    main()
