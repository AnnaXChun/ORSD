#!/usr/bin/env python3
"""Generate problem-only rubrics with vLLM and store them in a parquet dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

VERL_ROOT = Path(__file__).resolve().parents[2]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from examples.remote_chat_generation import RemoteChatGenerator  # noqa: E402

STANDALONE_RUBRIC_PRECOMPUTE_VERSION = 2

PROBLEM_RUBRIC_SYSTEM = """\
You are a rubric design expert. Given a math problem, generate exactly {k} concise rubric items that describe the
critical reasoning steps and key decision points needed to solve it correctly.

Requirements:
- Focus on decision points where a solver could go wrong: algebraic manipulations, substitutions, sign choices,
  constraints, or case analysis.
- Each item must be a short, verifiable statement about one specific reasoning step.
- State what the correct step is; avoid generic advice.
- Do not reveal or hint at the final numerical answer.
- Keep the items non-redundant.

Return JSON only: {{"rubrics": ["rubric 1", "rubric 2", ...]}}.
"""

PROBLEM_RUBRIC_USER = """\
Problem:
{problem}

Generate exactly {k} rubric items describing the key steps needed to solve this problem.
"""

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def normalize_messages(raw_prompt: Any) -> list[dict[str, str]]:
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
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(part for part in text_parts if part)
            messages.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": str(content),
                }
            )
        else:
            messages.append({"role": "user", "content": str(message)})
    return messages or [{"role": "user", "content": ""}]


def extract_problem_text(raw_prompt: Any) -> str:
    messages = normalize_messages(raw_prompt)
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"].strip()
    return messages[-1]["content"].strip()


def build_rubric_generation_messages(
    problem: str, rubrics_per_prompt: int
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": PROBLEM_RUBRIC_SYSTEM.format(k=rubrics_per_prompt),
        },
        {
            "role": "user",
            "content": PROBLEM_RUBRIC_USER.format(
                problem=problem, k=rubrics_per_prompt
            ),
        },
    ]


def parse_rubric_response(raw_text: str, max_items: int, max_chars: int) -> str:
    cleaned = _THINK_RE.sub("", raw_text or "").strip()
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    candidates = [cleaned]
    for left, right in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(left), cleaned.rfind(right)
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])

    parsed: Any = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue

    if isinstance(parsed, dict):
        items = parsed.get("rubrics", parsed.get("criteria", []))
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    descriptions = []
    for item in items:
        if isinstance(item, str):
            description = item
        elif isinstance(item, dict):
            description = item.get(
                "description", item.get("criterion", item.get("text", ""))
            )
        else:
            description = ""
        description = re.sub(r"\s+", " ", str(description)).strip()
        if description and description not in descriptions:
            descriptions.append(description)
        if len(descriptions) >= max_items:
            break

    if not descriptions:
        for line in cleaned.splitlines():
            description = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if description and len(description) >= 8:
                descriptions.append(re.sub(r"\s+", " ", description))
            if len(descriptions) >= max_items:
                break

    rubric_text = "\n".join(
        f"{index}. {description}"
        for index, description in enumerate(descriptions, start=1)
    )
    return rubric_text[:max_chars].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
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
    parser.add_argument(
        "--http-client-log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
    )
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--rubric-column", default="rubric")
    parser.add_argument("--valid-column", default="rubric_valid")
    parser.add_argument("--progress-file")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rubrics-per-prompt", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-rubric-chars", type=int, default=4000)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=-1, help="Optional row limit for a smoke test")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def problem_key(problem: str) -> str:
    return hashlib.sha256(problem.encode("utf-8")).hexdigest()


def load_progress(path: Path) -> dict[str, str]:
    completed: dict[str, str] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                rubric = str(record["rubric"]).strip()
                if rubric:
                    completed[str(record["key"])] = rubric
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid progress record at {path}:{line_number}") from exc
    return completed


def append_progress(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for key, rubric in records:
            handle.write(json.dumps({"key": key, "rubric": rubric}, ensure_ascii=False) + "\n")
        handle.flush()


def build_generation_prompts(tokenizer, problems: list[str], args: argparse.Namespace) -> list[str]:
    prompts = []
    for problem in problems:
        messages = build_rubric_generation_messages(problem, args.rubrics_per_prompt)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompts.append(prompt)
    return prompts


def generate_batch(
    llm: Any,
    tokenizer,
    sampling_params_cls: Any,
    remote_generator: RemoteChatGenerator | None,
    problems: list[str],
    args: argparse.Namespace,
) -> dict[str, str]:
    unresolved = list(problems)
    generated: dict[str, str] = {}
    for attempt in range(args.max_retries + 1):
        if not unresolved:
            break
        attempt_temperature = (
            args.temperature if attempt == 0 else min(args.temperature, 0.3)
        )
        if remote_generator is not None:
            messages_batch = [
                build_rubric_generation_messages(problem, args.rubrics_per_prompt)
                for problem in unresolved
            ]
            outputs = remote_generator.generate(
                messages_batch,
                temperature=attempt_temperature,
                top_p=args.top_p,
                top_k=-1,
                max_tokens=args.max_tokens,
                n=1,
                seed=42 + attempt * 1_000_000,
                enable_thinking=args.enable_thinking,
            )
        else:
            prompts = build_generation_prompts(tokenizer, unresolved, args)
            sampling_params = sampling_params_cls(
                temperature=attempt_temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=1,
            )
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        next_unresolved = []
        for problem, output in zip(unresolved, outputs, strict=True):
            raw_text = output.outputs[0].text
            rubric = parse_rubric_response(
                raw_text,
                max_items=args.rubrics_per_prompt,
                max_chars=args.max_rubric_chars,
            )
            if rubric:
                generated[problem] = rubric
            else:
                next_unresolved.append(problem)
        unresolved = next_unresolved

    if unresolved:
        example = unresolved[0][:160]
        raise RuntimeError(
            f"Failed to parse rubrics for {len(unresolved)}/{len(problems)} problems after "
            f"{args.max_retries + 1} attempts. First problem: {example!r}"
        )
    return generated


def main() -> None:
    args = parse_args()
    http_log_level = getattr(logging, args.http_client_log_level)
    for logger_name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(logger_name).setLevel(http_log_level)
    if args.remote_concurrency <= 0:
        raise ValueError("--remote-concurrency must be positive")
    if args.remote_timeout <= 0:
        raise ValueError("--remote-timeout must be positive")
    if args.remote_max_retries < 0:
        raise ValueError("--remote-max-retries cannot be negative")
    input_path = Path(args.input_parquet).expanduser().resolve()
    output_path = Path(args.output_parquet).expanduser().resolve()
    progress_path = (
        Path(args.progress_file).expanduser().resolve()
        if args.progress_file
        else output_path.with_suffix(".progress.jsonl")
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_path}")

    dataset = load_dataset("parquet", data_files=str(input_path), split="train")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.prompt_column not in dataset.column_names:
        raise KeyError(
            f"Prompt column {args.prompt_column!r} is absent. Available columns: {dataset.column_names}"
        )

    row_problems = [extract_problem_text(prompt) for prompt in dataset[args.prompt_column]]
    unique_problems = list(dict.fromkeys(row_problems))
    completed = load_progress(progress_path)
    pending = [problem for problem in unique_problems if problem_key(problem) not in completed]

    print(f"Rows: {len(dataset)}")
    print(f"Unique problems: {len(unique_problems)}")
    print(f"Resumed rubrics: {len(completed)}")
    print(f"Pending rubrics: {len(pending)}")

    if pending:
        llm = None
        tokenizer = None
        sampling_params_cls = None
        remote_generator = None
        if args.inference_backend == "vllm":
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
        for start in tqdm(range(0, len(pending), args.batch_size), desc="Generating rubrics"):
            problems = pending[start : start + args.batch_size]
            generated = generate_batch(
                llm,
                tokenizer,
                sampling_params_cls,
                remote_generator,
                problems,
                args,
            )
            records = [(problem_key(problem), generated[problem]) for problem in problems]
            append_progress(progress_path, records)
            completed.update(records)

    missing = [problem for problem in unique_problems if problem_key(problem) not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} rubrics after generation")

    rubrics = [completed[problem_key(problem)] for problem in row_problems]
    valid = [bool(rubric.strip()) for rubric in rubrics]
    for column in (args.rubric_column, args.valid_column):
        if column in dataset.column_names:
            dataset = dataset.remove_columns(column)
    dataset = dataset.add_column(args.rubric_column, rubrics)
    dataset = dataset.add_column(args.valid_column, valid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))

    print(f"Saved dataset: {output_path}")
    print(f"Valid rubrics: {sum(valid)}/{len(valid)}")
    print(f"Progress file: {progress_path}")


if __name__ == "__main__":
    main()
