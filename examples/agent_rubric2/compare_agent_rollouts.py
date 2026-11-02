#!/usr/bin/env python3
"""Compare paired baseline and rubric-conditioned agent rollouts.

Context audit matches the real rollout semantics:
- model_response = full raw generation, possibly including thinking
- history_response = assistant text actually retained in future context
- when available, history_response is used to reconstruct later prompts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-parquet", required=True)
    parser.add_argument("--rubric-parquet", required=True)
    parser.add_argument("--output-json", required=True)

    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer path for context-length auditing",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
    )

    parser.add_argument(
        "--max-action-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--history-window",
        type=int,
        default=8,
        help="Number of recent assistant/user turns retained in context audit.",
    )

    return parser.parse_args()


def load_json(
    value: Any,
) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(
            value or "[]"
        )

    return list(
        value or []
    )


def key(
    row: dict[str, Any],
) -> str:
    return str(
        row.get("task_id")
        or (
            row.get("extra_info")
            or {}
        ).get("task_id")
        or row.get("task")
    )


def invalid_step(
    step: dict[str, Any],
) -> bool:
    metrics = (
        step.get("metrics")
        or {}
    )

    if metrics.get(
        "invalid_action"
    ) is not None:
        return bool(
            metrics[
                "invalid_action"
            ]
        )

    response = str(
        step.get(
            "environment_response"
        )
        or ""
    ).lower()

    return any(
        marker in response
        for marker in (
            "invalid action",
            "not a valid action",
            "nothing happens",
        )
    )


def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trajectories = [
        trajectory
        for row in rows
        for trajectory in load_json(
            row.get(
                "rollout_trajectories_json"
            )
        )
    ]

    rewards = [
        float(
            trajectory.get(
                "official_reward",
                0.0,
            )
        )
        for trajectory in trajectories
    ]

    steps = [
        step
        for trajectory in trajectories
        for step in trajectory.get(
            "steps",
            [],
        )
    ]

    unique_actions = [
        len(
            {
                str(
                    step.get("action")
                    or ""
                ).strip()
                for step in trajectory.get(
                    "steps",
                    [],
                )
                if step.get("action")
            }
        )
        for trajectory in trajectories
    ]

    return {
        "num_tasks": len(rows),
        "num_trajectories": len(
            trajectories
        ),
        "official_reward_mean": (
            sum(rewards)
            / max(
                len(rewards),
                1,
            )
        ),
        "success_count": sum(
            reward >= 1.0 - 1e-8
            for reward in rewards
        ),
        "success_rate": (
            sum(
                reward >= 1.0 - 1e-8
                for reward in rewards
            )
            / max(
                len(rewards),
                1,
            )
        ),
        "invalid_action_rate": (
            sum(
                invalid_step(step)
                for step in steps
            )
            / max(
                len(steps),
                1,
            )
        ),
        "unique_actions_mean": (
            sum(unique_actions)
            / max(
                len(unique_actions),
                1,
            )
        ),
    }


def task_reward(
    row: dict[str, Any],
) -> float:
    rewards = [
        float(
            item.get(
                "official_reward",
                0.0,
            )
        )
        for item in load_json(
            row.get(
                "rollout_trajectories_json"
            )
        )
    ]

    return (
        sum(rewards)
        / max(
            len(rewards),
            1,
        )
    )


def normalize_messages(
    value: Any,
) -> list[dict[str, str]]:
    if (
        hasattr(
            value,
            "tolist",
        )
        and not isinstance(
            value,
            (
                str,
                bytes,
                list,
                tuple,
                dict,
            ),
        )
    ):
        value = value.tolist()

    if isinstance(
        value,
        str,
    ):
        try:
            value = json.loads(
                value
            )
        except json.JSONDecodeError:
            return [
                {
                    "role": "user",
                    "content": value,
                }
            ]

    if isinstance(
        value,
        dict,
    ):
        value = value.get(
            "messages",
            [value],
        )

    return [
        {
            "role": str(
                message.get(
                    "role",
                    "user",
                )
            ),
            "content": str(
                message.get(
                    "content",
                    "",
                )
            ),
        }
        for message in list(
            value or []
        )
        if isinstance(
            message,
            dict,
        )
    ]


def percentile(
    values: list[int],
    fraction: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    index = min(
        len(ordered) - 1,
        max(
            0,
            int(
                round(
                    (
                        len(ordered)
                        - 1
                    )
                    * fraction
                )
            ),
        ),
    )

    return float(
        ordered[index]
    )


def context_summary(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    enable_thinking: bool,
    max_model_len: int,
    max_action_tokens: int,
    history_window: int,
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    first_step_prompt_lengths: list[int] = []

    # Raw completion lengths measure compute/output behavior.
    raw_completion_lengths: list[int] = []

    # History completion lengths measure what is actually retained in context.
    history_completion_lengths: list[int] = []

    context_limited_steps = 0

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }

    for row in rows:
        base_messages = normalize_messages(
            row.get("prompt")
            or row.get("raw_prompt")
        )

        for trajectory in load_json(
            row.get(
                "rollout_trajectories_json"
            )
        ):
            messages = [
                dict(message)
                for message in base_messages
            ]

            # For rubric rollouts the initial observation already contains
            # the injected privileged hints.
            messages.append(
                {
                    "role": "user",
                    "content": str(
                        trajectory.get(
                            "initial_observation"
                        )
                        or ""
                    ),
                }
            )

            for step_index, step in enumerate(
                trajectory.get(
                    "steps",
                    [],
                )
            ):
                try:
                    prompt_text = (
                        tokenizer.apply_chat_template(
                            messages,
                            **template_kwargs,
                        )
                    )
                except TypeError:
                    fallback_kwargs = dict(
                        template_kwargs
                    )
                    fallback_kwargs.pop(
                        "enable_thinking"
                    )

                    prompt_text = (
                        tokenizer.apply_chat_template(
                            messages,
                            **fallback_kwargs,
                        )
                    )

                prompt_length = len(
                    tokenizer.encode(
                        prompt_text,
                        add_special_tokens=False,
                    )
                )

                raw_response = str(
                    step.get(
                        "model_response"
                    )
                    or ""
                )

                history_response = str(
                    step.get(
                        "history_response"
                    )
                    or raw_response
                )

                raw_completion_length = len(
                    tokenizer.encode(
                        raw_response,
                        add_special_tokens=False,
                    )
                )

                history_completion_length = len(
                    tokenizer.encode(
                        history_response,
                        add_special_tokens=False,
                    )
                )

                prompt_lengths.append(
                    prompt_length
                )

                raw_completion_lengths.append(
                    raw_completion_length
                )

                history_completion_lengths.append(
                    history_completion_length
                )

                if step_index == 0:
                    first_step_prompt_lengths.append(
                        prompt_length
                    )

                context_limited_steps += int(
                    prompt_length
                    + max_action_tokens
                    > max_model_len
                )

                # IMPORTANT:
                # use only the content that the rollout really retained.
                messages.append(
                    {
                        "role": "assistant",
                        "content": history_response,
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": str(
                            step.get(
                                "environment_response"
                            )
                            or ""
                        ),
                    }
                )
                if history_window > 0 and len(messages) > 3 + history_window * 2:
                    prefix = messages[:3]
                    history = messages[3:]
                    history = history[-history_window * 2:]
                    messages = prefix + history

    count = len(
        prompt_lengths
    )

    return {
        "num_generation_steps": count,

        "prompt_tokens_mean": (
            sum(prompt_lengths)
            / max(
                count,
                1,
            )
        ),

        "prompt_tokens_p95": percentile(
            prompt_lengths,
            0.95,
        ),

        "prompt_tokens_max": max(
            prompt_lengths,
            default=0,
        ),

        "first_step_prompt_tokens_mean": (
            sum(
                first_step_prompt_lengths
            )
            / max(
                len(
                    first_step_prompt_lengths
                ),
                1,
            )
        ),

        # Full Qwen output, including current-step thinking.
        "raw_completion_tokens_mean": (
            sum(
                raw_completion_lengths
            )
            / max(
                len(
                    raw_completion_lengths
                ),
                1,
            )
        ),

        "raw_completion_tokens_p95": percentile(
            raw_completion_lengths,
            0.95,
        ),

        "raw_completion_tokens_max": max(
            raw_completion_lengths,
            default=0,
        ),

        # Only ACTION text retained in next-turn history.
        "history_completion_tokens_mean": (
            sum(
                history_completion_lengths
            )
            / max(
                len(
                    history_completion_lengths
                ),
                1,
            )
        ),

        "history_completion_tokens_p95": percentile(
            history_completion_lengths,
            0.95,
        ),

        "history_completion_tokens_max": max(
            history_completion_lengths,
            default=0,
        ),

        "raw_completion_near_cap_frac": (
            sum(
                value
                >= 0.95
                * max_action_tokens
                for value
                in raw_completion_lengths
            )
            / max(
                len(
                    raw_completion_lengths
                ),
                1,
            )
        ),

        "context_limited_step_count": (
            context_limited_steps
        ),

        "context_limited_step_frac": (
            context_limited_steps
            / max(
                count,
                1,
            )
        ),

        "minimum_requested_generation_headroom": (
            max_model_len
            - max(
                (
                    value
                    + max_action_tokens
                    for value
                    in prompt_lengths
                ),
                default=0,
            )
        ),
    }


def rubric_text(
    row: dict[str, Any],
) -> str:
    return str(
        row.get("rubric")
        or ""
    ).strip()


def main() -> None:
    args = parse_args()

    baseline_rows = [
        dict(row)
        for row in load_dataset(
            "parquet",
            data_files=(
                args.baseline_parquet
            ),
            split="train",
        )
    ]

    rubric_rows = [
        dict(row)
        for row in load_dataset(
            "parquet",
            data_files=(
                args.rubric_parquet
            ),
            split="train",
        )
    ]

    baseline_by_key = {
        key(row): row
        for row in baseline_rows
    }

    rubric_by_key = {
        key(row): row
        for row in rubric_rows
    }

    common = sorted(
        baseline_by_key.keys()
        & rubric_by_key.keys()
    )

    if not common:
        raise RuntimeError(
            "No paired tasks were found "
            "between baseline and rubric rollouts"
        )

    differences = [
        task_reward(
            rubric_by_key[item]
        )
        - task_reward(
            baseline_by_key[item]
        )
        for item in common
    ]

    rubric_present_by_key = {
        item: bool(
            rubric_text(
                rubric_by_key[item]
            )
        )
        for item in common
    }

    valid_task_count = sum(
        rubric_present_by_key.values()
    )

    result = {
        "baseline": summarize(
            [
                baseline_by_key[item]
                for item in common
            ]
        ),

        "rubric_privileged": summarize(
            [
                rubric_by_key[item]
                for item in common
            ]
        ),

        "paired": {
            "num_tasks": len(common),

            "reward_delta": (
                sum(differences)
                / len(differences)
            ),

            "wins": sum(
                value > 1e-8
                for value in differences
            ),

            "ties": sum(
                abs(value) <= 1e-8
                for value in differences
            ),

            "losses": sum(
                value < -1e-8
                for value in differences
            ),
        },

        "rubric_coverage": (
            valid_task_count
            / len(common)
        ),

        "rubric_valid_task_count": (
            valid_task_count
        ),

        "rubric_missing_task_count": (
            len(common)
            - valid_task_count
        ),

        # Explicitly no fallback.
        "baseline_fallback_task_count": 0,
    }

    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = (
            AutoTokenizer.from_pretrained(
                args.tokenizer,
                trust_remote_code=True,
            )
        )

        baseline_context = (
            context_summary(
                [
                    baseline_by_key[item]
                    for item in common
                ],
                tokenizer,
                enable_thinking=(
                    args.enable_thinking
                ),
                max_model_len=(
                    args.max_model_len
                ),
                max_action_tokens=(
                    args.max_action_tokens
                ),
                history_window=(
                    args.history_window
                ),
            )
        )

        rubric_context = (
            context_summary(
                [
                    rubric_by_key[item]
                    for item in common
                ],
                tokenizer,
                enable_thinking=(
                    args.enable_thinking
                ),
                max_model_len=(
                    args.max_model_len
                ),
                max_action_tokens=(
                    args.max_action_tokens
                ),
                history_window=(
                    args.history_window
                ),
            )
        )

        result[
            "context_audit"
        ] = {
            "max_model_len": (
                args.max_model_len
            ),

            "max_tokens_per_action": (
                args.max_action_tokens
            ),

            "enable_thinking": (
                args.enable_thinking
            ),

            "past_thinking_retained": False,

            "history_window": (
                args.history_window
            ),

            "baseline": baseline_context,

            "rubric_privileged": (
                rubric_context
            ),

            "rubric_first_step_overhead_tokens": (
                rubric_context[
                    "first_step_prompt_tokens_mean"
                ]
                - baseline_context[
                    "first_step_prompt_tokens_mean"
                ]
            ),
        }

    output = (
        Path(args.output_json)
        .expanduser()
        .resolve()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"Summary saved to: {output}"
    )


if __name__ == "__main__":
    main()
