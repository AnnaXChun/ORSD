#!/usr/bin/env python3
"""Collect fixed-policy ALFWorld/WebShop trajectories with local vLLM.

Important context semantics:
- The model may use thinking during the CURRENT generation.
- Full raw model output is stored in trajectory["model_response"] for analysis.
- Only the extracted `ACTION: ...` is written back into conversation history.
- Therefore old thinking traces do NOT accumulate across turns.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

VERL_ROOT = Path(__file__).resolve().parents[2]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from examples.token_credit_grpo.precompute_rubrics import normalize_messages  # noqa: E402
from verl.interactions.agent_environment_utils import extract_action  # noqa: E402
from verl.interactions.alfworld_interaction import AlfworldInteraction  # noqa: E402
from verl.interactions.webshop_interaction import WebshopInteraction  # noqa: E402

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@dataclass
class Episode:
    row_index: int
    sample_index: int
    instance_id: str
    interaction: Any
    interaction_kwargs: dict[str, Any]
    messages: list[dict[str, str]]
    initial_observation: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    official_reward: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", choices=("alfworld", "webshop"), required=True)

    parser.add_argument("--alfworld-config", default=os.environ.get("ALFWORLD_CONFIG", ""))
    parser.add_argument("--alfworld-data", default=os.environ.get("ALFWORLD_DATA", ""))

    parser.add_argument("--webshop-root", default=os.environ.get("WEBSHOP_ROOT", ""))
    parser.add_argument(
        "--webshop-products-json",
        default=os.environ.get("WEBSHOP_PRODUCTS_JSON", ""),
    )

    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument(
        "--episode-batch-size",
        type=int,
        default=32,
        help="Maximum live environment episodes (tasks x rollouts) per vLLM batch",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override per-row environment limit",
    )

    parser.add_argument("--max-action-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=40960)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--inject-rubric",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject each row's non-empty rubric into the environment prompt",
    )

    parser.add_argument(
        "--replace-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--debug-first-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the first rendered actor prompt in each collection batch",
    )

    parser.add_argument(
        "--debug-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print prompt-token growth at every environment step",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=8,
        help="Number of recent assistant/user turns kept in conversation history",
    )
    return parser.parse_args()


def make_interaction(args: argparse.Namespace):
    if args.benchmark == "alfworld":
        if not args.alfworld_config or not args.alfworld_data:
            raise ValueError(
                "ALFWorld collection requires --alfworld-config and --alfworld-data"
            )

        return AlfworldInteraction(
            {
                "name": "alfworld",
                "config_path": args.alfworld_config,
                "data_dir": args.alfworld_data,
                "seed": args.seed,
            }
        )

    if not args.webshop_root:
        raise ValueError("WebShop collection requires --webshop-root")

    return WebshopInteraction(
        {
            "name": "webshop",
            "webshop_root": args.webshop_root,
            "file_path": args.webshop_products_json,
            "observation_mode": "text_rich",
            "seed": args.seed,
        }
    )
import ast
def interaction_kwargs(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:

    extra_info = row.get("extra_info") or {}

    if isinstance(extra_info, str):

        try:
            extra_info = json.loads(extra_info)

        except Exception:

            try:
                extra_info = ast.literal_eval(extra_info)

            except Exception:
                extra_info = {}


    values = dict(
        extra_info.get("interaction_kwargs") or {}
    )


    # WebShop official human goal index
    if args.benchmark == "webshop":

        if "goal_idx" not in values:

            if "goal_idx" in row:
                values["goal_idx"] = int(
                    row["goal_idx"]
                )

            else:
                raise ValueError(
                    "WebShop row missing goal_idx"
                )


    rubric = str(
        row.get("rubric") or ""
    ).strip()


    inject = bool(
        args.inject_rubric and rubric
    )


    values["inject_rubric"] = inject

    values["rubric"] = (
        rubric if inject else ""
    )

    if args.max_steps > 0:
        values["max_steps"] = args.max_steps


    return values
async def initialize_episodes(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    interaction: Any,
    row_offset: int,
) -> list[Episode]:
    episodes: list[Episode] = []

    try:
        for local_row_index, row in enumerate(rows):
            row_index = row_offset + local_row_index

            for sample_index in range(args.num_rollouts):
                instance_id = f"offline-{row_index}-{sample_index}"
                kwargs = interaction_kwargs(row, args)

                await interaction.start_interaction(
                    instance_id,
                    **kwargs,
                )

                try:
                    messages = normalize_messages(row["prompt"])

                    initial = await interaction.get_initial_observation(
                        instance_id,
                        **kwargs,
                    )

                except BaseException:
                    await interaction.finalize_interaction(
                        instance_id,
                        **kwargs,
                    )
                    raise

                messages.append(
                    {
                        "role": "user",
                        "content": initial,
                    }
                )

                episodes.append(
                    Episode(
                        row_index=row_index,
                        sample_index=sample_index,
                        instance_id=instance_id,
                        interaction=interaction,
                        interaction_kwargs=kwargs,
                        messages=messages,
                        initial_observation=initial,
                    )
                )

    except BaseException:
        for episode in episodes:
            await episode.interaction.finalize_interaction(
                episode.instance_id,
                **episode.interaction_kwargs,
            )
        raise

    return episodes


def render_prompts(
    tokenizer: Any,
    episodes: list[Episode],
    enable_thinking: bool,
) -> list[str]:
    prompts: list[str] = []

    for episode in episodes:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }

        try:
            prompt = tokenizer.apply_chat_template(
                episode.messages,
                **kwargs,
            )
        except TypeError:
            kwargs.pop("enable_thinking")
            prompt = tokenizer.apply_chat_template(
                episode.messages,
                **kwargs,
            )

        prompts.append(prompt)

    return prompts

def history_response_from_raw(raw_response: str) -> tuple[str, str]:
    """Keep only the externally visible action in future history."""

    action = str(extract_action(raw_response) or "").strip()

    if action:
        return action, f"ACTION: {action}"

    # Do not retain raw reasoning when action parsing fails.
    # This intentionally becomes an invalid environment action.
    return "", "ACTION: __INVALID__"

def trim_episode_history(
    episode: Episode,
    *,
    keep_prefix_messages: int = 3,
    max_history_turns: int = 8,
) -> None:
    """
    Keep the initial prompt permanently:

        system
        task
        initial observation (+ rubric)

    Afterwards only keep the newest assistant/user turns.
    """

    if max_history_turns <= 0:
        return

    prefix = episode.messages[:keep_prefix_messages]

    history = episode.messages[keep_prefix_messages:]

    max_history_messages = max_history_turns * 2

    if len(history) > max_history_messages:
        history = history[-max_history_messages:]

    episode.messages = prefix + history
async def run_collection_batch(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    interaction: Any,
    llm: Any,
    tokenizer: Any,
    sampling_params_cls: Any,
    row_offset: int,
) -> list[Episode]:
    episodes = await initialize_episodes(
        rows,
        args,
        interaction,
        row_offset,
    )

    max_steps = max(
        episode.interaction_kwargs.get("max_steps", 100)
        for episode in episodes
    )

    if args.inject_rubric:
        injected = sum(
            bool(episode.interaction_kwargs.get("rubric"))
            for episode in episodes
        )
        print(
            f"[rubric] batch rows={row_offset}-{row_offset + len(rows) - 1} "
            f"injected_episodes={injected}/{len(episodes)}",
            flush=True,
        )

    try:
        for step_index in range(max_steps):
            active = [
                episode
                for episode in episodes
                if not episode.done
            ]

            if not active:
                break

            prompts = render_prompts(
                tokenizer,
                active,
                args.enable_thinking,
            )

            prompt_token_counts = [
                len(
                    tokenizer.encode(
                        prompt,
                        add_special_tokens=False,
                    )
                )
                for prompt in prompts
            ]

            if args.debug_context and prompt_token_counts:
                print(
                    f"[context] step={step_index + 1} "
                    f"active={len(active)} "
                    f"min={min(prompt_token_counts)} "
                    f"mean={sum(prompt_token_counts) / len(prompt_token_counts):.1f} "
                    f"max={max(prompt_token_counts)}",
                    flush=True,
                )

            if args.debug_first_prompt and step_index == 0 and prompts:
                print(
                    "\n"
                    "========== ACTOR FIRST PROMPT ==========\n"
                    f"{prompts[0]}\n"
                    "========================================\n",
                    flush=True,
                )

            # Fail early with a useful error before vLLM's own validation.
            too_long = [
                (episode.instance_id, count)
                for episode, count in zip(
                    active,
                    prompt_token_counts,
                    strict=True,
                )
                if count >= args.max_model_len
            ]

            if too_long:
                details = ", ".join(
                    f"{instance_id}:{count}"
                    for instance_id, count in too_long
                )
                raise RuntimeError(
                    "Actor prompt exceeded model context before generation. "
                    f"max_model_len={args.max_model_len}; "
                    f"episodes={details}. "
                    "Old thinking is already excluded from history in this version; "
                    "if this still occurs, add observation/history compression."
                )

            params = [
                sampling_params_cls(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_action_tokens,
                    seed=(
                        args.seed
                        + episode.row_index * 100_000
                        + episode.sample_index * 1_000
                        + step_index
                    ),
                )
                for episode in active
            ]

            outputs = llm.generate(
                prompts,
                params,
                use_tqdm=False,
            )

            for episode, output in zip(
                active,
                outputs,
                strict=True,
            ):
                completion = output.outputs[0]

                # Full current-step generation, including possible <think>.
                raw_response = completion.text

                # Only ACTION is retained in future context.
                action, history_response = history_response_from_raw(
                    raw_response
                )

                episode.messages.append(
                    {
                        "role": "assistant",
                        "content": history_response,
                    }
                )

                terminate, environment_response, _, metrics = (
                    await episode.interaction.generate_response(
                        episode.instance_id,
                        episode.messages,
                        **episode.interaction_kwargs,
                    )
                )

                episode.steps.append(
                    {
                        "step": step_index,

                        # Full raw Qwen output for offline analysis.
                        "model_response": raw_response,

                        # Exact assistant text retained in future context.
                        "history_response": history_response,

                        "prompt_token_count": len(
                            output.prompt_token_ids
                        ),
                        "completion_token_count": len(
                            completion.token_ids
                        ),
                        "finish_reason": completion.finish_reason,

                        "action": action,
                        "environment_response": environment_response,
                        "metrics": metrics,
                    }
                )

                episode.messages.append(
                    {
                        "role": "user",
                        "content": environment_response,
                    }
                )

                trim_episode_history(
                    episode,
                    max_history_turns=args.history_window,
                )

                episode.done = bool(terminate)

            print(
                f"[rollout] "
                f"rows={row_offset}-{row_offset + len(rows) - 1} "
                f"step={step_index + 1} "
                f"active={sum(not episode.done for episode in episodes)}/"
                f"{len(episodes)}",
                flush=True,
            )

    finally:
        for episode in episodes:
            if episode.instance_id not in episode.interaction._instance_dict:
                continue

            episode.official_reward = (
                await episode.interaction.calculate_score(
                    episode.instance_id,
                    **episode.interaction_kwargs,
                )
            )

            await episode.interaction.finalize_interaction(
                episode.instance_id,
                **episode.interaction_kwargs,
            )

    return episodes


async def run_collection(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[Episode]:
    from vllm import LLM, SamplingParams

    interaction = make_interaction(args)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()

    rows_per_batch = max(
        1,
        args.episode_batch_size // args.num_rollouts,
    )

    episodes: list[Episode] = []

    for row_offset in range(
        0,
        len(rows),
        rows_per_batch,
    ):
        batch_rows = rows[
            row_offset:row_offset + rows_per_batch
        ]

        episodes.extend(
            await run_collection_batch(
                batch_rows,
                args,
                interaction,
                llm,
                tokenizer,
                SamplingParams,
                row_offset,
            )
        )

    return episodes


def main() -> None:
    args = parse_args()

    if (
        args.num_rollouts < 1
        or args.max_action_tokens < 1
        or args.episode_batch_size < args.num_rollouts
    ):
        raise ValueError(
            "num-rollouts/max-action-tokens must be positive and "
            "episode-batch-size >= num-rollouts"
        )

    dataset = load_dataset(
        "parquet",
        data_files=str(
            Path(args.input_parquet).resolve()
        ),
        split="train",
    )

    if args.limit > 0:
        dataset = dataset.select(
            range(
                min(
                    args.limit,
                    len(dataset),
                )
            )
        )

    rows = [
        dict(row)
        for row in dataset
    ]

    if args.inject_rubric:
        nonempty = sum(
            bool(
                str(
                    row.get("rubric") or ""
                ).strip()
            )
            for row in rows
        )

        print(
            f"[rubric] non-empty task rubrics="
            f"{nonempty}/{len(rows)}",
            flush=True,
        )

    episodes = asyncio.run(
        run_collection(
            rows,
            args,
        )
    )

    by_row: dict[int, list[dict[str, Any]]] = {
        index: []
        for index in range(len(rows))
    }

    for episode in episodes:
        by_row[
            episode.row_index
        ].append(
            {
                "source": (
                    f"fixed_policy_{args.model}_"
                    f"{'rubric' if args.inject_rubric else 'baseline'}"
                ),
                "sample_index": episode.sample_index,
                "official_reward": episode.official_reward,
                "success": (
                    episode.official_reward
                    >= 1.0 - 1e-8
                ),
                "initial_observation": episode.initial_observation,
                "steps": episode.steps,
            }
        )

    for row_index, row in enumerate(rows):
        previous = []

        if not args.replace_existing:
            try:
                previous = json.loads(
                    row.get(
                        "rollout_trajectories_json"
                    )
                    or "[]"
                )
            except json.JSONDecodeError:
                previous = []

        row["rollout_trajectories_json"] = (
            json.dumps(
                previous + by_row[row_index],
                ensure_ascii=False,
            )
        )

    output_path = (
        Path(args.output_parquet)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Dataset.from_list(
        rows
    ).to_parquet(
        str(output_path)
    )

    rewards = [
        episode.official_reward
        for episode in episodes
    ]

    print(
        f"Saved {len(episodes)} trajectories "
        f"to {output_path}"
    )

    print(
        "Official mean reward: "
        f"{sum(rewards) / max(len(rewards), 1):.4f}"
    )


if __name__ == "__main__":
    main()
