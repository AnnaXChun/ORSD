# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
from copy import deepcopy
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.turn_context import (
    TURN_REPLAY_KEY,
    AgentTrajectoryRecord,
    AgentTurnRecord,
    ContextBuilder,
    clip_trajectory_record,
    validate_trajectory_record,
)
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.agent_environment_utils import extract_action
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_legacy_generation_budget(
    *,
    prompt_token_count: int,
    response_token_count: int,
    prompt_length: int,
    response_length: int,
    max_model_len: Optional[int],
    max_response_tokens_per_turn: Optional[int],
    requested_max_tokens: Optional[int],
) -> tuple[int, int, int]:
    """Return a safe next-turn budget for VERL's append-only agent context.

    The legacy path retains every policy response and observation in one
    growing sequence.  Never rely on vLLM to truncate that sequence: doing so
    would make rollout and PPO/GRPO replay condition on different contexts.
    Instead, cap the next generation by both the configured trajectory budget
    and the actual model context.  A zero budget means the current trajectory
    must terminate as an ordinary failed/unfinished episode.
    """

    sequence_limit = int(prompt_length) + int(response_length)
    if max_model_len is not None:
        sequence_limit = min(sequence_limit, int(max_model_len))

    remaining_context_tokens = max(sequence_limit - int(prompt_token_count), 0)
    remaining_trajectory_tokens = max(int(response_length) - int(response_token_count), 0)
    turn_max_tokens = min(remaining_context_tokens, remaining_trajectory_tokens)

    if max_response_tokens_per_turn is not None:
        turn_max_tokens = min(turn_max_tokens, int(max_response_tokens_per_turn))
    if requested_max_tokens is not None:
        turn_max_tokens = min(turn_max_tokens, int(requested_max_tokens))

    return max(turn_max_tokens, 0), remaining_context_tokens, remaining_trajectory_tokens


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.context_builder = context_builder
        self.turn_replay_enabled = context_builder is not None

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.stop_after_interaction = False

        # The legacy path continues to use ``prompt_ids`` as one growing
        # sequence. The turn-replay path keeps the actual generation context
        # in ``prompt_ids`` and stores the training trajectory separately.
        self.trajectory_prompt_ids: list[int] = []
        self.flat_response_ids: list[int] = []
        self.fixed_prefix = (
            deepcopy(messages[: context_builder.fixed_prefix_messages]) if context_builder is not None else []
        )
        self.history_turns: list[dict[str, str]] = []
        self.turn_replay = (
            AgentTrajectoryRecord(
                history_window=context_builder.history_window,
                fixed_prefix_messages=context_builder.fixed_prefix_messages,
            )
            if context_builder is not None
            else None
        )
        self.pending_turn: Optional[AgentTurnRecord] = None

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition, e.g., tool session data.  Keep
        # keys that can vary by termination path present on every trajectory:
        # AgentLoop workers are concatenated later and DataProto.concat
        # requires identical non-tensor schemas across worker outputs.
        self.extra_fields: dict[str, Any] = {"termination_reason": ""}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config
        self.disable_rubric = config.get("credit_assignment", {}).get("type") in {"grpo", "attention_traceback", "attention_traceback_train"}
        self.attention_capture = config.get("credit_assignment", {}).get("type") in {"attention_traceback", "attention_traceback_train"}
        self.attention_traceback_config = config.get("attention_traceback", {})
        self.independent_sampling = self.attention_traceback_config.get("independent_sampling", False)

        # Initialize tools from config file
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        self.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        multi_turn_config = config.actor_rollout_ref.rollout.multi_turn
        self.history_window = int(multi_turn_config.get("history_window", 0) or 0)
        self.history_fixed_prefix_messages = int(
            multi_turn_config.get("history_fixed_prefix_messages", 3) or 3
        )
        self.history_response_mode = str(
            multi_turn_config.get("history_response_mode", "action_only") or "action_only"
        )
        self.history_min_generation_tokens = int(
            multi_turn_config.get("history_min_generation_tokens", 2048) or 2048
        )
        configured_turn_cap = multi_turn_config.get("max_response_tokens_per_turn", None)
        self.max_response_tokens_per_turn = (
            int(configured_turn_cap) if configured_turn_cap is not None else None
        )
        configured_max_model_len = config.actor_rollout_ref.rollout.max_model_len
        self.max_model_len = int(configured_max_model_len) if configured_max_model_len is not None else None

        # Initialize interactions from config file
        self.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        interaction_message_start = len(messages)

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        interaction_final_reward = None
        interaction_reward_details: dict[str, float] = {}
        interaction_initial_observation = None
        if self.interaction_config_file:
            interaction_kwargs = dict(kwargs["extra_info"]["interaction_kwargs"])
            if self.disable_rubric:
                interaction_kwargs.update(inject_rubric=False, rubric=None)
            if self.attention_capture:
                interaction_kwargs.update(use_interaction_reward=True)
                if interaction_kwargs.get("name") == "webshop":
                    interaction_kwargs.update(max_steps=15, use_small=True, human_goals=False, num_products=1000)
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
            try:
                initial_observation = await interaction.get_initial_observation(request_id, **interaction_kwargs)
            except BaseException:
                await interaction.finalize_interaction(request_id, **interaction_kwargs)
                raise
            if initial_observation:
                messages.append({"role": "user", "content": initial_observation})
                interaction_initial_observation = initial_observation
        try:
            context_builder = None
            if self.history_window > 0:
                if interaction is None:
                    raise ValueError(
                        "multi_turn.history_window currently requires an environment interaction with parseable actions; "
                        "leave it at 0 for VERL's original tool/chat rollout"
                    )
                if len(messages) != self.history_fixed_prefix_messages:
                    raise ValueError(
                        "history-window rollout requires the context to consist exactly of the fixed prefix "
                        "(system, task, initial observation) before the first action: "
                        f"messages={len(messages)}, configured={self.history_fixed_prefix_messages}"
                    )
                context_builder = ContextBuilder(
                    history_window=self.history_window,
                    fixed_prefix_messages=self.history_fixed_prefix_messages,
                    history_response_mode=self.history_response_mode,
                )
            # Create AgentData instance to encapsulate all state.
            agent_data = AgentData(
                messages=messages,
                image_data=images,
                video_data=videos,
                metrics=metrics,
                request_id=request_id,
                tools_kwargs=tools_kwargs,
                interaction=interaction,
                interaction_kwargs=interaction_kwargs,
                context_builder=context_builder,
            )
        except BaseException:
            if interaction is not None:
                try:
                    await interaction.finalize_interaction(request_id, **interaction_kwargs)
                except Exception:
                    logger.exception("Failed to finalize interaction %s during rollout setup", request_id)
            raise

        # State machine loop. Environment-backed interactions opt into using the
        # official terminal score as GRPO reward via ``use_interaction_reward``.
        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED
        except BaseException:
            if interaction is not None:
                try:
                    await interaction.finalize_interaction(request_id, **interaction_kwargs)
                except Exception:
                    logger.exception("Failed to finalize interaction %s after rollout error", request_id)
            raise
        else:
            if interaction is not None:
                try:
                    if interaction_kwargs.get("use_interaction_reward", False):
                        interaction_final_reward = float(
                            await interaction.calculate_score(request_id, **interaction_kwargs)
                        )
                        interaction_reward_details = await interaction.get_reward_details(
                            request_id, **interaction_kwargs
                        )
                finally:
                    await interaction.finalize_interaction(request_id, **interaction_kwargs)

        # Finalize output
        if agent_data.turn_replay_enabled:
            prompt_ids = agent_data.trajectory_prompt_ids
            response_ids = agent_data.flat_response_ids
        else:
            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
            reward_score=interaction_final_reward,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.update({k: v for k, v in agent_data.extra_fields.items() if k != "_attention_turns"})
        if agent_data.turn_replay_enabled:
            replay_record = clip_trajectory_record(
                agent_data.turn_replay.to_dict(),
                self.response_length,
            )
            validate_trajectory_record(
                replay_record,
                response_length=len(output.response_ids),
                flattened_response_ids=output.response_ids,
                response_mask=output.response_mask,
                response_attention_mask=[1] * len(output.response_ids),
            )
            output.extra_fields[TURN_REPLAY_KEY] = replay_record
        if interaction_initial_observation is not None:
            output.extra_fields["interaction_initial_observation"] = interaction_initial_observation
        if interaction is not None and agent_data.turn_replay_enabled:
            transcript_parts = []
            for turn in agent_data.turn_replay.turns:
                transcript_parts.append(turn.history_response)
                if turn.observation:
                    transcript_parts.append(f"OBSERVATION: {turn.observation}")
            output.extra_fields["interaction_trajectory_transcript"] = "\n".join(transcript_parts)
        elif interaction is not None:
            transcript_parts: list[str] = []
            for message in agent_data.messages[interaction_message_start:]:
                role = str(message.get("role", ""))
                content = str(message.get("content", "")).strip()
                if role == "assistant":
                    action = extract_action(content)
                    if action:
                        transcript_parts.append(f"ACTION: {action}")
                elif role == "user" and content:
                    transcript_parts.append(f"OBSERVATION: {content}")
            output.extra_fields["interaction_trajectory_transcript"] = "\n".join(transcript_parts)
        if interaction_final_reward is not None:
            output.extra_fields["reward_extra_info"] = {
                "score": interaction_final_reward,
                # Keep benchmark-native accuracy/reward visible even when the
                # policy is optimized with a shaped interaction reward.
                "acc": interaction_reward_details.get("official_reward", interaction_final_reward),
                "rubric_injected": float(bool(interaction_kwargs.get("inject_rubric", False))),
                # Agent-loop metrics are otherwise kept only in DataProto
                # metadata and are not aggregated by validation. Mirror the
                # full-history context diagnostics into reward_extra_info so
                # W&B exposes their mean@N rates alongside benchmark metrics.
                "legacy_context_budget_terminated": float(
                    agent_data.extra_fields.get("termination_reason") == "context_budget_exhausted"
                ),
                "legacy_context_limited_turns": float(
                    agent_data.metrics.get("legacy_context_limited_turns", 0.0)
                ),
                "legacy_context_remaining_tokens_min": float(
                    agent_data.metrics.get("legacy_context_remaining_tokens_min", -1.0)
                ),
                "legacy_context_prompt_tokens_max": float(
                    agent_data.metrics.get("legacy_context_prompt_tokens_max", 0.0)
                ),
                **interaction_reward_details,
            }
        if self.attention_capture:
            from verl.experimental.attention_traceback.capture import finish_record
            if interaction_final_reward is None:
                raise ValueError("Traceback requires real environment outcomes")
            output.extra_fields["attention_traceback_record"] = finish_record(
                agent_data, self.tokenizer, self.attention_traceback_config, kwargs,
                interaction_final_reward, interaction_reward_details, sampling_params,
            )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if agent_data.turn_replay_enabled:
            # vLLM's implicit max_tokens is bounded by prompt_length +
            # response_length as well as max_model_len. Reserve enough room
            # for one thinking/action response and, when necessary, remove the
            # oldest complete projected turn. Never truncate prompt token IDs:
            # the chosen messages and IDs are recorded verbatim for PPO replay.
            generation_budget = self.prompt_length + self.response_length
            if self.max_model_len is not None:
                generation_budget = min(generation_budget, self.max_model_len)
            generation_reserve = self.history_min_generation_tokens
            if self.max_response_tokens_per_turn is not None:
                generation_reserve = min(generation_reserve, self.max_response_tokens_per_turn)
            max_prompt_tokens = generation_budget - generation_reserve
            if max_prompt_tokens < 1:
                raise ValueError(
                    "history_min_generation_tokens leaves no room for the fixed prompt: "
                    f"prompt_length={self.prompt_length}, response_length={self.response_length}, "
                    f"max_model_len={self.max_model_len}, "
                    f"generation_reserve={generation_reserve}"
                )

            prompt_ids = None
            dropped_turns = 0
            last_prompt_length = 0
            for candidate_messages, candidate_dropped_turns in agent_data.context_builder.build_candidates(
                agent_data.fixed_prefix,
                agent_data.history_turns,
            ):
                candidate_prompt_ids = await self.apply_chat_template(
                    candidate_messages,
                    tools=self.tool_schemas,
                    images=agent_data.image_data,
                    videos=agent_data.video_data,
                )
                last_prompt_length = len(candidate_prompt_ids)
                if last_prompt_length <= max_prompt_tokens:
                    agent_data.messages = candidate_messages
                    prompt_ids = candidate_prompt_ids
                    dropped_turns = candidate_dropped_turns
                    break

            if prompt_ids is None:
                raise ValueError(
                    "The fixed agent prefix exceeds the safe rollout prompt budget; it cannot be shortened "
                    "without changing the task. Increase max_prompt_length/max_response_length or max_model_len, "
                    "or lower history_min_generation_tokens. "
                    f"fixed_prefix_tokens={last_prompt_length}, max_prompt_tokens={max_prompt_tokens}"
                )
            agent_data.metrics["history_turns_dropped_for_context"] = (
                agent_data.metrics.get("history_turns_dropped_for_context", 0) + dropped_turns
            )
            agent_data.metrics["history_context_prompt_tokens_max"] = max(
                agent_data.metrics.get("history_context_prompt_tokens_max", 0),
                len(prompt_ids),
            )
        else:
            prompt_ids = await self.apply_chat_template(
                agent_data.messages,
                tools=self.tool_schemas,
                images=agent_data.image_data,
                videos=agent_data.video_data,
            )
        agent_data.prompt_ids = prompt_ids
        if agent_data.turn_replay_enabled and not agent_data.trajectory_prompt_ids:
            agent_data.trajectory_prompt_ids = list(prompt_ids)
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        turn_sampling_params = dict(sampling_params)
        if (self.attention_capture or self.independent_sampling) and "seed" in sampling_params:
            from verl.experimental.attention_traceback.capture import sampling_seed
            turn_sampling_params["seed"] = sampling_seed(
                sampling_params["seed"], "turn", agent_data.assistant_turns)
        requested_max_tokens = turn_sampling_params.get("max_tokens", None)
        if not agent_data.turn_replay_enabled:
            (
                turn_max_tokens,
                remaining_context_tokens,
                remaining_trajectory_tokens,
            ) = compute_legacy_generation_budget(
                prompt_token_count=len(agent_data.prompt_ids),
                response_token_count=len(agent_data.response_mask),
                prompt_length=self.prompt_length,
                response_length=self.response_length,
                max_model_len=self.max_model_len,
                max_response_tokens_per_turn=self.max_response_tokens_per_turn,
                requested_max_tokens=requested_max_tokens,
            )
            agent_data.metrics["legacy_context_prompt_tokens_max"] = max(
                agent_data.metrics.get("legacy_context_prompt_tokens_max", 0),
                len(agent_data.prompt_ids),
            )
            agent_data.metrics["legacy_context_remaining_tokens_min"] = min(
                agent_data.metrics.get("legacy_context_remaining_tokens_min", remaining_context_tokens),
                remaining_context_tokens,
            )
            agent_data.metrics.setdefault("legacy_context_budget_terminated", 0.0)
            if turn_max_tokens <= 0:
                # Do not send an oversized request to vLLM and do not truncate
                # the prompt.  Normal interaction finalization below assigns
                # the unfinished episode its benchmark failure reward while
                # allowing the rest of the batch/training run to continue.
                agent_data.metrics["legacy_context_budget_terminated"] = 1.0
                agent_data.extra_fields["termination_reason"] = "context_budget_exhausted"
                return AgentState.TERMINATED
            context_independent_turn_cap = remaining_trajectory_tokens
            if self.max_response_tokens_per_turn is not None:
                context_independent_turn_cap = min(
                    context_independent_turn_cap,
                    self.max_response_tokens_per_turn,
                )
            if requested_max_tokens is not None:
                context_independent_turn_cap = min(
                    context_independent_turn_cap,
                    int(requested_max_tokens),
                )
            if remaining_context_tokens < context_independent_turn_cap:
                agent_data.metrics["legacy_context_limited_turns"] = (
                    agent_data.metrics.get("legacy_context_limited_turns", 0) + 1
                )
            turn_sampling_params["max_tokens"] = turn_max_tokens
        elif self.max_response_tokens_per_turn is not None:
            remaining_trajectory_tokens = self.response_length - len(agent_data.response_mask)
            if remaining_trajectory_tokens <= 0:
                return AgentState.TERMINATED
            turn_max_tokens = min(self.max_response_tokens_per_turn, remaining_trajectory_tokens)
            if requested_max_tokens is not None:
                turn_max_tokens = min(turn_max_tokens, int(requested_max_tokens))
            turn_sampling_params["max_tokens"] = turn_max_tokens

        if self.max_response_tokens_per_turn is not None:
            agent_data.metrics["max_response_tokens_per_turn"] = self.max_response_tokens_per_turn

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if self.attention_capture:
            from verl.experimental.attention_traceback.capture import action_span
            start, end, action, selection = action_span(self.tokenizer, output.token_ids)
            offset = len(agent_data.prompt_ids)
            agent_data.extra_fields.setdefault("_attention_turns", []).append({
                "turn_idx": agent_data.assistant_turns,
                "action_text": action,
                "response_text": self.tokenizer.decode(output.token_ids, skip_special_tokens=True),
                "action_token_start": offset + start, "action_token_end": offset + end,
                "turn_token_start": offset, "response_token_end": offset + len(output.token_ids),
                "action_token_ids": list(output.token_ids[start:end]),
                "action_token_logps": list((output.log_probs or [])[start:end]),
                "rollout_action_token_logps": list((output.log_probs or [])[start:end]),
                "sampling_seed": turn_sampling_params.get("seed"),
                "token_selection": selection,
            })
        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        if agent_data.turn_replay_enabled:
            flat_response_start = len(agent_data.flat_response_ids)
            agent_data.flat_response_ids += agent_data.response_ids
        else:
            agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            if agent_data.turn_replay_enabled:
                _, history_response = agent_data.context_builder.project_response(assistant_message)
                history_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(history_response, add_special_tokens=False),
                )
                agent_data.pending_turn = AgentTurnRecord(
                    turn_index=len(agent_data.turn_replay.turns),
                    context_messages=deepcopy(agent_data.messages),
                    context_token_ids=list(agent_data.prompt_ids),
                    full_response_ids=list(agent_data.response_ids),
                    history_response=history_response,
                    history_response_ids=list(history_response_ids),
                    old_logprobs=list(output.log_probs or []),
                    flat_response_start=flat_response_start,
                    flat_response_end=len(agent_data.flat_response_ids),
                    flat_observation_start=len(agent_data.flat_response_ids),
                    flat_observation_end=len(agent_data.flat_response_ids),
                )
                add_messages.append({"role": "assistant", "content": history_response})
            else:
                add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        if agent_data.turn_replay_enabled and agent_data.tool_calls:
            raise NotImplementedError(
                "turn-aware replay does not support tool-call messages; set multi_turn.history_window=0 "
                "for the legacy tool-agent path"
            )

        # Execute the final environment action before a length/turn limit ends
        # the trajectory. Otherwise a successful last action is silently lost
        # and the episode receives an incorrect failure reward.
        reached_limit = (
            (not ignore_termination and len(agent_data.response_mask) >= self.response_length)
            or (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
            or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        )
        if reached_limit:
            if self.interaction_config_file and not agent_data.tool_calls:
                agent_data.stop_after_interaction = True
            else:
                return AgentState.TERMINATED

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=new_images_this_turn,  # Using local variable
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        if agent_data.turn_replay_enabled:
            raise NotImplementedError("turn replay currently supports environment observations, not tool responses")
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)
        if metrics:
            agent_data.extra_fields.setdefault("interaction_turn_metrics", []).append(metrics)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        if self.attention_capture:
            from verl.experimental.attention_traceback.capture import observation_end
            turn = agent_data.extra_fields["_attention_turns"][-1]
            offset = len(agent_data.prompt_ids)
            content_end = offset + observation_end(self.tokenizer, response_ids, self.apply_chat_template_kwargs)
            reason = metrics.get("done_reason", "") if metrics else ""
            if not reason and agent_data.stop_after_interaction:
                reason = "max_steps" if (
                    self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns
                ) else "length_limit"
            turn.update(observation_text=str(interaction_responses),
                        observation_token_start=offset, observation_token_end=content_end,
                        turn_token_end=content_end, reward=float((metrics or {}).get("step_reward", reward or 0)),
                        done=bool(should_terminate_sequence or agent_data.stop_after_interaction),
                        done_reason=reason, environment_metrics=metrics or {})

        # Update the flattened training sequence. In turn-replay mode the next
        # generation context is rebuilt from projected messages instead of
        # extending this sequence.
        if agent_data.turn_replay_enabled:
            if agent_data.pending_turn is None:
                raise RuntimeError("interaction observation has no pending generation turn")
            observation_start = len(agent_data.flat_response_ids)
            agent_data.flat_response_ids += response_ids
            agent_data.pending_turn.observation = str(interaction_responses)
            agent_data.pending_turn.observation_ids = list(response_ids)
            agent_data.pending_turn.flat_observation_start = observation_start
            agent_data.pending_turn.flat_observation_end = len(agent_data.flat_response_ids)
            agent_data.turn_replay.turns.append(agent_data.pending_turn)
            agent_data.history_turns.append(
                {
                    "history_response": agent_data.pending_turn.history_response,
                    "observation": str(interaction_responses),
                }
            )
            agent_data.pending_turn = None
            agent_data.messages = agent_data.context_builder.build(
                agent_data.fixed_prefix,
                agent_data.history_turns,
            )
        else:
            agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence or agent_data.stop_after_interaction:
            return AgentState.TERMINATED
        elif agent_data.turn_replay_enabled:
            return AgentState.PENDING
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
