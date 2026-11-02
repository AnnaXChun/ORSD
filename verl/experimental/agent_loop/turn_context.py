# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Structured multi-turn context records for action-based agent rollouts.

The policy response and the text projected into future history are deliberately
different objects.  The former retains every generated token (including
reasoning); the latter contains only the externally visible action.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Optional

from verl.interactions.agent_environment_utils import extract_action

TURN_REPLAY_KEY = "turn_replay"
TURN_REPLAY_SCHEMA_VERSION = 1


@dataclass
class AgentTurnRecord:
    """One policy generation together with its exact rollout context."""

    turn_index: int
    context_messages: list[dict[str, Any]]
    context_token_ids: list[int]
    full_response_ids: list[int]
    history_response: str
    history_response_ids: list[int]
    observation: str = ""
    observation_ids: list[int] = field(default_factory=list)
    old_logprobs: list[float] = field(default_factory=list)
    flat_response_start: int = 0
    flat_response_end: int = 0
    flat_observation_start: int = 0
    flat_observation_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTrajectoryRecord:
    """Replay manifest carried as one object per trajectory in ``DataProto``."""

    history_window: int
    fixed_prefix_messages: int
    turns: list[AgentTurnRecord] = field(default_factory=list)
    schema_version: int = TURN_REPLAY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "history_window": self.history_window,
            "fixed_prefix_messages": self.fixed_prefix_messages,
            "turns": [turn.to_dict() for turn in self.turns],
        }


ContextTransform = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


class ContextBuilder:
    """Build the exact context visible to an action-based agent.

    A non-positive ``history_window`` intentionally means that this builder is
    disabled.  The caller must then use VERL's legacy growing-token history.
    """

    def __init__(
        self,
        history_window: int,
        fixed_prefix_messages: int = 3,
        history_response_mode: str = "action_only",
    ):
        if history_window <= 0:
            raise ValueError("ContextBuilder requires history_window > 0")
        if fixed_prefix_messages < 1:
            raise ValueError("fixed_prefix_messages must be positive")
        self.history_window = int(history_window)
        self.fixed_prefix_messages = int(fixed_prefix_messages)
        normalized_mode = str(history_response_mode).strip().lower()
        if normalized_mode not in {"action_only", "full_response"}:
            raise ValueError(
                "history_response_mode must be 'action_only' or 'full_response', "
                f"got {history_response_mode!r}"
            )
        self.history_response_mode = normalized_mode

    def build(
        self,
        fixed_prefix: list[dict[str, Any]],
        history_turns: list[dict[str, Any]],
        *,
        context_transform: Optional[ContextTransform] = None,
    ) -> list[dict[str, Any]]:
        messages = deepcopy(fixed_prefix[: self.fixed_prefix_messages])
        for turn in history_turns[-self.history_window :]:
            messages.append({"role": "assistant", "content": str(turn["history_response"])})
            messages.append({"role": "user", "content": str(turn["observation"])})
        if context_transform is not None:
            messages = context_transform(messages)
        return messages

    def build_candidates(
        self,
        fixed_prefix: list[dict[str, Any]],
        history_turns: list[dict[str, Any]],
        *,
        context_transform: Optional[ContextTransform] = None,
    ) -> Iterator[tuple[list[dict[str, Any]], int]]:
        """Yield contexts from the configured window down to prefix-only.

        Each retry removes exactly one oldest action/observation pair. This is
        intentionally turn based: callers must never make an oversized prompt
        fit by slicing token IDs, because replay must retain the exact context
        used for generation.
        """

        visible_turns = list(history_turns[-self.history_window :])
        for dropped_turns in range(len(visible_turns) + 1):
            yield (
                self.build(
                    fixed_prefix,
                    visible_turns[dropped_turns:],
                    context_transform=context_transform,
                ),
                dropped_turns,
            )

    def project_response(self, full_response: str) -> tuple[str, str]:
        """Return the parsed action and the assistant text visible next turn."""

        action = str(extract_action(full_response) or "").strip()
        if self.history_response_mode == "full_response":
            retained_response = str(full_response).strip()
            return action, retained_response or "ACTION: __INVALID__"
        if action:
            return action, f"ACTION: {action}"
        return "", "ACTION: __INVALID__"


def clip_trajectory_record(record: dict[str, Any], max_response_length: int) -> dict[str, Any]:
    """Clip a replay manifest to the flattened response retained by VERL.

    Clipping is performed through stored turn boundaries.  Context tokens are
    never truncated or sliced.
    """

    clipped = deepcopy(record)
    kept_turns: list[dict[str, Any]] = []
    for turn in clipped.get("turns", []):
        start = int(turn["flat_response_start"])
        end = int(turn["flat_response_end"])
        if start >= max_response_length:
            break
        retained_end = min(end, max_response_length)
        retained_tokens = max(0, retained_end - start)
        turn["full_response_ids"] = list(turn.get("full_response_ids", []))[:retained_tokens]
        turn["old_logprobs"] = list(turn.get("old_logprobs", []))[:retained_tokens]
        turn["flat_response_end"] = retained_end

        # The global response budget may end inside a policy generation.  In
        # that case none of the following environment observation is present
        # in VERL's retained flattened response.  Anchor an empty observation
        # span at the clipped response end; leaving it at the original response
        # end makes the replay manifest internally inconsistent.
        if retained_end < end:
            turn["observation"] = ""
            turn["observation_ids"] = []
            turn["flat_observation_start"] = retained_end
            turn["flat_observation_end"] = retained_end
            kept_turns.append(turn)
            break

        obs_start = int(turn.get("flat_observation_start", end))
        obs_end = int(turn.get("flat_observation_end", obs_start))
        retained_obs_end = min(obs_end, max_response_length)
        retained_obs = max(0, retained_obs_end - obs_start)
        turn["observation_ids"] = list(turn.get("observation_ids", []))[:retained_obs]
        turn["flat_observation_end"] = retained_obs_end
        kept_turns.append(turn)
    clipped["turns"] = kept_turns
    return clipped


def validate_trajectory_record(
    record: Any,
    response_length: Optional[int] = None,
    *,
    flattened_response_ids: Optional[Any] = None,
    response_mask: Optional[Any] = None,
    response_attention_mask: Optional[Any] = None,
) -> None:
    """Validate turn ordering and the exact flattened-token mapping.

    Length-only span checks cannot detect an off-by-one shift that still has
    the expected shape.  When the flattened tensors are supplied, verify the
    token contents and the policy/observation masks as well.  This makes turn
    replay fail before PPO can silently recompute ``log pi(y_t | c'_t)`` under
    a context/token layout different from rollout.
    """

    if not isinstance(record, dict):
        raise TypeError(f"turn replay record must be a dict, got {type(record)!r}")
    if int(record.get("schema_version", -1)) != TURN_REPLAY_SCHEMA_VERSION:
        raise ValueError(f"unsupported turn replay schema: {record.get('schema_version')!r}")
    def _as_list(value: Any) -> Optional[list[Any]]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)

    flat_ids = _as_list(flattened_response_ids)
    flat_policy_mask = _as_list(response_mask)
    flat_attention_mask = _as_list(response_attention_mask)
    inferred_length = response_length
    for supplied in (flat_ids, flat_policy_mask, flat_attention_mask):
        if supplied is None:
            continue
        if inferred_length is None:
            inferred_length = len(supplied)
        elif len(supplied) != inferred_length:
            raise ValueError(
                "turn replay flattened inputs have inconsistent lengths: "
                f"expected={inferred_length}, got={len(supplied)}"
            )

    expected_policy_mask = [False] * inferred_length if inferred_length is not None else None
    expected_attention_mask = [False] * inferred_length if inferred_length is not None else None
    previous_end = 0
    for expected_index, turn in enumerate(record.get("turns", [])):
        if int(turn.get("turn_index", -1)) != expected_index:
            raise ValueError("turn indices must be contiguous and zero based")
        start = int(turn["flat_response_start"])
        end = int(turn["flat_response_end"])
        token_count = len(turn.get("full_response_ids", []))
        if start < previous_end or end < start or end - start != token_count:
            raise ValueError(
                f"invalid turn token mapping at turn {expected_index}: "
                f"previous_end={previous_end}, start={start}, end={end}, tokens={token_count}"
            )
        if len(turn.get("old_logprobs", [])) not in (0, token_count):
            raise ValueError(f"old_logprobs length does not match response at turn {expected_index}")
        if response_length is not None and end > response_length:
            raise ValueError(f"turn {expected_index} exceeds response length {response_length}")
        if flat_ids is not None and flat_ids[start:end] != list(turn.get("full_response_ids", [])):
            raise ValueError(
                f"flattened policy tokens do not match full_response_ids at turn {expected_index}"
            )
        if expected_policy_mask is not None:
            expected_policy_mask[start:end] = [True] * token_count
        if expected_attention_mask is not None:
            expected_attention_mask[start:end] = [True] * token_count
        observation_start = int(turn.get("flat_observation_start", end))
        observation_end = int(turn.get("flat_observation_end", observation_start))
        observation_count = len(turn.get("observation_ids", []))
        if observation_start != end or observation_end < observation_start:
            raise ValueError(f"invalid observation span at turn {expected_index}")
        if observation_end - observation_start != observation_count:
            raise ValueError(f"observation token mapping does not match at turn {expected_index}")
        if response_length is not None and observation_end > response_length:
            raise ValueError(f"turn {expected_index} observation exceeds response length {response_length}")
        if flat_ids is not None and flat_ids[observation_start:observation_end] != list(
            turn.get("observation_ids", [])
        ):
            raise ValueError(
                f"flattened observation tokens do not match observation_ids at turn {expected_index}"
            )
        if expected_attention_mask is not None:
            expected_attention_mask[observation_start:observation_end] = [True] * observation_count
        if not turn.get("context_token_ids"):
            raise ValueError(f"turn {expected_index} has no generation context")
        previous_end = observation_end

    if flat_policy_mask is not None and [bool(value) for value in flat_policy_mask] != expected_policy_mask:
        raise ValueError("response_mask does not exactly match the saved per-turn policy spans")
    if flat_attention_mask is not None and [bool(value) for value in flat_attention_mask] != expected_attention_mask:
        raise ValueError("response attention mask does not exactly match saved policy/observation spans")
