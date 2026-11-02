# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from verl.experimental.agent_loop.tool_agent_loop import AgentData, compute_legacy_generation_budget
from verl.experimental.agent_loop.turn_context import (
    AgentTrajectoryRecord,
    AgentTurnRecord,
    ContextBuilder,
    clip_trajectory_record,
    validate_trajectory_record,
)


def test_legacy_generation_budget_caps_request_before_model_context_overflow():
    budget, remaining_context, remaining_trajectory = compute_legacy_generation_budget(
        prompt_token_count=34_000,
        response_token_count=31_000,
        prompt_length=2_048,
        response_length=32_768,
        max_model_len=40_960,
        max_response_tokens_per_turn=2_048,
        requested_max_tokens=2_048,
    )

    assert remaining_context == 816
    assert remaining_trajectory == 1_768
    assert budget == 816


def test_legacy_generation_budget_returns_zero_at_context_limit():
    budget, remaining_context, remaining_trajectory = compute_legacy_generation_budget(
        prompt_token_count=34_816,
        response_token_count=32_000,
        prompt_length=2_048,
        response_length=32_768,
        max_model_len=40_960,
        max_response_tokens_per_turn=2_048,
        requested_max_tokens=None,
    )

    assert remaining_context == 0
    assert remaining_trajectory == 768
    assert budget == 0


def test_agent_data_has_stable_termination_reason_schema():
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="request-0",
        tools_kwargs={},
    )

    assert agent_data.extra_fields == {"termination_reason": ""}


def test_context_builder_keeps_fixed_prefix_and_last_complete_turns():
    builder = ContextBuilder(history_window=2, fixed_prefix_messages=3)
    fixed = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "user", "content": "initial observation"},
    ]
    turns = [
        {"history_response": f"ACTION: action-{index}", "observation": f"observation-{index}"}
        for index in range(4)
    ]

    context = builder.build(fixed, turns)

    assert context == fixed + [
        {"role": "assistant", "content": "ACTION: action-2"},
        {"role": "user", "content": "observation-2"},
        {"role": "assistant", "content": "ACTION: action-3"},
        {"role": "user", "content": "observation-3"},
    ]


def test_context_candidates_drop_oldest_complete_turns_only():
    builder = ContextBuilder(history_window=3, fixed_prefix_messages=3)
    fixed = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "user", "content": "initial observation"},
    ]
    turns = [
        {"history_response": f"ACTION: action-{index}", "observation": f"observation-{index}"}
        for index in range(4)
    ]

    candidates = list(builder.build_candidates(fixed, turns))

    assert [dropped for _, dropped in candidates] == [0, 1, 2, 3]
    assert candidates[0][0] == fixed + [
        {"role": "assistant", "content": "ACTION: action-1"},
        {"role": "user", "content": "observation-1"},
        {"role": "assistant", "content": "ACTION: action-2"},
        {"role": "user", "content": "observation-2"},
        {"role": "assistant", "content": "ACTION: action-3"},
        {"role": "user", "content": "observation-3"},
    ]
    assert candidates[-1][0] == fixed


def test_action_projection_removes_thinking_without_changing_policy_response():
    builder = ContextBuilder(history_window=8)
    full_response = "<think>private chain of thought</think>\nACTION: open drawer 1"

    action, history_response = builder.project_response(full_response)

    assert action == "open drawer 1"
    assert history_response == "ACTION: open drawer 1"
    assert "private chain of thought" in full_response


def test_full_response_projection_retains_thinking_inside_history_window():
    builder = ContextBuilder(history_window=2, history_response_mode="full_response")
    full_response = "<think>inspect product attributes</think>\nACTION: click[Buy Now]"

    action, history_response = builder.project_response(full_response)

    assert action == "click[Buy Now]"
    assert history_response == full_response


def test_clip_manifest_uses_turn_boundaries_and_preserves_context():
    turn = AgentTurnRecord(
        turn_index=0,
        context_messages=[{"role": "user", "content": "context"}],
        context_token_ids=[10, 11],
        full_response_ids=[20, 21, 22],
        history_response="ACTION: x",
        history_response_ids=[30],
        observation="obs",
        observation_ids=[40, 41],
        old_logprobs=[-1.0, -2.0, -3.0],
        flat_response_start=0,
        flat_response_end=3,
        flat_observation_start=3,
        flat_observation_end=5,
    )
    record = AgentTrajectoryRecord(history_window=8, fixed_prefix_messages=3, turns=[turn]).to_dict()

    clipped = clip_trajectory_record(record, max_response_length=4)

    assert clipped["turns"][0]["context_token_ids"] == [10, 11]
    assert clipped["turns"][0]["full_response_ids"] == [20, 21, 22]
    assert clipped["turns"][0]["observation_ids"] == [40]
    validate_trajectory_record(clipped, response_length=4)


def test_clip_manifest_inside_policy_response_drops_following_observation():
    turn = AgentTurnRecord(
        turn_index=0,
        context_messages=[{"role": "user", "content": "context"}],
        context_token_ids=[10, 11],
        full_response_ids=[20, 21, 22],
        history_response="ACTION: x",
        history_response_ids=[30],
        observation="obs",
        observation_ids=[40, 41],
        old_logprobs=[-1.0, -2.0, -3.0],
        flat_response_start=0,
        flat_response_end=3,
        flat_observation_start=3,
        flat_observation_end=5,
    )
    record = AgentTrajectoryRecord(history_window=8, fixed_prefix_messages=3, turns=[turn]).to_dict()

    clipped = clip_trajectory_record(record, max_response_length=2)
    clipped_turn = clipped["turns"][0]

    assert clipped_turn["full_response_ids"] == [20, 21]
    assert clipped_turn["old_logprobs"] == [-1.0, -2.0]
    assert clipped_turn["observation"] == ""
    assert clipped_turn["observation_ids"] == []
    assert clipped_turn["flat_response_end"] == 2
    assert clipped_turn["flat_observation_start"] == 2
    assert clipped_turn["flat_observation_end"] == 2
    validate_trajectory_record(clipped, response_length=2)
