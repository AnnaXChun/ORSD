# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.turn_context import AgentTrajectoryRecord, AgentTurnRecord
from verl.trainer.ppo.rubric_token_credit import build_rubric_conditioned_turn_replay
from verl.trainer.ppo.turn_replay import flatten_turn_entries, pack_turn_entries, scatter_turn_outputs


def _record() -> dict:
    turns = [
        AgentTurnRecord(
            turn_index=0,
            context_messages=[{"role": "user", "content": "first"}],
            context_token_ids=[1, 2],
            full_response_ids=[3, 4],
            history_response="ACTION: first",
            history_response_ids=[30],
            observation="obs-1",
            observation_ids=[5],
            flat_response_start=0,
            flat_response_end=2,
            flat_observation_start=2,
            flat_observation_end=3,
        ),
        AgentTurnRecord(
            turn_index=1,
            context_messages=[{"role": "user", "content": "second"}],
            context_token_ids=[6, 7, 8],
            full_response_ids=[9],
            history_response="ACTION: second",
            history_response_ids=[31],
            observation="obs-2",
            observation_ids=[10, 11],
            flat_response_start=3,
            flat_response_end=4,
            flat_observation_start=4,
            flat_observation_end=6,
        ),
    ]
    return AgentTrajectoryRecord(history_window=8, fixed_prefix_messages=3, turns=turns).to_dict()


def test_pack_replays_each_turn_under_its_own_exact_context():
    entries = flatten_turn_entries([_record()], response_length=6)

    packed, _ = pack_turn_entries(entries, pad_token_id=0, device="cpu")

    assert packed["input_ids"].tolist() == [[0, 1, 2, 3, 4], [6, 7, 8, 9, 0]]
    assert packed["attention_mask"].tolist() == [[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert packed["position_ids"].tolist() == [[0, 0, 1, 2, 3], [0, 1, 2, 3, 0]]


def test_flatten_turn_entries_validates_exact_tokens_and_masks():
    responses = torch.tensor([[3, 4, 5, 9, 10, 11]])
    response_mask = torch.tensor([[1, 1, 0, 1, 0, 0]])
    response_attention = torch.ones_like(response_mask)

    entries = flatten_turn_entries(
        [_record()],
        response_length=6,
        responses=responses,
        response_mask=response_mask,
        response_attention_mask=response_attention,
    )

    assert len(entries) == 2

    misaligned = responses.clone()
    misaligned[0, 3] = 99
    with pytest.raises(ValueError, match="flattened policy tokens"):
        flatten_turn_entries(
            [_record()],
            response_length=6,
            responses=misaligned,
            response_mask=response_mask,
            response_attention_mask=response_attention,
        )

    wrong_mask = response_mask.clone()
    wrong_mask[0, 2] = 1
    with pytest.raises(ValueError, match="response_mask"):
        flatten_turn_entries(
            [_record()],
            response_length=6,
            responses=responses,
            response_mask=wrong_mask,
            response_attention_mask=response_attention,
        )


def test_scatter_restores_flat_token_layout_and_preserves_gradients():
    first = torch.tensor([0.1, 0.2], requires_grad=True)
    second = torch.tensor([0.3], requires_grad=True)

    scattered = scatter_turn_outputs(
        {(0, 0): first, (0, 1): second},
        [_record()],
        response_length=6,
        device="cpu",
        dtype=torch.float32,
    )

    assert torch.allclose(scattered, torch.tensor([[0.1, 0.2, 0.0, 0.3, 0.0, 0.0]]))
    scattered.sum().backward()
    assert torch.equal(first.grad, torch.ones_like(first))
    assert torch.equal(second.grad, torch.ones_like(second))


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        return [ord(character) for character in text]


def test_rubric_transform_changes_only_per_turn_context():
    record = _record()
    batch = SimpleNamespace(
        batch={"responses": torch.zeros((1, 6), dtype=torch.long)},
        non_tensor_batch={"turn_replay": np.array([record], dtype=object)},
    )

    transformed = build_rubric_conditioned_turn_replay(
        batch,
        _Tokenizer(),
        ["Prefer actions supported by the current observation."],
        teacher_max_prompt_length=40960,
        teacher_enable_thinking=True,
    )[0]

    for original_turn, transformed_turn in zip(record["turns"], transformed["turns"], strict=True):
        assert transformed_turn["context_token_ids"] != original_turn["context_token_ids"]
        assert "Key criteria for solving" in transformed_turn["context_messages"][0]["content"]
        assert transformed_turn["full_response_ids"] == original_turn["full_response_ids"]
        assert transformed_turn["flat_response_start"] == original_turn["flat_response_start"]
        assert transformed_turn["flat_response_end"] == original_turn["flat_response_end"]
