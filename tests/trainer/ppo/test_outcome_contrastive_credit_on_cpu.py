# Copyright 2026

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import apply_outcome_contrastive_token_credit
from verl.trainer.ppo.outcome_contrastive_credit import (
    build_outcome_contrastive_inputs,
    jensen_shannon_from_logits,
    select_outcome_contrastive_references,
)


def test_reference_selection_uses_nonself_positive_and_negative_examples():
    group_ids = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4, dtype=object)
    scores = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    positive, negative, active, metrics = select_outcome_contrastive_references(
        group_ids,
        scores,
        min_correct=2,
        seed=7,
    )

    assert active[:4].all()
    assert not active[4:8].any()
    assert active[8:11].all()
    assert not active[11]
    for target_index in torch.nonzero(active, as_tuple=False).squeeze(-1).tolist():
        assert scores[positive[target_index]] == 1
        assert scores[negative[target_index]] == 0
        assert positive[target_index].item() != target_index
        assert negative[target_index].item() != target_index
    assert metrics["outcome_contrastive/eligible_group_frac"] == pytest.approx(2 / 3)


def test_full_vocabulary_js_is_symmetric_nonnegative_and_zero_for_identity():
    positive = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]]])
    negative = torch.tensor([[[-1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]])

    divergence = jensen_shannon_from_logits(positive, negative, chunk_size=1)
    reverse = jensen_shannon_from_logits(negative, positive, chunk_size=2)
    identity = jensen_shannon_from_logits(positive, positive)

    assert torch.allclose(divergence, reverse, atol=1e-7)
    assert divergence[0, 0] > 0
    assert divergence[0, 1] == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(identity, torch.zeros_like(identity), atol=1e-7)
    assert divergence.max().item() <= torch.log(torch.tensor(2.0)).item() + 1e-6


def test_outcome_contrastive_weights_gate_only_active_grpo_tokens():
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [0.5, 0.5, 0.5]])
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]])
    divergence = torch.tensor([[0.1, 0.2, 0.4], [0.3, 0.6, 0.0], [9.0, 9.0, 9.0]])
    active = torch.tensor([True, True, False])

    weighted, weights, metrics = apply_outcome_contrastive_token_credit(
        advantages,
        response_mask,
        divergence,
        active,
        scale_quantile=0.75,
    )

    assert torch.all((weights[:2][response_mask[:2].bool()] >= 0) & (weights[:2][response_mask[:2].bool()] < 1))
    assert torch.equal(weights[2], torch.ones_like(weights[2]))
    assert weights[0, 0] < weights[0, 2]
    assert torch.equal(weighted, advantages * weights)
    assert metrics["token_credit/oc_active_trajectory_frac"] == pytest.approx(2 / 3)
    assert metrics["token_credit/weight_mean"] < 1.0


class _FakeTokenizer:
    padding_side = "right"
    truncation_side = "right"

    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        rendered = "\n".join(message["content"] for message in messages)
        self.rendered.append(rendered)
        return rendered

    def __call__(self, prompt_texts, **kwargs):
        max_length = kwargs["max_length"]
        input_ids = torch.zeros((len(prompt_texts), max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, prompt in enumerate(prompt_texts):
            token_count = min(max_length, max(1, len(prompt) // 10))
            input_ids[index, -token_count:] = index + 1
            attention_mask[index, -token_count:] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_conditioned_inputs_reuse_target_response_and_label_reference_outcomes():
    responses = torch.tensor([[31, 32, 0], [41, 0, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    batch = SimpleNamespace(
        batch={"responses": responses, "response_mask": response_mask},
        non_tensor_batch={
            "raw_prompt": np.array(
                [
                    [{"role": "user", "content": "Problem A"}],
                    [{"role": "user", "content": "Problem B"}],
                ],
                dtype=object,
            )
        },
    )
    tokenizer = _FakeTokenizer()
    inputs = build_outcome_contrastive_inputs(
        batch,
        tokenizer,
        ["correct A", "correct B"],
        ["incorrect A", "incorrect B"],
        teacher_max_prompt_length=8,
        teacher_enable_thinking=False,
    )

    assert torch.equal(inputs["positive_input_ids"][:, -3:], responses)
    assert torch.equal(inputs["negative_input_ids"][:, -3:], responses)
    assert inputs["positive_input_ids"].shape == (2, 11)
    assert any("VERIFIED_CORRECT" in prompt for prompt in tokenizer.rendered)
    assert any("VERIFIED_INCORRECT" in prompt for prompt in tokenizer.rendered)
