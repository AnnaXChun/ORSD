# Copyright 2026
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
"""Outcome-contrastive references and full-vocabulary token credit utilities."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from verl.trainer.ppo.rubric_token_credit import normalize_messages

_REFERENCE_BLOCK = """\

The following response is privileged outcome feedback for the same problem.
Verification status: {status}

<reference_response>
{response}
</reference_response>

Use the verification status and reference response only as evidence while predicting the target response.
"""


def _stable_pick(candidates: list[int], *, seed: int, group_id: Any, target_index: int, label: str) -> int:
    """Pick a reproducible candidate without depending on Python's randomized hash."""
    if not candidates:
        return -1
    payload = f"{seed}|{group_id}|{target_index}|{label}".encode()
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
    return candidates[offset % len(candidates)]


def select_outcome_contrastive_references(
    group_ids: np.ndarray,
    scores: torch.Tensor,
    *,
    success_threshold: float = 0.5,
    min_correct: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Choose non-self verified-correct and verified-incorrect references.

    A group is eligible only when ``min_correct <= n_correct < group_size``.
    Correct targets never use themselves as the positive reference. Incorrect
    targets never use themselves as the negative reference; when a group has
    only one incorrect rollout, that rollout falls back to ordinary GRPO while
    the correct rollouts can still receive contrastive credit.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got shape {tuple(scores.shape)}")
    if len(group_ids) != scores.shape[0]:
        raise ValueError(f"group_ids length {len(group_ids)} does not match scores length {scores.shape[0]}")
    if min_correct < 1:
        raise ValueError(f"min_correct must be positive, got {min_correct}")

    scores_cpu = scores.detach().float().cpu()
    correct = scores_cpu >= success_threshold
    positive_indices = torch.full((scores.shape[0],), -1, dtype=torch.long)
    negative_indices = torch.full((scores.shape[0],), -1, dtype=torch.long)
    active = torch.zeros(scores.shape[0], dtype=torch.bool)

    grouped: dict[Any, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(index)

    eligible_groups = 0
    mixed_groups = 0
    for group_id, member_indices in grouped.items():
        correct_indices = [index for index in member_indices if bool(correct[index])]
        incorrect_indices = [index for index in member_indices if not bool(correct[index])]
        if correct_indices and incorrect_indices:
            mixed_groups += 1
        if len(correct_indices) < min_correct or not incorrect_indices:
            continue

        eligible_groups += 1
        for target_index in member_indices:
            positive_candidates = (
                [index for index in correct_indices if index != target_index]
                if bool(correct[target_index])
                else correct_indices
            )
            negative_candidates = (
                [index for index in incorrect_indices if index != target_index]
                if not bool(correct[target_index])
                else incorrect_indices
            )
            positive_index = _stable_pick(
                positive_candidates,
                seed=seed,
                group_id=group_id,
                target_index=target_index,
                label="positive",
            )
            negative_index = _stable_pick(
                negative_candidates,
                seed=seed,
                group_id=group_id,
                target_index=target_index,
                label="negative",
            )
            if positive_index < 0 or negative_index < 0:
                continue
            positive_indices[target_index] = positive_index
            negative_indices[target_index] = negative_index
            active[target_index] = True

    group_count = max(1, len(grouped))
    metrics = {
        "outcome_contrastive/group_count": float(len(grouped)),
        "outcome_contrastive/mixed_group_frac": mixed_groups / group_count,
        "outcome_contrastive/eligible_group_frac": eligible_groups / group_count,
        "outcome_contrastive/active_trajectory_frac": active.float().mean().item(),
        "outcome_contrastive/correct_trajectory_frac": correct.float().mean().item(),
    }
    return positive_indices, negative_indices, active, metrics


def decode_masked_responses(batch: Any, tokenizer: Any) -> list[str]:
    """Decode each rollout up to its response mask without padding tokens."""
    responses = batch.batch["responses"].detach().cpu()
    response_mask = batch.batch["response_mask"].detach().cpu().bool()
    decoded = []
    for response, mask in zip(responses, response_mask, strict=True):
        decoded.append(tokenizer.decode(response[mask].tolist(), skip_special_tokens=True))
    return decoded


def _with_outcome_reference(raw_prompt: Any, reference: str, status: str) -> list[dict[str, str]]:
    messages = deepcopy(normalize_messages(raw_prompt))
    block = _REFERENCE_BLOCK.format(status=status, response=reference.strip())
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = f"{messages[0]['content'].rstrip()}\n{block}"
    else:
        messages.insert(0, {"role": "system", "content": block.strip()})
    return messages


def build_outcome_contrastive_inputs(
    batch: Any,
    tokenizer: Any,
    positive_references: list[str],
    negative_references: list[str],
    *,
    teacher_max_prompt_length: int,
    teacher_enable_thinking: bool,
) -> dict[str, torch.Tensor]:
    """Build positive/negative current-actor prompts with identical target responses."""
    raw_prompts = list(batch.non_tensor_batch["raw_prompt"])
    expected = len(raw_prompts)
    if len(positive_references) != expected or len(negative_references) != expected:
        raise ValueError(
            "Reference counts must match the active rollout batch: "
            f"{len(positive_references)=}, {len(negative_references)=}, {expected=}"
        )

    def render(reference_texts: list[str], status: str) -> list[str]:
        prompt_texts = []
        for raw_prompt, reference in zip(raw_prompts, reference_texts, strict=True):
            messages = _with_outcome_reference(raw_prompt, reference, status)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": teacher_enable_thinking,
            }
            try:
                prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking")
                prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
            prompt_texts.append(prompt_text)
        return prompt_texts

    positive_texts = render(positive_references, "VERIFIED_CORRECT")
    negative_texts = render(negative_references, "VERIFIED_INCORRECT")

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    old_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    try:
        positive_encoded = tokenizer(
            positive_texts,
            padding="max_length",
            truncation=True,
            max_length=teacher_max_prompt_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        negative_encoded = tokenizer(
            negative_texts,
            padding="max_length",
            truncation=True,
            max_length=teacher_max_prompt_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.truncation_side = old_truncation_side

    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    device = responses.device

    def join(encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = encoded["input_ids"].to(device=device, dtype=responses.dtype)
        prompt_mask = encoded["attention_mask"].to(device=device, dtype=response_mask.dtype)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
        input_ids = torch.cat([prompts, responses], dim=-1)
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return input_ids, attention_mask, position_ids

    positive_input_ids, positive_attention_mask, positive_position_ids = join(positive_encoded)
    negative_input_ids, negative_attention_mask, negative_position_ids = join(negative_encoded)
    return {
        "positive_input_ids": positive_input_ids,
        "positive_attention_mask": positive_attention_mask,
        "positive_position_ids": positive_position_ids,
        "negative_input_ids": negative_input_ids,
        "negative_attention_mask": negative_attention_mask,
        "negative_position_ids": negative_position_ids,
        "response_mask": response_mask,
    }


def jensen_shannon_from_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute symmetric full-vocabulary JS divergence in response-token chunks."""
    if positive_logits.shape != negative_logits.shape:
        raise ValueError(
            f"positive logits shape {tuple(positive_logits.shape)} does not match "
            f"negative logits shape {tuple(negative_logits.shape)}"
        )
    if positive_logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, tokens, vocab], got {positive_logits.ndim} dimensions")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    batch_size, token_count, _ = positive_logits.shape
    divergence = torch.empty((batch_size, token_count), dtype=torch.float32, device=positive_logits.device)
    log_two = math.log(2.0)
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        positive_log_probs = F.log_softmax(
            positive_logits[:, start:end, :] / temperature,
            dim=-1,
            dtype=torch.float32,
        )
        negative_log_probs = F.log_softmax(
            negative_logits[:, start:end, :] / temperature,
            dim=-1,
            dtype=torch.float32,
        )
        mixture_log_probs = torch.logaddexp(positive_log_probs, negative_log_probs) - log_two
        positive_kl = (positive_log_probs.exp() * (positive_log_probs - mixture_log_probs)).sum(dim=-1)
        negative_kl = (negative_log_probs.exp() * (negative_log_probs - mixture_log_probs)).sum(dim=-1)
        divergence[:, start:end] = 0.5 * (positive_kl + negative_kl)
    return divergence.clamp_min_(0.0)
