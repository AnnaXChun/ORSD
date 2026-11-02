# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Packing and scattering helpers for exact multi-turn policy replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from verl.experimental.agent_loop.turn_context import TURN_REPLAY_KEY, validate_trajectory_record


@dataclass(frozen=True)
class TurnEntry:
    trajectory_index: int
    turn_index: int
    record: dict[str, Any]


def has_turn_replay(non_tensor_batch: dict[str, Any]) -> bool:
    values = non_tensor_batch.get(TURN_REPLAY_KEY)
    return values is not None and len(values) > 0 and all(value is not None for value in values)


def flatten_turn_entries(
    records: Iterable[dict[str, Any]],
    response_length: int,
    *,
    responses: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
    response_attention_mask: torch.Tensor | None = None,
) -> list[TurnEntry]:
    entries: list[TurnEntry] = []
    for trajectory_index, record in enumerate(records):
        validate_trajectory_record(
            record,
            response_length=response_length,
            flattened_response_ids=None if responses is None else responses[trajectory_index],
            response_mask=None if response_mask is None else response_mask[trajectory_index],
            response_attention_mask=(
                None if response_attention_mask is None else response_attention_mask[trajectory_index]
            ),
        )
        for turn_index, turn in enumerate(record.get("turns", [])):
            if turn.get("full_response_ids"):
                entries.append(TurnEntry(trajectory_index, turn_index, turn))
    return entries


def pack_turn_entries(
    entries: list[TurnEntry],
    *,
    pad_token_id: int,
    device: torch.device | int | str,
) -> tuple[dict[str, torch.Tensor], list[TurnEntry]]:
    """Pack independent ``context + full response`` sequences for one forward."""

    if not entries:
        raise ValueError("cannot pack an empty turn replay batch")
    prompt_width = max(len(entry.record["context_token_ids"]) for entry in entries)
    response_width = max(len(entry.record["full_response_ids"]) for entry in entries)
    batch_size = len(entries)

    prompts = torch.full(
        (batch_size, prompt_width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    prompt_mask = torch.zeros((batch_size, prompt_width), dtype=torch.long, device=device)
    responses = torch.full(
        (batch_size, response_width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    response_attention = torch.zeros((batch_size, response_width), dtype=torch.long, device=device)

    for row, entry in enumerate(entries):
        context_ids = torch.as_tensor(entry.record["context_token_ids"], dtype=torch.long, device=device)
        response_ids = torch.as_tensor(entry.record["full_response_ids"], dtype=torch.long, device=device)
        prompts[row, prompt_width - context_ids.numel() :] = context_ids
        prompt_mask[row, prompt_width - context_ids.numel() :] = 1
        responses[row, : response_ids.numel()] = response_ids
        response_attention[row, : response_ids.numel()] = 1

    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.cat([prompt_mask, response_attention], dim=-1)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, entries


def scatter_turn_outputs(
    outputs: dict[tuple[int, int], torch.Tensor],
    records: list[dict[str, Any]],
    *,
    response_length: int,
    device: torch.device | int | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore turn outputs to VERL's original trajectory/token layout.

    ``torch.cat`` is used instead of assigning into a detached destination so
    that actor-update gradients flow from the flattened policy loss back into
    every per-turn forward.
    """

    rows: list[torch.Tensor] = []
    for trajectory_index, record in enumerate(records):
        cursor = 0
        segments: list[torch.Tensor] = []
        for turn_index, turn in enumerate(record.get("turns", [])):
            token_output = outputs.get((trajectory_index, turn_index))
            if token_output is None:
                raise KeyError(
                    f"missing replay output for trajectory={trajectory_index}, turn={turn_index}"
                )
            start = int(turn["flat_response_start"])
            end = int(turn["flat_response_end"])
            if start < cursor:
                raise ValueError("overlapping turn replay spans")
            if start > cursor:
                segments.append(torch.zeros(start - cursor, dtype=dtype, device=device))
            expected = end - start
            if token_output.numel() != expected:
                raise ValueError(
                    f"turn output length mismatch for trajectory={trajectory_index}, turn={turn_index}: "
                    f"got {token_output.numel()}, expected {expected}"
                )
            segments.append(token_output)
            cursor = end
        if cursor < response_length:
            segments.append(torch.zeros(response_length - cursor, dtype=dtype, device=device))
        row = torch.cat(segments) if segments else torch.zeros(response_length, dtype=dtype, device=device)
        if row.numel() != response_length:
            raise ValueError(
                f"scattered replay row has length {row.numel()}, expected {response_length}"
            )
        rows.append(row)
    return torch.stack(rows, dim=0)
