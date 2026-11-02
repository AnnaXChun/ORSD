"""Structured, driver-side diagnostics for paper analyses.

The logger writes one JSON object per optimizer update. It records only values
already present in the training batch and never performs model inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

TURN_REPLAY_KEY = "turn_replay"


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _row_scalar(non_tensor_batch: dict[str, Any], key: str, index: int) -> Any:
    values = non_tensor_batch.get(key)
    if values is None or len(values) <= index:
        return None
    return _json_value(values[index])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    active = values[mask]
    return float(active.float().mean().item()) if active.numel() else 0.0


def build_training_analysis_record(batch, metrics: dict[str, Any], step: int, seed: Any) -> dict[str, Any]:
    """Create an auditable update record from post-advantage training data."""
    response_mask = batch.batch["response_mask"].bool()
    rewards = batch.batch["token_level_rewards"].sum(dim=-1)
    final_advantages = batch.batch["advantages"]
    base_advantages = batch.batch.get("grpo_advantages", final_advantages)
    weights = batch.batch.get("token_credit_weights", torch.ones_like(final_advantages))
    raw_gap = batch.batch.get("token_credit_raw_gap", torch.zeros_like(final_advantages))
    eligibility = batch.batch.get("rubric_teacher_eligible", response_mask).bool()
    if eligibility.ndim == 1:
        eligibility = eligibility.unsqueeze(-1).expand_as(response_mask)
    turn_records = batch.non_tensor_batch.get(TURN_REPLAY_KEY)
    turn_metrics = batch.non_tensor_batch.get("interaction_turn_metrics")

    trajectories = []
    for index in range(len(batch)):
        mask = response_mask[index]
        base = base_advantages[index]
        final = final_advantages[index]
        correction = final - base
        base_abs = _masked_mean(base.abs(), mask)
        correction_abs = _masked_mean(correction.abs(), mask)
        trajectory = {
            "trajectory_index": index,
            "trajectory_id": _row_scalar(batch.non_tensor_batch, "request_id", index) or f"step{step}-row{index}",
            "uid": _row_scalar(batch.non_tensor_batch, "uid", index),
            "task_id": _row_scalar(batch.non_tensor_batch, "task_id", index),
            "training_return": float(rewards[index].item()),
            "grpo_advantage": _masked_mean(base, mask),
            "mean_final_advantage": _masked_mean(final, mask),
            "mean_raw_opd_gap": _masked_mean(raw_gap[index], mask & eligibility[index]),
            "mean_abs_raw_opd_gap": _masked_mean(raw_gap[index].abs(), mask & eligibility[index]),
            "mean_abs_advantage_correction": correction_abs,
            "mean_abs_grpo_advantage": base_abs,
            "correction_to_grpo_abs_ratio": correction_abs / (base_abs + 1e-8),
            "trajectory_average_correction": _masked_mean(correction, mask),
            "token_sign_flip_rate": _masked_mean(((base * final) < 0).float(), mask),
            "weight_min": float(weights[index][mask].min().item()) if mask.any() else 1.0,
            "weight_max": float(weights[index][mask].max().item()) if mask.any() else 1.0,
            "rubric_valid": bool(_row_scalar(batch.non_tensor_batch, "online_rubric_valid", index) or False),
            "rubric_generator_bundle_size": len(
                _row_scalar(batch.non_tensor_batch, "rubric_generator_bundle", index) or []
            ),
            "environment": {
                key: _row_scalar(batch.non_tensor_batch, key, index)
                for key in (
                    "official_reward", "success", "invalid_action_count", "invalid_action_rate",
                    "repeat_no_progress_count", "repeat_no_progress_rate", "episode_steps",
                )
                if _row_scalar(batch.non_tensor_batch, key, index) is not None
            },
            "turns": [],
        }
        if turn_records is not None and index < len(turn_records) and turn_records[index]:
            metrics_for_turns = turn_metrics[index] if turn_metrics is not None and index < len(turn_metrics) else []
            for turn_index, turn in enumerate(turn_records[index].get("turns", [])):
                start = int(turn.get("flat_response_start", 0))
                end = min(int(turn.get("flat_response_end", start)), correction.shape[-1])
                turn_mask = mask[start:end]
                observed = metrics_for_turns[turn_index] if turn_index < len(metrics_for_turns) else {}
                action_type = "invalid" if observed.get("invalid_action", 0) else (
                    "no_progress" if observed.get("repeat_no_progress_action", 0) else "normal"
                )
                trajectory["turns"].append(
                    {
                        "turn_index": turn_index,
                        "action_text": str(turn.get("history_response", "")),
                        "action_type": action_type,
                        "invalid_action": bool(observed.get("invalid_action", 0)),
                        "no_progress_action": bool(observed.get("repeat_no_progress_action", 0)),
                        "mean_raw_opd_gap": _masked_mean(raw_gap[index, start:end], turn_mask),
                        "advantage_correction": _masked_mean(correction[start:end], turn_mask),
                        "optimized_tokens": int(turn_mask.sum().item()),
                    }
                )
        trajectories.append(trajectory)

    scalar_metrics = {
        key: _json_value(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.number)) and (
            key.startswith("rubric_pool/")
            or key.startswith("token_credit/")
            or key.startswith("timing_s/")
            or key in {"training/global_step", "training/epoch"}
        )
    }
    return {
        "schema_version": 1,
        "step": int(step),
        "seed": _json_value(seed),
        "rollout_count": len(trajectories),
        "generated_response_tokens": int(response_mask.sum().item()),
        "metrics": scalar_metrics,
        "trajectories": trajectories,
    }


def maybe_write_training_analysis(config, batch, metrics: dict[str, Any], step: int) -> Path | None:
    trainer_config = config.trainer
    if not bool(trainer_config.get("analysis_log_enable", False)):
        return None
    every = int(trainer_config.get("analysis_log_every_n_steps", 1))
    if every <= 0:
        raise ValueError("trainer.analysis_log_every_n_steps must be positive")
    if step % every:
        return None
    output_dir = trainer_config.get("analysis_log_dir") or str(
        Path(trainer_config.default_local_dir) / "logs" / "analysis"
    )
    path = Path(output_dir) / "train_updates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = config.data.get("seed", None)
    if seed is None:
        seed = config.actor_rollout_ref.actor.get("seed", None)
    record = build_training_analysis_record(batch, metrics, step, seed)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    return path
