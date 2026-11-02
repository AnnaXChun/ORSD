# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = [
    "register_adv_est",
    "get_adv_estimator_fn",
    "AdvantageEstimator",
    "apply_additive_logprob_gap_credit",
    "apply_centered_opd_token_credit",
    "apply_opsd_reverse_kl_advantage",
    "apply_outcome_contrastive_token_credit",
    "apply_rank_attenuated_token_credit",
    "apply_reward_polarized_token_credit",
]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def apply_opsd_reverse_kl_advantage(
    response_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    *,
    eligibility_mask: Optional[torch.Tensor] = None,
    grpo_advantages: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build the sampled-token OPSD reverse-KL advantage.

    OPSD treats the rubric-conditioned policy as a privileged teacher and the
    plain-context policy as the student.  For every sampled action token the
    detached advantage is

    ``log p_teacher(y_t | c_t, rubric) - log p_student(y_t | c_t)``.

    The environment return and GRPO advantage do not enter this objective.
    Rows without a valid saved rubric receive zero advantage instead of a
    silent GRPO fallback, which keeps an OPSD comparison methodologically pure.
    ``grpo_advantages`` is accepted only to report the scale of the objective
    that OPSD replaces.
    """
    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError(
            "teacher_log_probs and student_log_probs must have identical shapes, "
            f"got {tuple(teacher_log_probs.shape)} and {tuple(student_log_probs.shape)}"
        )
    if response_mask.shape != student_log_probs.shape:
        raise ValueError(
            "response_mask must match sampled-token log probabilities, "
            f"got {tuple(response_mask.shape)} and {tuple(student_log_probs.shape)}"
        )
    if grpo_advantages is not None and grpo_advantages.shape != student_log_probs.shape:
        raise ValueError(
            "grpo_advantages must match sampled-token log probabilities, "
            f"got {tuple(grpo_advantages.shape)} and {tuple(student_log_probs.shape)}"
        )

    with torch.no_grad():
        valid_tokens = response_mask.bool()
        if eligibility_mask is None:
            eligible_tokens = valid_tokens
        else:
            if eligibility_mask.ndim == 1:
                if eligibility_mask.shape[0] != student_log_probs.shape[0]:
                    raise ValueError(
                        "trajectory eligibility_mask must have one value per batch item, "
                        f"got {tuple(eligibility_mask.shape)} for batch {student_log_probs.shape[0]}"
                    )
                eligibility_mask = eligibility_mask.unsqueeze(-1).expand_as(student_log_probs)
            elif eligibility_mask.shape != student_log_probs.shape:
                raise ValueError(
                    "token eligibility_mask must match sampled-token log probabilities, "
                    f"got {tuple(eligibility_mask.shape)} and {tuple(student_log_probs.shape)}"
                )
            eligible_tokens = valid_tokens & eligibility_mask.bool()

        raw_gap = (teacher_log_probs.float() - student_log_probs.float()).detach()
        finite_gap = torch.isfinite(raw_gap)
        active_tokens = eligible_tokens & finite_gap
        opsd_advantages_float = torch.where(active_tokens, raw_gap, torch.zeros_like(raw_gap))
        opsd_advantages = opsd_advantages_float.to(student_log_probs.dtype)

        valid_count = valid_tokens.sum().clamp(min=1)
        valid_trajectories = valid_tokens.any(dim=-1)
        valid_trajectory_count = valid_trajectories.sum().clamp(min=1)
        active_count = active_tokens.sum()
        active_trajectories = active_tokens.any(dim=-1) & valid_trajectories
        eligible_trajectories = eligible_tokens.any(dim=-1) & valid_trajectories

        zero = torch.zeros((), device=student_log_probs.device)
        if active_count.item() > 0:
            active_gap = raw_gap[active_tokens]
            gap_mean = active_gap.mean()
            gap_abs_mean = active_gap.abs().mean()
            gap_std = active_gap.std(unbiased=False)
            teacher_preferred_frac = (active_gap > 0).float().mean()
            student_preferred_frac = (active_gap < 0).float().mean()
        else:
            gap_mean = zero
            gap_abs_mean = zero
            gap_std = zero
            teacher_preferred_frac = zero
            student_preferred_frac = zero

        grpo_abs_mean = zero
        if grpo_advantages is not None and valid_tokens.any().item():
            grpo_abs_mean = grpo_advantages.detach().float()[valid_tokens].abs().mean()

        metrics = {
            "opsd/objective_is_pure": 1.0,
            "opsd/active_token_frac": (active_count / valid_count).float().item(),
            "opsd/eligible_trajectory_frac": (
                eligible_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "opsd/active_trajectory_frac": (
                active_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "opsd/finite_gap_frac": ((finite_gap & valid_tokens).sum() / valid_count).float().item(),
            "opsd/logprob_gap_mean": gap_mean.item(),
            "opsd/logprob_gap_abs_mean": gap_abs_mean.item(),
            "opsd/logprob_gap_std": gap_std.item(),
            "opsd/teacher_preferred_frac": teacher_preferred_frac.item(),
            "opsd/student_preferred_frac": student_preferred_frac.item(),
            "opsd/advantage_abs_mean": gap_abs_mean.item(),
            "opsd/replaced_grpo_advantage_abs_mean": grpo_abs_mean.item(),
        }

    return opsd_advantages, metrics


def apply_additive_logprob_gap_credit(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    *,
    coefficient: float = 0.01,
    eligibility_mask: Optional[torch.Tensor] = None,
    trajectory_scores: Optional[torch.Tensor] = None,
    success_threshold: float = 0.5,
    reference_log_probs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Add a small comparison-context sampled-token log-probability gap to GRPO.

    For response token ``t`` in trajectory ``i``:

    ``A_it = A_i + coefficient * (log p_context(y_it) - log p_plain(y_it))``.

    The log-probability gap is detached. Padding, non-finite gaps, and tokens
    excluded by ``eligibility_mask`` retain the original GRPO advantage.
    Unlike the legacy multiplicative rule, this direct additive rule can create
    a small token-level signal when the group-normalized GRPO advantage is zero.
    ``trajectory_scores`` reports whether the context gap is outcome-aligned;
    ``reference_log_probs`` is diagnostic only and never enters the update.

    Returns:
        Adjusted advantages, detached additive adjustments, and diagnostics.
    """
    tensors = {
        "response_mask": response_mask,
        "teacher_log_probs": teacher_log_probs,
        "student_log_probs": student_log_probs,
    }
    for name, tensor in tensors.items():
        if tensor.shape != advantages.shape:
            raise ValueError(f"{name} must have shape {tuple(advantages.shape)}, got {tuple(tensor.shape)}")
    if reference_log_probs is not None and reference_log_probs.shape != advantages.shape:
        raise ValueError(
            "reference_log_probs must match advantages, "
            f"got {tuple(reference_log_probs.shape)} and {tuple(advantages.shape)}"
        )
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(f"token credit coefficient must be finite and non-negative, got {coefficient}")
    if trajectory_scores is not None and trajectory_scores.shape != advantages.shape[:1]:
        raise ValueError(
            "trajectory_scores must have one scalar per batch item, "
            f"got {tuple(trajectory_scores.shape)} for batch {advantages.shape[0]}"
        )
    if not np.isfinite(success_threshold):
        raise ValueError(f"success_threshold must be finite, got {success_threshold}")

    with torch.no_grad():
        valid_tokens = response_mask.bool()
        if eligibility_mask is None:
            eligible_tokens = valid_tokens
        else:
            if eligibility_mask.ndim == 1:
                if eligibility_mask.shape[0] != advantages.shape[0]:
                    raise ValueError(
                        "trajectory eligibility_mask must have one value per batch item, "
                        f"got {tuple(eligibility_mask.shape)} for batch {advantages.shape[0]}"
                    )
                eligibility_mask = eligibility_mask.unsqueeze(-1).expand_as(advantages)
            elif eligibility_mask.shape != advantages.shape:
                raise ValueError(
                    "token eligibility_mask must match advantages, "
                    f"got {tuple(eligibility_mask.shape)} and {tuple(advantages.shape)}"
                )
            eligible_tokens = valid_tokens & eligibility_mask.bool()

        raw_gap = (teacher_log_probs.float() - student_log_probs.float()).detach()
        finite_gap = torch.isfinite(raw_gap)
        active_tokens = eligible_tokens & finite_gap
        effective_gap = torch.where(active_tokens, raw_gap, torch.zeros_like(raw_gap))
        adjustment_float = torch.where(
            active_tokens,
            coefficient * raw_gap,
            torch.zeros_like(raw_gap),
        )
        adjustments = adjustment_float.to(advantages.dtype)
        adjusted_advantages = advantages + adjustments

        grpo_advantages_float = advantages.detach().float()
        adjusted_advantages_float = adjusted_advantages.detach().float()

        valid_count = valid_tokens.sum().clamp(min=1)
        valid_trajectories = valid_tokens.any(dim=-1)
        valid_trajectory_count = valid_trajectories.sum().clamp(min=1)
        active_count = active_tokens.sum()
        active_trajectories = active_tokens.any(dim=-1) & valid_trajectories
        eligible_trajectories = eligible_tokens.any(dim=-1) & valid_trajectories
        nonzero_gap_tokens = active_tokens & (raw_gap.abs() > 1e-8)
        nonzero_adjustment_tokens = active_tokens & (adjustment_float.abs() > 1e-8)
        zero_advantage_tokens = valid_tokens & (advantages.detach().abs() <= 1e-8)
        activated_zero_advantage = zero_advantage_tokens & nonzero_adjustment_tokens
        zero_advantage_count = zero_advantage_tokens.sum().clamp(min=1)

        if valid_tokens.any().item():
            valid_grpo_advantages = grpo_advantages_float[valid_tokens]
            valid_adjusted_advantages = adjusted_advantages_float[valid_tokens]
            valid_adjustments = adjustment_float[valid_tokens]
            valid_effective_gap = effective_gap[valid_tokens]
            effective_gap_mean = valid_effective_gap.mean()
            effective_gap_abs_mean = valid_effective_gap.abs().mean()
            grpo_advantage_mean = valid_grpo_advantages.mean()
            grpo_advantage_abs_mean = valid_grpo_advantages.abs().mean()
            grpo_advantage_std = valid_grpo_advantages.std(unbiased=False)
            adjusted_advantage_mean = valid_adjusted_advantages.mean()
            adjusted_advantage_abs_mean = valid_adjusted_advantages.abs().mean()
            adjusted_advantage_std = valid_adjusted_advantages.std(unbiased=False)
            advantage_delta_mean = valid_adjustments.mean()
            advantage_delta_abs_mean = valid_adjustments.abs().mean()
            if grpo_advantage_abs_mean.item() > 1e-8:
                adjustment_to_grpo_abs_ratio = advantage_delta_abs_mean / grpo_advantage_abs_mean
            else:
                adjustment_to_grpo_abs_ratio = torch.zeros((), device=advantages.device)
        else:
            effective_gap_mean = torch.zeros((), device=advantages.device)
            effective_gap_abs_mean = torch.zeros((), device=advantages.device)
            grpo_advantage_mean = torch.zeros((), device=advantages.device)
            grpo_advantage_abs_mean = torch.zeros((), device=advantages.device)
            grpo_advantage_std = torch.zeros((), device=advantages.device)
            adjusted_advantage_mean = torch.zeros((), device=advantages.device)
            adjusted_advantage_abs_mean = torch.zeros((), device=advantages.device)
            adjusted_advantage_std = torch.zeros((), device=advantages.device)
            advantage_delta_mean = torch.zeros((), device=advantages.device)
            advantage_delta_abs_mean = torch.zeros((), device=advantages.device)
            adjustment_to_grpo_abs_ratio = torch.zeros((), device=advantages.device)

        if active_count.item() > 0:
            active_gap = raw_gap[active_tokens]
            active_adjustment = adjustment_float[active_tokens]
            active_grpo_advantages = grpo_advantages_float[active_tokens]
            raw_gap_mean = active_gap.mean()
            raw_gap_abs_mean = active_gap.abs().mean()
            teacher_preferred_frac = (active_gap > 0).float().mean()
            student_preferred_frac = (active_gap < 0).float().mean()
            adjustment_mean = active_adjustment.mean()
            adjustment_abs_mean = active_adjustment.abs().mean()
            adjustment_std = active_adjustment.std(unbiased=False)
            adjustment_min = active_adjustment.min()
            adjustment_max = active_adjustment.max()
            active_grpo_advantage_mean = active_grpo_advantages.mean()
            active_grpo_advantage_abs_mean = active_grpo_advantages.abs().mean()
            if active_grpo_advantage_abs_mean.item() > 1e-8:
                adjustment_to_active_grpo_abs_ratio = adjustment_abs_mean / active_grpo_advantage_abs_mean
            else:
                adjustment_to_active_grpo_abs_ratio = torch.zeros((), device=advantages.device)
        else:
            zero = torch.zeros((), device=advantages.device)
            raw_gap_mean = zero
            raw_gap_abs_mean = zero
            teacher_preferred_frac = zero
            student_preferred_frac = zero
            adjustment_mean = zero
            adjustment_abs_mean = zero
            adjustment_std = zero
            adjustment_min = zero
            adjustment_max = zero
            active_grpo_advantage_mean = zero
            active_grpo_advantage_abs_mean = zero
            adjustment_to_active_grpo_abs_ratio = zero

        # Diagnose whether the context gap is actually outcome-aligned before
        # trusting it as token credit. The trajectory statistic is the mean
        # sampled-token gap over active response tokens.
        zero = torch.zeros((), device=advantages.device)
        rubric_gap_correct_mean = zero
        rubric_gap_wrong_mean = zero
        rubric_gap_outcome_margin = zero
        rubric_gap_reward_correlation = zero
        correct_trajectory_count = zero
        wrong_trajectory_count = zero
        scored_trajectory_count = zero
        if trajectory_scores is not None:
            scores = trajectory_scores.detach().float().to(advantages.device)
            active_per_trajectory = active_tokens.sum(dim=-1)
            trajectory_gap_mean = effective_gap.sum(dim=-1) / active_per_trajectory.clamp(min=1)
            scored_trajectories = active_per_trajectory > 0
            scored_trajectories &= torch.isfinite(scores) & torch.isfinite(trajectory_gap_mean)
            correct_trajectories = scored_trajectories & (scores >= success_threshold)
            wrong_trajectories = scored_trajectories & (scores < success_threshold)
            correct_trajectory_count = correct_trajectories.sum().float()
            wrong_trajectory_count = wrong_trajectories.sum().float()
            scored_trajectory_count = scored_trajectories.sum().float()

            if correct_trajectories.any().item():
                rubric_gap_correct_mean = trajectory_gap_mean[correct_trajectories].mean()
            if wrong_trajectories.any().item():
                rubric_gap_wrong_mean = trajectory_gap_mean[wrong_trajectories].mean()
            if correct_trajectories.any().item() and wrong_trajectories.any().item():
                rubric_gap_outcome_margin = rubric_gap_correct_mean - rubric_gap_wrong_mean

            if scored_trajectories.sum().item() > 1:
                scored_gaps = trajectory_gap_mean[scored_trajectories]
                scored_rewards = scores[scored_trajectories]
                centered_gaps = scored_gaps - scored_gaps.mean()
                centered_rewards = scored_rewards - scored_rewards.mean()
                denominator = torch.sqrt(
                    centered_gaps.square().sum() * centered_rewards.square().sum()
                )
                if denominator.item() > 1e-12:
                    rubric_gap_reward_correlation = (
                        centered_gaps * centered_rewards
                    ).sum() / denominator

        reference_drift_mean = zero
        reference_drift_abs_mean = zero
        rubric_to_reference_drift_abs_ratio = zero
        reference_drift_available = 0.0
        if reference_log_probs is not None:
            reference_drift = (
                reference_log_probs.detach().float().to(advantages.device)
                - student_log_probs.detach().float()
            )
            finite_reference_drift = torch.isfinite(reference_drift)
            reference_drift_tokens = active_tokens & finite_reference_drift
            if reference_drift_tokens.any().item():
                active_reference_drift = reference_drift[reference_drift_tokens]
                reference_drift_mean = active_reference_drift.mean()
                reference_drift_abs_mean = active_reference_drift.abs().mean()
                reference_drift_available = 1.0
                if reference_drift_abs_mean.item() > 1e-12:
                    rubric_to_reference_drift_abs_ratio = raw_gap_abs_mean / reference_drift_abs_mean

        metrics = {
            "token_credit/additive_coefficient": float(coefficient),
            "token_credit/active_frac": (active_count / valid_count).float().item(),
            "token_credit/active_trajectory_frac": (
                active_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/eligible_frac": (eligible_tokens.sum() / valid_count).float().item(),
            "token_credit/eligible_trajectory_frac": (
                eligible_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/finite_gap_frac": ((finite_gap & valid_tokens).sum() / valid_count).float().item(),
            "token_credit/nonzero_gap_frac": (nonzero_gap_tokens.sum() / valid_count).float().item(),
            "token_credit/nonzero_adjustment_frac": (
                nonzero_adjustment_tokens.sum() / valid_count
            ).float().item(),
            "token_credit/zero_advantage_activated_frac": (
                activated_zero_advantage.sum() / zero_advantage_count
            ).float().item(),
            "token_credit/raw_gap_mean": raw_gap_mean.item(),
            "token_credit/raw_gap_abs_mean": raw_gap_abs_mean.item(),
            "token_credit/gap_abs_mean": raw_gap_abs_mean.item(),
            "token_credit/active_gap_mean": raw_gap_mean.item(),
            "token_credit/active_gap_abs_mean": raw_gap_abs_mean.item(),
            "token_credit/effective_gap_mean": effective_gap_mean.item(),
            "token_credit/effective_gap_abs_mean": effective_gap_abs_mean.item(),
            "token_credit/teacher_preferred_frac": teacher_preferred_frac.item(),
            "token_credit/student_preferred_frac": student_preferred_frac.item(),
            "token_credit/grpo_advantage_mean": grpo_advantage_mean.item(),
            "token_credit/grpo_advantage_abs_mean": grpo_advantage_abs_mean.item(),
            "token_credit/grpo_advantage_std": grpo_advantage_std.item(),
            "token_credit/active_grpo_advantage_mean": active_grpo_advantage_mean.item(),
            "token_credit/active_grpo_advantage_abs_mean": active_grpo_advantage_abs_mean.item(),
            "token_credit/adjusted_advantage_mean": adjusted_advantage_mean.item(),
            "token_credit/adjusted_advantage_abs_mean": adjusted_advantage_abs_mean.item(),
            "token_credit/adjusted_advantage_std": adjusted_advantage_std.item(),
            "token_credit/advantage_delta_mean": advantage_delta_mean.item(),
            "token_credit/advantage_delta_abs_mean": advantage_delta_abs_mean.item(),
            "token_credit/adjustment_to_grpo_abs_ratio": adjustment_to_grpo_abs_ratio.item(),
            "token_credit/adjustment_to_grpo_abs_pct": (100.0 * adjustment_to_grpo_abs_ratio).item(),
            "token_credit/adjustment_to_active_grpo_abs_ratio": (
                adjustment_to_active_grpo_abs_ratio.item()
            ),
            "token_credit/adjustment_to_active_grpo_abs_pct": (
                100.0 * adjustment_to_active_grpo_abs_ratio
            ).item(),
            "token_credit/adjustment_mean": adjustment_mean.item(),
            "token_credit/adjustment_abs_mean": adjustment_abs_mean.item(),
            "token_credit/adjustment_std": adjustment_std.item(),
            "token_credit/adjustment_min": adjustment_min.item(),
            "token_credit/adjustment_max": adjustment_max.item(),
            "token_credit/rubric_gap_correct_mean": rubric_gap_correct_mean.item(),
            "token_credit/rubric_gap_wrong_mean": rubric_gap_wrong_mean.item(),
            "token_credit/rubric_gap_outcome_margin": rubric_gap_outcome_margin.item(),
            "token_credit/rubric_gap_reward_correlation": rubric_gap_reward_correlation.item(),
            "token_credit/rubric_gap_correct_trajectory_count": correct_trajectory_count.item(),
            "token_credit/rubric_gap_wrong_trajectory_count": wrong_trajectory_count.item(),
            "token_credit/rubric_gap_scored_trajectory_frac": (
                scored_trajectory_count / valid_trajectory_count
            ).item(),
            "token_credit/reference_drift_available": reference_drift_available,
            "token_credit/reference_drift_mean": reference_drift_mean.item(),
            "token_credit/reference_drift_abs_mean": reference_drift_abs_mean.item(),
            "token_credit/rubric_to_reference_drift_abs_ratio": (
                rubric_to_reference_drift_abs_ratio.item()
            ),
        }

    return adjusted_advantages, adjustments, metrics


def apply_centered_opd_token_credit(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    *,
    trajectory_scores: torch.Tensor,
    group_ids,
    coefficient: float = 0.1,
    clip_value: float = 2.0,
    multiplier_min: float = 0.5,
    multiplier_max: float = 1.5,
    eligibility_mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Redistribute GRPO credit within a trajectory using centered OPD gaps.

    The raw sampled-token gap is ``log p_teacher - log p_student``. It is
    centered and standardized independently in each trajectory, then clipped.
    A task-group gate is the positive part of the Spearman correlation between
    environment reward and each trajectory's *uncentered* mean gap. The token
    multiplier is therefore
    ``1 + coefficient * gate * sign(A_GRPO) * normalized_gap``. Including the
    GRPO sign preserves the direction of the original additive OPD signal:
    positive normalized gaps receive relatively more credit in both successful
    and failed trajectories, while the bounded multiplier prevents sign flips.

    Active multipliers are projected onto the configured bounds while retaining
    mean one. Consequently, for the trajectory-constant advantages produced by
    GRPO, this route changes only within-trajectory token credit and cannot
    change the trajectory-level advantage or reverse its sign.
    """
    tensors = {
        "response_mask": response_mask,
        "teacher_log_probs": teacher_log_probs,
        "student_log_probs": student_log_probs,
    }
    for name, tensor in tensors.items():
        if tensor.shape != advantages.shape:
            raise ValueError(f"{name} must have shape {tuple(advantages.shape)}, got {tuple(tensor.shape)}")
    if trajectory_scores.shape != advantages.shape[:1]:
        raise ValueError(
            "trajectory_scores must have one scalar per batch item, "
            f"got {tuple(trajectory_scores.shape)} for batch {advantages.shape[0]}"
        )
    group_ids = np.asarray(group_ids, dtype=object).reshape(-1).tolist()
    if len(group_ids) != advantages.shape[0]:
        raise ValueError(
            "group_ids must have one value per batch item, "
            f"got {len(group_ids)} for batch {advantages.shape[0]}"
        )
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError(f"centered OPD coefficient must be finite and nonnegative, got {coefficient}")
    if not np.isfinite(clip_value) or clip_value <= 0:
        raise ValueError(f"centered OPD clip_value must be positive and finite, got {clip_value}")
    if (
        not np.isfinite(multiplier_min)
        or not np.isfinite(multiplier_max)
        or multiplier_min <= 0
        or multiplier_min > 1
        or multiplier_max < 1
        or multiplier_min >= multiplier_max
    ):
        raise ValueError("centered OPD multiplier bounds must satisfy 0 < min <= 1 <= max and min < max")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"centered OPD epsilon must be positive and finite, got {epsilon}")

    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranks

    def spearman(values_x: np.ndarray, values_y: np.ndarray) -> float:
        if len(values_x) < 2:
            return 0.0
        rank_x = average_ranks(values_x)
        rank_y = average_ranks(values_y)
        centered_x = rank_x - rank_x.mean()
        centered_y = rank_y - rank_y.mean()
        denominator = np.sqrt(np.square(centered_x).sum() * np.square(centered_y).sum())
        if denominator <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(centered_x, centered_y) / denominator, -1.0, 1.0))

    with torch.no_grad():
        valid_tokens = response_mask.bool()
        if eligibility_mask is None:
            eligible_tokens = valid_tokens
        else:
            if eligibility_mask.ndim == 1:
                if eligibility_mask.shape[0] != advantages.shape[0]:
                    raise ValueError(
                        "trajectory eligibility_mask must have one value per batch item, "
                        f"got {tuple(eligibility_mask.shape)} for batch {advantages.shape[0]}"
                    )
                eligibility_mask = eligibility_mask.unsqueeze(-1).expand_as(advantages)
            elif eligibility_mask.shape != advantages.shape:
                raise ValueError(
                    "token eligibility_mask must match advantages, "
                    f"got {tuple(eligibility_mask.shape)} and {tuple(advantages.shape)}"
                )
            eligible_tokens = valid_tokens & eligibility_mask.bool()

        raw_gap = (teacher_log_probs.float() - student_log_probs.float()).detach()
        active_tokens = eligible_tokens & torch.isfinite(raw_gap)
        active_counts = active_tokens.sum(dim=-1, keepdim=True)
        active_trajectories = active_counts.squeeze(-1) > 0
        active_float = active_tokens.float()
        trajectory_gap_mean = (torch.where(active_tokens, raw_gap, 0.0).sum(dim=-1, keepdim=True)
                               / active_counts.clamp(min=1))
        centered_gap = torch.where(active_tokens, raw_gap - trajectory_gap_mean, 0.0)
        trajectory_variance = (
            centered_gap.square().sum(dim=-1, keepdim=True) / active_counts.clamp(min=1)
        )
        trajectory_std = trajectory_variance.sqrt()
        normalized_gap = torch.where(
            active_tokens,
            centered_gap / trajectory_std.clamp(min=epsilon),
            0.0,
        )
        normalized_gap = torch.where(
            trajectory_std > epsilon,
            normalized_gap,
            torch.zeros_like(normalized_gap),
        )
        clipped_gap = normalized_gap.clamp(min=-clip_value, max=clip_value)

        scores = trajectory_scores.detach().float().cpu().numpy()
        gap_means = trajectory_gap_mean.squeeze(-1).detach().cpu().numpy()
        finite_scores = np.isfinite(scores)
        gates = torch.zeros(advantages.shape[0], dtype=torch.float32, device=advantages.device)
        grouped_indices = defaultdict(list)
        for index, group_id in enumerate(group_ids):
            grouped_indices[group_id].append(index)
        group_gate_values = []
        eligible_group_gate_values = []
        for indices in grouped_indices.values():
            usable = [
                index
                for index in indices
                if active_trajectories[index].item()
                and finite_scores[index]
                and np.isfinite(gap_means[index])
            ]
            correlation = spearman(scores[usable], gap_means[usable]) if len(usable) >= 2 else 0.0
            gate = max(0.0, correlation)
            group_gate_values.append(gate)
            if len(usable) >= 2:
                eligible_group_gate_values.append(gate)
            if usable:
                gates[usable] = gate

        valid_counts = valid_tokens.sum(dim=-1, keepdim=True).clamp(min=1)
        trajectory_advantage = (
            torch.where(valid_tokens, advantages.detach().float(), 0.0).sum(dim=-1, keepdim=True)
            / valid_counts
        )
        advantage_polarity = trajectory_advantage.sign()
        raw_weights = 1.0 + coefficient * gates.unsqueeze(-1) * advantage_polarity * clipped_gap
        token_weights_float = torch.ones_like(raw_gap)

        # Clipping can shift the mean away from one. Project each active row by
        # a scalar offset through the box, which preserves both the bounds and
        # the exact mean-one constraint without modifying ineligible tokens.
        for row in torch.nonzero(active_trajectories, as_tuple=False).flatten().tolist():
            row_mask = active_tokens[row]
            values = raw_weights[row, row_mask]
            low = float(multiplier_min - values.max().item())
            high = float(multiplier_max - values.min().item())
            for _ in range(48):
                midpoint = 0.5 * (low + high)
                mean = values.add(midpoint).clamp(multiplier_min, multiplier_max).mean().item()
                if mean < 1.0:
                    low = midpoint
                else:
                    high = midpoint
            projected = values.add(0.5 * (low + high)).clamp(multiplier_min, multiplier_max)
            token_weights_float[row, row_mask] = projected

        token_weights = token_weights_float.to(advantages.dtype)
        weighted_advantages = advantages * token_weights
        valid_count = valid_tokens.sum().clamp(min=1)
        active_count = active_tokens.sum()
        valid_trajectory_count = valid_tokens.any(dim=-1).sum().clamp(min=1)
        active_trajectory_count = active_trajectories.sum()
        if active_count.item() > 0:
            active_raw_gap = raw_gap[active_tokens]
            active_centered_gap = centered_gap[active_tokens]
            active_normalized_gap = normalized_gap[active_tokens]
            active_clipped_gap = clipped_gap[active_tokens]
            active_weights = token_weights_float[active_tokens]
            clip_fraction = (active_normalized_gap.abs() > clip_value).float().mean()
            raw_gap_mean = active_raw_gap.mean()
            raw_gap_abs_mean = active_raw_gap.abs().mean()
            centered_gap_abs_mean = active_centered_gap.abs().mean()
            normalized_gap_abs_mean = active_normalized_gap.abs().mean()
            clipped_gap_abs_mean = active_clipped_gap.abs().mean()
            weight_std = active_weights.std(unbiased=False)
            weight_min = active_weights.min()
            weight_max = active_weights.max()
        else:
            zero = torch.zeros((), device=advantages.device)
            clip_fraction = zero
            raw_gap_mean = zero
            raw_gap_abs_mean = zero
            centered_gap_abs_mean = zero
            normalized_gap_abs_mean = zero
            clipped_gap_abs_mean = zero
            weight_std = zero
            weight_min = torch.ones((), device=advantages.device)
            weight_max = torch.ones((), device=advantages.device)

        active_weight_means = (
            (token_weights_float * active_float).sum(dim=-1) / active_counts.squeeze(-1).clamp(min=1)
        )
        weight_mean_error = torch.where(
            active_trajectories,
            (active_weight_means - 1.0).abs(),
            torch.zeros_like(active_weight_means),
        )
        valid_float = valid_tokens.float()
        masked_counts = valid_tokens.sum(dim=-1).clamp(min=1)
        original_means = (advantages.float() * valid_float).sum(dim=-1) / masked_counts
        adjusted_means = (weighted_advantages.float() * valid_float).sum(dim=-1) / masked_counts
        advantage_mean_error = torch.where(
            valid_tokens.any(dim=-1),
            (adjusted_means - original_means).abs(),
            torch.zeros_like(original_means),
        )
        group_gate_array = np.asarray(group_gate_values, dtype=np.float64)
        eligible_group_gate_array = np.asarray(eligible_group_gate_values, dtype=np.float64)
        gate_mean = float(group_gate_array.mean()) if group_gate_array.size else 0.0
        gate_max = float(group_gate_array.max()) if group_gate_array.size else 0.0
        gate_active_frac = (
            float((eligible_group_gate_array > 0).mean()) if eligible_group_gate_array.size else 0.0
        )
        metrics = {
            "token_credit/centered_opd_coefficient": float(coefficient),
            "token_credit/active_frac": (active_count / valid_count).float().item(),
            "token_credit/active_trajectory_frac": (
                active_trajectory_count / valid_trajectory_count
            ).float().item(),
            "token_credit/group_gate_mean": gate_mean,
            "token_credit/group_gate_max": gate_max,
            "token_credit/group_gate_active_frac": gate_active_frac,
            "token_credit/group_gate_accepted_count": float((eligible_group_gate_array > 0).sum()),
            "token_credit/group_gate_eligible_count": float(eligible_group_gate_array.size),
            "token_credit/group_gate_active_frac_all_groups": (
                float((group_gate_array > 0).mean()) if group_gate_array.size else 0.0
            ),
            "token_credit/nonzero_advantage_trajectory_frac": (
                (advantage_polarity.squeeze(-1) != 0).sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/raw_gap_mean": raw_gap_mean.item(),
            "token_credit/raw_gap_abs_mean": raw_gap_abs_mean.item(),
            "token_credit/centered_gap_abs_mean": centered_gap_abs_mean.item(),
            "token_credit/normalized_gap_abs_mean": normalized_gap_abs_mean.item(),
            "token_credit/clipped_gap_abs_mean": clipped_gap_abs_mean.item(),
            "token_credit/clipfrac": clip_fraction.item(),
            "token_credit/weight_std": weight_std.item(),
            "token_credit/weight_min": weight_min.item(),
            "token_credit/weight_max": weight_max.item(),
            "token_credit/trajectory_weight_mean_error_max": weight_mean_error.max().item(),
            "token_credit/trajectory_advantage_mean_error_max": advantage_mean_error.max().item(),
        }

    return weighted_advantages, token_weights, metrics


def apply_rank_attenuated_token_credit(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    *,
    attenuation: float = 0.2,
    conflict_fraction: float = 0.25,
    min_gap_range: float = 1e-4,
    eligibility_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Selectively weaken questionable trajectory-credit inheritance by rank.

    The rubric-conditioned sampled-token gap is used only to rank eligible
    response tokens within each trajectory. For a positive GRPO trajectory,
    tokens in the lowest ``conflict_fraction`` of ranks inherit only
    ``1 - attenuation`` of the trajectory advantage. For a negative trajectory,
    the ordering is reversed, so the highest-gap tokens receive the same
    conservative discount. No token is amplified, assigned a new sign, or
    activated when its trajectory-level GRPO advantage is zero.

    A trajectory whose finite eligible gaps span no more than
    ``min_gap_range`` is left unchanged. Padding and ineligible tokens never
    participate in ranking and retain their original advantages.
    """
    tensors = {
        "response_mask": response_mask,
        "teacher_log_probs": teacher_log_probs,
        "student_log_probs": student_log_probs,
    }
    for name, tensor in tensors.items():
        if tensor.shape != advantages.shape:
            raise ValueError(f"{name} must have shape {tuple(advantages.shape)}, got {tuple(tensor.shape)}")
    if not np.isfinite(attenuation) or not 0.0 <= attenuation < 1.0:
        raise ValueError(f"rank attenuation must be finite and in [0, 1), got {attenuation}")
    if not np.isfinite(conflict_fraction) or not 0.0 < conflict_fraction <= 0.5:
        raise ValueError(
            f"rank conflict_fraction must be finite and in (0, 0.5], got {conflict_fraction}"
        )
    if not np.isfinite(min_gap_range) or min_gap_range < 0.0:
        raise ValueError(f"rank min_gap_range must be finite and nonnegative, got {min_gap_range}")

    with torch.no_grad():
        valid_tokens = response_mask.bool()
        if eligibility_mask is None:
            eligible_tokens = valid_tokens
        else:
            if eligibility_mask.ndim == 1:
                if eligibility_mask.shape[0] != advantages.shape[0]:
                    raise ValueError(
                        "trajectory eligibility_mask must have one value per batch item, "
                        f"got {tuple(eligibility_mask.shape)} for batch {advantages.shape[0]}"
                    )
                eligibility_mask = eligibility_mask.unsqueeze(-1).expand_as(advantages)
            elif eligibility_mask.shape != advantages.shape:
                raise ValueError(
                    "token eligibility_mask must match advantages, "
                    f"got {tuple(eligibility_mask.shape)} and {tuple(advantages.shape)}"
                )
            eligible_tokens = valid_tokens & eligibility_mask.bool()

        raw_gap = (teacher_log_probs.float() - student_log_probs.float()).detach()
        finite_gap = torch.isfinite(raw_gap)
        valid_counts = valid_tokens.sum(dim=-1, keepdim=True).clamp(min=1)
        trajectory_advantage = (
            torch.where(valid_tokens, advantages.detach().float(), 0.0).sum(dim=-1, keepdim=True)
            / valid_counts
        )
        nonzero_trajectory = trajectory_advantage.abs().squeeze(-1) > 1e-8
        rankable_tokens = eligible_tokens & finite_gap & nonzero_trajectory.unsqueeze(-1)

        token_weights_float = torch.ones_like(raw_gap)
        rank_scores = torch.zeros_like(raw_gap)
        conflict_tokens = torch.zeros_like(valid_tokens)
        active_tokens = torch.zeros_like(valid_tokens)
        active_trajectories = torch.zeros(advantages.shape[0], dtype=torch.bool, device=advantages.device)
        no_spread_trajectories = torch.zeros_like(active_trajectories)
        gap_ranges = torch.zeros(advantages.shape[0], dtype=torch.float32, device=advantages.device)
        for row in torch.nonzero(rankable_tokens.any(dim=-1), as_tuple=False).flatten().tolist():
            row_mask = rankable_tokens[row]
            values = raw_gap[row, row_mask]
            if values.numel() < 2:
                no_spread_trajectories[row] = True
                continue
            gap_range = values.max() - values.min()
            gap_ranges[row] = gap_range
            if gap_range.item() <= min_gap_range:
                no_spread_trajectories[row] = True
                continue

            # Average ranks make tied gap values indistinguishable instead of
            # attenuating an arbitrary token selected by sort order.
            _, inverse, counts = torch.unique(
                values, sorted=True, return_inverse=True, return_counts=True
            )
            starts = counts.cumsum(dim=0) - counts
            average_ranks = starts.to(torch.float32) + 0.5 * (counts.to(torch.float32) - 1.0)
            scores = 2.0 * average_ranks[inverse] / float(values.numel() - 1) - 1.0
            rank_scores[row, row_mask] = scores
            consistency = trajectory_advantage[row].sign() * scores
            budget = int(values.numel() * conflict_fraction)
            row_conflicts = torch.zeros_like(consistency, dtype=torch.bool)
            if budget > 0:
                # Never split an equal-consistency tie group merely to fill the
                # requested fraction. This keeps the selection deterministic
                # and caps attenuation at the configured token budget.
                _, consistency_inverse, consistency_counts = torch.unique(
                    consistency, sorted=True, return_inverse=True, return_counts=True
                )
                complete_groups = consistency_counts.cumsum(dim=0) <= budget
                row_conflicts = complete_groups[consistency_inverse]
            active_tokens[row, row_mask] = True
            active_trajectories[row] = True
            conflict_tokens[row, row_mask] = row_conflicts

        token_weights_float[conflict_tokens] = 1.0 - float(attenuation)
        token_weights = token_weights_float.to(advantages.dtype)
        adjusted_advantages = advantages * token_weights
        attenuated_tokens = conflict_tokens & (token_weights_float < 1.0 - 1e-8)

        valid_count = valid_tokens.sum().clamp(min=1)
        valid_trajectories = valid_tokens.any(dim=-1)
        valid_trajectory_count = valid_trajectories.sum().clamp(min=1)
        eligible_trajectories = eligible_tokens.any(dim=-1) & valid_trajectories
        rankable_trajectories = rankable_tokens.any(dim=-1) & valid_trajectories
        active_count = active_tokens.sum()
        conflict_count = conflict_tokens.sum()
        attenuated_count = attenuated_tokens.sum()
        attenuated_trajectories = attenuated_tokens.any(dim=-1) & valid_trajectories

        zero = torch.zeros((), dtype=torch.float32, device=advantages.device)
        if active_count.item() > 0:
            active_gap_abs_mean = raw_gap[active_tokens].abs().mean()
            active_rank_abs_mean = rank_scores[active_tokens].abs().mean()
            active_weights = token_weights_float[active_tokens]
            weight_mean = active_weights.mean()
            weight_std = active_weights.std(unbiased=False)
            weight_min = active_weights.min()
            weight_max = active_weights.max()
            active_gap_ranges = gap_ranges[active_trajectories]
            gap_range_mean = active_gap_ranges.mean()
            gap_range_min = active_gap_ranges.min()
            gap_range_max = active_gap_ranges.max()
        else:
            active_gap_abs_mean = zero
            active_rank_abs_mean = zero
            weight_mean = torch.ones((), device=advantages.device)
            weight_std = zero
            weight_min = torch.ones((), device=advantages.device)
            weight_max = torch.ones((), device=advantages.device)
            gap_range_mean = zero
            gap_range_min = zero
            gap_range_max = zero

        valid_float = valid_tokens.float()
        original_means = (advantages.detach().float() * valid_float).sum(dim=-1) / valid_counts.squeeze(-1)
        adjusted_means = (adjusted_advantages.detach().float() * valid_float).sum(dim=-1) / valid_counts.squeeze(-1)
        mean_abs_reduction = torch.where(
            active_trajectories,
            original_means.abs() - adjusted_means.abs(),
            torch.zeros_like(original_means),
        )
        trajectory_scales = torch.where(
            active_trajectories,
            adjusted_means.abs() / original_means.abs().clamp(min=1e-8),
            torch.ones_like(original_means),
        )
        sign_flips = valid_tokens & (advantages.detach() * adjusted_advantages.detach() < 0)

        metrics = {
            "token_credit/rank_attenuation_strength": float(attenuation),
            "token_credit/rank_conflict_fraction": float(conflict_fraction),
            "token_credit/rank_min_gap_range": float(min_gap_range),
            "token_credit/active_frac": (active_count / valid_count).float().item(),
            "token_credit/active_trajectory_frac": (
                active_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/eligible_frac": (eligible_tokens.sum() / valid_count).float().item(),
            "token_credit/eligible_trajectory_frac": (
                eligible_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/rankable_trajectory_frac": (
                rankable_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/no_spread_trajectory_frac": (
                no_spread_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/conflict_frac": (conflict_count / valid_count).float().item(),
            "token_credit/conflict_active_frac": (
                conflict_count / active_count.clamp(min=1)
            ).float().item(),
            "token_credit/attenuated_frac": (attenuated_count / valid_count).float().item(),
            "token_credit/attenuated_active_frac": (
                attenuated_count / active_count.clamp(min=1)
            ).float().item(),
            "token_credit/attenuated_trajectory_frac": (
                attenuated_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/raw_gap_abs_mean": active_gap_abs_mean.item(),
            "token_credit/active_gap_abs_mean": active_gap_abs_mean.item(),
            "token_credit/active_rank_abs_mean": active_rank_abs_mean.item(),
            "token_credit/rank_gap_range_mean": gap_range_mean.item(),
            "token_credit/rank_gap_range_min": gap_range_min.item(),
            "token_credit/rank_gap_range_max": gap_range_max.item(),
            "token_credit/weight_mean": weight_mean.item(),
            "token_credit/weight_std": weight_std.item(),
            "token_credit/weight_min": weight_min.item(),
            "token_credit/weight_max": weight_max.item(),
            "token_credit/trajectory_advantage_mean_abs_reduction": (
                mean_abs_reduction[active_trajectories].mean().item()
                if active_trajectories.any().item()
                else 0.0
            ),
            "token_credit/trajectory_advantage_scale_mean": (
                trajectory_scales[active_trajectories].mean().item()
                if active_trajectories.any().item()
                else 1.0
            ),
            "token_credit/advantage_sign_flip_frac": (
                sign_flips.sum() / valid_count
            ).float().item(),
        }

    return adjusted_advantages, token_weights, metrics


def apply_reward_polarized_token_credit(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    *,
    alpha: float = 0.1,
    scale_quantile: float = 0.5,
    min_scale: float = 1e-4,
    eligibility_mask: Optional[torch.Tensor] = None,
    preserve_trajectory_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Redistribute a GRPO trajectory advantage across sampled response tokens.

    For token ``t`` in trajectory ``i``, let ``A_i`` be the group-normalized
    GRPO advantage and let ``raw_gap_it = log p_teacher - log p_student`` on
    the sampled token. To remove trajectory-wide calibration and prompt shifts,
    the raw gap is centered by its per-trajectory median and divided by a
    per-trajectory robust absolute-deviation quantile. The positive token
    multiplier is

    ``1 + alpha * tanh(sign(A_i) * relative_gap_it)``.

    The multiplier is detached from both policies and, by default, normalized
    to mean one within each trajectory. Consequently this function cannot
    create signal for an all-correct/all-wrong group, cannot reverse the sign
    determined by the verifier when ``alpha < 1``, and does not secretly change
    the average learning rate of a trajectory.

    ``eligibility_mask`` may be a trajectory mask of shape ``[batch]`` or a
    token mask matching ``advantages``. A rubric pipeline can use it to limit
    teacher-based credit to validated trajectories. Ineligible samples retain
    ordinary GRPO weights.

    Returns:
        A tuple containing reweighted advantages, detached token weights, and
        scalar diagnostics suitable for trainer logging.
    """
    tensors = {
        "response_mask": response_mask,
        "teacher_log_probs": teacher_log_probs,
        "student_log_probs": student_log_probs,
    }
    for name, tensor in tensors.items():
        if tensor.shape != advantages.shape:
            raise ValueError(f"{name} must have shape {tuple(advantages.shape)}, got {tuple(tensor.shape)}")
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"token credit alpha must be in [0, 1), got {alpha}")
    if not 0.0 < scale_quantile <= 1.0:
        raise ValueError(f"token credit scale_quantile must be in (0, 1], got {scale_quantile}")
    if min_scale <= 0.0:
        raise ValueError(f"token credit min_scale must be positive, got {min_scale}")

    with torch.no_grad():
        valid_tokens = response_mask.bool()
        if eligibility_mask is None:
            eligible_tokens = valid_tokens
        else:
            if eligibility_mask.ndim == 1:
                if eligibility_mask.shape[0] != advantages.shape[0]:
                    raise ValueError(
                        "trajectory eligibility_mask must have one value per batch item, "
                        f"got {tuple(eligibility_mask.shape)} for batch {advantages.shape[0]}"
                    )
                eligibility_mask = eligibility_mask.unsqueeze(-1).expand_as(advantages)
            elif eligibility_mask.shape != advantages.shape:
                raise ValueError(
                    "token eligibility_mask must match advantages, "
                    f"got {tuple(eligibility_mask.shape)} and {tuple(advantages.shape)}"
                )
            eligible_tokens = valid_tokens & eligibility_mask.bool()

        raw_token_gap = (teacher_log_probs - student_log_probs).detach()
        nonzero_advantage = advantages.detach().abs() > 1e-8
        finite_gap = torch.isfinite(raw_token_gap)
        active_tokens = eligible_tokens & nonzero_advantage & finite_gap
        token_weights = torch.ones_like(advantages)

        active_count = active_tokens.sum()
        if active_count.item() > 0:
            active_trajectory_mask = active_tokens.any(dim=-1)
            nan = torch.tensor(float("nan"), dtype=torch.float32, device=advantages.device)
            active_gap = raw_token_gap.float().masked_fill(~active_tokens, nan)
            active_gap = torch.where(
                active_trajectory_mask.unsqueeze(-1), active_gap, torch.zeros_like(active_gap)
            )

            # Quantiles are computed independently for every trajectory. The
            # vectorized nanquantile calls avoid one GPU kernel launch per row.
            trajectory_centers = torch.nanquantile(active_gap, 0.5, dim=-1, keepdim=True)
            trajectory_centers = torch.where(
                active_trajectory_mask.unsqueeze(-1), trajectory_centers, torch.zeros_like(trajectory_centers)
            )
            centered_gap_float = torch.where(
                active_tokens, raw_token_gap.float() - trajectory_centers, torch.zeros_like(active_gap)
            )
            active_abs_deviation = centered_gap_float.abs().masked_fill(~active_tokens, nan)
            active_abs_deviation = torch.where(
                active_trajectory_mask.unsqueeze(-1),
                active_abs_deviation,
                torch.zeros_like(active_abs_deviation),
            )
            trajectory_scales = torch.nanquantile(
                active_abs_deviation, scale_quantile, dim=-1, keepdim=True
            )
            trajectory_scales = torch.where(
                active_trajectory_mask.unsqueeze(-1),
                trajectory_scales.clamp(min=min_scale),
                torch.ones_like(trajectory_scales),
            )
            relative_gap_float = torch.where(
                active_tokens, centered_gap_float / trajectory_scales, torch.zeros_like(centered_gap_float)
            )
            centered_gap = centered_gap_float.to(raw_token_gap.dtype)
            relative_gap = relative_gap_float.to(raw_token_gap.dtype)

            reward_polarity = advantages.detach().sign()
            aligned_gap = reward_polarity * relative_gap
            raw_weights = 1.0 + alpha * torch.tanh(aligned_gap)
            token_weights = torch.where(active_tokens, raw_weights, token_weights)

            if preserve_trajectory_mean:
                mask_float = valid_tokens.to(token_weights.dtype)
                mean_weight = (token_weights * mask_float).sum(dim=-1, keepdim=True) / mask_float.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1)
                token_weights = torch.where(
                    valid_tokens,
                    token_weights / mean_weight.clamp(min=min_scale),
                    token_weights,
                )

            active_weights = token_weights[active_tokens].float()
            aligned_fraction = (
                ((reward_polarity * centered_gap) > 0) & active_tokens
            ).float().sum() / active_count.clamp(min=1)
            raw_gap_abs_mean = raw_token_gap[active_tokens].abs().float().mean()
            centered_gap_abs_mean = centered_gap[active_tokens].abs().float().mean()
            relative_gap_abs_mean = relative_gap[active_tokens].abs().float().mean()
            active_trajectory_scales = trajectory_scales.squeeze(-1)[active_trajectory_mask]
            active_trajectory_centers = trajectory_centers.squeeze(-1)[active_trajectory_mask]
            scale_value = active_trajectory_scales.mean()
            scale_min = active_trajectory_scales.min()
            scale_max = active_trajectory_scales.max()
            center_abs_mean = active_trajectory_centers.abs().mean()
            weight_std = active_weights.std(unbiased=False)
            weight_min = active_weights.min()
            weight_max = active_weights.max()
        else:
            aligned_fraction = torch.zeros((), device=advantages.device)
            raw_gap_abs_mean = torch.zeros((), device=advantages.device)
            centered_gap_abs_mean = torch.zeros((), device=advantages.device)
            relative_gap_abs_mean = torch.zeros((), device=advantages.device)
            scale_value = torch.zeros((), device=advantages.device)
            scale_min = torch.zeros((), device=advantages.device)
            scale_max = torch.zeros((), device=advantages.device)
            center_abs_mean = torch.zeros((), device=advantages.device)
            weight_std = torch.zeros((), device=advantages.device)
            weight_min = torch.ones((), device=advantages.device)
            weight_max = torch.ones((), device=advantages.device)

        valid_count = valid_tokens.sum().clamp(min=1)
        valid_trajectories = valid_tokens.any(dim=-1)
        valid_trajectory_count = valid_trajectories.sum().clamp(min=1)
        eligible_finite_tokens = eligible_tokens & finite_gap
        eligible_finite_count = eligible_finite_tokens.sum()
        nonzero_gap_tokens = eligible_finite_tokens & (raw_token_gap.abs() > 1e-8)
        centered_nonzero_gap_tokens = (
            active_tokens & (centered_gap.abs() > 1e-8) if active_count.item() > 0 else active_tokens
        )
        eligible_trajectories = eligible_tokens.any(dim=-1) & valid_trajectories
        nonzero_advantage_trajectories = (nonzero_advantage & valid_tokens).any(dim=-1) & valid_trajectories
        active_trajectories = active_tokens.any(dim=-1) & valid_trajectories
        if eligible_finite_count.item() > 0:
            eligible_gap_abs_mean = raw_token_gap[eligible_finite_tokens].abs().float().mean()
        else:
            eligible_gap_abs_mean = torch.zeros((), device=advantages.device)
        metrics = {
            "token_credit/active_frac": (active_count / valid_count).float().item(),
            "token_credit/active_trajectory_frac": (
                active_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/eligible_frac": (eligible_tokens.sum() / valid_count).float().item(),
            "token_credit/eligible_trajectory_frac": (
                eligible_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/nonzero_advantage_frac": (
                (nonzero_advantage & valid_tokens).sum() / valid_count
            ).float().item(),
            "token_credit/nonzero_advantage_trajectory_frac": (
                nonzero_advantage_trajectories.sum() / valid_trajectory_count
            ).float().item(),
            "token_credit/finite_gap_frac": ((finite_gap & valid_tokens).sum() / valid_count).float().item(),
            "token_credit/nonzero_gap_frac": (nonzero_gap_tokens.sum() / valid_count).float().item(),
            "token_credit/centered_nonzero_gap_frac": (
                centered_nonzero_gap_tokens.sum() / valid_count
            ).float().item(),
            "token_credit/eligible_gap_abs_mean": eligible_gap_abs_mean.item(),
            "token_credit/raw_gap_abs_mean": raw_gap_abs_mean.item(),
            "token_credit/gap_abs_mean": centered_gap_abs_mean.item(),
            "token_credit/centered_gap_abs_mean": centered_gap_abs_mean.item(),
            "token_credit/relative_gap_abs_mean": relative_gap_abs_mean.item(),
            "token_credit/trajectory_center_abs_mean": center_abs_mean.item(),
            "token_credit/scale": scale_value.item(),
            "token_credit/scale_min": scale_min.item(),
            "token_credit/scale_max": scale_max.item(),
            "token_credit/alignment_frac": aligned_fraction.item(),
            "token_credit/weight_std": weight_std.item(),
            "token_credit/weight_min": weight_min.item(),
            "token_credit/weight_max": weight_max.item(),
        }
        weighted_advantages = advantages * token_weights

    return weighted_advantages, token_weights, metrics


def apply_outcome_contrastive_token_credit(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    js_divergence: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    scale_quantile: float = 0.75,
    min_scale: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Gate GRPO token advantages by outcome-conditioned full-vocabulary JS.

    For eligible token ``t`` in trajectory ``i``, the current actor is evaluated
    under a verified-correct and a verified-incorrect reference trajectory. If
    their next-token distributions have Jensen-Shannon divergence ``C_it``, the
    detached GRPO multiplier is ``C_it / (C_it + c)`` where ``c`` is a robust
    batch quantile. Ineligible trajectories keep multiplier one and therefore
    exactly recover ordinary GRPO.

    Unlike the legacy polarity-gap mode, this rule intentionally does not
    normalize weights to mean one: it follows SC-GRPO's information gate and
    suppresses tokens whose policy distribution is insensitive to the outcome
    contrast.
    """
    if response_mask.shape != advantages.shape:
        raise ValueError(
            f"response_mask must have shape {tuple(advantages.shape)}, got {tuple(response_mask.shape)}"
        )
    if js_divergence.shape != advantages.shape:
        raise ValueError(
            f"js_divergence must have shape {tuple(advantages.shape)}, got {tuple(js_divergence.shape)}"
        )
    if not 0.0 < scale_quantile <= 1.0:
        raise ValueError(f"scale_quantile must be in (0, 1], got {scale_quantile}")
    if min_scale <= 0.0:
        raise ValueError(f"min_scale must be positive, got {min_scale}")
    if active_mask.ndim == 1:
        if active_mask.shape[0] != advantages.shape[0]:
            raise ValueError(
                f"active_mask length {active_mask.shape[0]} does not match batch size {advantages.shape[0]}"
            )
        active_mask = active_mask.unsqueeze(-1).expand_as(advantages)
    elif active_mask.shape != advantages.shape:
        raise ValueError(
            f"active_mask must have shape [batch] or {tuple(advantages.shape)}, got {tuple(active_mask.shape)}"
        )

    with torch.no_grad():
        valid_tokens = response_mask.bool()
        finite_js = torch.isfinite(js_divergence)
        active_tokens = valid_tokens & active_mask.bool() & finite_js
        token_weights = torch.ones_like(advantages)
        active_count = active_tokens.sum()

        if active_count.item() > 0:
            active_js = js_divergence.detach()[active_tokens].float().clamp_min(0.0)
            scale = torch.quantile(active_js, scale_quantile).clamp_min(min_scale)
            raw_weights = active_js / (active_js + scale)
            token_weights[active_tokens] = raw_weights.to(token_weights.dtype)
            js_mean = active_js.mean()
            js_std = active_js.std(unbiased=False)
            weight_mean = raw_weights.mean()
            weight_std = raw_weights.std(unbiased=False)
            weight_min = raw_weights.min()
            weight_max = raw_weights.max()
        else:
            scale = torch.zeros((), device=advantages.device)
            js_mean = torch.zeros((), device=advantages.device)
            js_std = torch.zeros((), device=advantages.device)
            weight_mean = torch.ones((), device=advantages.device)
            weight_std = torch.zeros((), device=advantages.device)
            weight_min = torch.ones((), device=advantages.device)
            weight_max = torch.ones((), device=advantages.device)

        valid_count = valid_tokens.sum().clamp(min=1)
        valid_trajectory_count = valid_tokens.any(dim=-1).sum().clamp(min=1)
        active_trajectory_count = active_tokens.any(dim=-1).sum()
        weighted_advantages = advantages * token_weights
        metrics = {
            "token_credit/oc_active_frac": (active_count / valid_count).float().item(),
            "token_credit/oc_active_trajectory_frac": (
                active_trajectory_count / valid_trajectory_count
            ).float().item(),
            "token_credit/oc_js_mean": js_mean.item(),
            "token_credit/oc_js_std": js_std.item(),
            "token_credit/oc_scale": scale.item(),
            "token_credit/weight_mean": weight_mean.item(),
            "token_credit/weight_std": weight_std.item(),
            "token_credit/weight_min": weight_min.item(),
            "token_credit/weight_max": weight_max.item(),
        }

    return weighted_advantages, token_weights, metrics


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = False,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_objective_abs = agg_loss(
        loss_mat=pg_losses.abs(), loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        # Unlike the signed loss, this scale does not vanish when centered
        # positive and negative advantages cancel.
        "actor/pg_objective_abs": pg_objective_abs.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics
