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

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    apply_additive_logprob_gap_credit,
    apply_centered_opd_token_credit,
    apply_opsd_reverse_kl_advantage,
    apply_rank_attenuated_token_credit,
    apply_reward_polarized_token_credit,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)


def test_opsd_reverse_kl_uses_raw_gap_and_zeroes_ineligible_tokens():
    teacher = torch.tensor([[0.3, -0.2, 7.0], [0.4, 0.5, 0.6]])
    student = torch.tensor([[0.1, -0.5, 2.0], [0.1, 0.1, 0.1]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    eligible = torch.tensor([True, False])
    grpo = torch.full_like(teacher, 3.0)

    advantages, metrics = apply_opsd_reverse_kl_advantage(
        response_mask=response_mask,
        teacher_log_probs=teacher,
        student_log_probs=student,
        eligibility_mask=eligible,
        grpo_advantages=grpo,
    )

    torch.testing.assert_close(advantages, torch.tensor([[0.2, 0.3, 0.0], [0.0, 0.0, 0.0]]))
    assert metrics["opsd/objective_is_pure"] == 1.0
    assert metrics["opsd/eligible_trajectory_frac"] == pytest.approx(0.5)
    assert metrics["opsd/replaced_grpo_advantage_abs_mean"] == pytest.approx(3.0)


def test_opsd_reverse_kl_masks_nonfinite_gaps_instead_of_using_grpo():
    teacher = torch.tensor([[float("nan"), 0.0]])
    student = torch.tensor([[0.0, -0.25]])
    response_mask = torch.ones_like(teacher)

    advantages, metrics = apply_opsd_reverse_kl_advantage(
        response_mask=response_mask,
        teacher_log_probs=teacher,
        student_log_probs=student,
        grpo_advantages=torch.full_like(teacher, -4.0),
    )

    torch.testing.assert_close(advantages, torch.tensor([[0.0, 0.25]]))
    assert metrics["opsd/finite_gap_frac"] == pytest.approx(0.5)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_token_credit_alpha_zero_exactly_recovers_grpo():
    advantages = torch.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    teacher_log_probs = torch.tensor([[-0.1, -2.0, 0.0], [-3.0, -0.2, -1.0]])
    student_log_probs = torch.tensor([[-1.0, -0.3, 0.0], [-0.2, -2.0, -1.5]])

    weighted_advantages, weights, metrics = apply_reward_polarized_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        alpha=0.0,
    )

    assert torch.equal(weighted_advantages, advantages)
    assert torch.equal(weights, torch.ones_like(weights))
    assert metrics["token_credit/weight_std"] == 0.0


def test_additive_gap_credit_is_direct_teacher_minus_student_update():
    advantages = torch.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    teacher_log_probs = torch.tensor([[-0.1, -2.0, 0.0], [-3.0, -0.2, -1.0]])
    student_log_probs = torch.tensor([[-1.0, -0.3, 0.0], [-0.2, -2.0, -1.5]])

    adjusted, adjustments, metrics = apply_additive_logprob_gap_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        coefficient=0.01,
    )

    expected_adjustments = 0.01 * (teacher_log_probs - student_log_probs) * response_mask
    assert torch.allclose(adjustments, expected_adjustments)
    assert torch.allclose(adjusted, advantages + expected_adjustments)
    assert metrics["token_credit/additive_coefficient"] == pytest.approx(0.01)
    assert metrics["token_credit/raw_gap_mean"] == pytest.approx(-0.26)
    assert metrics["token_credit/raw_gap_abs_mean"] == pytest.approx(1.54)
    assert metrics["token_credit/effective_gap_mean"] == pytest.approx(-0.26)
    assert metrics["token_credit/effective_gap_abs_mean"] == pytest.approx(1.54)
    assert metrics["token_credit/grpo_advantage_mean"] == pytest.approx(-0.2)
    assert metrics["token_credit/grpo_advantage_abs_mean"] == pytest.approx(1.0)
    assert metrics["token_credit/adjusted_advantage_mean"] == pytest.approx(-0.2026)
    assert metrics["token_credit/advantage_delta_mean"] == pytest.approx(-0.0026)
    assert metrics["token_credit/advantage_delta_abs_mean"] == pytest.approx(0.0154)
    assert metrics["token_credit/adjustment_to_grpo_abs_ratio"] == pytest.approx(0.0154)
    assert metrics["token_credit/adjustment_to_grpo_abs_pct"] == pytest.approx(1.54)
    assert metrics["token_credit/adjustment_to_active_grpo_abs_ratio"] == pytest.approx(0.0154)
    assert metrics["token_credit/adjustment_to_active_grpo_abs_pct"] == pytest.approx(1.54)
    assert metrics["token_credit/adjustment_abs_mean"] > 0.0


def test_additive_gap_credit_respects_eligibility_and_can_activate_zero_grpo_advantage():
    advantages = torch.zeros(3, 2)
    response_mask = torch.ones_like(advantages)
    teacher_log_probs = torch.tensor([[-0.1, -2.0], [-0.1, -2.0], [-2.0, -0.1]])
    student_log_probs = torch.tensor([[-2.0, -0.1], [-2.0, -0.1], [-0.1, -2.0]])

    adjusted, adjustments, metrics = apply_additive_logprob_gap_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        coefficient=0.01,
        eligibility_mask=torch.tensor([1, 0, 1]),
    )

    assert torch.count_nonzero(adjusted[0]) == 2
    assert torch.equal(adjusted[1], advantages[1])
    assert torch.count_nonzero(adjusted[2]) == 2
    assert torch.equal(adjustments[1], torch.zeros(2))
    assert metrics["token_credit/eligible_trajectory_frac"] == pytest.approx(2 / 3)
    assert metrics["token_credit/zero_advantage_activated_frac"] == pytest.approx(2 / 3)
    assert metrics["token_credit/grpo_advantage_abs_mean"] == 0.0
    assert metrics["token_credit/adjustment_to_grpo_abs_ratio"] == 0.0
    assert metrics["token_credit/adjustment_to_grpo_abs_pct"] == 0.0


def test_additive_gap_credit_coefficient_zero_exactly_recovers_grpo():
    advantages = torch.tensor([[1.0, -1.0]])
    response_mask = torch.ones_like(advantages)
    teacher_log_probs = torch.tensor([[-0.1, -2.0]])
    student_log_probs = torch.tensor([[-2.0, -0.1]])

    adjusted, adjustments, metrics = apply_additive_logprob_gap_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        coefficient=0.0,
    )

    assert torch.equal(adjusted, advantages)
    assert torch.equal(adjustments, torch.zeros_like(adjustments))
    assert metrics["token_credit/nonzero_adjustment_frac"] == 0.0


def test_additive_gap_credit_reports_outcome_alignment():
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    response_mask = torch.ones_like(advantages)
    student_log_probs = torch.tensor([[-2.0, -2.0], [-1.0, -1.0]])
    rubric_actor_log_probs = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])

    _, _, metrics = apply_additive_logprob_gap_credit(
        advantages,
        response_mask,
        rubric_actor_log_probs,
        student_log_probs,
        coefficient=0.01,
        trajectory_scores=torch.tensor([1.0, 0.0]),
        success_threshold=0.5,
        reference_log_probs=student_log_probs + 0.5,
    )

    assert metrics["token_credit/rubric_gap_correct_mean"] == pytest.approx(1.0)
    assert metrics["token_credit/rubric_gap_wrong_mean"] == pytest.approx(-1.0)
    assert metrics["token_credit/rubric_gap_outcome_margin"] == pytest.approx(2.0)
    assert metrics["token_credit/rubric_gap_reward_correlation"] == pytest.approx(1.0)
    assert metrics["token_credit/rubric_gap_correct_trajectory_count"] == 1.0
    assert metrics["token_credit/rubric_gap_wrong_trajectory_count"] == 1.0
    assert metrics["token_credit/rubric_gap_scored_trajectory_frac"] == 1.0
    assert metrics["token_credit/reference_drift_available"] == 1.0
    assert metrics["token_credit/reference_drift_mean"] == pytest.approx(0.5)
    assert metrics["token_credit/reference_drift_abs_mean"] == pytest.approx(0.5)
    assert metrics["token_credit/rubric_to_reference_drift_abs_ratio"] == pytest.approx(2.0)


@pytest.mark.parametrize("coefficient", [-0.01, float("nan"), float("inf")])
def test_additive_gap_credit_rejects_invalid_coefficient(coefficient):
    tensor = torch.ones(2, 3)
    with pytest.raises(ValueError, match="coefficient"):
        apply_additive_logprob_gap_credit(
            tensor,
            tensor,
            tensor,
            tensor,
            coefficient=coefficient,
        )


def test_centered_opd_preserves_trajectory_advantage_after_clipping():
    advantages = torch.tensor(
        [[-1.0, -1.0, -1.0], [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
    )
    response_mask = torch.ones_like(advantages)
    offsets = torch.tensor([[-3.0], [-1.0], [1.0], [3.0]])
    teacher_log_probs = offsets + torch.tensor([[-4.0, 2.0, 2.0]])
    student_log_probs = torch.zeros_like(teacher_log_probs)

    adjusted, weights, metrics = apply_centered_opd_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        trajectory_scores=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        group_ids=np.array(["task"] * 4, dtype=object),
        coefficient=1.0,
        clip_value=0.75,
        multiplier_min=0.5,
        multiplier_max=1.5,
    )

    assert torch.allclose(weights.mean(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(adjusted.mean(dim=-1), advantages.mean(dim=-1), atol=1e-6)
    assert weights.min() >= 0.5
    assert weights.max() <= 1.5
    assert weights[0, 0] > weights[0, 1]  # sign(A)<0 reverses the normalized-gap ordering.
    assert weights[3, 0] < weights[3, 1]  # sign(A)>0 keeps the normalized-gap ordering.
    assert torch.all(adjusted[:2] < 0)
    assert torch.all(adjusted[2:] > 0)
    assert metrics["token_credit/group_gate_mean"] == pytest.approx(1.0)
    assert metrics["token_credit/clipfrac"] > 0
    assert metrics["token_credit/trajectory_weight_mean_error_max"] < 1e-6
    assert metrics["token_credit/trajectory_advantage_mean_error_max"] < 1e-6


def test_centered_opd_bounded_projection_preserves_full_masked_mean_with_ineligible_tokens():
    advantages = torch.tensor(
        [[-1.0, -1.0, -1.0, 0.0], [-0.5, -0.5, -0.5, 0.0], [0.5, 0.5, 0.5, 0.0], [1.0, 1.0, 1.0, 0.0]]
    )
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]] * 4)
    offsets = torch.tensor([[-3.0], [-1.0], [1.0], [3.0]])
    teacher_log_probs = offsets + torch.tensor([[-4.0, 2.0, 2.0, 99.0]])

    adjusted, weights, metrics = apply_centered_opd_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        torch.zeros_like(teacher_log_probs),
        trajectory_scores=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        group_ids=["task"] * 4,
        coefficient=1.0,
        clip_value=0.75,
        multiplier_min=0.5,
        multiplier_max=1.5,
        eligibility_mask=torch.tensor([1, 1, 0, 1]),
    )

    masked_weight_mean = (weights * response_mask).sum(-1) / response_mask.sum(-1)
    masked_advantage_mean = (adjusted * response_mask).sum(-1) / response_mask.sum(-1)
    original_advantage_mean = (advantages * response_mask).sum(-1) / response_mask.sum(-1)
    assert torch.allclose(masked_weight_mean, torch.ones(4), atol=1e-6)
    assert torch.allclose(masked_advantage_mean, original_advantage_mean, atol=1e-6)
    assert torch.equal(weights[2], torch.ones(4))
    assert torch.equal(weights[:, -1], torch.ones(4))
    assert metrics["token_credit/trajectory_weight_mean_error_max"] < 1e-6


def test_centered_opd_disables_redistribution_for_reversed_group_signal():
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]])
    response_mask = torch.ones_like(advantages)
    teacher_log_probs = torch.tensor([[4.0, 2.0], [3.0, 1.0], [2.0, 0.0], [1.0, -1.0]])
    student_log_probs = torch.zeros_like(teacher_log_probs)

    adjusted, weights, metrics = apply_centered_opd_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        trajectory_scores=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        group_ids=["task"] * 4,
    )

    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(adjusted, advantages)
    assert metrics["token_credit/group_gate_mean"] == 0.0
    assert metrics["token_credit/group_gate_active_frac"] == 0.0


def test_centered_opd_zero_coefficient_and_ineligible_rows_recover_grpo():
    advantages = torch.tensor([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]])
    response_mask = torch.ones_like(advantages)
    teacher_log_probs = torch.tensor([[-2.0, 0.0], [-1.0, 1.0], [0.0, 2.0]])
    student_log_probs = torch.zeros_like(teacher_log_probs)

    adjusted, weights, _ = apply_centered_opd_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        trajectory_scores=torch.tensor([0.0, 0.5, 1.0]),
        group_ids=["task"] * 3,
        coefficient=0.0,
        eligibility_mask=torch.tensor([1, 0, 1]),
    )

    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(adjusted, advantages)


def test_token_credit_respects_reward_polarity_and_preserves_mean():
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    response_mask = torch.ones_like(advantages)
    # For the positive trajectory, token 0 is teacher-preferred. For the
    # negative trajectory, token 0 is teacher-disfavoured.
    teacher_log_probs = torch.tensor([[-0.1, -2.0], [-2.0, -0.1]])
    student_log_probs = torch.tensor([[-1.1, -1.0], [-1.0, -1.1]])

    weighted_advantages, weights, _ = apply_reward_polarized_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        alpha=0.5,
    )

    assert weights[0, 0] > weights[0, 1]
    assert weights[1, 0] > weights[1, 1]
    assert torch.allclose(weights.mean(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(weighted_advantages[0] > 0)
    assert torch.all(weighted_advantages[1] < 0)


def test_rank_attenuation_uses_outcome_polarity_without_amplifying_credit():
    advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])
    response_mask = torch.ones_like(advantages)
    student_log_probs = torch.zeros_like(advantages)
    teacher_log_probs = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])

    adjusted, weights, metrics = apply_rank_attenuated_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        attenuation=0.2,
        conflict_fraction=0.25,
        min_gap_range=1e-4,
    )

    assert torch.equal(weights[0], torch.tensor([0.8, 1.0, 1.0, 1.0]))
    assert torch.equal(weights[1], torch.tensor([1.0, 1.0, 1.0, 0.8]))
    assert torch.all(weights <= 1.0)
    assert torch.all(weights >= 0.8)
    assert torch.all(adjusted[0] > 0)
    assert torch.all(adjusted[1] < 0)
    assert adjusted[0, 0] == pytest.approx(0.8)
    assert adjusted[1, 3] == pytest.approx(-0.8)
    assert metrics["token_credit/attenuated_active_frac"] == pytest.approx(0.25)
    assert metrics["token_credit/trajectory_advantage_scale_mean"] == pytest.approx(0.95)
    assert metrics["token_credit/advantage_sign_flip_frac"] == 0.0


def test_rank_attenuation_respects_mask_eligibility_zero_advantage_and_no_spread():
    advantages = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    teacher_log_probs = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 999.0],
        ]
    )
    eligibility = torch.tensor([1, 1, 0, 1])

    adjusted, weights, metrics = apply_rank_attenuated_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        torch.zeros_like(teacher_log_probs),
        attenuation=0.2,
        conflict_fraction=0.5,
        min_gap_range=1e-4,
        eligibility_mask=eligibility,
    )

    assert torch.equal(weights[0], torch.ones(4))  # Uniform gap is not ranked.
    assert torch.equal(weights[1], torch.ones(4))  # Zero GRPO advantage stays zero.
    assert torch.equal(weights[2], torch.ones(4))  # Ineligible trajectory stays GRPO.
    assert torch.equal(weights[3], torch.tensor([0.8, 1.0, 1.0, 1.0]))
    assert torch.equal(adjusted[1], advantages[1])
    assert torch.equal(adjusted[2], advantages[2])
    assert adjusted[3, 3] == 0.0
    assert metrics["token_credit/no_spread_trajectory_frac"] == pytest.approx(0.25)
    assert metrics["token_credit/advantage_sign_flip_frac"] == 0.0


def test_rank_attenuation_does_not_split_tied_gap_ranks_to_fill_budget():
    advantages = torch.ones(1, 4)
    response_mask = torch.ones_like(advantages)
    teacher_log_probs = torch.tensor([[0.0, 0.0, 1.0, 2.0]])

    adjusted, weights, metrics = apply_rank_attenuated_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        torch.zeros_like(teacher_log_probs),
        attenuation=0.2,
        conflict_fraction=0.25,
    )

    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(adjusted, advantages)
    assert metrics["token_credit/attenuated_frac"] == 0.0


def test_rank_attenuation_zero_strength_exactly_recovers_grpo():
    advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    response_mask = torch.ones_like(advantages)
    teacher_log_probs = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    adjusted, weights, metrics = apply_rank_attenuated_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        torch.zeros_like(teacher_log_probs),
        attenuation=0.0,
        conflict_fraction=0.25,
    )

    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(adjusted, advantages)
    assert metrics["token_credit/conflict_active_frac"] == pytest.approx(0.25)
    assert metrics["token_credit/attenuated_active_frac"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attenuation": 1.0}, "attenuation"),
        ({"conflict_fraction": 0.0}, "conflict_fraction"),
        ({"min_gap_range": -1.0}, "min_gap_range"),
    ],
)
def test_rank_attenuation_rejects_invalid_hyperparameters(kwargs, message):
    tensor = torch.ones(2, 3)
    with pytest.raises(ValueError, match=message):
        apply_rank_attenuated_token_credit(tensor, tensor, tensor, tensor, **kwargs)


def test_token_credit_is_invariant_to_per_trajectory_logprob_offsets():
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.ones_like(advantages)
    teacher_log_probs = torch.tensor([[-0.2, -1.0, -2.0], [-2.0, -1.0, -0.2]])
    student_log_probs = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

    _, weights, _ = apply_reward_polarized_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        alpha=0.5,
    )
    _, shifted_weights, _ = apply_reward_polarized_token_credit(
        advantages,
        response_mask,
        teacher_log_probs + torch.tensor([[7.0], [-4.0]]),
        student_log_probs,
        alpha=0.5,
    )

    assert torch.allclose(weights, shifted_weights, atol=1e-6)


def test_token_credit_uniform_gap_has_no_token_level_signal():
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.ones_like(advantages)
    student_log_probs = torch.tensor([[-2.0, -1.5, -0.5], [-1.0, -2.0, -3.0]])
    teacher_log_probs = student_log_probs + torch.tensor([[3.0], [-2.0]])

    weighted_advantages, weights, metrics = apply_reward_polarized_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        alpha=0.5,
    )

    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)
    assert torch.allclose(weighted_advantages, advantages, atol=1e-6)
    assert metrics["token_credit/raw_gap_abs_mean"] > 0
    assert metrics["token_credit/centered_gap_abs_mean"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["token_credit/weight_std"] == pytest.approx(0.0, abs=1e-6)


def test_token_credit_zero_advantage_and_ineligible_trajectories_remain_grpo():
    advantages = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    response_mask = torch.ones_like(advantages)
    teacher_log_probs = torch.tensor([[-0.1, -2.0], [-0.1, -2.0], [-2.0, -0.1]])
    student_log_probs = torch.tensor([[-2.0, -0.1], [-2.0, -0.1], [-0.1, -2.0]])
    eligibility_mask = torch.tensor([1, 0, 1])

    weighted_advantages, weights, metrics = apply_reward_polarized_token_credit(
        advantages,
        response_mask,
        teacher_log_probs,
        student_log_probs,
        eligibility_mask=eligibility_mask,
    )

    assert torch.equal(weighted_advantages[0], advantages[0])
    assert torch.equal(weighted_advantages[1], advantages[1])
    assert torch.equal(weights[0], torch.ones(2))
    assert torch.equal(weights[1], torch.ones(2))
    assert metrics["token_credit/active_frac"] == pytest.approx(1 / 3)
    assert metrics["token_credit/active_trajectory_frac"] == pytest.approx(1 / 3)
    assert metrics["token_credit/eligible_frac"] == pytest.approx(2 / 3)
    assert metrics["token_credit/eligible_trajectory_frac"] == pytest.approx(2 / 3)
    assert metrics["token_credit/nonzero_advantage_frac"] == pytest.approx(2 / 3)
    assert metrics["token_credit/nonzero_advantage_trajectory_frac"] == pytest.approx(2 / 3)
    assert metrics["token_credit/finite_gap_frac"] == pytest.approx(1.0)
    assert metrics["token_credit/nonzero_gap_frac"] == pytest.approx(2 / 3)
    assert metrics["token_credit/eligible_gap_abs_mean"] == pytest.approx(1.9)


def test_token_credit_diagnostics_distinguish_ineligible_from_zero_advantage():
    response_mask = torch.ones(2, 2)
    teacher_log_probs = torch.tensor([[-0.1, -2.0], [-0.1, -2.0]])
    student_log_probs = torch.tensor([[-2.0, -0.1], [-2.0, -0.1]])

    _, ineligible_weights, ineligible_metrics = apply_reward_polarized_token_credit(
        torch.ones(2, 2),
        response_mask,
        teacher_log_probs,
        student_log_probs,
        eligibility_mask=torch.zeros(2),
    )
    assert torch.equal(ineligible_weights, torch.ones_like(ineligible_weights))
    assert ineligible_metrics["token_credit/eligible_trajectory_frac"] == 0.0
    assert ineligible_metrics["token_credit/nonzero_advantage_trajectory_frac"] == 1.0
    assert ineligible_metrics["token_credit/active_trajectory_frac"] == 0.0

    _, zero_advantage_weights, zero_advantage_metrics = apply_reward_polarized_token_credit(
        torch.zeros(2, 2),
        response_mask,
        teacher_log_probs,
        student_log_probs,
        eligibility_mask=torch.ones(2),
    )
    assert torch.equal(zero_advantage_weights, torch.ones_like(zero_advantage_weights))
    assert zero_advantage_metrics["token_credit/eligible_trajectory_frac"] == 1.0
    assert zero_advantage_metrics["token_credit/nonzero_advantage_trajectory_frac"] == 0.0
    assert zero_advantage_metrics["token_credit/active_trajectory_frac"] == 0.0
    assert zero_advantage_metrics["token_credit/eligible_gap_abs_mean"] == pytest.approx(1.9)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 1.0}, "alpha"),
        ({"scale_quantile": 0.0}, "scale_quantile"),
        ({"min_scale": 0.0}, "min_scale"),
    ],
)
def test_token_credit_rejects_invalid_hyperparameters(kwargs, message):
    tensor = torch.ones(2, 3)
    with pytest.raises(ValueError, match=message):
        apply_reward_polarized_token_credit(tensor, tensor, tensor, tensor, **kwargs)


if __name__ == "__main__":
    unittest.main()
