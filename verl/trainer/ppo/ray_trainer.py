# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.turn_context import TURN_REPLAY_KEY
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    canonicalize_validation_data_source,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
    validation_data_source_rollup,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        grpo_advantages = advantages.clone()
        token_credit_config = config.get("token_credit", None) if config is not None else None
        if token_credit_config is not None and token_credit_config.get("enable", False):
            mode = token_credit_config.get("mode", "additive_gap")
            if mode in (
                "additive_gap",
                "centered_opd",
                "rank_attenuation",
                "polarity_gap",
                "opsd_reverse_kl",
            ):
                teacher_key = token_credit_config.get("teacher_log_prob_key", "ref_log_prob")
                student_key = token_credit_config.get("student_log_prob_key", "old_log_probs")
                missing_keys = [key for key in (teacher_key, student_key) if key not in data.batch]
                if missing_keys:
                    raise KeyError(
                        "Token credit requires sampled-token log probabilities in the training batch; "
                        f"missing {missing_keys}. Available keys: {list(data.batch.keys())}"
                    )

                active_mask_key = token_credit_config.get("active_mask_key", None)
                eligibility_mask = None
                if active_mask_key is not None:
                    if active_mask_key not in data.batch:
                        raise KeyError(
                            f"Token credit active_mask_key={active_mask_key!r} is absent from the batch. "
                            f"Available keys: {list(data.batch.keys())}"
                        )
                    eligibility_mask = data.batch[active_mask_key]

                if mode == "opsd_reverse_kl":
                    advantages, token_credit_metrics = core_algos.apply_opsd_reverse_kl_advantage(
                        response_mask=grpo_calculation_mask,
                        teacher_log_probs=data.batch[teacher_key],
                        student_log_probs=data.batch[student_key],
                        eligibility_mask=eligibility_mask,
                        grpo_advantages=grpo_advantages,
                    )
                    # Pure OPSD does not use environment returns in the actor
                    # objective. Keep reward-derived GRPO values only in the
                    # dedicated diagnostic tensor below.
                    returns = advantages.clone()
                elif mode in ("additive_gap", "centered_opd"):
                    # GRPO itself is still computed from token_level_rewards
                    # above.  Rubric diagnostics, however, should use the
                    # benchmark outcome configured for rubric generation and
                    # gating. Current launchers use strict success for ALFWorld
                    # and the continuous training reward for WebShop.
                    trajectory_scores = (
                        data.batch["rubric_outcome_scores"]
                        if "rubric_outcome_scores" in data.batch
                        else data.batch["token_level_rewards"].sum(dim=-1)
                    )
                    if mode == "additive_gap":
                        base_coefficient = float(token_credit_config.get("alpha", 0.01))
                        schedule_scale = float(data.meta_info.get("token_credit_schedule_scale", 1.0))
                        schedule_progress = float(data.meta_info.get("token_credit_training_progress", 0.0))
                        decay_enable = token_credit_config.get("linear_decay_enable", False)
                        warmup_enable = token_credit_config.get("linear_warmup_enable", False)
                        if not np.isfinite(schedule_scale) or not 0 <= schedule_scale <= 1:
                            raise ValueError("token_credit_schedule_scale must be in [0, 1]")
                        if not np.isfinite(schedule_progress) or not 0 <= schedule_progress <= 1:
                            raise ValueError("token_credit_training_progress must be in [0, 1]")
                        advantages, _, token_credit_metrics = core_algos.apply_additive_logprob_gap_credit(
                            advantages=advantages,
                            response_mask=grpo_calculation_mask,
                            teacher_log_probs=data.batch[teacher_key],
                            student_log_probs=data.batch[student_key],
                            coefficient=base_coefficient * schedule_scale,
                            eligibility_mask=eligibility_mask,
                            trajectory_scores=trajectory_scores,
                            success_threshold=token_credit_config.get("outcome_success_threshold", 0.5),
                            reference_log_probs=(
                                data.batch["ref_log_prob"] if "ref_log_prob" in data.batch else None
                            ),
                        )
                        token_credit_metrics.update(
                            {
                                "token_credit/additive_base_coefficient": base_coefficient,
                                "token_credit/schedule_scale": schedule_scale,
                                "token_credit/schedule_progress": schedule_progress,
                                "token_credit/decay_progress": schedule_progress if decay_enable else 0.0,
                                "token_credit/warmup_progress": schedule_progress if warmup_enable else 0.0,
                                "token_credit/decay_start_step": float(
                                    token_credit_config.get("linear_decay_start_step", 0)
                                ),
                                "token_credit/warmup_steps": float(
                                    token_credit_config.get("linear_warmup_steps", 0)
                                ),
                            }
                        )
                    else:
                        base_coefficient = float(token_credit_config.get("alpha", 0.1))
                        decay_enable = token_credit_config.get("linear_decay_enable", False)
                        warmup_enable = token_credit_config.get("linear_warmup_enable", False)
                        progress = float(data.meta_info.get("token_credit_training_progress", 0.0))
                        schedule_scale = float(data.meta_info.get("token_credit_schedule_scale", 1.0))
                        if not isinstance(decay_enable, bool) or not isinstance(warmup_enable, bool):
                            raise ValueError("token_credit schedule enable flags must be boolean")
                        if decay_enable and warmup_enable:
                            raise ValueError("token_credit linear decay and linear warmup are mutually exclusive")
                        if not np.isfinite(progress) or not 0 <= progress <= 1:
                            raise ValueError("token_credit_training_progress must be in [0, 1]")
                        if not np.isfinite(schedule_scale) or not 0 <= schedule_scale <= 1:
                            raise ValueError("token_credit_schedule_scale must be in [0, 1]")
                        advantages, token_credit_weights, token_credit_metrics = (
                            core_algos.apply_centered_opd_token_credit(
                                advantages=advantages,
                                response_mask=grpo_calculation_mask,
                                teacher_log_probs=data.batch[teacher_key],
                                student_log_probs=data.batch[student_key],
                                trajectory_scores=trajectory_scores,
                                group_ids=data.non_tensor_batch["uid"],
                                coefficient=base_coefficient * schedule_scale,
                                clip_value=token_credit_config.get("centered_clip", 2.0),
                                multiplier_min=token_credit_config.get("centered_multiplier_min", 0.5),
                                multiplier_max=token_credit_config.get("centered_multiplier_max", 1.5),
                                eligibility_mask=eligibility_mask,
                            )
                        )
                        data.batch["token_credit_weights"] = token_credit_weights
                        data.batch["token_credit_raw_gap"] = (
                            data.batch[teacher_key] - data.batch[student_key]
                        ) * grpo_calculation_mask
                        token_credit_metrics.update(
                            {
                                "token_credit/centered_opd_base_coefficient": base_coefficient,
                                "token_credit/schedule_scale": schedule_scale,
                                "token_credit/schedule_progress": progress,
                                "token_credit/decay_progress": progress if decay_enable else 0.0,
                                "token_credit/warmup_progress": progress if warmup_enable else 0.0,
                                "token_credit/decay_start_step": float(
                                    token_credit_config.get("linear_decay_start_step", 0)
                                ),
                                "token_credit/warmup_steps": float(
                                    token_credit_config.get("linear_warmup_steps", 0)
                                ),
                            }
                        )
                elif mode == "rank_attenuation":
                    base_attenuation = float(token_credit_config.get("rank_attenuation_strength", 0.2))
                    decay_enable = token_credit_config.get("linear_decay_enable", False)
                    warmup_enable = token_credit_config.get("linear_warmup_enable", False)
                    progress = float(data.meta_info.get("token_credit_training_progress", 0.0))
                    schedule_scale = float(data.meta_info.get("token_credit_schedule_scale", 1.0))
                    if not np.isfinite(progress) or not 0 <= progress <= 1:
                        raise ValueError("token_credit_training_progress must be in [0, 1]")
                    if not np.isfinite(schedule_scale) or not 0 <= schedule_scale <= 1:
                        raise ValueError("token_credit_schedule_scale must be in [0, 1]")
                    advantages, token_credit_weights, token_credit_metrics = (
                        core_algos.apply_rank_attenuated_token_credit(
                            advantages=advantages,
                            response_mask=grpo_calculation_mask,
                            teacher_log_probs=data.batch[teacher_key],
                            student_log_probs=data.batch[student_key],
                            attenuation=base_attenuation * schedule_scale,
                            conflict_fraction=token_credit_config.get("rank_conflict_fraction", 0.25),
                            min_gap_range=token_credit_config.get("rank_min_gap_range", 1e-4),
                            eligibility_mask=eligibility_mask,
                        )
                    )
                    data.batch["token_credit_weights"] = token_credit_weights
                    data.batch["token_credit_raw_gap"] = (
                        data.batch[teacher_key] - data.batch[student_key]
                    ) * grpo_calculation_mask
                    token_credit_metrics.update(
                        {
                            "token_credit/rank_attenuation_base": base_attenuation,
                            "token_credit/schedule_scale": schedule_scale,
                            "token_credit/schedule_progress": progress,
                            "token_credit/decay_progress": progress if decay_enable else 0.0,
                            "token_credit/warmup_progress": progress if warmup_enable else 0.0,
                            "token_credit/decay_start_step": float(
                                token_credit_config.get("linear_decay_start_step", 0)
                            ),
                            "token_credit/warmup_steps": float(
                                token_credit_config.get("linear_warmup_steps", 0)
                            ),
                        }
                    )
                else:
                    advantages, _, token_credit_metrics = core_algos.apply_reward_polarized_token_credit(
                        advantages=advantages,
                        response_mask=grpo_calculation_mask,
                        teacher_log_probs=data.batch[teacher_key],
                        student_log_probs=data.batch[student_key],
                        alpha=token_credit_config.get("alpha", 0.1),
                        scale_quantile=token_credit_config.get("scale_quantile", 0.5),
                        min_scale=token_credit_config.get("min_scale", 1e-4),
                        eligibility_mask=eligibility_mask,
                        preserve_trajectory_mean=token_credit_config.get("preserve_trajectory_mean", True),
                    )
            elif mode == "outcome_contrastive_js":
                js_key = token_credit_config.get("outcome_js_key", "outcome_contrastive_js")
                active_mask_key = token_credit_config.get(
                    "outcome_active_mask_key", "outcome_contrastive_active"
                )
                missing_keys = [key for key in (js_key, active_mask_key) if key not in data.batch]
                if missing_keys:
                    raise KeyError(
                        "Outcome-contrastive token credit was enabled but its actor outputs are missing: "
                        f"{missing_keys}. Available keys: {list(data.batch.keys())}"
                    )
                advantages, _, token_credit_metrics = core_algos.apply_outcome_contrastive_token_credit(
                    advantages=advantages,
                    response_mask=grpo_calculation_mask,
                    js_divergence=data.batch[js_key],
                    active_mask=data.batch[active_mask_key],
                    scale_quantile=token_credit_config.get("scale_quantile", 0.75),
                    min_scale=token_credit_config.get("min_scale", 1e-4),
                )
            else:
                raise ValueError(f"Unsupported token credit mode: {mode}")
            data.meta_info["token_credit_metrics"] = token_credit_metrics
        data.batch["grpo_advantages"] = grpo_advantages
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        self._rubric_vllm_client = None

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        reward_for_val: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            reward_for_val: Whether this is for validation
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If reward_for_val=False and sum_reward=True: summed reward_tensor (1D tensor)
            Otherwise: tuple of (reward_tensor, reward_extra_infos_dict)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if not reward_for_val and sum_reward:
                return reward_tensor

            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            reward_extra_infos_dict = (
                {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
            )
            return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if reward_for_val:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_infos_dict = result.get("reward_extra_info", {})
            return reward_tensor, reward_extra_infos_dict
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        token_credit_config = self.config.algorithm.get("token_credit", None)
        rubric_config = token_credit_config.get("rubric_pipeline", None) if token_credit_config is not None else None
        if (
            rubric_config is not None
            and rubric_config.get("enable", False)
            and rubric_config.get("backend", "http") == "dataset"
        ):
            dataset_rubric_keys = {
                rubric_config.get("rubric_column", "rubric"),
                rubric_config.get("valid_column", "rubric_valid"),
            }
            # These columns are needed after rollout generation, when the
            # current old actor builds rubric-conditioned scoring inputs.
            reward_model_keys.update(dataset_rubric_keys & batch.non_tensor_batch.keys())

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        common_reward_extra_keys: set[str] | None = None

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            if self.config.get("credit_assignment", {}).get("type") == "attention_traceback":
                from verl.experimental.attention_traceback.capture import export_records
                export_records(test_output_gen_batch.non_tensor_batch["attention_traceback_record"],
                               self.config.attention_traceback.capture_dir)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            current_reward_extra_keys = set(reward_extra_info).difference({"reward"})
            if common_reward_extra_keys is None:
                common_reward_extra_keys = current_reward_extra_keys
            else:
                dropped_keys = common_reward_extra_keys.difference(current_reward_extra_keys)
                for key in dropped_keys:
                    reward_extra_infos_dict.pop(key, None)
                common_reward_extra_keys.intersection_update(current_reward_extra_keys)

            for key in common_reward_extra_keys:
                values = reward_extra_info[key]
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            batch_size = reward_tensor.shape[0]
            raw_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * batch_size)
            extra_infos = test_batch.non_tensor_batch.get("extra_info", [None] * batch_size)
            reference_answers = test_batch.non_tensor_batch.get("reference_answer_json", [None] * batch_size)
            normalized_data_sources = np.array(
                [
                    canonicalize_validation_data_source(
                        raw_data_sources[index],
                        extra_infos[index],
                        reference_answers[index],
                    )
                    for index in range(batch_size)
                ],
                dtype=object,
            )
            data_source_lst.append(normalized_data_sources)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        split_rollup_sources = np.array(
            [validation_data_source_rollup(data_source) for data_source in data_sources], dtype=object
        )
        if not np.array_equal(split_rollup_sources, np.asarray(data_sources, dtype=object)):
            split_rollups = process_validation_metrics(
                split_rollup_sources, sample_uids, reward_extra_infos_dict
            )
            for data_source, var2metric2val in split_rollups.items():
                if data_source not in data_src2var2metric2val:
                    data_src2var2metric2val[data_source] = var2metric2val
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            display_data_source = {
                "AIME24": "aime24",
                "AIME25": "aime25",
                "aime_2024": "aime24",
                "aime_2025": "aime25",
            }.get(data_source, data_source)
            core_vars = {"acc" if "acc" in var2metric2val else "reward"}
            if str(display_data_source).lower() == "webshop" and "success" in var2metric2val:
                core_vars.add("success")
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name in core_vars)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{display_data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            rm_resource_pool=rm_resource_pool,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self._turn_replay_enabled(batch):
            raise NotImplementedError(
                "history-window turn replay currently supports GRPO without a critic; "
                "PPO/GAE also requires turn-aware value replay"
            )
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        self._configure_turn_replay(batch)
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _turn_replay_enabled(self, batch: DataProto) -> bool:
        history_window = int(
            self.config.actor_rollout_ref.rollout.multi_turn.get("history_window", 0) or 0
        )
        if history_window <= 0:
            return False
        if TURN_REPLAY_KEY not in batch.non_tensor_batch:
            raise KeyError(
                "multi_turn.history_window is enabled but rollout output has no turn_replay manifest"
            )
        return True

    def _configure_turn_replay(self, batch: DataProto) -> bool:
        enabled = self._turn_replay_enabled(batch)
        batch.meta_info["turn_replay_enabled"] = enabled
        if not enabled:
            return False
        if self.use_legacy_worker_impl == "disable":
            raise NotImplementedError(
                "turn replay is not implemented for trainer.use_legacy_worker_impl=disable"
            )
        actor_strategy = str(self.config.actor_rollout_ref.actor.get("strategy", "fsdp"))
        if actor_strategy not in ("fsdp", "fsdp2"):
            raise NotImplementedError(
                f"turn replay currently requires the data-parallel FSDP actor, got {actor_strategy!r}"
            )
        batch.meta_info["turn_replay_micro_batch_size"] = int(
            self.config.actor_rollout_ref.rollout.multi_turn.get("turn_replay_micro_batch_size", 4)
        )
        return True

    def _compute_rubric_actor_log_prob(self, batch: DataProto) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Score the same response under a rubric-conditioned teacher context.

        Standard token-credit modes use the current pre-update actor. Pure OPSD
        may instead use VERL's frozen reference policy, matching the fixed
        teacher used by the released OPSD launchers while retaining identical
        online environment rollouts.
        """
        token_credit_config = self.config.algorithm.token_credit
        rubric_config = token_credit_config.rubric_pipeline
        if "raw_prompt" not in batch.non_tensor_batch:
            raise KeyError(
                "Rubric-conditioned token credit requires raw_prompt in the rollout batch. "
                f"Available non-tensor keys: {list(batch.non_tensor_batch.keys())}"
            )

        from verl.trainer.ppo.rubric_token_credit import (
            RubricVLLMClient,
            build_rubric_conditioned_inputs,
            build_rubric_conditioned_turn_replay,
            load_precomputed_rubrics,
        )

        raw_prompts = list(batch.non_tensor_batch["raw_prompt"])
        backend = str(rubric_config.get("backend", "http"))
        online_backends = ("actor_rollout", "actor_rollout_group_hints")
        if backend == "dataset" or backend in online_backends:
            rubric_column = (
                "online_rubric" if backend in online_backends else rubric_config.get("rubric_column", "rubric")
            )
            valid_column = (
                "online_rubric_valid"
                if backend in online_backends
                else rubric_config.get("valid_column", "rubric_valid")
            )
            rubric_texts, eligible, rubric_metrics = load_precomputed_rubrics(
                batch.non_tensor_batch,
                raw_prompts,
                rubric_column=rubric_column,
                valid_column=valid_column,
                fail_on_error=rubric_config.get("fail_on_error", True),
            )
        elif backend == "http":
            if self._rubric_vllm_client is None:
                self._rubric_vllm_client = RubricVLLMClient(rubric_config)
            rubric_texts, eligible, rubric_metrics = self._rubric_vllm_client.generate_batch(raw_prompts)
        else:
            raise ValueError(f"Unsupported rubric pipeline backend: {backend!r}")
        dataset_match_online_gates = backend == "dataset" and bool(
            rubric_config.get("dataset_match_online_gates", False)
        )
        if dataset_match_online_gates:
            eligible, dataset_gate_metrics = self._filter_dataset_rubric_contrast(
                batch, eligible, rubric_config
            )
            rubric_metrics.update(dataset_gate_metrics)
        rubric_batch = deepcopy(batch)
        if self._turn_replay_enabled(batch):
            rubric_batch.non_tensor_batch[TURN_REPLAY_KEY] = np.array(
                build_rubric_conditioned_turn_replay(
                    batch=batch,
                    tokenizer=self.tokenizer,
                    rubric_texts=rubric_texts,
                    teacher_max_prompt_length=rubric_config.teacher_max_prompt_length,
                    teacher_enable_thinking=rubric_config.teacher_enable_thinking,
                ),
                dtype=object,
            )
        else:
            rubric_inputs = build_rubric_conditioned_inputs(
                batch=batch,
                tokenizer=self.tokenizer,
                rubric_texts=rubric_texts,
                teacher_max_prompt_length=rubric_config.teacher_max_prompt_length,
                teacher_enable_thinking=rubric_config.teacher_enable_thinking,
            )
            for key, value in rubric_inputs.items():
                rubric_batch.batch[key] = value

        token_credit_mode = str(token_credit_config.get("mode", "additive_gap"))
        teacher_source = str(token_credit_config.get("opsd_teacher_source", "current_actor"))
        if token_credit_mode != "opsd_reverse_kl":
            teacher_source = "current_actor"

        if teacher_source == "reference_policy":
            if not self.use_reference_policy:
                raise RuntimeError("OPSD reference-policy teacher was requested but no reference worker exists")
            rubric_teacher_output = self._compute_ref_log_prob(rubric_batch)
            rubric_teacher_log_probs = rubric_teacher_output.batch["ref_log_prob"]
        elif teacher_source == "current_actor":
            rubric_teacher_output, _ = self._compute_old_log_prob(rubric_batch)
            rubric_teacher_log_probs = rubric_teacher_output.batch["old_log_probs"]
        else:
            raise ValueError(f"Unsupported OPSD teacher source: {teacher_source!r}")

        eligible = eligible.to(device=rubric_teacher_log_probs.device)
        if (backend == "actor_rollout_group_hints" or dataset_match_online_gates) and bool(
            rubric_config.get("require_outcome_aligned_gap", True)
        ):
            eligible, alignment_metrics = self._filter_outcome_aligned_rubric_groups(
                batch,
                rubric_teacher_log_probs,
                batch.batch["old_log_probs"],
                eligible,
                rubric_config,
            )
            rubric_metrics.update(alignment_metrics)
        rubric_metrics["rubric_pipeline/scorer_is_current_actor"] = float(teacher_source == "current_actor")
        rubric_metrics["rubric_pipeline/scorer_is_reference_policy"] = float(
            teacher_source == "reference_policy"
        )
        rubric_metrics["rubric_pipeline/scorer_thinking"] = float(rubric_config.teacher_enable_thinking)
        return rubric_teacher_log_probs, eligible, rubric_metrics

    def _uses_online_actor_rubrics(self) -> bool:
        token_credit_config = self.config.algorithm.get("token_credit", None)
        if token_credit_config is None or not token_credit_config.get("enable", False):
            return False
        rubric_config = token_credit_config.get("rubric_pipeline", None)
        return bool(
            rubric_config is not None
            and rubric_config.get("enable", False)
            and str(rubric_config.get("backend", "http"))
            in ("actor_rollout", "actor_rollout_group_hints")
        )

    @staticmethod
    def _online_rollout_score(
        non_tensor_batch: dict[str, Any],
        index: int,
        score_source: str = "strict_success",
    ) -> float:
        """Return the outcome used by the online rubric pipeline.

        ``strict_success`` preserves the original binary agent outcome.
        ``training_reward`` lets WebShop R2 use its continuous shaped reward
        for both GRPO and rubric grouping. ``official_reward`` exposes the
        benchmark-native continuous score without R2 penalties.
        """
        score_source = str(score_source).strip().casefold()
        if score_source not in ("strict_success", "training_reward", "official_reward"):
            raise ValueError(
                "rubric outcome_score_source must be one of: "
                f"strict_success, training_reward, official_reward; got {score_source!r}"
            )

        data_sources = non_tensor_batch.get("data_source")
        data_source = ""
        if data_sources is not None and data_sources[index] is not None:
            data_source = str(data_sources[index]).strip().casefold()

        reward_info = non_tensor_batch.get("reward_extra_info")
        if reward_info is not None:
            value = reward_info[index]
            if isinstance(value, dict):
                if score_source == "training_reward":
                    for key in ("training_reward", "score"):
                        if value.get(key) is not None:
                            return float(value[key])
                if score_source == "official_reward":
                    for key in ("official_reward", "acc", "won"):
                        if value.get(key) is not None:
                            return float(value[key])
                # Agent interactions expose an explicit terminal-success bit.
                # Prefer it even when AgentLoop output does not echo
                # ``data_source`` into the generated DataProto.
                if value.get("success") is not None:
                    return float(value["success"])
                if data_source in ("alfworld", "webshop"):
                    for key in ("official_reward", "acc", "won"):
                        if value.get(key) is not None:
                            outcome = float(value[key])
                            return float(outcome == 1.0) if data_source == "webshop" else outcome
                for key in ("training_reward", "score", "acc", "reward"):
                    if value.get(key) is not None:
                        return float(value[key])

        if score_source == "training_reward":
            for key in ("training_reward", "score"):
                values = non_tensor_batch.get(key)
                if values is not None and values[index] is not None:
                    return float(values[index])
        if score_source == "official_reward":
            for key in ("official_reward", "acc", "won"):
                values = non_tensor_batch.get(key)
                if values is not None and values[index] is not None:
                    return float(values[index])

        if data_source in ("alfworld", "webshop"):
            for key in ("success", "official_reward", "acc", "won"):
                values = non_tensor_batch.get(key)
                if values is not None and values[index] is not None:
                    outcome = float(values[index])
                    if key == "success":
                        return outcome
                    return float(outcome == 1.0) if data_source == "webshop" else outcome
        # ``score`` is the reward actually optimized by GRPO.  ``acc`` remains
        # the benchmark-native WebShop reward when R2 shaping is enabled.
        for key in ("training_reward", "score", "acc"):
            values = non_tensor_batch.get(key)
            if values is not None and values[index] is not None:
                return float(values[index])
        return 0.0

    def _rubric_outcome_score(self, non_tensor_batch: dict[str, Any], index: int) -> float:
        rubric_config = self.config.algorithm.token_credit.rubric_pipeline
        return self._online_rollout_score(
            non_tensor_batch,
            index,
            score_source=rubric_config.get("outcome_score_source", "strict_success"),
        )

    def _filter_dataset_rubric_contrast(self, batch, eligible, rubric_config):
        """Match online group-hint eligibility using the *current* rollout outcomes."""
        filtered = eligible.clone()
        if bool(rubric_config.get("require_group_outcome_contrast", True)):
            groups = {}
            for index, uid in enumerate(batch.non_tensor_batch["uid"]):
                groups.setdefault(str(uid), []).append(index)
            threshold = float(rubric_config.get("correctness_threshold", 0.5))
            epsilon = float(rubric_config.get("outcome_contrast_epsilon", 1e-6))
            for indices in groups.values():
                scores = [self._rubric_outcome_score(batch.non_tensor_batch, i) for i in indices]
                labels = [score >= threshold for score in scores]
                contrast = max(scores) - min(scores) > epsilon and any(labels) and not all(labels)
                if not contrast:
                    filtered[indices] = False
        return filtered, {
            "rubric_pipeline/dataset_match_online_gates": 1.0,
            "rubric_pipeline/dataset_contrast_eligible_frac": float(filtered.float().mean().item()),
        }

    def _filter_outcome_aligned_rubric_groups(
        self,
        batch: DataProto,
        teacher_log_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        eligible: torch.Tensor,
        rubric_config: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Keep privileged credit only when its sampled-token gap ranks outcomes correctly.

        A rubric can improve free-running reward yet still lower the likelihood
        of the particular successful actions sampled in the current group.  A
        raw additive gap from such a group points against GRPO.  For each
        contrastive group, compare the mean token gap of the highest-reward
        trajectories with that of the lowest-reward trajectories and retain
        the group only when the former is larger.
        """

        if "uid" not in batch.non_tensor_batch:
            raise KeyError("outcome-aligned rubric gating requires uid in the rollout batch")
        if teacher_log_probs.shape != student_log_probs.shape:
            raise ValueError("teacher and student log-probabilities must have identical shapes")

        response_mask = batch.batch["response_mask"].bool()
        finite = torch.isfinite(teacher_log_probs) & torch.isfinite(student_log_probs)
        token_mask = response_mask & finite
        token_gap = (teacher_log_probs.float() - student_log_probs.float()).detach()
        trajectory_gap = torch.where(token_mask, token_gap, torch.zeros_like(token_gap)).sum(dim=-1)
        trajectory_gap = trajectory_gap / token_mask.sum(dim=-1).clamp(min=1)

        filtered = eligible.to(device=teacher_log_probs.device, dtype=torch.bool).clone()
        uids = [str(value) for value in batch.non_tensor_batch["uid"]]
        groups: dict[str, list[int]] = {}
        for index, uid in enumerate(uids):
            groups.setdefault(uid, []).append(index)

        epsilon = float(rubric_config.get("outcome_contrast_epsilon", 1e-6))
        minimum_margin = float(rubric_config.get("min_outcome_gap_alignment", 0.0))
        checked = 0
        aligned = 0
        margins: list[float] = []
        for indices in groups.values():
            if not filtered[indices].any().item():
                continue
            scores = [self._rubric_outcome_score(batch.non_tensor_batch, index) for index in indices]
            lowest = min(scores)
            highest = max(scores)
            if highest - lowest <= epsilon:
                filtered[indices] = False
                continue
            checked += 1
            low_indices = [index for index, score in zip(indices, scores, strict=True) if score <= lowest + epsilon]
            high_indices = [
                index for index, score in zip(indices, scores, strict=True) if score >= highest - epsilon
            ]
            margin = float(
                trajectory_gap[high_indices].mean().item() - trajectory_gap[low_indices].mean().item()
            )
            margins.append(margin)
            if margin > minimum_margin:
                aligned += 1
            else:
                filtered[indices] = False

        metrics = {
            "rubric_pipeline/outcome_gap_alignment_enabled": 1.0,
            "rubric_pipeline/outcome_gap_alignment_groups_checked": float(checked),
            "rubric_pipeline/outcome_aligned_gap_group_frac": float(aligned / max(checked, 1)),
            "rubric_pipeline/outcome_gap_margin_mean": float(sum(margins) / max(len(margins), 1)),
            "rubric_pipeline/final_teacher_eligible_frac": float(filtered.float().mean().item()),
        }
        return filtered, metrics

    def _auxiliary_actor_generation_meta_info(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """Build worker-invariant metadata for actor-backed rubric generation."""
        rollout_config = self.config.actor_rollout_ref.rollout
        max_model_len = rollout_config.get("max_model_len", None)
        if max_model_len is None:
            max_model_len = int(rollout_config.prompt_length) + int(rollout_config.response_length)
        max_model_len = int(max_model_len)
        max_tokens = int(sampling_params["max_tokens"])
        if max_model_len < 1 or max_tokens < 1:
            raise ValueError(
                "Auxiliary actor generation requires positive padding lengths: "
                f"max_model_len={max_model_len}, max_tokens={max_tokens}"
            )
        return {
            "global_steps": self.global_steps,
            "skip_reward": True,
            # Auxiliary rubric prompts routinely exceed data.max_prompt_length.
            # Pad every worker to the same declared widths so DataProto.concat
            # never depends on the longest sample assigned to that worker.
            "output_prompt_length": max_model_len,
            "output_response_length": max_tokens,
            "auxiliary_sampling_params": dict(sampling_params),
        }

    def _generate_auxiliary_actor_sequences(self, batch: DataProto) -> tuple[DataProto, int]:
        """Generate an auxiliary actor batch without requiring one row per worker.

        ``AgentLoopManager.generate_sequences`` partitions its input into one
        equal chunk per agent-loop worker. Online rubric and judge batches are
        formed from unique prompt groups, so they can be smaller than (or not
        divisible by) the worker count even when the main rollout batch is
        perfectly divisible. Reuse VERL's standard validation padding scheme,
        then remove the duplicated scheduling-only rows before parsing output.
        """

        num_workers = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        if num_workers < 1:
            raise ValueError(f"actor_rollout_ref.rollout.agent.num_workers must be positive, got {num_workers}")
        padded_batch, pad_size = pad_dataproto_to_divisor(batch, num_workers)
        padded_output = self.async_rollout_manager.generate_sequences(padded_batch)
        output = unpad_dataproto(padded_output, pad_size=pad_size)
        output.meta_info.pop("timing", None)
        return output, pad_size

    def _generate_online_actor_group_hints(
        self,
        rollout_output: DataProto,
        rollout_metadata: dict[str, Any] | None = None,
        allow_uncorrelated_fallback: bool = False,
        snapshot_single_bundle: bool = False,
    ) -> tuple[DataProto, dict[str, float]]:
        """Distill one actor-visible rubric block from every current rollout group.

        The first-version prompt sees the same-task attempts and their terminal
        outcomes together.  By default it samples one rubric exactly as before.
        The bundle-selection mode repeats that identical prompt, scores each
        complete generated bundle across the sibling trajectories, and sends
        only the highest-Spearman bundle to teacher replay.  Items from different
        bundles are never mixed, and no generator loss is attached to this path.
        """
        if self.config.algorithm.token_credit.rubric_pipeline.get("pool_enable", False):
            from verl.trainer.ppo.online_rubric_pool import generate_online_rubric_pool

            return generate_online_rubric_pool(self, rollout_output, rollout_metadata)
        if not self.async_rollout_mode:
            raise RuntimeError("actor_rollout_group_hints requires rollout.mode=async")
        metadata = rollout_output.non_tensor_batch if rollout_metadata is None else rollout_metadata
        if "uid" not in metadata or "raw_prompt" not in metadata:
            raise KeyError("Online actor group hints require uid and raw_prompt in repeated rollout metadata")
        if len(metadata["uid"]) != len(rollout_output) or len(metadata["raw_prompt"]) != len(rollout_output):
            raise ValueError(
                "Online actor group-hint metadata must remain row-aligned with rollout output: "
                f"uids={len(metadata['uid'])}, raw_prompts={len(metadata['raw_prompt'])}, "
                f"rollouts={len(rollout_output)}"
            )

        from verl.trainer.ppo.rubric_token_credit import (
            compact_online_trajectory,
            extract_problem_text,
            fit_task_group_hint_messages,
            order_task_group_trajectory_pairs,
            parse_task_group_hint_response,
        )

        rubric_config = self.config.algorithm.token_credit.rubric_pipeline
        max_hints = int(rubric_config.get("rubrics_per_prompt", 5))
        min_hints = int(rubric_config.get("min_rubrics_per_prompt", 1))
        max_trajectories = int(rubric_config.get("max_trajectories_per_prompt", 4))
        max_trajectory_chars = int(rubric_config.get("max_trajectory_chars", 6000))
        max_rubric_chars = int(rubric_config.get("max_rubric_chars", 1600))
        max_item_chars = int(rubric_config.get("max_rubric_item_chars", 280))
        correctness_threshold = float(rubric_config.get("correctness_threshold", 0.5))
        require_outcome_contrast = bool(rubric_config.get("require_group_outcome_contrast", True))
        outcome_contrast_epsilon = float(rubric_config.get("outcome_contrast_epsilon", 1e-6))
        generator_enable_thinking = bool(rubric_config.get("generator_enable_thinking", False))
        candidate_count = int(rubric_config.get("group_hint_candidate_count", 1))
        select_by_correlation = bool(rubric_config.get("group_hint_select_by_correlation", False))
        select_lowest_correlation = bool(rubric_config.get("group_hint_select_lowest_correlation", False))
        fallback_on_unranked = bool(rubric_config.get("group_hint_fallback_on_unranked", False))
        force_first_candidate = bool(
            rubric_config.get("group_hint_force_first_candidate", False)
        )
        random_candidate = bool(rubric_config.get("group_hint_random_candidate", False))
        fixed_candidate_index = int(rubric_config.get("group_hint_fixed_candidate_index", -1))
        evidence_order = str(
            rubric_config.get("group_hint_evidence_order", "outcome_anchored_random")
        )
        prompt_style = str(
            rubric_config.get("group_hint_prompt_style", "process_rubric_5_to_k")
        )
        guarded_selection = bool(rubric_config.get("group_hint_guarded_selection", False))
        min_selected_correlation = float(
            rubric_config.get("group_hint_min_selected_correlation", 0.0)
        )
        min_correlation_improvement = float(
            rubric_config.get("group_hint_min_correlation_improvement", 0.0)
        )
        if snapshot_single_bundle:
            select_lowest_correlation = False
            fixed_candidate_index = -1
            random_candidate = False
            candidate_count = 1
            select_by_correlation = False
            guarded_selection = False
        rubric_max_tokens = int(rubric_config.get("max_tokens", 512))
        rollout_config = self.config.actor_rollout_ref.rollout
        rubric_model_context = int(
            rollout_config.get(
                "max_model_len",
                int(rollout_config.prompt_length) + int(rollout_config.response_length),
            )
        )
        rubric_prompt_token_budget = rubric_model_context - rubric_max_tokens
        if min_hints < 1 or max_hints < min_hints:
            raise ValueError(
                "actor_rollout_group_hints requires rubrics_per_prompt >= min_rubrics_per_prompt >= 1"
            )
        if max_trajectories < 1:
            raise ValueError("actor_rollout_group_hints requires max_trajectories_per_prompt >= 1")
        if require_outcome_contrast and max_trajectories < 2:
            raise ValueError(
                "contrast-gated actor_rollout_group_hints requires max_trajectories_per_prompt >= 2"
            )
        if candidate_count < 1:
            raise ValueError("group_hint_candidate_count must be at least 1")
        if fixed_candidate_index < -1 or fixed_candidate_index >= candidate_count:
            raise ValueError("group_hint_fixed_candidate_index must be -1 or a valid zero-based candidate index")
        if fixed_candidate_index >= 0 and (
            random_candidate or select_by_correlation or force_first_candidate or guarded_selection
        ):
            raise ValueError("Fixed bundle index cannot be combined with random, judge, first-only or guarded selection")
        if random_candidate and (select_by_correlation or force_first_candidate or guarded_selection):
            raise ValueError("Random bundle selection cannot be combined with judge, first-only or guarded selection")
        if select_lowest_correlation and (not select_by_correlation or guarded_selection):
            raise ValueError("Lowest-correlation selection requires judging and guarded selection disabled")
        if select_by_correlation and candidate_count < 2:
            raise ValueError("group-hint correlation selection requires at least two candidate bundles")
        if force_first_candidate and select_by_correlation:
            raise ValueError(
                "group_hint_force_first_candidate cannot be combined with correlation selection"
            )
        if select_by_correlation and not rubric_config.get("pool_judge_model"):
            raise ValueError("group-hint correlation selection requires pool_judge_model")
        if guarded_selection and not select_by_correlation:
            raise ValueError("guarded group-hint selection requires correlation selection")
        if evidence_order not in {"outcome_anchored_random", "legacy_alternating_extremes"}:
            raise ValueError(f"Unsupported group_hint_evidence_order: {evidence_order!r}")
        if prompt_style not in {"process_rubric_5_to_k", "legacy_process_rubric_3_to_k"}:
            raise ValueError(f"Unsupported group_hint_prompt_style: {prompt_style!r}")
        if not math.isfinite(min_selected_correlation):
            raise ValueError("group_hint_min_selected_correlation must be finite")
        if not math.isfinite(min_correlation_improvement) or min_correlation_improvement < 0:
            raise ValueError("group_hint_min_correlation_improvement must be finite and nonnegative")
        if rubric_prompt_token_budget < 1:
            raise ValueError(
                "Rubric generation leaves no prompt context: "
                f"model_context={rubric_model_context}, max_tokens={rubric_max_tokens}"
            )

        uids = [str(value) for value in metadata["uid"]]
        groups: dict[str, list[int]] = {}
        for index, uid in enumerate(uids):
            groups.setdefault(uid, []).append(index)

        # Generated AgentLoop batches contain environment reward details but
        # do not necessarily echo dataset metadata.  Restore data_source for
        # benchmark-specific outcome semantics before grouping trajectories.
        outcome_metadata = dict(rollout_output.non_tensor_batch)
        if "data_source" in metadata:
            outcome_metadata["data_source"] = metadata["data_source"]

        prompt_length = rollout_output.batch["prompts"].shape[-1]
        response_attention = rollout_output.batch["attention_mask"][:, prompt_length:]
        raw_prompts = list(metadata["raw_prompt"])
        structured_transcripts = rollout_output.non_tensor_batch.get("interaction_trajectory_transcript")

        generation_messages: list[list[dict[str, str]]] = []
        generation_indices: list[int] = []
        generation_uids: list[str] = []
        problems_by_uid: dict[str, str] = {}
        transcripts_by_uid: dict[str, list[str]] = {}
        outcomes_by_uid: dict[str, list[float]] = {}
        selection_outcomes_by_uid: dict[str, list[float]] = {}
        attempts_seen_by_uid: dict[str, int] = {}
        contrastive_by_uid: dict[str, bool] = {}
        prompt_fit_by_uid: dict[str, dict[str, Any]] = {}
        by_uid: dict[str, tuple[str, bool, int]] = {
            uid: ("", False, 0) for uid in groups
        }
        for uid, indices in groups.items():
            outcomes = [self._rubric_outcome_score(outcome_metadata, index) for index in indices]
            outcomes_by_uid[uid] = outcomes
            outcome_labels = [outcome >= correctness_threshold for outcome in outcomes]
            has_contrast = (
                bool(outcomes)
                and max(outcomes) - min(outcomes) > outcome_contrast_epsilon
                and any(outcome_labels)
                and not all(outcome_labels)
            )
            contrastive_by_uid[uid] = has_contrast
            if require_outcome_contrast and not has_contrast and not allow_uncorrelated_fallback:
                attempts_seen_by_uid[uid] = 0
                continue

            original_pairs = list(zip(indices, outcomes, strict=True))
            selected_pairs = order_task_group_trajectory_pairs(
                original_pairs,
                seed=f"{self.global_steps}:{uid}",
                mode=evidence_order,
            )[:max_trajectories]
            # Correlation selection must use the exact evidence shown to both
            # the generator and judge. This keeps the guarded best-of-N route
            # comparable to the historical four-trajectory single-bundle run.
            selection_outcomes_by_uid[uid] = [outcome for _, outcome in selected_pairs]
            evidence: list[dict[str, Any]] = []
            for rollout_index, outcome in selected_pairs:
                transcript = ""
                if structured_transcripts is not None:
                    transcript = str(structured_transcripts[rollout_index] or "").strip()
                if not transcript:
                    active = response_attention[rollout_index].bool()
                    token_ids = (
                        rollout_output.batch["responses"][rollout_index][active].detach().cpu().tolist()
                    )
                    transcript = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                evidence.append(
                    {
                        "transcript": transcript,
                        "outcome": outcome,
                        "success": bool(outcome >= correctness_threshold),
                    }
                )
            first_index = indices[0]
            problem = extract_problem_text(raw_prompts[first_index])
            problems_by_uid[uid] = problem
            transcripts_by_uid[uid] = [
                compact_online_trajectory(item["transcript"], max_chars=max_trajectory_chars)
                for item in evidence
            ]
            attempts_seen_by_uid[uid] = len(evidence)
            prompt, prompt_fit = fit_task_group_hint_messages(
                problem,
                evidence,
                self.tokenizer,
                max_hints=max_hints,
                max_trajectories=max_trajectories,
                max_trajectory_chars=max_trajectory_chars,
                max_input_tokens=rubric_prompt_token_budget,
                correctness_threshold=correctness_threshold,
                enable_thinking=generator_enable_thinking,
                prompt_style=prompt_style,
            )
            prompt_fit_by_uid[uid] = prompt_fit
            if prompt is None:
                continue
            for _ in range(candidate_count):
                # Every sample receives the same prompt and differs only through
                # actor sampling.  Keep separate list objects to prevent a later
                # in-place template mutation from coupling candidate rows.
                generation_messages.append([dict(message) for message in prompt])
                generation_indices.append(first_index)
                generation_uids.append(uid)

        auxiliary_padding_rows = 0
        candidate_texts_by_uid: dict[str, list[str]] = {uid: [] for uid in groups}
        candidate_bundles_by_uid: dict[str, list[list[str]]] = {uid: [] for uid in groups}
        bundle_diagnostics_by_uid: dict[str, list[dict[str, Any]]] = {uid: [] for uid in groups}
        judge_failed_requests = 0
        judge_request_count = 0
        fallback_uids: set[str] = set()
        if generation_messages:
            rubric_batch = rollout_output.select_idxs(generation_indices)
            rubric_batch.non_tensor_batch["raw_prompt"] = np.array(generation_messages, dtype=object)
            rubric_batch.non_tensor_batch["agent_name"] = np.array(
                ["single_turn_agent"] * len(generation_messages), dtype=object
            )
            rubric_batch.non_tensor_batch["chat_template_kwargs_override"] = np.array(
                [{"enable_thinking": generator_enable_thinking} for _ in generation_messages], dtype=object
            )
            rubric_batch.meta_info = self._auxiliary_actor_generation_meta_info(
                {
                    "temperature": float(rubric_config.get("temperature", 0.3)),
                    "top_p": float(rubric_config.get("top_p", 0.9)),
                    "top_k": -1,
                    "max_tokens": rubric_max_tokens,
                }
            )
            # Keep the padded auxiliary batch within the same total context
            # used by vLLM: prompt budget + generated rubric <= model context.
            rubric_batch.meta_info["output_prompt_length"] = rubric_prompt_token_budget
            rubric_output, auxiliary_padding_rows = self._generate_auxiliary_actor_sequences(rubric_batch)
            rubric_prompt_length = rubric_output.batch["prompts"].shape[-1]
            rubric_attention = rubric_output.batch["attention_mask"][:, rubric_prompt_length:]

            for uid, response_ids, attention in zip(
                generation_uids,
                rubric_output.batch["responses"],
                rubric_attention,
                strict=True,
            ):
                raw_text = self.tokenizer.decode(response_ids[attention.bool()].detach().cpu().tolist())
                rubric_text, items = parse_task_group_hint_response(
                    raw_text,
                    max_items=max_hints,
                    min_items=min_hints,
                    max_item_chars=max_item_chars,
                    max_rubric_chars=max_rubric_chars,
                )
                candidate_texts_by_uid[uid].append(rubric_text)
                candidate_bundles_by_uid[uid].append(items)

        if select_by_correlation:
            from verl.trainer.ppo.rubric_pool import (
                CloudRubricJudge,
                build_pool,
                select_best_bundle,
            )

            # CloudRubricJudge shares the current satisfaction prompt and HTTP
            # reliability path.  pool_enable remains False so generation stays
            # on this aggregate group-hint route.
            judge_config = dict(rubric_config)
            judge_config["pool_enable"] = True
            judge = getattr(self, "_cloud_group_hint_judge", None)
            if judge is None:
                judge = CloudRubricJudge(judge_config)
                self._cloud_group_hint_judge = judge
            pools_by_uid = {
                uid: build_pool(candidate_bundles_by_uid[uid]) for uid in groups
            }
            jobs = [
                {
                    "uid": uid,
                    "problem": problems_by_uid[uid],
                    "transcripts": transcripts_by_uid[uid],
                    "pool": pools_by_uid[uid],
                }
                for uid in groups
                if contrastive_by_uid.get(uid, False) and pools_by_uid[uid]
            ]
            judged = judge.score_groups(jobs)
            for uid in groups:
                result = judged.get(uid)
                if result is None:
                    continue
                judge_request_count += int(result.get("requests", 0))
                judge_failed_requests += int(result.get("failed_requests", 0))
                selected, diagnostics = select_best_bundle(
                    candidate_bundles_by_uid[uid],
                    pools_by_uid[uid],
                    result["matrix"],
                    selection_outcomes_by_uid[uid],
                    min_judge_coverage=float(rubric_config.get("pool_min_judge_coverage", 0.75)),
                    epsilon=outcome_contrast_epsilon,
                    select_lowest=select_lowest_correlation,
                    baseline_index=0 if guarded_selection else None,
                    min_selected_correlation=min_selected_correlation if guarded_selection else None,
                    min_correlation_improvement=(
                        min_correlation_improvement if guarded_selection else 0.0
                    ),
                )
                bundle_diagnostics_by_uid[uid] = diagnostics
                if selected is not None:
                    text = candidate_texts_by_uid[uid][selected]
                    items = candidate_bundles_by_uid[uid][selected]
                    by_uid[uid] = (text, bool(items), len(items))
            if allow_uncorrelated_fallback or fallback_on_unranked:
                # Correlation is undefined for uniform outcomes and may be
                # unavailable after judging fails. The opt-in first-version
                # route still passes one intact parsed bundle to teacher.
                for uid in groups:
                    if by_uid[uid][1]:
                        continue
                    for text, items in zip(
                        candidate_texts_by_uid[uid], candidate_bundles_by_uid[uid], strict=True
                    ):
                        if items:
                            by_uid[uid] = (text, True, len(items))
                            fallback_uids.add(uid)
                            break
        else:
            import random

            # Independent RNG: selecting a bundle must not advance rollout RNG state.
            selection_rng = random.Random(
                f"{self.config.data.get('seed', 1)}:{getattr(self, 'global_steps', 0)}:rubric_random"
            )
            random_indices = []
            # The strict control ablation consumes candidate zero only. The
            # compatibility path otherwise uses the first successfully parsed
            # generation. Neither path invokes a verifier.
            for uid in groups:
                candidates = list(
                    zip(
                        candidate_texts_by_uid[uid],
                        candidate_bundles_by_uid[uid],
                        strict=True,
                    )
                )
                if random_candidate and not candidates:
                    continue
                if fixed_candidate_index >= 0:
                    # Preserve generation order; never substitute another parsed candidate.
                    candidates = candidates[fixed_candidate_index:fixed_candidate_index + 1]
                elif random_candidate:
                    selected_index = selection_rng.randrange(len(candidates))
                    random_indices.append(selected_index)
                    # Choose uniformly BEFORE considering parser validity. No reranking
                    # or fallback to another bundle when the chosen one is malformed.
                    candidates = candidates[selected_index:selected_index + 1]
                elif force_first_candidate:
                    candidates = candidates[:1]
                for text, items in candidates:
                    if items:
                        by_uid[uid] = (text, True, len(items))
                        break

        rubric_texts = [by_uid[uid][0] for uid in uids]
        rubric_valid = [by_uid[uid][1] for uid in uids]
        rollout_output.non_tensor_batch["online_rubric"] = np.array(rubric_texts, dtype=object)
        rollout_output.non_tensor_batch["online_rubric_valid"] = np.array(rubric_valid, dtype=bool)
        rollout_output.non_tensor_batch["online_rubric_item_count"] = np.array(
            [by_uid[uid][2] for uid in uids], dtype=np.int64
        )

        unique_valid = sum(int(value[1]) for value in by_uid.values())
        labels_by_uid = {
            uid: [score >= correctness_threshold for score in outcomes]
            for uid, outcomes in outcomes_by_uid.items()
        }
        all_correct = sum(bool(labels) and all(labels) for labels in labels_by_uid.values())
        all_wrong = sum(not any(labels) for labels in labels_by_uid.values())
        mixed = len(groups) - all_correct - all_wrong
        all_scores = [score for outcomes in outcomes_by_uid.values() for score in outcomes]
        outcome_ranges = [max(outcomes) - min(outcomes) for outcomes in outcomes_by_uid.values() if outcomes]
        source_values = list(metadata.get("data_source", [""] * len(uids)))
        strict_success_mask = [
            str(source).strip().casefold() in ("alfworld", "webshop") for source in source_values
        ]
        strict_success_scores = [
            self._online_rollout_score(outcome_metadata, index, score_source="strict_success")
            for index, is_strict in enumerate(strict_success_mask)
            if is_strict
        ]
        selected_bundle_correlations = [
            float(item["spearman_correlation"])
            for diagnostics in bundle_diagnostics_by_uid.values()
            for item in diagnostics
            if item.get("selected") and item.get("spearman_correlation") is not None
        ]
        guarded_replacements = sum(
            item.get("selection_reason") == "guard_replaced_baseline"
            for diagnostics in bundle_diagnostics_by_uid.values()
            for item in diagnostics
        )
        guarded_baseline_kept = sum(
            item.get("selection_reason") == "guard_kept_baseline"
            for diagnostics in bundle_diagnostics_by_uid.values()
            for item in diagnostics
        )
        parsed_candidate_count = sum(
            bool(items) for bundles in candidate_bundles_by_uid.values() for items in bundles
        )
        prompt_fit_records = list(prompt_fit_by_uid.values())
        metrics = {
            "rubric_pipeline/online_actor": 1.0,
            "rubric_pipeline/task_group_hints": 1.0,
            "rubric_pipeline/auxiliary_prompt_count": float(len(generation_messages)),
            "rubric_pipeline/auxiliary_padding_rows": float(auxiliary_padding_rows),
            # No criterion-level filtering occurs on this route. Correlation
            # is used only to choose one intact generated bundle.
            "rubric_pipeline/correlation_filter": 0.0,
            "rubric_pipeline/bundle_selection_enabled": float(select_by_correlation),
            "rubric_pipeline/bundle_force_first_candidate": float(force_first_candidate),
            "rubric_pipeline/bundle_fixed_candidate_index": float(fixed_candidate_index),
            "rubric_pipeline/bundle_random_selection_enabled": float(random_candidate),
            **({
                f"rubric_pipeline/random_candidate_{index}_frac": random_indices.count(index) / max(len(random_indices), 1)
                for index in range(candidate_count)
            } if random_candidate else {}),
            "rubric_pipeline/bundle_guarded_selection_enabled": float(guarded_selection),
            "rubric_pipeline/bundle_guard_replacement_group_frac": float(
                guarded_replacements / max(len(groups), 1)
            ),
            "rubric_pipeline/bundle_guard_baseline_group_frac": float(
                guarded_baseline_kept / max(len(groups), 1)
            ),
            "rubric_pipeline/bundle_unranked_fallback_on": float(fallback_on_unranked),
            "rubric_pipeline/bundle_unranked_fallback_group_frac": float(
                len(fallback_uids) / max(len(groups), 1)
            ),
            "rubric_pipeline/candidates_per_group": float(candidate_count),
            "rubric_pipeline/generated_candidate_bundle_count": float(len(generation_messages)),
            "rubric_pipeline/parsed_candidate_bundle_frac": float(
                parsed_candidate_count / max(len(generation_messages), 1)
            ),
            "rubric_pipeline/generator_prompt_token_budget": float(rubric_prompt_token_budget),
            "rubric_pipeline/generator_prompt_tokens_mean": float(
                sum(record["prompt_tokens"] for record in prompt_fit_records)
                / max(len(prompt_fit_records), 1)
            ),
            "rubric_pipeline/generator_prompt_context_truncated_frac": float(
                sum(bool(record["context_truncated"]) for record in prompt_fit_records)
                / max(len(prompt_fit_records), 1)
            ),
            "rubric_pipeline/generator_prompt_context_overflow_frac": float(
                sum(bool(record["context_overflow"]) for record in prompt_fit_records)
                / max(len(prompt_fit_records), 1)
            ),
            "rubric_pipeline/selected_bundle_spearman_mean": float(
                sum(selected_bundle_correlations) / max(len(selected_bundle_correlations), 1)
            ),
            "rubric_pipeline/judge_request_count": float(judge_request_count),
            "rubric_pipeline/judge_request_failure_frac": float(
                judge_failed_requests / max(judge_request_count, 1)
            ),
            "rubric_pipeline/group_outcome_contrast_gate": float(require_outcome_contrast),
            "rubric_pipeline/generator_thinking": float(generator_enable_thinking),
            "rubric_pipeline/success_rate": float(sum(rubric_valid) / max(len(rubric_valid), 1)),
            "rubric_pipeline/unique_success_rate": float(unique_valid / max(len(groups), 1)),
            "rubric_pipeline/teacher_eligible_group_frac": float(unique_valid / max(len(groups), 1)),
            "rubric_pipeline/items_mean": float(
                sum(value[2] for value in by_uid.values()) / max(len(by_uid), 1)
            ),
            "rubric_pipeline/chars_mean": float(sum(map(len, rubric_texts)) / max(len(rubric_texts), 1)),
            "rubric_pipeline/attempts_seen_mean": float(
                sum(attempts_seen_by_uid.values()) / max(len(attempts_seen_by_uid), 1)
            ),
            "rubric_pipeline/outcome_mean": float(sum(all_scores) / max(len(all_scores), 1)),
            "rubric_pipeline/outcome_positive_threshold": float(correctness_threshold),
            "rubric_pipeline/outcome_positive_frac": float(
                sum(score >= correctness_threshold for score in all_scores) / max(len(all_scores), 1)
            ),
            "rubric_pipeline/outcome_source_training_reward": float(
                str(rubric_config.get("outcome_score_source", "strict_success")).casefold()
                == "training_reward"
            ),
            "rubric_pipeline/strict_success_outcome_frac": float(
                sum(strict_success_mask) / max(len(strict_success_mask), 1)
            ),
            "rubric_pipeline/strict_success_rate": float(
                sum(strict_success_scores) / max(len(strict_success_scores), 1)
            ),
            "rubric_pipeline/outcome_group_range_mean": float(
                sum(outcome_ranges) / max(len(outcome_ranges), 1)
            ),
            "rubric_pipeline/mixed_outcome_group_frac": float(mixed / max(len(groups), 1)),
            "rubric_pipeline/contrastive_outcome_group_frac": float(
                sum(contrastive_by_uid.values()) / max(len(groups), 1)
            ),
            "rubric_pipeline/uniform_outcome_group_frac": float(
                (all_correct + all_wrong) / max(len(groups), 1)
            ),
            "rubric_pipeline/all_correct_group_frac": float(all_correct / max(len(groups), 1)),
            "rubric_pipeline/all_wrong_group_frac": float(all_wrong / max(len(groups), 1)),
        }
        return rollout_output, metrics

    def _generate_online_actor_rubrics(
        self,
        rollout_output: DataProto,
        rollout_metadata: dict[str, Any] | None = None,
        allow_uncorrelated_fallback: bool = False,
        snapshot_single_bundle: bool = False,
    ) -> tuple[DataProto, dict[str, float]]:
        """Generate, score, and filter fresh self-rubrics on mixed-outcome groups."""
        rubric_config = self.config.algorithm.token_credit.rubric_pipeline
        backend = str(rubric_config.get("backend", "actor_rollout"))
        if backend == "actor_rollout_group_hints":
            return self._generate_online_actor_group_hints(
                rollout_output,
                rollout_metadata=rollout_metadata,
                allow_uncorrelated_fallback=allow_uncorrelated_fallback,
                snapshot_single_bundle=snapshot_single_bundle,
            )
        if backend != "actor_rollout":
            raise ValueError(f"Unsupported online actor rubric backend: {backend!r}")
        if not self.async_rollout_mode:
            raise RuntimeError("actor_rollout rubric generation currently requires rollout.mode=async")
        metadata = rollout_output.non_tensor_batch if rollout_metadata is None else rollout_metadata
        if "uid" not in metadata or "raw_prompt" not in metadata:
            raise KeyError("Online actor rubrics require uid and raw_prompt in repeated rollout metadata")
        if len(metadata["uid"]) != len(rollout_output) or len(metadata["raw_prompt"]) != len(rollout_output):
            raise ValueError(
                "Online actor-rubric metadata must remain row-aligned with rollout output: "
                f"uids={len(metadata['uid'])}, raw_prompts={len(metadata['raw_prompt'])}, "
                f"rollouts={len(rollout_output)}"
            )

        from verl.trainer.ppo.rubric_token_credit import (
            build_agent_rubric_generation_messages,
            build_agent_rubric_judge_messages,
            compact_online_trajectory,
            extract_problem_text,
            filter_rubrics_by_outcome_correlation,
            format_rubric_items,
            parse_rubric_judge_response,
            parse_rubric_response,
            rubric_text_to_items,
            validate_alfworld_rubric_text,
        )

        rubrics_per_prompt = int(rubric_config.get("rubrics_per_prompt", 6))
        min_rubrics = int(rubric_config.get("min_rubrics_per_prompt", 4))
        min_reliable_rubrics = int(rubric_config.get("min_reliable_rubrics_per_prompt", 4))
        reliability_threshold = float(rubric_config.get("reliability_threshold", 0.5))
        correctness_threshold = float(rubric_config.get("correctness_threshold", 0.5))
        max_trajectories = int(rubric_config.get("max_trajectories_per_prompt", 1))
        max_trajectory_chars = int(rubric_config.get("max_trajectory_chars", 6000))
        max_candidates = int(rubric_config.get("max_candidate_rubrics_per_group", 24))
        generator_enable_thinking = bool(rubric_config.get("generator_enable_thinking", True))
        judge_enable_thinking = bool(rubric_config.get("judge_enable_thinking", True))
        actor_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        actor_enable_thinking = bool(actor_chat_template_kwargs.get("enable_thinking", False))
        if min_rubrics < 1 or rubrics_per_prompt < min_rubrics:
            raise ValueError("Online rubric generation requires rubrics_per_prompt >= min_rubrics_per_prompt >= 1")
        if min_reliable_rubrics < 1:
            raise ValueError("min_reliable_rubrics_per_prompt must be at least 1")
        if max_trajectories != 1:
            raise ValueError(
                "Each rubric-generation request must see exactly one rollout; "
                "set max_trajectories_per_prompt=1"
            )

        uids = [str(value) for value in metadata["uid"]]
        groups: dict[str, list[int]] = {}
        for index, uid in enumerate(uids):
            groups.setdefault(uid, []).append(index)

        outcome_metadata = dict(rollout_output.non_tensor_batch)
        if "data_source" in metadata:
            outcome_metadata["data_source"] = metadata["data_source"]

        prompt_length = rollout_output.batch["prompts"].shape[-1]
        response_attention = rollout_output.batch["attention_mask"][:, prompt_length:]
        raw_prompts = list(metadata["raw_prompt"])
        data_sources = metadata.get("data_source")
        generation_messages: list[list[dict[str, str]]] = []
        generation_indices: list[int] = []
        generation_keys: list[tuple[str, str]] = []
        outcomes_by_uid: dict[str, list[float]] = {}
        evidence_by_uid: dict[str, list[dict[str, str]]] = {}
        benchmark_by_uid: dict[str, str] = {}
        problem_by_uid: dict[str, str] = {}
        mixed_uids: list[str] = []
        for uid, indices in groups.items():
            outcomes = [self._rubric_outcome_score(outcome_metadata, index) for index in indices]
            outcomes_by_uid[uid] = outcomes
            labels = [score >= correctness_threshold for score in outcomes]
            evidence: list[dict[str, str]] = []
            for evidence_number, rollout_index in enumerate(indices, start=1):
                active = response_attention[rollout_index].bool()
                token_ids = rollout_output.batch["responses"][rollout_index][active].detach().cpu().tolist()
                transcript = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                evidence.append(
                    {
                        "evidence_id": f"rollout_{evidence_number}",
                        "transcript": compact_online_trajectory(
                            transcript,
                            max_chars=max_trajectory_chars,
                        ),
                    }
                )
            first_index = indices[0]
            benchmark = str(data_sources[first_index]) if data_sources is not None else ""
            evidence_by_uid[uid] = evidence
            benchmark_by_uid[uid] = benchmark
            problem = extract_problem_text(raw_prompts[first_index])
            problem_by_uid[uid] = problem
            if not labels or all(labels) or not any(labels):
                continue
            mixed_uids.append(uid)
            for rollout_position, rollout_index in enumerate(indices):
                one_evidence = [evidence[rollout_position]]
                generation_messages.append(
                    build_agent_rubric_generation_messages(
                        problem,
                        one_evidence,
                        rubrics_per_prompt,
                        benchmark=benchmark,
                    )
                )
                generation_indices.append(rollout_index)
                generation_keys.append((uid, evidence[rollout_position]["evidence_id"]))

        candidates_by_uid: dict[str, list[str]] = {uid: [] for uid in groups}
        parsed_item_counts: list[int] = []
        grounded_item_counts: list[int] = []
        grounding_rejected_items: list[int] = []
        grounding_rejection_reason_totals: dict[str, int] = {}
        if generation_messages:
            rubric_batch = rollout_output.select_idxs(generation_indices)
            rubric_batch.non_tensor_batch["raw_prompt"] = np.array(generation_messages, dtype=object)
            rubric_batch.non_tensor_batch["agent_name"] = np.array(
                ["single_turn_agent"] * len(generation_messages), dtype=object
            )
            rubric_batch.non_tensor_batch["chat_template_kwargs_override"] = np.array(
                [{"enable_thinking": generator_enable_thinking} for _ in generation_messages], dtype=object
            )
            rubric_batch.meta_info = self._auxiliary_actor_generation_meta_info(
                {
                    "temperature": float(rubric_config.get("temperature", 0.2)),
                    "top_p": float(rubric_config.get("top_p", 0.95)),
                    "top_k": -1,
                    "max_tokens": int(rubric_config.get("max_tokens", 2048)),
                }
            )
            rubric_output, _ = self._generate_auxiliary_actor_sequences(rubric_batch)
            rubric_prompt_length = rubric_output.batch["prompts"].shape[-1]
            rubric_attention = rubric_output.batch["attention_mask"][:, rubric_prompt_length:]
            for (uid, evidence_id), response_ids, attention in zip(
                generation_keys,
                rubric_output.batch["responses"],
                rubric_attention,
                strict=True,
            ):
                raw_text = self.tokenizer.decode(response_ids[attention.bool()].detach().cpu().tolist())
                parsed = parse_rubric_response(
                    raw_text,
                    rubrics_per_prompt,
                    int(rubric_config.get("max_rubric_chars", 4000)),
                )
                parsed_items = rubric_text_to_items(parsed)
                parsed_item_counts.append(len(parsed_items))
                if "alfworld" in benchmark_by_uid.get(uid, "").lower():
                    source_evidence = [
                        item for item in evidence_by_uid[uid] if item["evidence_id"] == evidence_id
                    ]
                    parsed, grounding = validate_alfworld_rubric_text(
                        parsed,
                        source_evidence,
                        max_items=rubrics_per_prompt,
                    )
                else:
                    grounding = {"rejected_items": 0, "rejection_reasons": {}}
                grounded_items = rubric_text_to_items(parsed)
                grounded_item_counts.append(len(grounded_items))
                grounding_rejected_items.append(int(grounding["rejected_items"]))
                for reason, count in grounding.get("rejection_reasons", {}).items():
                    grounding_rejection_reason_totals[reason] = (
                        grounding_rejection_reason_totals.get(reason, 0) + int(count)
                    )
                existing_keys = {
                    item.rsplit("; SOURCE", 1)[0].strip().lower() for item in candidates_by_uid[uid]
                }
                for item in grounded_items:
                    semantic_key = item.rsplit("; SOURCE", 1)[0].strip().lower()
                    if semantic_key not in existing_keys and len(candidates_by_uid[uid]) < max_candidates:
                        candidates_by_uid[uid].append(item)
                        existing_keys.add(semantic_key)

        judge_messages: list[list[dict[str, str]]] = []
        judge_indices: list[int] = []
        judge_keys: list[str] = []
        judge_candidate_uids = [uid for uid in mixed_uids if len(candidates_by_uid[uid]) >= min_rubrics]
        for uid in judge_candidate_uids:
            for rollout_position, rollout_index in enumerate(groups[uid]):
                judge_messages.append(
                    build_agent_rubric_judge_messages(
                        problem_by_uid[uid],
                        evidence_by_uid[uid][rollout_position]["transcript"],
                        candidates_by_uid[uid],
                    )
                )
                judge_indices.append(rollout_index)
                judge_keys.append(uid)

        judge_vectors_by_uid: dict[str, list[list[float]]] = {uid: [] for uid in groups}
        judge_parse_successes = 0
        if judge_messages:
            judge_batch = rollout_output.select_idxs(judge_indices)
            judge_batch.non_tensor_batch["raw_prompt"] = np.array(judge_messages, dtype=object)
            judge_batch.non_tensor_batch["agent_name"] = np.array(
                ["single_turn_agent"] * len(judge_messages), dtype=object
            )
            judge_batch.non_tensor_batch["chat_template_kwargs_override"] = np.array(
                [{"enable_thinking": judge_enable_thinking} for _ in judge_messages], dtype=object
            )
            judge_batch.meta_info = self._auxiliary_actor_generation_meta_info(
                {
                    "temperature": float(rubric_config.get("judge_temperature", 0.0)),
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": int(rubric_config.get("judge_max_tokens", 1024)),
                }
            )
            judge_output, _ = self._generate_auxiliary_actor_sequences(judge_batch)
            judge_prompt_length = judge_output.batch["prompts"].shape[-1]
            judge_attention = judge_output.batch["attention_mask"][:, judge_prompt_length:]
            for uid, response_ids, attention in zip(
                judge_keys,
                judge_output.batch["responses"],
                judge_attention,
                strict=True,
            ):
                raw_text = self.tokenizer.decode(response_ids[attention.bool()].detach().cpu().tolist())
                vector = parse_rubric_judge_response(raw_text, len(candidates_by_uid[uid]))
                if vector is not None:
                    judge_vectors_by_uid[uid].append(vector)
                    judge_parse_successes += 1

        by_uid: dict[str, tuple[str, bool]] = {uid: ("", False) for uid in groups}
        filter_stats_by_uid: dict[str, dict[str, Any]] = {}
        for uid in mixed_uids:
            vectors = judge_vectors_by_uid[uid]
            if len(vectors) != len(groups[uid]):
                filter_stats_by_uid[uid] = {
                    "eligible_group": False,
                    "outcome_group": "mixed_judge_failure",
                    "candidate_items": len(candidates_by_uid[uid]),
                    "reliable_items": 0,
                    "reliability_scores": [],
                    "pearson_scores": [],
                    "selected_indices": [],
                    "constant_items": 0,
                }
                continue
            score_matrix = [list(row) for row in zip(*vectors, strict=True)]
            selected, filter_stats = filter_rubrics_by_outcome_correlation(
                candidates_by_uid[uid],
                score_matrix,
                outcomes_by_uid[uid],
                reliability_threshold=reliability_threshold,
                correctness_threshold=correctness_threshold,
            )
            filter_stats_by_uid[uid] = filter_stats
            valid = len(selected) >= min_reliable_rubrics
            by_uid[uid] = (
                format_rubric_items(
                    selected,
                    max_chars=int(rubric_config.get("max_rubric_chars", 4000)),
                )
                if valid
                else "",
                valid,
            )

        for uid in groups:
            if uid not in filter_stats_by_uid:
                labels = [score > correctness_threshold for score in outcomes_by_uid[uid]]
                outcome_group = "all_correct" if labels and all(labels) else "all_wrong"
                filter_stats_by_uid[uid] = {
                    "eligible_group": False,
                    "outcome_group": outcome_group,
                    "candidate_items": 0,
                    "reliable_items": 0,
                    "reliability_scores": [],
                    "pearson_scores": [],
                    "selected_indices": [],
                    "constant_items": 0,
                }

        rubric_texts = [by_uid[uid][0] for uid in uids]
        rubric_valid = [by_uid[uid][1] for uid in uids]
        rollout_output.non_tensor_batch["online_rubric"] = np.array(rubric_texts, dtype=object)
        rollout_output.non_tensor_batch["online_rubric_valid"] = np.array(rubric_valid, dtype=bool)
        all_reliability = [
            value for stats in filter_stats_by_uid.values() for value in stats["reliability_scores"]
        ]
        selected_reliability = [
            stats["reliability_scores"][index]
            for stats in filter_stats_by_uid.values()
            for index in stats["selected_indices"]
        ]
        unique_valid = sum(value[1] for value in by_uid.values())
        all_correct_groups = sum(
            stats["outcome_group"] == "all_correct" for stats in filter_stats_by_uid.values()
        )
        all_wrong_groups = sum(
            stats["outcome_group"] == "all_wrong" for stats in filter_stats_by_uid.values()
        )
        matrix_complete_groups = sum(
            stats.get("matrix_shape_valid", False) for stats in filter_stats_by_uid.values()
        )
        metrics = {
            "rubric_pipeline/online_actor": 1.0,
            "rubric_pipeline/correlation_filter": 1.0,
            "rubric_pipeline/actor_thinking": float(actor_enable_thinking),
            "rubric_pipeline/generator_thinking": float(generator_enable_thinking),
            "rubric_pipeline/judge_thinking": float(judge_enable_thinking),
            "rubric_pipeline/success_rate": float(sum(rubric_valid) / max(len(rubric_valid), 1)),
            "rubric_pipeline/unique_success_rate": float(unique_valid / max(len(by_uid), 1)),
            "rubric_pipeline/teacher_eligible_group_frac": float(unique_valid / max(len(by_uid), 1)),
            "rubric_pipeline/items_mean": float(
                sum(stats["reliable_items"] for stats in filter_stats_by_uid.values())
                / max(len(filter_stats_by_uid), 1)
            ),
            "rubric_pipeline/candidate_items_mean": float(
                sum(len(candidates_by_uid[uid]) for uid in groups) / max(len(groups), 1)
            ),
            "rubric_pipeline/parsed_items_mean": float(
                sum(parsed_item_counts) / max(len(parsed_item_counts), 1)
            ),
            "rubric_pipeline/grounded_item_acceptance_rate": float(
                sum(grounded_item_counts) / max(sum(parsed_item_counts), 1)
            ),
            "rubric_pipeline/grounding_rejected_items_mean": float(
                sum(grounding_rejected_items) / max(len(grounding_rejected_items), 1)
            ),
            "rubric_pipeline/chars_mean": float(sum(map(len, rubric_texts)) / max(len(rubric_texts), 1)),
            "rubric_pipeline/outcome_mean": float(
                sum(score for scores in outcomes_by_uid.values() for score in scores)
                / max(sum(map(len, outcomes_by_uid.values())), 1)
            ),
            "rubric_pipeline/mixed_outcome_group_frac": float(
                len(mixed_uids) / max(len(groups), 1)
            ),
            "rubric_pipeline/uniform_outcome_group_frac": float(
                (len(groups) - len(mixed_uids)) / max(len(groups), 1)
            ),
            "rubric_pipeline/all_correct_group_frac": float(all_correct_groups / max(len(groups), 1)),
            "rubric_pipeline/all_wrong_group_frac": float(all_wrong_groups / max(len(groups), 1)),
            "rubric_pipeline/matrix_complete_group_frac": float(
                matrix_complete_groups / max(len(groups), 1)
            ),
            "rubric_pipeline/judge_parse_success_rate": float(
                judge_parse_successes / max(len(judge_messages), 1)
            ),
            "rubric_pipeline/reliability_mean": float(
                sum(all_reliability) / max(len(all_reliability), 1)
            ),
            "rubric_pipeline/reliability_max": float(max(all_reliability, default=0.0)),
            "rubric_pipeline/selected_reliability_mean": float(
                sum(selected_reliability) / max(len(selected_reliability), 1)
            ),
            "rubric_pipeline/reliable_items_mean_mixed": float(
                sum(filter_stats_by_uid[uid]["reliable_items"] for uid in mixed_uids)
                / max(len(mixed_uids), 1)
            ),
            "rubric_pipeline/constant_item_frac": float(
                sum(stats["constant_items"] for stats in filter_stats_by_uid.values())
                / max(sum(stats["candidate_items"] for stats in filter_stats_by_uid.values()), 1)
            ),
            "rubric_pipeline/reliable_item_frac": float(
                sum(stats["reliable_items"] for stats in filter_stats_by_uid.values())
                / max(sum(stats["candidate_items"] for stats in filter_stats_by_uid.values()), 1)
            ),
        }
        metrics.update(
            {
                f"rubric_pipeline/reject_{reason}_mean": float(count / max(len(by_uid), 1))
                for reason, count in grounding_rejection_reason_totals.items()
            }
        )
        return rollout_output, metrics

    def _rubric_snapshot_output_path(self) -> Path:
        """Return the atomic parquet target for the configured full-train snapshot."""
        rubric_config = self.config.algorithm.token_credit.rubric_pipeline
        configured = rubric_config.get("snapshot_output_path", None)
        if configured:
            return Path(os.path.expandvars(os.path.expanduser(str(configured)))).resolve()
        step = int(rubric_config.get("snapshot_step", 300))
        return (
            Path(str(self.config.trainer.default_local_dir)).resolve()
            / "rubric_snapshots"
            / f"train_step_{step}.parquet"
        )

    def _should_export_training_rubric_snapshot(self) -> bool:
        rubric_config = self.config.algorithm.token_credit.rubric_pipeline
        return bool(rubric_config.get("snapshot_enable", False)) and self.global_steps == int(
            rubric_config.get("snapshot_step", 300)
        ) and not self._rubric_snapshot_output_path().exists()

    def _should_stop_after_rubric_snapshot(self) -> bool:
        config = self.config.algorithm.token_credit.rubric_pipeline
        return (
            bool(config.get("snapshot_enable", False))
            and bool(config.get("snapshot_stop_after_export", False))
            and self.global_steps >= int(config.get("snapshot_step", 300))
            and self._rubric_snapshot_output_path().exists()
        )

    def _export_training_rubric_snapshot(self) -> dict[str, float]:
        """Generate one complete rubric bundle for every training-set row.

        Rows remain in the exact dataset order.  ``rubric`` stores the complete
        numbered 5--8 criterion bundle; it is never exploded into individual
        criteria.  Successful rows are checkpointed to a JSONL sidecar after
        every batch so an interrupted sweep resumes only the missing rows. The
        final parquet is published atomically only after every row is valid.
        """
        rubric_config = self.config.algorithm.token_credit.rubric_pipeline
        output_path = self._rubric_snapshot_output_path()
        if output_path.exists():
            return {
                "rubric_snapshot/exported": 0.0,
                "rubric_snapshot/task_count": float(len(self.train_dataset)),
                "rubric_snapshot/valid_frac": 1.0,
            }
        if str(rubric_config.get("backend", "")) != "actor_rollout_group_hints":
            raise RuntimeError("Full training rubric snapshots require actor_rollout_group_hints")
        dataframe = getattr(self.train_dataset, "dataframe", None)
        if dataframe is None:
            raise RuntimeError("Training dataset does not expose its source dataframe for snapshot export")

        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        row_count = len(self.train_dataset)
        batch_size = int(rubric_config.get("snapshot_batch_size", 8))
        max_attempts = int(rubric_config.get("snapshot_max_generation_attempts", 4))
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        selected_rubrics = [""] * row_count
        selected_counts = [0] * row_count
        output_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path = output_path.with_suffix(".progress.jsonl")

        progress_by_row: dict[int, tuple[str, int]] = {}
        if progress_path.exists():
            for line_number, line in enumerate(progress_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if int(record.get("snapshot_step", -1)) != self.global_steps:
                        raise ValueError("snapshot step does not match the current checkpoint")
                    row_index = int(record["row_index"])
                    rubric_text = str(record["rubric"]).strip()
                    item_count = int(record["selected_count"])
                    if not 0 <= row_index < row_count:
                        raise ValueError(f"row index {row_index} is outside [0, {row_count})")
                    if not rubric_text or not 5 <= item_count <= 8:
                        raise ValueError("record does not contain a complete 5--8 criterion bundle")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(f"Ignoring invalid rubric snapshot progress line {line_number}: {exc}")
                    continue
                progress_by_row[row_index] = (rubric_text, item_count)

            # Canonicalize the sidecar after loading so a truncated final line
            # from an interrupted append cannot corrupt future records.
            progress_temporary_path = progress_path.with_name(
                f".{progress_path.name}.{uuid.uuid4().hex}.tmp"
            )
            with progress_temporary_path.open("w", encoding="utf-8") as progress_file:
                for row_index, (rubric_text, item_count) in sorted(progress_by_row.items()):
                    progress_file.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "snapshot_step": self.global_steps,
                                "row_index": row_index,
                                "rubric": rubric_text,
                                "selected_count": item_count,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                progress_file.flush()
                os.fsync(progress_file.fileno())
            os.replace(progress_temporary_path, progress_path)

        for row_index, (rubric_text, item_count) in progress_by_row.items():
            selected_rubrics[row_index] = rubric_text
            selected_counts[row_index] = item_count
        resumed_count = len(progress_by_row)

        def append_progress(row_indices: list[int]) -> None:
            if not row_indices:
                return
            with progress_path.open("a", encoding="utf-8") as progress_file:
                for row_index in row_indices:
                    progress_file.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "snapshot_step": self.global_steps,
                                "row_index": row_index,
                                "rubric": selected_rubrics[row_index],
                                "selected_count": selected_counts[row_index],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                progress_file.flush()
                os.fsync(progress_file.fileno())

        print(
            f"Starting step-{self.global_steps} full-train rubric snapshot: "
            f"rows={row_count}, batch_size={batch_size}, rollouts_per_task={rollout_n}, "
            f"resumed_rows={resumed_count}, output={output_path}"
        )
        for start in tqdm(range(0, row_count, batch_size), desc="Rubric snapshot"):
            stop = min(start + batch_size, row_count)
            batch_indices = [index for index in range(start, stop) if not selected_rubrics[index]]
            if not batch_indices:
                continue
            batch_dict = default_collate_fn([self.train_dataset[index] for index in batch_indices])
            batch = DataProto.from_single_dict(batch_dict)
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            batch.non_tensor_batch["uid"] = np.array(
                [f"rubric_snapshot_step_{self.global_steps}_row_{index}" for index in batch_indices],
                dtype=object,
            )
            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = self.global_steps
            repeated = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
            metadata = {
                key: repeated.non_tensor_batch[key]
                for key in ("uid", "raw_prompt", "data_source")
                if key in repeated.non_tensor_batch
            }
            expected_uids = [
                f"rubric_snapshot_step_{self.global_steps}_row_{index}"
                for index in batch_indices
                for _ in range(rollout_n)
            ]
            if list(metadata.get("uid", [])) != expected_uids:
                raise RuntimeError("Snapshot rollout repeat changed the task-to-row ordering")
            # Keep the rollout replicas awake for the complete dataset sweep.
            # Sleeping here would offload vLLM's weights and KV-cache buffers
            # after every snapshot batch, while the next batch does not perform
            # an update_weights()/wake_up transition.  The following request
            # would then execute Triton block-table kernels on offloaded CPU
            # tensors and kill the EngineCore.
            generated = self.async_rollout_manager.generate_sequences(repeated)
            unresolved = set(range(len(batch_indices)))
            newly_completed: list[int] = []
            for _attempt in range(max_attempts):
                generated, _ = self._generate_online_actor_rubrics(
                    generated,
                    rollout_metadata=metadata,
                    allow_uncorrelated_fallback=True,
                    snapshot_single_bundle=True,
                )
                rubric_values = generated.non_tensor_batch["online_rubric"]
                valid_values = generated.non_tensor_batch["online_rubric_valid"]
                item_counts = generated.non_tensor_batch["online_rubric_item_count"]
                for local_index in list(unresolved):
                    repeated_index = local_index * rollout_n
                    text = str(rubric_values[repeated_index]).strip()
                    count = int(item_counts[repeated_index])
                    group_slice = slice(repeated_index, repeated_index + rollout_n)
                    if any(str(value).strip() != text for value in rubric_values[group_slice]):
                        raise RuntimeError("Snapshot rubric bundle differs inside one task rollout group")
                    if any(int(value) != count for value in item_counts[group_slice]):
                        raise RuntimeError("Snapshot rubric item count differs inside one task rollout group")
                    if bool(valid_values[repeated_index]) and text and 5 <= count <= 8:
                        row_index = batch_indices[local_index]
                        selected_rubrics[row_index] = text
                        selected_counts[row_index] = count
                        newly_completed.append(row_index)
                        unresolved.remove(local_index)
                if not unresolved:
                    break
            append_progress(newly_completed)
            if unresolved:
                missing = [batch_indices[index] for index in sorted(unresolved)]
                print(
                    "Rubric snapshot deferred rows that did not produce a complete 5--8 criterion "
                    f"bundle after {max_attempts} attempts: {missing}"
                )

        invalid_rows = [
            index
            for index, (text, count) in enumerate(zip(selected_rubrics, selected_counts, strict=True))
            if not text or not 5 <= count <= 8
        ]
        if invalid_rows:
            raise RuntimeError(
                "Refusing to publish incomplete rubric snapshot; completed rows were saved to "
                f"{progress_path}. Restart from the step-{self.global_steps} checkpoint to retry only "
                f"invalid rows={invalid_rows}"
            )

        collision_columns = [
            name
            for name in (
                "rubric",
                "rubric_valid",
                "rubric_selected_count",
                "rubric_snapshot_step",
                "rubric_snapshot_row_index",
            )
            if name in dataframe.column_names
        ]
        enriched = dataframe.remove_columns(collision_columns) if collision_columns else dataframe
        enriched = enriched.add_column("rubric", selected_rubrics)
        enriched = enriched.add_column("rubric_valid", [True] * row_count)
        enriched = enriched.add_column("rubric_selected_count", selected_counts)
        enriched = enriched.add_column("rubric_snapshot_step", [self.global_steps] * row_count)
        enriched = enriched.add_column("rubric_snapshot_row_index", list(range(row_count)))

        temporary_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            enriched.to_parquet(str(temporary_path))
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "snapshot_step": self.global_steps,
                    "task_count": row_count,
                    "rubric_bundle_size_min": min(selected_counts),
                    "rubric_bundle_size_max": max(selected_counts),
                    "rubric_parquet": str(output_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if progress_path.exists():
            progress_path.unlink()
        print(f"Saved complete training rubric bundles: {output_path}")
        return {
            "rubric_snapshot/exported": 1.0,
            "rubric_snapshot/task_count": float(row_count),
            "rubric_snapshot/valid_frac": 1.0,
            "rubric_snapshot/resumed_task_count": float(resumed_count),
        }

    def _compute_outcome_contrastive_js(
        self, batch: DataProto
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Compute Experiment-12 current-actor JS on eligible partial-solve groups."""
        if self.use_legacy_worker_impl == "disable":
            raise NotImplementedError(
                "outcome_contrastive_js currently requires the legacy FSDP ActorRolloutRefWorker"
            )
        if "raw_prompt" not in batch.non_tensor_batch:
            raise KeyError(
                "Outcome-contrastive token credit requires raw_prompt in the rollout batch. "
                f"Available non-tensor keys: {list(batch.non_tensor_batch.keys())}"
            )

        from verl.trainer.ppo.outcome_contrastive_credit import (
            build_outcome_contrastive_inputs,
            decode_masked_responses,
            select_outcome_contrastive_references,
        )

        config = self.config.algorithm.token_credit
        scores = batch.batch["token_level_scores"].sum(dim=-1)
        positive_indices, negative_indices, active, route_metrics = select_outcome_contrastive_references(
            batch.non_tensor_batch["uid"],
            scores,
            success_threshold=config.get("outcome_success_threshold", 0.5),
            min_correct=config.get("outcome_min_correct", 2),
            seed=self.global_steps,
        )

        response_mask = batch.batch["response_mask"]
        full_divergence = torch.zeros_like(response_mask, dtype=torch.float32)
        active = active.to(device=response_mask.device)
        active_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            return full_divergence, active, route_metrics

        decoded_responses = decode_masked_responses(batch, self.tokenizer)
        active_index_list = active_indices.cpu().tolist()
        positive_references = [decoded_responses[positive_indices[index].item()] for index in active_index_list]
        negative_references = [decoded_responses[negative_indices[index].item()] for index in active_index_list]
        active_batch = batch[active_index_list]
        contrastive_inputs = build_outcome_contrastive_inputs(
            active_batch,
            self.tokenizer,
            positive_references,
            negative_references,
            teacher_max_prompt_length=config.get("outcome_teacher_max_prompt_length", 10240),
            teacher_enable_thinking=config.get("outcome_teacher_enable_thinking", False),
        )
        contrastive_batch = DataProto.from_dict(
            tensors=contrastive_inputs,
            meta_info={
                "micro_batch_size": config.get("outcome_micro_batch_size_per_gpu", 1),
                "distribution_temperature": config.get("outcome_distribution_temperature", 1.0),
                "js_chunk_size": config.get("outcome_js_chunk_size", 128),
            },
        )
        padded_batch, pad_size = pad_dataproto_to_divisor(
            contrastive_batch,
            self.actor_rollout_wg.world_size,
        )
        padded_output = self.actor_rollout_wg.compute_outcome_contrastive_js(padded_batch)
        output = unpad_dataproto(padded_output, pad_size=pad_size)
        active_divergence = output.batch["outcome_contrastive_js"].to(full_divergence.device)
        full_divergence[active_indices] = active_divergence
        route_metrics["outcome_contrastive/reference_chars_positive_mean"] = sum(
            len(reference) for reference in positive_references
        ) / len(positive_references)
        route_metrics["outcome_contrastive/reference_chars_negative_mean"] = sum(
            len(reference) for reference in negative_references
        ) / len(negative_references)
        return full_divergence, active, route_metrics

    def _compute_old_log_prob(self, batch: DataProto):
        self._configure_turn_replay(batch)
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        from verl.trainer.ppo.rubric_generator_loss import (
            REPLAY_KEY,
            SETTINGS_KEY,
            has_signal,
            linear_warmup,
            loss_settings,
            post_start_linear_decay,
            validate_execution,
        )

        rollout_config = self.config.actor_rollout_ref.rollout
        token_credit = self.config.algorithm.get("token_credit") or {}
        rubric_config = token_credit.get("rubric_pipeline") or {}
        generator_settings = (
            loss_settings(rubric_config)
            if token_credit.get("enable", False) and rubric_config.get("enable", False) else None
        )
        # Do not let stale metadata activate an auxiliary loss in a baseline.
        batch.meta_info.pop(SETTINGS_KEY, None)
        if generator_settings is not None:
            global_step = getattr(self, "global_steps", 1)
            total_steps = getattr(self, "total_training_steps", 1)
            if generator_settings["linear_warmup_enable"]:
                schedule_scale, progress = linear_warmup(
                    global_step,
                    generator_settings["linear_warmup_steps"],
                    generator_settings["linear_warmup_initial_scale"],
                )
            elif generator_settings["linear_decay_enable"]:
                schedule_scale, progress = post_start_linear_decay(
                    global_step,
                    total_steps,
                    generator_settings["linear_decay_start_step"],
                    generator_settings["linear_decay_final_scale"],
                )
            else:
                schedule_scale, progress = 1.0, 0.0
            generator_settings["base_coef"] = generator_settings["coef"]
            generator_settings["coef"] *= schedule_scale
            generator_settings["schedule_scale"] = schedule_scale
            generator_settings["schedule_progress"] = progress
            generator_settings["decay_progress"] = (
                progress if generator_settings["linear_decay_enable"] else 0.0
            )
            generator_settings["warmup_progress"] = (
                progress if generator_settings["linear_warmup_enable"] else 0.0
            )
            validate_execution(generator_settings, self.config.actor_rollout_ref.actor,
                               rollout_config, self.use_legacy_worker_impl)
            records = batch.non_tensor_batch.get(REPLAY_KEY)
            if records is None or len(records) != len(batch):
                raise ValueError("Enabled rubric generator loss requires row-aligned generation replay records")
            # Shared by every DP rank. A local zero-advantage rank MUST still
            # execute the same FSDP forward/backward calls as an active rank.
            # At the final decay step the coefficient can be exactly zero.
            # Skip replay forwards/backwards because they cannot contribute a
            # gradient, while retaining the scheduled coefficient in metrics.
            generator_settings["has_signal"] = generator_settings["coef"] > 0 and has_signal(records)
            batch.meta_info[SETTINGS_KEY] = generator_settings
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        self._configure_turn_replay(batch)
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        if self.config.get("credit_assignment", {}).get("type") == "attention_traceback":
            pipeline = self.config.algorithm.token_credit.rubric_pipeline
            if (not self.config.trainer.get("val_only") or not self.config.trainer.get("val_before_train")
                    or self.config.trainer.get("resume_mode") != "disable"
                    or self.config.algorithm.token_credit.enable
                    or any(pipeline.get(key, False) for key in
                           ("enable", "pool_enable", "pool_generator_loss_enable", "snapshot_enable"))):
                raise RuntimeError("Unsafe traceback configuration: use the diagnostic entry point")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # If an export was interrupted after the step checkpoint was saved,
        # resume the sweep from that checkpoint before the next actor update.
        if self._should_export_training_rubric_snapshot():
            terminal_snapshot_recovery = self.global_steps >= self.total_training_steps
            if (
                terminal_snapshot_recovery
                and self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
            ):
                # The original terminal validation may have completed but its
                # metrics were not committed when the later snapshot raised.
                # Recompute and log them before the long recovery sweep so a
                # second export interruption cannot lose step-N validation.
                recovered_val_metrics = self._validate()
                assert recovered_val_metrics, f"{recovered_val_metrics=}"
                pprint(f"Recovered final validation metrics: {recovered_val_metrics}")
                logger.log(data=recovered_val_metrics, step=self.global_steps)
            resumed_snapshot_metrics = self._export_training_rubric_snapshot()
            logger.log(data=resumed_snapshot_metrics, step=self.global_steps)
            # A checkpoint at the configured terminal step has no remaining
            # optimizer work.  Finish after recovering the missing artifact
            # instead of entering the dataloader and performing step N+1.
            if terminal_snapshot_recovery:
                return

        # Also stop on restart when the artifact was already completed.
        if self._should_stop_after_rubric_snapshot():
            return

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        if self.config.get("credit_assignment", {}).get("type") == "attention_traceback":
            raise RuntimeError("Traceback diagnostic refused to enter training: validation did not return")

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                online_rubric_metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                # AgentLoopWorker returns generated tensors, rewards, and
                # trajectory extras, but intentionally does not echo dataset
                # metadata.  Online rubric generation runs before the normal
                # repeated training batch is unioned with that output, so keep
                # the row-aligned grouping/problem metadata separately.
                online_rubric_metadata = None
                if self._uses_online_actor_rubrics():
                    online_rubric_metadata = {
                        key: gen_batch_output.non_tensor_batch[key]
                        for key in ("uid", "raw_prompt", "data_source")
                        if key in gen_batch_output.non_tensor_batch
                    }

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile(global_step=self.global_steps)
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            if self._uses_online_actor_rubrics():
                                with marked_timer("online_actor_rubric", timing_raw, color="magenta"):
                                    gen_batch_output, online_rubric_metrics = self._generate_online_actor_rubrics(
                                        gen_batch_output,
                                        rollout_metadata=online_rubric_metadata,
                                    )
                                    pool_expected = bool(
                                        self.config.algorithm.token_credit.rubric_pipeline.get("pool_enable", False)
                                    )
                                    if pool_expected and online_rubric_metrics.get("rubric_pool/enabled") != 1.0:
                                        raise RuntimeError(
                                            "rubric pool is enabled in the resolved config, but the online rubric "
                                            "call returned no rubric_pool/enabled sentinel; refusing to continue "
                                            "through a silent group-hint fallback"
                                        )
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    metrics.update(online_rubric_metrics)

                    # Keep a row-aligned copy of the outcome used by the
                    # rubric pipeline.  It travels through batch balancing and
                    # is used only for rubric-gap diagnostics; GRPO continues
                    # to consume the continuous/shaped environment reward.
                    if self._uses_online_actor_rubrics():
                        batch.batch["rubric_outcome_scores"] = torch.tensor(
                            [
                                self._rubric_outcome_score(batch.non_tensor_batch, index)
                                for index in range(len(batch))
                            ],
                            dtype=torch.float32,
                            device=batch.batch["responses"].device,
                        )

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, reward_for_val=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                debug_metrics = calculate_debug_metrics(batch)
                                metrics.update(debug_metrics)
                                if self._turn_replay_enabled(batch):
                                    replay_mask = batch.batch["response_mask"].bool()
                                    replay_delta = (
                                        batch.batch["old_log_probs"].float()
                                        - batch.batch["rollout_log_probs"].float()
                                    )[replay_mask]
                                    metrics["turn_replay/token_alignment_valid"] = 1.0
                                    metrics["turn_replay/rollout_old_logprob_abs_diff_mean"] = float(
                                        replay_delta.abs().mean().item() if replay_delta.numel() else 0.0
                                    )
                                    metrics["turn_replay/rollout_old_logprob_abs_diff_max"] = float(
                                        replay_delta.abs().max().item() if replay_delta.numel() else 0.0
                                    )
                                    metrics["turn_replay/rollout_old_logprob_signed_diff_mean"] = float(
                                        replay_delta.mean().item() if replay_delta.numel() else 0.0
                                    )

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    token_credit_config = self.config.algorithm.get("token_credit", None)
                    token_credit_enabled = token_credit_config is not None and token_credit_config.get("enable", False)
                    token_credit_mode = (
                        token_credit_config.get("mode", "additive_gap") if token_credit_enabled else None
                    )
                    sampled_gap_mode = token_credit_mode in (
                        "additive_gap",
                        "centered_opd",
                        "rank_attenuation",
                        "polarity_gap",
                        "opsd_reverse_kl",
                    )
                    teacher_key = (
                        token_credit_config.get("teacher_log_prob_key", "ref_log_prob")
                        if token_credit_enabled and sampled_gap_mode
                        else None
                    )
                    rubric_config = token_credit_config.get("rubric_pipeline", None) if token_credit_enabled else None
                    rubric_enabled = (
                        sampled_gap_mode
                        and rubric_config is not None
                        and rubric_config.get("enable", False)
                    )
                    expected_rubric_teacher_key = (
                        "rubric_teacher_log_probs"
                        if token_credit_mode == "opsd_reverse_kl"
                        else "rubric_actor_log_probs"
                    )
                    if rubric_enabled and teacher_key != expected_rubric_teacher_key:
                        raise ValueError(
                            "Rubric-conditioned token credit received the wrong teacher batch key. "
                            "Set algorithm.token_credit.teacher_log_prob_key="
                            f"{expected_rubric_teacher_key!r}; "
                            f"got {teacher_key!r}."
                        )

                    needs_standard_reference = (
                        self.config.algorithm.use_kl_in_reward
                        or self.config.actor_rollout_ref.actor.use_kl_loss
                        or teacher_key == "ref_log_prob"
                    )
                    if self.use_reference_policy and needs_standard_reference:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if rubric_enabled:
                        with marked_timer("rubric_actor", timing_raw, color="magenta"):
                            rubric_teacher_log_probs, rubric_eligible, rubric_metrics = (
                                self._compute_rubric_actor_log_prob(batch)
                            )
                            batch.batch[expected_rubric_teacher_key] = rubric_teacher_log_probs
                            batch.batch["rubric_teacher_eligible"] = rubric_eligible
                            metrics.update(rubric_metrics)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if token_credit_enabled and token_credit_mode == "outcome_contrastive_js":
                            with marked_timer("outcome_contrastive", timing_raw, color="magenta"):
                                contrastive_js, contrastive_active, contrastive_metrics = (
                                    self._compute_outcome_contrastive_js(batch)
                                )
                            js_key = token_credit_config.get("outcome_js_key", "outcome_contrastive_js")
                            active_key = token_credit_config.get(
                                "outcome_active_mask_key", "outcome_contrastive_active"
                            )
                            batch.batch[js_key] = contrastive_js
                            batch.batch[active_key] = contrastive_active
                            metrics.update(contrastive_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        decay_start_step = (
                            token_credit_config.get("linear_decay_start_step", 0)
                            if token_credit_config is not None
                            else 0
                        )
                        warmup_steps = (
                            token_credit_config.get("linear_warmup_steps", 0)
                            if token_credit_config is not None
                            else 0
                        )
                        decay_enable = (
                            token_credit_config.get("linear_decay_enable", False)
                            if token_credit_config is not None
                            else False
                        )
                        warmup_enable = (
                            token_credit_config.get("linear_warmup_enable", False)
                            if token_credit_config is not None
                            else False
                        )
                        if (
                            isinstance(decay_start_step, bool)
                            or int(decay_start_step) != decay_start_step
                            or decay_start_step < 0
                        ):
                            raise ValueError("token_credit.linear_decay_start_step must be a nonnegative integer")
                        if (
                            isinstance(warmup_steps, bool)
                            or int(warmup_steps) != warmup_steps
                            or warmup_steps < int(warmup_enable)
                        ):
                            raise ValueError(
                                "token_credit.linear_warmup_steps must be positive when warmup is enabled"
                            )
                        if decay_enable and warmup_enable:
                            raise ValueError("token_credit linear decay and linear warmup are mutually exclusive")
                        from verl.trainer.ppo.rubric_generator_loss import linear_warmup, post_start_linear_decay

                        if warmup_enable:
                            schedule_scale, schedule_progress = linear_warmup(
                                self.global_steps,
                                warmup_steps,
                                token_credit_config.get("linear_warmup_initial_scale", 0.0),
                            )
                        elif decay_enable:
                            schedule_scale, schedule_progress = post_start_linear_decay(
                                self.global_steps,
                                self.total_training_steps,
                                decay_start_step,
                                token_credit_config.get("linear_decay_final_scale", 0.0),
                            )
                        else:
                            schedule_scale, schedule_progress = 1.0, 0.0
                        batch.meta_info["token_credit_schedule_scale"] = schedule_scale
                        batch.meta_info["token_credit_training_progress"] = schedule_progress
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        if self.config.get("credit_assignment", {}).get("type") == "attention_traceback_train":
                            from verl.experimental.attention_traceback.training import apply_online_credit
                            with marked_timer("attention_credit", timing_raw, color="magenta"):
                                metrics.update(apply_online_credit(
                                    batch, self.actor_rollout_wg, self.config, self.global_steps))
                        token_credit_metrics = batch.meta_info.pop("token_credit_metrics", None)
                        if token_credit_metrics is not None:
                            metrics.update(token_credit_metrics)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # Decide before advancing global_steps.  The export itself is
                # deliberately deferred until after this step's metrics and
                # validation have been committed: a long or interrupted
                # snapshot must not hide the terminal validation in W&B.
                rubric_snapshot_due = self._should_export_training_rubric_snapshot()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                from verl.trainer.ppo.analysis_logging import maybe_write_training_analysis

                maybe_write_training_analysis(self.config, batch, metrics, self.global_steps)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.refresh()

                # Checkpointing has already completed above, and the terminal
                # validation plus ordinary step metrics are now durable.  Run
                # the optional full-dataset export last and log its result as
                # a separate update at the same training step.
                if rubric_snapshot_due:
                    snapshot_metrics = self._export_training_rubric_snapshot()
                    logger.log(data=snapshot_metrics, step=self.global_steps)

                if self._should_stop_after_rubric_snapshot():
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    progress_bar.close()
                    return

                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
