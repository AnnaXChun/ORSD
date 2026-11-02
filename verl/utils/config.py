# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def _validate_rubric_pipeline_backend(rubric_pipeline: DictConfig | dict) -> str:
    """Validate only resources actually consumed by the selected backend."""

    backend = str(rubric_pipeline.get("backend", "http"))
    valid_backends = {"http", "dataset", "actor_rollout", "actor_rollout_group_hints"}
    if backend not in valid_backends:
        raise ValueError(
            f"algorithm.token_credit.rubric_pipeline.backend must be one of {sorted(valid_backends)}, "
            f"got {backend!r}"
        )
    if backend == "http":
        if not rubric_pipeline.get("model", None):
            raise ValueError("HTTP rubric pipeline requires algorithm.token_credit.rubric_pipeline.model")
        if not rubric_pipeline.get("base_url", None):
            raise ValueError("HTTP rubric pipeline requires algorithm.token_credit.rubric_pipeline.base_url")
        if rubric_pipeline.get("max_workers", 0) <= 0:
            raise ValueError("HTTP rubric pipeline requires rubric_pipeline.max_workers to be positive")
    if rubric_pipeline.get("pool_enable", False):
        from verl.trainer.ppo.rubric_pool import validate_pool_config

        validate_pool_config(rubric_pipeline)
    if backend == "actor_rollout_group_hints" and rubric_pipeline.get(
        "group_hint_select_by_correlation", False
    ):
        candidate_count = int(rubric_pipeline.get("group_hint_candidate_count", 1))
        if candidate_count < 2:
            raise ValueError("group-hint correlation selection requires group_hint_candidate_count >= 2")
        if rubric_pipeline.get("pool_enable", False):
            raise ValueError("group-hint bundle selection and rubric pool cannot be enabled together")
        if rubric_pipeline.get("pool_generator_loss_enable", False):
            raise ValueError("group-hint bundle selection does not support generator loss")
        from verl.trainer.ppo.rubric_pool import validate_pool_config

        judge_config = dict(rubric_pipeline)
        judge_config["pool_enable"] = True
        validate_pool_config(judge_config)
    evidence_order = str(rubric_pipeline.get("group_hint_evidence_order", "outcome_anchored_random"))
    if evidence_order not in {"outcome_anchored_random", "legacy_alternating_extremes"}:
        raise ValueError(f"Unsupported group_hint_evidence_order: {evidence_order!r}")
    prompt_style = str(rubric_pipeline.get("group_hint_prompt_style", "process_rubric_5_to_k"))
    if prompt_style not in {"process_rubric_5_to_k", "legacy_process_rubric_3_to_k"}:
        raise ValueError(f"Unsupported group_hint_prompt_style: {prompt_style!r}")
    guarded_selection = bool(rubric_pipeline.get("group_hint_guarded_selection", False))
    force_first_candidate = bool(
        rubric_pipeline.get("group_hint_force_first_candidate", False)
    )
    if force_first_candidate and rubric_pipeline.get("group_hint_select_by_correlation", False):
        raise ValueError(
            "group_hint_force_first_candidate cannot be combined with correlation selection"
        )
    if guarded_selection:
        if backend != "actor_rollout_group_hints":
            raise ValueError("guarded group-hint selection requires backend='actor_rollout_group_hints'")
        if not rubric_pipeline.get("group_hint_select_by_correlation", False):
            raise ValueError("guarded group-hint selection requires group_hint_select_by_correlation=True")
        min_correlation = float(rubric_pipeline.get("group_hint_min_selected_correlation", 0.0))
        min_improvement = float(rubric_pipeline.get("group_hint_min_correlation_improvement", 0.0))
        if not math.isfinite(min_correlation):
            raise ValueError("group_hint_min_selected_correlation must be finite")
        if not math.isfinite(min_improvement) or min_improvement < 0:
            raise ValueError("group_hint_min_correlation_improvement must be finite and nonnegative")
    if rubric_pipeline.get("snapshot_enable", False):
        if backend != "actor_rollout_group_hints":
            raise ValueError("rubric snapshot export requires backend='actor_rollout_group_hints'")
        for key in ("snapshot_step", "snapshot_batch_size", "snapshot_max_generation_attempts"):
            value = rubric_pipeline.get(key, 0)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"rubric_pipeline.{key} must be a positive integer")
    return backend


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "reward_model": "micro_batch_size",
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    # Check for reward model micro-batch size conflicts
    if (
        config.reward_model.enable
        and not config.reward_model.use_dynamic_bsz
        and not config.reward_model.use_reward_loop
    ):
        check_mutually_exclusive(
            config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
        )

    if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    token_credit_config = config.algorithm.get("token_credit", None)
    if token_credit_config is not None and token_credit_config.get("enable", False):
        if config.algorithm.adv_estimator != "grpo":
            raise ValueError(
                "algorithm.token_credit.enable=True requires algorithm.adv_estimator=grpo, "
                f"got {config.algorithm.adv_estimator!r}"
            )
        token_credit_mode = token_credit_config.get("mode", "additive_gap")
        valid_token_credit_modes = {
            "additive_gap",
            "centered_opd",
            "rank_attenuation",
            "polarity_gap",
            "opsd_reverse_kl",
            "outcome_contrastive_js",
        }
        if token_credit_mode not in valid_token_credit_modes:
            raise ValueError(
                "algorithm.token_credit.mode must be one of "
                f"{sorted(valid_token_credit_modes)}, got {token_credit_mode!r}"
            )
        sampled_gap_mode = token_credit_mode in {
            "additive_gap", "centered_opd", "rank_attenuation", "polarity_gap", "opsd_reverse_kl"
        }
        opsd_teacher_source = token_credit_config.get("opsd_teacher_source", "current_actor")
        if token_credit_mode == "opsd_reverse_kl" and opsd_teacher_source not in {
            "current_actor", "reference_policy"
        }:
            raise ValueError(
                "algorithm.token_credit.opsd_teacher_source must be current_actor or reference_policy, "
                f"got {opsd_teacher_source!r}"
            )
        alpha = token_credit_config.get("alpha", 0.01 if token_credit_mode == "additive_gap" else 0.1)
        if token_credit_mode in {"additive_gap", "centered_opd", "polarity_gap"} and not 0.0 <= alpha < 1.0:
            raise ValueError(f"algorithm.token_credit.alpha must be in [0, 1), got {alpha}")
        decay_enable = token_credit_config.get("linear_decay_enable", False)
        warmup_enable = token_credit_config.get("linear_warmup_enable", False)
        if not isinstance(decay_enable, bool) or not isinstance(warmup_enable, bool):
            raise ValueError("token_credit linear schedule enable flags must be boolean")
        if decay_enable and warmup_enable:
            raise ValueError("token_credit linear decay and linear warmup are mutually exclusive")
        warmup_steps = token_credit_config.get("linear_warmup_steps", 0)
        if (
            isinstance(warmup_steps, bool)
            or int(warmup_steps) != warmup_steps
            or warmup_steps < int(warmup_enable)
        ):
            raise ValueError("token_credit.linear_warmup_steps must be positive when warmup is enabled")
        warmup_initial_scale = float(token_credit_config.get("linear_warmup_initial_scale", 0.0))
        if not 0 <= warmup_initial_scale <= 1:
            raise ValueError("token_credit.linear_warmup_initial_scale must be in [0, 1]")
        scale_quantile = token_credit_config.get("scale_quantile", 0.5)
        if not 0.0 < scale_quantile <= 1.0:
            raise ValueError(f"algorithm.token_credit.scale_quantile must be in (0, 1], got {scale_quantile}")
        min_scale = token_credit_config.get("min_scale", 1e-4)
        if min_scale <= 0.0:
            raise ValueError(f"algorithm.token_credit.min_scale must be positive, got {min_scale}")
        rank_attenuation_strength = token_credit_config.get("rank_attenuation_strength", 0.2)
        if token_credit_mode == "rank_attenuation" and (
            not math.isfinite(rank_attenuation_strength) or not 0.0 <= rank_attenuation_strength < 1.0
        ):
            raise ValueError(
                "algorithm.token_credit.rank_attenuation_strength must be finite and in [0, 1), "
                f"got {rank_attenuation_strength}"
            )
        rank_conflict_fraction = token_credit_config.get("rank_conflict_fraction", 0.25)
        if token_credit_mode == "rank_attenuation" and not 0.0 < rank_conflict_fraction <= 0.5:
            raise ValueError(
                "algorithm.token_credit.rank_conflict_fraction must be in (0, 0.5], "
                f"got {rank_conflict_fraction}"
            )
        rank_min_gap_range = token_credit_config.get("rank_min_gap_range", 1e-4)
        if token_credit_mode == "rank_attenuation" and (
            not math.isfinite(rank_min_gap_range) or rank_min_gap_range < 0.0
        ):
            raise ValueError(
                "algorithm.token_credit.rank_min_gap_range must be finite and nonnegative, "
                f"got {rank_min_gap_range}"
            )
        if sampled_gap_mode and token_credit_config.get(
            "teacher_log_prob_key", "ref_log_prob"
        ) == token_credit_config.get("student_log_prob_key", "old_log_probs"):
            raise ValueError("Token credit teacher_log_prob_key and student_log_prob_key must be different")

        rubric_pipeline = token_credit_config.get("rubric_pipeline", None)
        rubric_enabled = rubric_pipeline is not None and rubric_pipeline.get("enable", False)
        teacher_key = token_credit_config.get("teacher_log_prob_key", "ref_log_prob")
        active_mask_key = token_credit_config.get("active_mask_key", None)
        if rubric_enabled and not sampled_gap_mode:
            raise ValueError(
                "algorithm.token_credit.rubric_pipeline is supported only for additive_gap, centered_opd, "
                "rank_attenuation, polarity_gap, or opsd_reverse_kl"
            )
        if rubric_enabled:
            _validate_rubric_pipeline_backend(rubric_pipeline)
            from verl.trainer.ppo.rubric_generator_loss import loss_settings, validate_execution

            generator_settings = loss_settings(rubric_pipeline)
            validate_execution(
                generator_settings, config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout,
                config.trainer.get("use_legacy_worker_impl", "auto"),
            )
            if generator_settings is not None and int(rubric_pipeline.get("max_tokens", 2048)) > int(
                config.actor_rollout_ref.rollout.response_length
            ):
                raise ValueError("Rubric generator max_tokens must not exceed rollout.response_length")
            expected_teacher_key = (
                "rubric_teacher_log_probs"
                if token_credit_mode == "opsd_reverse_kl"
                else "rubric_actor_log_probs"
            )
            if teacher_key != expected_teacher_key:
                raise ValueError(
                    "Enabled rubric pipeline requires "
                    f"algorithm.token_credit.teacher_log_prob_key={expected_teacher_key} "
                    f"for mode={token_credit_mode}"
                )
            if active_mask_key != "rubric_teacher_eligible":
                inactive_behavior = (
                    "receive zero OPSD advantage"
                    if token_credit_mode == "opsd_reverse_kl"
                    else "fall back to ordinary GRPO"
                )
                raise ValueError(
                    "Enabled rubric pipeline requires "
                    "algorithm.token_credit.active_mask_key=rubric_teacher_eligible so failed generations "
                    f"{inactive_behavior}"
                )
            if rubric_pipeline.get("rubrics_per_prompt", 0) <= 0:
                raise ValueError("rubric_pipeline.rubrics_per_prompt must be positive")
            if rubric_pipeline.get("teacher_max_prompt_length", 0) <= 0:
                raise ValueError("rubric_pipeline.teacher_max_prompt_length must be positive")
        elif sampled_gap_mode and teacher_key in {"rubric_actor_log_probs", "rubric_teacher_log_probs"}:
            raise ValueError(
                "teacher_log_prob_key=rubric_actor_log_probs requires "
                "algorithm.token_credit.rubric_pipeline.enable=True"
            )

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")
