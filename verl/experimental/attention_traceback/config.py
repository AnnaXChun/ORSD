"""Explicit routing, preserving the existing ORSD route by default."""

def configure_credit_assignment(config):
    from omegaconf import OmegaConf, open_dict

    route = config.get("credit_assignment", {}).get("type", "orsd")
    if route not in {"grpo", "orsd", "attention_traceback", "attention_traceback_train"}:
        raise ValueError(f"Unknown credit_assignment.type: {route}")
    if route == "orsd":
        return
    with open_dict(config):
        credit = config.algorithm.token_credit
        credit.enable = False
        pipeline = credit.rubric_pipeline
        for key in ("enable", "pool_enable", "pool_generator_loss_enable", "snapshot_enable"):
            if key in pipeline:
                pipeline[key] = False
        if route == "grpo":
            config.algorithm.adv_estimator = "grpo"
            return
        if route == "attention_traceback_train":
            validate_training_config(config)
            config.algorithm.adv_estimator = "grpo"
            config.algorithm.use_kl_in_reward = False
            # Training uses fresh pre-update actor snapshots, never a frozen replay policy.
            model_path = str(config.actor_rollout_ref.model.path)
            config.attention_traceback.model_path = model_path
            config.attention_traceback.model_fingerprint = "initial_checkpoint_only; replay_uses_current_actor"
            config.attention_traceback.tokenizer_path = str(config.actor_rollout_ref.model.get("tokenizer_path") or model_path)
            kwargs = config.data.get("apply_chat_template_kwargs")
            config.attention_traceback.chat_template_kwargs = OmegaConf.to_container(kwargs, resolve=True) if kwargs else {}
            config.attention_traceback.rollout_engine_seed = int(config.actor_rollout_ref.rollout.get("seed", 0))
            return
        config.trainer.val_only = True
        config.trainer.val_before_train = True
        config.trainer.resume_mode = "disable"
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False
        config.actor_rollout_ref.actor.use_kl_loss = False
        config.reward_model.enable = False
        config.actor_rollout_ref.rollout.multi_turn.history_window = 0
        # One worker avoids validation padding being executed as extra episodes.
        config.actor_rollout_ref.rollout.agent.num_workers = 1
        if not config.get("attention_traceback", {}).get("capture_dir"):
            raise ValueError("attention_traceback.capture_dir must be an absolute shared server path")
        import os
        if not os.path.isabs(config.attention_traceback.capture_dir):
            raise ValueError("capture_dir must be absolute")
        model_path = config.actor_rollout_ref.model.path
        from transformers import AutoConfig
        model = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if (model.model_type, model.hidden_size, model.num_hidden_layers) != ("qwen3", 2048, 28):
            raise ValueError("Phase A only supports the existing Qwen3-1.7B checkpoint")
        if config.actor_rollout_ref.model.get("lora_rank", 0) or config.actor_rollout_ref.model.get("lora_adapter_path"):
            raise ValueError("Export a merged HF checkpoint for exact replay of LoRA policies")
        if config.actor_rollout_ref.model.get("override_config", {}):
            raise ValueError("Export model overrides into the HF checkpoint config before exact replay")
        from .capture import checkpoint_fingerprint
        config.attention_traceback.model_fingerprint = checkpoint_fingerprint(model_path)
        config.attention_traceback.model_path = str(model_path)
        kwargs = config.data.get("apply_chat_template_kwargs")
        config.attention_traceback.chat_template_kwargs = OmegaConf.to_container(kwargs, resolve=True) if kwargs is not None else {}
        config.attention_traceback.tokenizer_path = str(config.actor_rollout_ref.model.get("tokenizer_path") or model_path)
        config.attention_traceback.rollout_engine_seed = int(config.actor_rollout_ref.rollout.get("seed", 0))


def validate_training_config(config):
    import math
    c = config.attention_traceback
    if config.trainer.get("use_legacy_worker_impl") != "enable":
        raise ValueError("Traceback training currently requires the legacy FSDP worker")
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"} or config.trainer.nnodes != 1:
        raise ValueError("Traceback training supports single-node FSDP/FSDP2 only")
    if config.actor_rollout_ref.rollout.multi_turn.history_window != 0:
        raise ValueError("Lossless traceback training requires full history")
    if config.actor_rollout_ref.model.get("lora_rank", 0) or config.actor_rollout_ref.model.get("lora_adapter_path"):
        raise ValueError("Traceback training requires a full-parameter actor")
    if config.actor_rollout_ref.model.get("override_config", {}):
        raise ValueError("Model config overrides are unsupported for snapshot replay")
    from transformers import AutoConfig
    model = AutoConfig.from_pretrained(config.actor_rollout_ref.model.path, local_files_only=True)
    if (model.model_type, model.hidden_size, model.num_hidden_layers) != ("qwen3", 2048, 28):
        raise ValueError("Traceback training supports Qwen3-1.7B only")
    if not 0 <= c.alpha < 1 or not 0 < c.depth_decay <= 1:
        raise ValueError("Require 0 <= alpha < 1 and 0 < depth_decay <= 1")
    if not math.isfinite(c.delta_threshold) or c.delta_threshold < 0 or c.max_trace_depth < 1:
        raise ValueError("A finite nonnegative threshold and positive depth are required")
    if c.selection not in {"attention", "previous", "random"}:
        raise ValueError("Unknown traceback selection control")
    if c.intervention != "mask_span" or c.mask_query_origin != "action":
        raise ValueError("Training MVP uses action-origin mask_span, as in the primary audit")
    devices = str(c.replay_visible_devices).split(",")
    if len(devices) != 4 or set(devices) != {"4", "5", "6", "7"}:
        raise ValueError("This experiment reserves only physical GPUs 4,5,6,7")
    if c.workers != 4 or config.trainer.n_gpus_per_node != 4:
        raise ValueError("ORSD-matched training requires four GPUs and four replay workers")
