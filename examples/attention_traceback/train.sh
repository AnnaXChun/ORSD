#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BENCHMARK="${1:-webshop}"
MODE="${2:-train}"
if (( $# >= 1 )); then shift; fi
if (( $# >= 1 )); then shift; fi
source "$ROOT/examples/attention_traceback/orsd_paths.sh"
# Reuse the existing ORSD W&B credential. Do not echo this value.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# These are physical devices on this server; never inherit a preceding 0-3 run.
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export RAY_INCLUDE_DASHBOARD=False
unset RAY_ADDRESS
BATCH=8; SAMPLES=8; STEPS=300; TEST=50; VAL_N=4; SAVE=50
PROMPT=2048; RESPONSE=32768; MODEL_LEN=40960; MICRO=1
WORKERS=1; LOADER_WORKERS=1; GPU_UTIL=0.72; BATCHED_TOKENS=32768; MAX_SEQS=64
TURN_CAP=2048; MAX_TURNS=15
case "$BENCHMARK" in
  webshop)
    TRAIN="${WEBSHOP_TRAIN_PARQUET:-$WEBSHOP_SMALL_DATA_DIR/webshop_train.parquet}"
    VAL="['$WEBSHOP_VAL_PARQUET']"
    INTERACTION="$ROOT/examples/agent_rubric/config/webshop_interaction_config.yaml"
    export WEBSHOP_USE_SMALL=True WEBSHOP_HUMAN_GOALS=False WEBSHOP_NUM_PRODUCTS=1000
    export WEBSHOP_PRODUCTS_JSON="$WEBSHOP_ROOT/data/items_shuffle_1000.json"
    export WEBSHOP_REWARD_MODE=hybrid WEBSHOP_R2_INVALID_ACTION_COEF=0.05 WEBSHOP_R2_REPEAT_ACTION_COEF=0.025
    REFERENCE=examples/agent_rubric2/run_webshop_small_qwen3_1_7b_orsd_opd0p001_4gpu.sh
    ;;
  alfworld)
    TRAIN="${ALFWORLD_TRAIN_PARQUET:-$ROOT/Data/agent_rubric/alfworld/alfworld_train.parquet}"
    VAL="['${ALFWORLD_VAL_SEEN_PARQUET:-$ROOT/Data/agent_rubric/alfworld/alfworld_valid_seen.parquet}','$ALFWORLD_VAL_UNSEEN_PARQUET']"
    INTERACTION="$ROOT/examples/agent_rubric/config/alfworld_interaction_config.yaml"
    export ALFWORLD_R2_REWARD_ENABLE=False ALFWORLD_ENV_POOL_ENABLE=True
    RESPONSE=16384; MODEL_LEN=32768; MICRO=2; WORKERS=4; LOADER_WORKERS=2
    GPU_UTIL=0.60; BATCHED_TOKENS=8192; MAX_SEQS=256; TURN_CAP=null; MAX_TURNS=null; SAVE=25
    REFERENCE=examples/agent_rubric2/run_alfworld_qwen3_1_7b_first_version_standalone.sh
    ;;
  *) echo 'Supported training benchmarks: webshop, alfworld. ScienceWorld interface audit remains pending.' >&2; exit 2 ;;
esac
VAL_BEFORE=True
EXTRA=()
case "$MODE" in
  train) ;;
  config) EXTRA+=(--cfg job --resolve) ;;
  smoke)
    # Two real optimizer steps test refresh of the actor snapshot after an update.
    BATCH=1; SAMPLES=4; STEPS=2; MICRO=1; WORKERS=1; LOADER_WORKERS=0
    TEST=-1; SAVE=-1; VAL_BEFORE=False
    ;;
  *) echo 'Usage: train.sh [webshop|alfworld] [train|smoke|config] [Hydra overrides...]' >&2; exit 2 ;;
esac
NAME="${EXPERIMENT_NAME:-qwen3_1_7b_${BENCHMARK}_attention_credit_b${BATCH}_n${SAMPLES}_s${STEPS}_${MODE}}"
OUTPUT="${OUTPUT_DIR:-$ROOT/checkpoints/attention_credit/$NAME}"
mkdir -p "$OUTPUT"
printf 'ORSD reference: %s\nBudget: %s prompts x %s samples x %s steps = %s trajectories\nGPUs: %s\nOutput: %s\n' "$REFERENCE" "$BATCH" "$SAMPLES" "$STEPS" "$((BATCH*SAMPLES*STEPS))" "$CUDA_VISIBLE_DEVICES" "$OUTPUT"
# All algorithm switches are resolved by the explicit training route. No rubric
# judge, generator loss, or ORSD teacher can leak in through inherited env vars.
exec python3 -m verl.trainer.main_ppo --config-name attention_traceback_train \
  "${EXTRA[@]}" \
  "data.train_files=['$TRAIN']" "data.val_files=$VAL" \
  "data.train_batch_size=$BATCH" "data.dataloader_num_workers=$LOADER_WORKERS" \
  "data.max_prompt_length=$PROMPT" "data.max_response_length=$RESPONSE" \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 "actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO" \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL" \
  "actor_rollout_ref.rollout.max_model_len=$MODEL_LEN" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=$BATCHED_TOKENS" \
  "actor_rollout_ref.rollout.max_num_seqs=$MAX_SEQS" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.enable_prefix_caching=True \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO" \
  "actor_rollout_ref.rollout.n=$SAMPLES" "actor_rollout_ref.rollout.agent.num_workers=$WORKERS" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$INTERACTION" \
  "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=null \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
  actor_rollout_ref.rollout.multi_turn.history_window=0 \
  actor_rollout_ref.rollout.multi_turn.history_fixed_prefix_messages=3 \
  actor_rollout_ref.rollout.multi_turn.history_response_mode=action_only \
  ++actor_rollout_ref.rollout.multi_turn.history_min_generation_tokens=2048 \
  "actor_rollout_ref.rollout.multi_turn.max_response_tokens_per_turn=$TURN_CAP" \
  "actor_rollout_ref.rollout.multi_turn.turn_replay_micro_batch_size=$MICRO" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO" actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=True algorithm.use_kl_in_reward=False \
  reward_model.enable=False reward_model.use_reward_loop=False \
  trainer.critic_warmup=0 'trainer.logger=[console,wandb]' trainer.project_name=attention_credit \
  "trainer.experiment_name=$NAME" "trainer.default_local_dir=$OUTPUT" \
  trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.total_epochs=5 \
  "trainer.total_training_steps=$STEPS" "trainer.save_freq=$SAVE" "trainer.test_freq=$TEST" \
  "trainer.val_before_train=$VAL_BEFORE" trainer.val_only=False \
  ++ray_kwargs.ray_init.include_dashboard=False ++ray_kwargs.ray_init.address=local \
  "$@"
