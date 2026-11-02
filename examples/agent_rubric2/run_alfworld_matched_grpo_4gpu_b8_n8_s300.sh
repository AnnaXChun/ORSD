#!/usr/bin/env bash
set -euo pipefail

# Matched control for the ALFWorld rubric-pool experiment. It keeps the same
# environment reward, prompt batch, rollout count, actor settings, hardware,
# and 300-step budget while disabling every rubric-derived training signal.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"
export ALFWORLD_R2_REWARD_ENABLE=False

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _ALFWORLD_GRPO_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_ALFWORLD_GRPO_VISIBLE_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export N_GPUS=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ALFWORLD_4GPU_VLLM_GPU_UTIL:-0.50}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ALFWORLD_4GPU_BATCHED_TOKENS:-32768}"
export ROLLOUT_MAX_NUM_SEQS="${ALFWORLD_4GPU_MAX_NUM_SEQS:-64}"
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True

export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export AGENT_LOOP_NUM_WORKERS="${ALFWORLD_4GPU_AGENT_LOOP_WORKERS:-8}"
export DATALOADER_NUM_WORKERS="${ALFWORLD_4GPU_DATALOADER_WORKERS:-2}"
export MICRO_BATCH_SIZE=2
export HISTORY_WINDOW=0
export ACTOR_THINKING="${ACTOR_THINKING:-True}"

export _ALFWORLD_MATCHED_ENTRY=1
export TOKEN_CREDIT_ENABLE=False
export TOKEN_CREDIT_MODE=additive_gap
export TOKEN_CREDIT_ALPHA=0.01
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export RUBRIC_PIPELINE_ENABLE=False
export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_LOSS_COEF=0.0
export RUBRIC_GENERATOR_LINEAR_DECAY_ENABLE=False
export RUBRIC_GENERATOR_LINEAR_WARMUP_ENABLE=False
export RUBRIC_GENERATOR_THINKING=False

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export ALFWORLD_MATCHED_EXPERIMENT_NAME="${ALFWORLD_MATCHED_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_grpo_4gpu_b8_n8_official_reward_s300}"
export OUTPUT_DIR="${ALFWORLD_GRPO_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${ALFWORLD_MATCHED_EXPERIMENT_NAME}}"

echo "ALFWorld matched GRPO: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, total=$((TRAIN_BATCH_SIZE * ROLLOUT_N)), steps=${MAX_STEPS}, official reward"
echo "Rubric pipeline=off, token credit=off, generator loss=off"
echo "Actor profile: thinking=${ACTOR_THINKING}, history_window=${HISTORY_WINDOW}, micro_batch=${MICRO_BATCH_SIZE}, agent_workers=${AGENT_LOOP_NUM_WORKERS}"
echo "Four-GPU profile: devices=${CUDA_VISIBLE_DEVICES}, TP=${ROLLOUT_TP}, vLLM_util=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "Experiment: ${ALFWORLD_MATCHED_EXPERIMENT_NAME}; checkpoints=${OUTPUT_DIR}"

exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_group_hint_grpo.sh" \
  algorithm.token_credit.enable=False \
  algorithm.token_credit.mode=additive_gap \
  algorithm.token_credit.linear_decay_enable=False \
  algorithm.token_credit.linear_warmup_enable=False \
  algorithm.token_credit.rubric_pipeline.enable=False \
  algorithm.token_credit.rubric_pipeline.pool_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_enable=False \
  trainer.total_training_steps=300 \
  "$@"
