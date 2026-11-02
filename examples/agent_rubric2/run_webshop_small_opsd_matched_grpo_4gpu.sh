#!/usr/bin/env bash
set -euo pipefail

# Matched GRPO control for run_webshop_small_opsd_reverse_kl_4gpu.sh.
# It intentionally reads the same enriched parquet so task order and sampling
# population are identical; rubric columns remain hidden and unused.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPSD_RUBRIC_PARQUET:?Set OPSD_RUBRIC_PARQUET to the same saved-rubric parquet used by OPSD}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WEBSHOP_TRAIN_PARQUET="${OPSD_RUBRIC_PARQUET}"
export WEBSHOP_REWARD_MODE=hybrid
export WEBSHOP_R2_INVALID_ACTION_COEF=0.05
export WEBSHOP_R2_REPEAT_ACTION_COEF=0.025

export N_GPUS=4
export ROLLOUT_TP=1
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export TEST_FREQ=50
export SAVE_FREQ=50
export MICRO_BATCH_SIZE=1
export LEARNING_RATE=1e-6
export HISTORY_WINDOW=4
export HISTORY_RESPONSE_MODE=full_response
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export MAX_RESPONSE_LENGTH=32768
export MAX_MODEL_LENGTH=40960
export MAX_ASSISTANT_TURNS=15
export VAL_N=4
export VAL_TEMPERATURE=0.6

export ROLLOUT_GPU_MEMORY_UTILIZATION="${OPSD_VLLM_GPU_UTIL:-0.72}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export ROLLOUT_MAX_NUM_SEQS=64
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True
export WEBSHOP_AGENT_LOOP_NUM_WORKERS=1
export WEBSHOP_DATALOADER_NUM_WORKERS=1

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export WEBSHOP_MATCHED_EXPERIMENT_NAME="${OPSD_MATCHED_GRPO_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_opsd_matched_grpo_b8_n8_history4_hybrid_4gpu_s300}"

echo "WebShop-small matched GRPO for OPSD comparison"
echo "Protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, steps=${MAX_STEPS}, validation_every=${TEST_FREQ}"
echo "Dataset: ${WEBSHOP_TRAIN_PARQUET} (rubric columns ignored)"
echo "Rubric pipeline=off, token credit=off, generator loss=off"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_matched_grpo.sh" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  "$@"
