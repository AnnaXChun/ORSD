#!/usr/bin/env bash
set -euo pipefail

# Matched GRPO control for run_webshop_small_first_version_4gpu.sh.
# The rollout, context, reward, optimizer, validation, and hardware budgets are
# aligned; only rubric generation, token-credit shaping, and generator updates
# are disabled.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _WEBSHOP_MATCHED_H4_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_WEBSHOP_MATCHED_H4_VISIBLE_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

# Match the current Qwen3-1.7B WebShop-small first-version experiment.
export N_GPUS=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY_UTILIZATION="${WEBSHOP_MATCHED_H4_VLLM_GPU_UTIL:-0.72}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export ROLLOUT_MAX_NUM_SEQS=64
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True

export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export TEST_FREQ=50
export SAVE_FREQ=50
export MICRO_BATCH_SIZE=1
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export LEARNING_RATE=1e-6

# Retain the fixed prefix and latest four complete action/observation turns.
export HISTORY_WINDOW=4
export HISTORY_RESPONSE_MODE=full_response
export ACTOR_THINKING=True
export MAX_ASSISTANT_TURNS=15
export MAX_RESPONSE_LENGTH=32768
export MAX_MODEL_LENGTH=40960
export VAL_N=4
export VAL_TEMPERATURE=0.6

export WEBSHOP_REWARD_MODE=hybrid
export WEBSHOP_R2_INVALID_ACTION_COEF=0.05
export WEBSHOP_R2_REPEAT_ACTION_COEF=0.025
export WEBSHOP_AGENT_LOOP_NUM_WORKERS=1
export WEBSHOP_DATALOADER_NUM_WORKERS=1

# Explicitly isolate the baseline from any rubric variables left in the shell.
export _WEBSHOP_MATCHED_ENTRY=1
export TOKEN_CREDIT_ENABLE=False
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export RUBRIC_PIPELINE_ENABLE=False
export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_LOSS_COEF=0.0
export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=False
export RUBRIC_SNAPSHOT_ENABLE=False

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export WEBSHOP_MATCHED_EXPERIMENT_NAME="${WEBSHOP_MATCHED_H4_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_matched_grpo_history4_hybrid_4gpu_b8_n8_s300}"

echo "WebShop-small matched GRPO: hybrid reward, history_window=4"
echo "Protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, total=$((TRAIN_BATCH_SIZE * ROLLOUT_N)), steps=${MAX_STEPS}"
echo "Validation: every=${TEST_FREQ}, rollouts/prompt=${VAL_N}; checkpoints every=${SAVE_FREQ}"
echo "Actor: thinking=${ACTOR_THINKING}, history_response=${HISTORY_RESPONSE_MODE}, learning_rate=${LEARNING_RATE}"
echo "Rubric pipeline=off, token credit=off, generator loss=off, rubric snapshot=off"
echo "Experiment: ${WEBSHOP_MATCHED_EXPERIMENT_NAME}"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_matched_grpo.sh" "$@"
