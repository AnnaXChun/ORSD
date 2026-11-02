#!/usr/bin/env bash
set -euo pipefail

# Same ALFWorld data/hardware/validation budget as the OPSD launcher, with all
# privileged-rubric paths disabled and ordinary GRPO/PPO retained.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ALFWORLD_OPSD_RUBRIC_PARQUET:?Set ALFWORLD_OPSD_RUBRIC_PARQUET to the same parquet used by OPSD}"

export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"
export ALFWORLD_R2_REWARD_ENABLE=False
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _ALFWORLD_OPSD_GRPO_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_ALFWORLD_OPSD_GRPO_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in /workspace/checkpoints/qwen/Qwen3-1.7B /workspace/checkpoints/Qwen3-1.7B "${ROOT}/checkpoints/Qwen3-1.7B"; do
    if [[ -f "${candidate}/config.json" ]]; then MODEL_PATH="${candidate}"; break; fi
  done
fi
if [[ -z "${MODEL_PATH:-}" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Cannot find Qwen3-1.7B. Set MODEL_PATH to the exact model directory." >&2
  exit 1
fi
export BENCHMARK=alfworld
export MODEL_PATH
export TRAIN_PARQUET="${ALFWORLD_OPSD_RUBRIC_PARQUET}"
export ALFWORLD_VAL_SEEN_PARQUET="${ALFWORLD_VAL_SEEN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_valid_seen.parquet}"
export ALFWORLD_VAL_UNSEEN_PARQUET="${ALFWORLD_VAL_UNSEEN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_valid_unseen.parquet}"
export VAL_PARQUET="${ALFWORLD_VAL_UNSEEN_PARQUET}"
export VAL_FILES="['${ALFWORLD_VAL_SEEN_PARQUET}', '${ALFWORLD_VAL_UNSEEN_PARQUET}']"

export N_GPUS=4 ROLLOUT_TP=1 TRAIN_BATCH_SIZE=8 ROLLOUT_N=8 MAX_STEPS=300
export TEST_FREQ=50 SAVE_FREQ=25 MICRO_BATCH_SIZE=2 LEARNING_RATE=1e-6
export HISTORY_WINDOW=0 MAX_RESPONSE_LENGTH=16384 MAX_MODEL_LENGTH=40960
export VAL_N=4 VAL_TEMPERATURE=0.6 AGENT_LOOP_NUM_WORKERS=4 DATALOADER_NUM_WORKERS=2
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ALFWORLD_OPSD_VLLM_GPU_UTIL:-0.60}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192 ROLLOUT_MAX_NUM_SEQS=256
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True ROLLOUT_ENABLE_PREFIX_CACHING=True
export TOKEN_CREDIT_ENABLE=False RUBRIC_PIPELINE_ENABLE=False RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False RUBRIC_SNAPSHOT_ENABLE=False ACTOR_THINKING=True
export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export EXPERIMENT_NAME="${ALFWORLD_OPSD_MATCHED_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_opsd_matched_grpo_b8_n8_official01_4gpu_s300}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

echo "ALFWorld matched GRPO for OPSD: 8x8, 300 steps, official reward, seen+unseen validation"
echo "Dataset=${TRAIN_PARQUET}; rubric columns ignored"

exec bash "${ROOT}/examples/agent_rubric/run_agent_grpo.sh" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  "$@"
