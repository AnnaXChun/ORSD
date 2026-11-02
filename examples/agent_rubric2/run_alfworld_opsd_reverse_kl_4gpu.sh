#!/usr/bin/env bash
set -euo pipefail

# Pure fixed-teacher OPSD with real ALFWorld rollouts managed by VERL.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ALFWORLD_OPSD_RUBRIC_PARQUET:?Set ALFWORLD_OPSD_RUBRIC_PARQUET to the saved train parquet with rubric columns}"

export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"
export ALFWORLD_R2_REWARD_ENABLE=False
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _ALFWORLD_OPSD_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_ALFWORLD_OPSD_GPUS[@]} != 4 )); then
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

for path in "${ALFWORLD_CONFIG}" "${ALFWORLD_DATA}" "${TRAIN_PARQUET}" "${ALFWORLD_VAL_SEEN_PARQUET}" "${ALFWORLD_VAL_UNSEEN_PARQUET}"; do
  if [[ ! -e "${path}" ]]; then echo "Missing ALFWorld/OPSD asset: ${path}" >&2; exit 1; fi
done

export N_GPUS=4
export ROLLOUT_TP=1
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export TEST_FREQ=50
export SAVE_FREQ=25
export MICRO_BATCH_SIZE=2
export LEARNING_RATE=1e-6
export HISTORY_WINDOW=0
export MAX_RESPONSE_LENGTH=16384
export MAX_MODEL_LENGTH=40960
export VAL_N=4
export VAL_TEMPERATURE=0.6
export AGENT_LOOP_NUM_WORKERS=4
export DATALOADER_NUM_WORKERS=2
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ALFWORLD_OPSD_VLLM_GPU_UTIL:-0.60}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
export ROLLOUT_MAX_NUM_SEQS=256
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True

export TOKEN_CREDIT_ENABLE=True
export TOKEN_CREDIT_MODE=opsd_reverse_kl
export TOKEN_CREDIT_TEACHER_LOG_PROB_KEY=rubric_teacher_log_probs
export OPSD_TEACHER_SOURCE=reference_policy
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export RUBRIC_PIPELINE_ENABLE=True
export RUBRIC_BACKEND=dataset
export TARGET_VALID_RATE=1.0
export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_SNAPSHOT_ENABLE=False
export ACTOR_THINKING=True
export TEACHER_THINKING=True
export RUBRIC_GENERATOR_THINKING=False
export TEACHER_MAX_PROMPT_LENGTH=4096

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export EXPERIMENT_NAME="${ALFWORLD_OPSD_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_opsd_reversekl_savedrubric_b8_n8_official01_4gpu_s300}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

echo "ALFWorld OPSD: VERL rollout + frozen-reference sampled-token reverse-KL loss"
echo "Protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, steps=${MAX_STEPS}, seen+unseen validation every ${TEST_FREQ}"
echo "Dataset rubric=${TRAIN_PARQUET}; online rubric generation=off; GRPO advantage excluded from actor update"

exec bash "${ROOT}/examples/agent_rubric/run_agent_grpo.sh" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
  "$@"
