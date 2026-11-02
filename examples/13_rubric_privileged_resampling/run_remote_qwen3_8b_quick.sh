#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Fast full-AIME capability diagnostic. N=4 matches the original OPSD
# rollout group size while avoiding the 700-1000 long API calls of N=12.
export INFERENCE_BACKEND="dashscope"
export MODEL_PATH="${MODEL_PATH:-qwen3-8b}"
export REMOTE_API_KEY_ENV="${REMOTE_API_KEY_ENV:-RUBRIC_API_KEY}"
export REMOTE_BASE_URL="${REMOTE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export REMOTE_CONCURRENCY="${REMOTE_CONCURRENCY:-16}"
export REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-120}"
export REMOTE_MAX_RETRIES="${REMOTE_MAX_RETRIES:-2}"

export DATASET="${DATASET:-aime24}"
export SELECTION_NUM_SAMPLES="${SELECTION_NUM_SAMPLES:-4}"
export NUM_SAMPLES="${NUM_SAMPLES:-4}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export CANDIDATE_ROOT="${CANDIDATE_ROOT:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/candidate_rubrics}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/remote_qwen3_8b_quick_n4}"

exec bash "${SCRIPT_DIR}/run_dataset.sh" "$@"
