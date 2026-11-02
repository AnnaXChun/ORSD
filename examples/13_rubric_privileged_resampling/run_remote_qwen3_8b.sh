#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Keep the full Experiment 13 protocol unchanged and replace only the model
# inference backend/checkpoint with DashScope qwen3-8b.
export INFERENCE_BACKEND="dashscope"
export MODEL_PATH="${MODEL_PATH:-qwen3-8b}"
export REMOTE_API_KEY_ENV="${REMOTE_API_KEY_ENV:-RUBRIC_API_KEY}"
export REMOTE_BASE_URL="${REMOTE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export REMOTE_CONCURRENCY="${REMOTE_CONCURRENCY:-8}"
export REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-180}"
export REMOTE_MAX_RETRIES="${REMOTE_MAX_RETRIES:-3}"

DATASET="${DATASET:-aime24}"
# Reuse the same candidate-rubric parquet as the local-model run whenever it
# already exists, so the capability comparison changes only the solver model.
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/candidate_rubrics}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/remote_qwen3_8b}"

export DATASET CANDIDATE_ROOT OUTPUT_ROOT
exec bash "${SCRIPT_DIR}/run_dataset.sh" "$@"
