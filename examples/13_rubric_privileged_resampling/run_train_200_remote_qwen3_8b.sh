#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/verl_wsm/Data}"

# Remote Qwen3-8B scale check on the same rubric-augmented training parquet
# and N=4 protocol used by the original 64-problem experiment.
export INFERENCE_BACKEND="dashscope"
export MODEL_PATH="${MODEL_PATH:-qwen3-8b}"
export REMOTE_API_KEY_ENV="${REMOTE_API_KEY_ENV:-RUBRIC_API_KEY}"
export REMOTE_BASE_URL="${REMOTE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export REMOTE_CONCURRENCY="${REMOTE_CONCURRENCY:-16}"
export REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-120}"
export REMOTE_MAX_RETRIES="${REMOTE_MAX_RETRIES:-2}"

export DATASET_NAME="train200"
export INPUT_PARQUET="${INPUT_PARQUET:-${DATA_ROOT}/train_with_rubrics.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/train_200_qwen3_8b_remote}"
export LIMIT="${LIMIT:-200}"
export SELECTION_NUM_SAMPLES="${SELECTION_NUM_SAMPLES:-4}"
export NUM_SAMPLES="${NUM_SAMPLES:-4}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export MAX_TOKENS="${MAX_TOKENS:-8192}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:--1}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
