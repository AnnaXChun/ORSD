#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/verl_wsm/Data}"

# Scale the original 64-problem local smoke test to 1,000 unique training
# problems without changing its model, rollout group, or OPSD filter.
export INFERENCE_BACKEND="vllm"
export MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7b}"
export DATASET_NAME="train1000"
export INPUT_PARQUET="${INPUT_PARQUET:-${DATA_ROOT}/train_with_rubrics.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/train_1000_qwen3_1_7b}"
export LIMIT="${LIMIT:-1000}"
export SELECTION_NUM_SAMPLES="${SELECTION_NUM_SAMPLES:-4}"
export NUM_SAMPLES="${NUM_SAMPLES:-4}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export MAX_TOKENS="${MAX_TOKENS:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:--1}"
export EVAL_GPUS="${EVAL_GPUS:-0,1}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
