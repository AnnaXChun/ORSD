#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/verl_wsm/Data}"
MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7b}"
DATASET="${DATASET:-aime24}"

case "${DATASET}" in
    aime24)
        source_name="aime_2024.parquet"
        ;;
    aime25)
        source_name="aime25_test.parquet"
        ;;
    math500)
        source_name="math500_test.parquet"
        ;;
    amc23)
        source_name="amc23_test.parquet"
        ;;
    *)
        echo "Unsupported DATASET=${DATASET}; choose aime24, aime25, math500, or amc23." >&2
        exit 1
        ;;
esac

source_parquet="${DATA_ROOT}/${source_name}"
candidate_root="${CANDIDATE_ROOT:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/candidate_rubrics}"
candidate_parquet="${candidate_root}/${DATASET}_with_rubrics.parquet"
dataset_output_dir="${OUTPUT_ROOT:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling}/${DATASET}"
precompute_python="${VERL_ROOT}/examples/token_credit_grpo/precompute_rubrics.py"
precompute_shell="${VERL_ROOT}/examples/token_credit_grpo/precompute_rubrics.sh"

if [[ ! -f "${source_parquet}" ]]; then
    echo "Local evaluation parquet does not exist: ${source_parquet}" >&2
    exit 1
fi
if [[ ! -f "${precompute_python}" || ! -f "${precompute_shell}" ]]; then
    echo "Rubric precompute entry is missing under examples/token_credit_grpo." >&2
    exit 1
fi
if ! grep -q '^STANDALONE_RUBRIC_PRECOMPUTE_VERSION = 2$' "${precompute_python}"; then
    echo "The server has an old precompute_rubrics.py that imports the VERL runtime." >&2
    echo "Sync the updated examples/token_credit_grpo/precompute_rubrics.py before running Experiment 13." >&2
    exit 1
fi

mkdir -p "${candidate_root}" "${dataset_output_dir}"

if [[ ! -f "${candidate_parquet}" ]]; then
    echo "Generating problem-only candidate rubrics for ${DATASET}..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS:-0,1}" \
    INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}" \
    REMOTE_API_KEY_ENV="${REMOTE_API_KEY_ENV:-RUBRIC_API_KEY}" \
    REMOTE_BASE_URL="${REMOTE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
    REMOTE_CONCURRENCY="${REMOTE_CONCURRENCY:-8}" \
    REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-180}" \
    REMOTE_MAX_RETRIES="${REMOTE_MAX_RETRIES:-3}" \
    INPUT_PARQUET="${source_parquet}" \
    OUTPUT_PARQUET="${candidate_parquet}" \
    MODEL_PATH="${MODEL_PATH}" \
    RUBRIC_TP_SIZE="${TENSOR_PARALLEL_SIZE:-2}" \
    RUBRIC_BATCH_SIZE="${RUBRIC_BATCH_SIZE:-16}" \
    RUBRIC_PRECOMPUTE_LOG="${candidate_root}/${DATASET}_precompute.log" \
    bash "${precompute_shell}"
else
    echo "Reusing candidate rubric parquet: ${candidate_parquet}"
fi

echo "Running selected-rubric resampling on ${DATASET}..."
DATASET_NAME="${DATASET}" \
INPUT_PARQUET="${candidate_parquet}" \
OUTPUT_DIR="${dataset_output_dir}" \
MODEL_PATH="${MODEL_PATH}" \
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}" \
REMOTE_API_KEY_ENV="${REMOTE_API_KEY_ENV:-RUBRIC_API_KEY}" \
REMOTE_BASE_URL="${REMOTE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
REMOTE_CONCURRENCY="${REMOTE_CONCURRENCY:-8}" \
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-180}" \
REMOTE_MAX_RETRIES="${REMOTE_MAX_RETRIES:-3}" \
LIMIT="${LIMIT:--1}" \
SELECTION_NUM_SAMPLES="${SELECTION_NUM_SAMPLES:-12}" \
NUM_SAMPLES="${NUM_SAMPLES:-12}" \
bash "${SCRIPT_DIR}/run.sh" "$@"
