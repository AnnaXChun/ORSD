#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/verl_wsm/Data}"
MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7b}"

if [[ -n "${INPUT_PARQUET:-}" ]]; then
    input_parquet="${INPUT_PARQUET}"
elif [[ -f "${DATA_ROOT}/train_with_rubrics.parquet" ]]; then
    input_parquet="${DATA_ROOT}/train_with_rubrics.parquet"
else
    input_parquet="${DATA_ROOT}/train.parquet"
fi

output_parquet="${OUTPUT_PARQUET:-${DATA_ROOT}/train_with_validated_rubrics.parquet}"
log_file="${VALIDATED_RUBRIC_LOG:-${DATA_ROOT}/precompute_validated_rubrics.log}"
generation_backend="${GENERATION_BACKEND:-vllm}"
judge_api_key_env="${JUDGE_API_KEY_ENV:-RUBRIC_API_KEY}"

if [[ ! -f "${input_parquet}" ]]; then
    echo "Input parquet does not exist: ${input_parquet}" >&2
    exit 1
fi
if [[ "${generation_backend}" == "vllm" && ! -e "${MODEL_PATH}" ]]; then
    echo "Generator model does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ -z "${!judge_api_key_env:-}" ]]; then
    echo "Export ${judge_api_key_env} for answer-aware rubric validation." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="false"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

mkdir -p "$(dirname "${output_parquet}")" "$(dirname "${log_file}")"
cd "${VERL_ROOT}"
exec > >(tee -a "${log_file}") 2>&1

python3 -u examples/token_credit_grpo/precompute_validated_rubrics.py \
    --input-parquet "${input_parquet}" \
    --output-parquet "${output_parquet}" \
    --generator-model "${MODEL_PATH}" \
    --generation-backend "${generation_backend}" \
    --remote-base-url "${REMOTE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
    --generator-api-key-env "${GENERATOR_API_KEY_ENV:-RUBRIC_API_KEY}" \
    --judge-api-key-env "${judge_api_key_env}" \
    --judge-model "${JUDGE_MODEL:-qwen3.5-plus}" \
    --remote-concurrency "${REMOTE_CONCURRENCY:-16}" \
    --remote-timeout "${REMOTE_TIMEOUT:-180}" \
    --remote-max-retries "${REMOTE_MAX_RETRIES:-3}" \
    --http-client-log-level "${HTTP_CLIENT_LOG_LEVEL:-WARNING}" \
    --tensor-parallel-size "${RUBRIC_TP_SIZE:-8}" \
    --gpu-memory-utilization "${RUBRIC_GPU_MEMORY_UTILIZATION:-0.8}" \
    --max-model-len "${RUBRIC_MAX_MODEL_LEN:-8192}" \
    --batch-size "${VALIDATION_BATCH_SIZE:-32}" \
    --candidate-count "${CANDIDATE_COUNT:-12}" \
    --min-selected "${MIN_SELECTED_RUBRICS:-4}" \
    --max-selected "${MAX_SELECTED_RUBRICS:-8}" \
    --validation-threshold "${VALIDATION_THRESHOLD:-0.75}" \
    --max-validation-rounds "${MAX_VALIDATION_ROUNDS:-4}" \
    --target-valid-rate "${TARGET_VALID_RATE:-0.70}" \
    --generator-max-tokens "${RUBRIC_MAX_TOKENS:-2048}" \
    --judge-max-tokens "${JUDGE_MAX_TOKENS:-2048}" \
    --temperature "${RUBRIC_TEMPERATURE:-1.0}" \
    --top-p "${RUBRIC_TOP_P:-0.95}" \
    --limit "${LIMIT:--1}" \
    --seed "${SEED:-42}" \
    "$@"
