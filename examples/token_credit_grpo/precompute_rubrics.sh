#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/verl_wsm/Data}"
MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7b}"
INPUT_PARQUET="${INPUT_PARQUET:-${DATA_ROOT}/train.parquet}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-${DATA_ROOT}/train_with_rubrics.parquet}"
LOG_FILE="${RUBRIC_PRECOMPUTE_LOG:-${DATA_ROOT}/precompute_rubrics.log}"
inference_backend="${INFERENCE_BACKEND:-vllm}"
remote_api_key_env="${REMOTE_API_KEY_ENV:-RUBRIC_API_KEY}"

if [[ ! -e "${INPUT_PARQUET}" ]]; then
    echo "Required path does not exist: ${INPUT_PARQUET}" >&2
    exit 1
fi
if [[ "${inference_backend}" == "vllm" && ! -e "${MODEL_PATH}" ]]; then
    echo "Required local model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ "${inference_backend}" == "dashscope" && -z "${!remote_api_key_env:-}" ]]; then
    echo "Export ${remote_api_key_env} for DashScope rubric generation." >&2
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

mkdir -p "$(dirname "${OUTPUT_PARQUET}")" "$(dirname "${LOG_FILE}")"
cd "${VERL_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

python3 -u examples/token_credit_grpo/precompute_rubrics.py \
    --input-parquet "${INPUT_PARQUET}" \
    --output-parquet "${OUTPUT_PARQUET}" \
    --model "${MODEL_PATH}" \
    --inference-backend "${inference_backend}" \
    --remote-base-url "${REMOTE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
    --remote-api-key-env "${remote_api_key_env}" \
    --remote-concurrency "${REMOTE_CONCURRENCY:-8}" \
    --remote-timeout "${REMOTE_TIMEOUT:-180}" \
    --remote-max-retries "${REMOTE_MAX_RETRIES:-3}" \
    --http-client-log-level "${HTTP_CLIENT_LOG_LEVEL:-WARNING}" \
    --tensor-parallel-size "${RUBRIC_TP_SIZE:-8}" \
    --gpu-memory-utilization "${RUBRIC_GPU_MEMORY_UTILIZATION:-0.8}" \
    --max-model-len "${RUBRIC_MAX_MODEL_LEN:-8192}" \
    --batch-size "${RUBRIC_BATCH_SIZE:-64}" \
    --rubrics-per-prompt "${RUBRICS_PER_PROMPT:-8}" \
    --max-tokens "${RUBRIC_MAX_TOKENS:-2048}" \
    --temperature "${RUBRIC_TEMPERATURE:-1.0}" \
    --top-p "${RUBRIC_TOP_P:-0.95}" \
    --max-retries "${RUBRIC_MAX_RETRIES:-2}" \
    "$@"
