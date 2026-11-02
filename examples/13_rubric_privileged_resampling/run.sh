#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/verl_wsm/Data}"
MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7b}"
OPSD_ROOT="${OPSD_ROOT:-/workspace/verl_wsm/OPSD}"
DATASET_NAME="${DATASET_NAME:-aime24}"
INPUT_PARQUET="${INPUT_PARQUET:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/candidate_rubrics/${DATASET_NAME}_with_rubrics.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_ROOT}/experiment_outputs/13_rubric_privileged_resampling/${DATASET_NAME}}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/summary.json}"
PROGRESS_FILE="${PROGRESS_FILE:-${OUTPUT_DIR}/paired_results.jsonl}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run.log}"
inference_backend="${INFERENCE_BACKEND:-vllm}"
remote_api_key_env="${REMOTE_API_KEY_ENV:-RUBRIC_API_KEY}"

if [[ ! -e "${INPUT_PARQUET}" ]]; then
    echo "Required input parquet does not exist: ${INPUT_PARQUET}" >&2
    exit 1
fi
if [[ "${inference_backend}" == "vllm" && ! -e "${MODEL_PATH}" ]]; then
    echo "Required local model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ "${inference_backend}" == "dashscope" && -z "${!remote_api_key_env:-}" ]]; then
    echo "Export ${remote_api_key_env} for DashScope model generation." >&2
    exit 1
fi

selection_mode="${SELECTION_MODE:-opsd_filtered}"
rubric_api_key_env="${RUBRIC_API_KEY_ENV:-RUBRIC_API_KEY}"
if [[ "${selection_mode}" == "opsd_filtered" ]]; then
    if [[ ! -f "${OPSD_ROOT}/rubric_pipeline.py" ]]; then
        echo "OPSD rubric selector does not exist: ${OPSD_ROOT}/rubric_pipeline.py" >&2
        exit 1
    fi
    if [[ -z "${!rubric_api_key_env:-}" ]]; then
        echo "Export ${rubric_api_key_env} for OPSD rubric reliability filtering." >&2
        exit 1
    fi
fi

# Deliberately override an outer CUDA_VISIBLE_DEVICES. Set EVAL_GPUS to choose
# another pair, for example EVAL_GPUS=6,7.
export CUDA_VISIBLE_DEVICES="${EVAL_GPUS:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="false"
# vLLM tensor-parallel workers must not inherit an already initialized CUDA
# context from the Python parent process.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

gpu_count="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
tp_size="${TENSOR_PARALLEL_SIZE:-2}"
if [[ "${inference_backend}" == "vllm" && "${gpu_count}" -ne "${tp_size}" ]]; then
    echo "Visible GPU count (${gpu_count}) must equal TENSOR_PARALLEL_SIZE (${tp_size})." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${VERL_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

thinking_flag="--enable-thinking"
if [[ "${ENABLE_THINKING:-1}" == "0" ]]; then
    thinking_flag="--no-enable-thinking"
fi

completion_flag=()
if [[ "${SAVE_COMPLETIONS:-0}" == "1" ]]; then
    completion_flag+=(--save-completions)
fi

python3 -u examples/13_rubric_privileged_resampling/evaluate_rubric_sampling.py \
    --input-parquet "${INPUT_PARQUET}" \
    --model "${MODEL_PATH}" \
    --inference-backend "${inference_backend}" \
    --remote-base-url "${REMOTE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
    --remote-api-key-env "${remote_api_key_env}" \
    --remote-concurrency "${REMOTE_CONCURRENCY:-8}" \
    --remote-timeout "${REMOTE_TIMEOUT:-180}" \
    --remote-max-retries "${REMOTE_MAX_RETRIES:-3}" \
    --output-json "${OUTPUT_JSON}" \
    --progress-file "${PROGRESS_FILE}" \
    --prompt-column "${PROMPT_COLUMN:-prompt}" \
    --rubric-column "${RUBRIC_COLUMN:-rubric}" \
    --valid-column "${RUBRIC_VALID_COLUMN-rubric_valid}" \
    --default-data-source "${DEFAULT_DATA_SOURCE:-math_dapo}" \
    --selection-mode "${selection_mode}" \
    --opsd-root "${OPSD_ROOT}" \
    --judge-api-key-env "${rubric_api_key_env}" \
    --judge-base-url "${RUBRIC_JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
    --judge-model "${RUBRIC_JUDGE_MODEL:-qwen3.5-plus}" \
    --judge-timeout "${RUBRIC_JUDGE_TIMEOUT:-60}" \
    --judge-max-concurrency "${RUBRIC_JUDGE_MAX_CONCURRENCY:-8}" \
    --selection-prompt-concurrency "${SELECTION_PROMPT_CONCURRENCY:-2}" \
    --selection-num-samples "${SELECTION_NUM_SAMPLES:-4}" \
    --reliability-threshold "${RELIABILITY_THRESHOLD:-0.5}" \
    --max-selected-rubrics "${MAX_SELECTED_RUBRICS:-8}" \
    --limit "${LIMIT:--1}" \
    --num-samples "${NUM_SAMPLES:-4}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top-p "${TOP_P:-0.95}" \
    --top-k "${TOP_K:--1}" \
    --max-tokens "${MAX_TOKENS:-8192}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --seed "${SEED:-42}" \
    --tensor-parallel-size "${tp_size}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
    --http-client-log-level "${HTTP_CLIENT_LOG_LEVEL:-WARNING}" \
    "${thinking_flag}" \
    "${completion_flag[@]}" \
    "$@"
