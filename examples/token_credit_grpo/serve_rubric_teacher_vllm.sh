#!/usr/bin/env bash
set -euo pipefail

: "${RUBRIC_TEACHER_GPUS:?Set RUBRIC_TEACHER_GPUS to GPU IDs not used by VERL training}"

MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7b}"
SERVED_MODEL_NAME="${RUBRIC_TEACHER_MODEL:-rubric-teacher}"
HOST="${RUBRIC_TEACHER_HOST:-0.0.0.0}"
PORT="${RUBRIC_TEACHER_PORT:-8001}"

export CUDA_VISIBLE_DEVICES="${RUBRIC_TEACHER_GPUS}"
export NCCL_DEBUG="WARN"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${RUBRIC_TEACHER_TP_SIZE:-1}" \
    --gpu-memory-utilization "${RUBRIC_TEACHER_GPU_MEMORY_UTILIZATION:-0.8}" \
    --max-model-len "${RUBRIC_TEACHER_MAX_MODEL_LEN:-8192}" \
    --dtype bfloat16 \
    --trust-remote-code
