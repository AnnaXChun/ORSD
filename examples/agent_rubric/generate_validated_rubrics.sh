#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

: "${INPUT_PARQUET:?Set INPUT_PARQUET to a parquet containing complete rollout_trajectories_json}"
: "${OUTPUT_PARQUET:?Set OUTPUT_PARQUET for the validated training parquet}"
: "${RUBRIC_GENERATOR_MODEL:?Set RUBRIC_GENERATOR_MODEL to a local model path or remote model name}"

python3 "${ROOT}/examples/agent_rubric/precompute_trajectory_rubrics.py" \
  --input-parquet "${INPUT_PARQUET}" \
  --output-parquet "${OUTPUT_PARQUET}" \
  --generator-model "${RUBRIC_GENERATOR_MODEL}" \
  --generation-backend "${GENERATION_BACKEND:-vllm}" \
  --judge-backend "${JUDGE_BACKEND:-dashscope}" \
  --judge-model "${RUBRIC_JUDGE_MODEL:-qwen3.5-plus}" \
  --tensor-parallel-size "${RUBRIC_TP:-4}" \
  --gpu-memory-utilization "${RUBRIC_GPU_MEMORY_UTILIZATION:-0.75}" \
  --max-model-len "${RUBRIC_MAX_MODEL_LEN:-65536}" \
  --batch-size "${RUBRIC_BATCH_SIZE:-16}" \
  --candidate-count "${RUBRIC_CANDIDATES:-10}" \
  --min-trajectories "${MIN_TRAJECTORIES_PER_TASK:-4}" \
  --min-selected "${MIN_RUBRICS_PER_TASK:-4}" \
  --max-selected "${MAX_RUBRICS_PER_TASK:-8}" \
  --validation-threshold "${RUBRIC_VALIDATION_THRESHOLD:-0.80}" \
  --max-validation-rounds "${RUBRIC_VALIDATION_ROUNDS:-5}" \
  --target-valid-rate "${TARGET_VALID_RATE:-0.60}" \
  --remote-concurrency "${RUBRIC_REMOTE_CONCURRENCY:-16}" \
  --generator-max-tokens "${RUBRIC_GENERATOR_MAX_TOKENS:-2048}" \
  --judge-max-tokens "${RUBRIC_JUDGE_MAX_TOKENS:-3072}" \
  "$@"
