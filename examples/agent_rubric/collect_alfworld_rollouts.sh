#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${ALFWORLD_CONFIG:?Set ALFWORLD_CONFIG to the official base_config.yaml}"
: "${ALFWORLD_DATA:?Set ALFWORLD_DATA to the downloaded ALFWorld data directory}"
: "${INPUT_PARQUET:?Set INPUT_PARQUET to preprocessed ALFWorld parquet}"
: "${OUTPUT_PARQUET:?Set OUTPUT_PARQUET for trajectory-enriched parquet}"
: "${ROLLOUT_MODEL:?Set ROLLOUT_MODEL to a local Hugging Face model path}"

python3 "${ROOT}/examples/agent_rubric/collect_rollout_trajectories.py" \
  --benchmark alfworld \
  --alfworld-config "${ALFWORLD_CONFIG}" \
  --alfworld-data "${ALFWORLD_DATA}" \
  --input-parquet "${INPUT_PARQUET}" \
  --output-parquet "${OUTPUT_PARQUET}" \
  --model "${ROLLOUT_MODEL}" \
  --num-rollouts "${NUM_ROLLOUTS:-4}" \
  --episode-batch-size "${ROLLOUT_EPISODE_BATCH_SIZE:-32}" \
  --tensor-parallel-size "${ROLLOUT_TP:-2}" \
  --gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}" \
  --max-model-len "${ROLLOUT_MAX_MODEL_LEN:-40960}" \
  --max-action-tokens "${MAX_ACTION_TOKENS:-64}" \
  "$@"
