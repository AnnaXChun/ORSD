#!/usr/bin/env bash
set -euo pipefail

# Repair the high cloud-judge group failure rate without changing OPD,
# generator loss, rubric generation, or quality thresholds.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_OPD001_GEN0003_PARTIAL_JUDGE=1

export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"
export RUBRIC_POOL_JUDGE_CONCURRENCY="${RUBRIC_POOL_JUDGE_CONCURRENCY:-4}"
export RUBRIC_POOL_JUDGE_RETRIES="${RUBRIC_POOL_JUDGE_RETRIES:-4}"
export RUBRIC_POOL_MIN_JUDGE_COVERAGE="${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-0.75}"

echo "Repair experiment: OPD=0.01; genloss=0.003; judge concurrency=4; retries=4; coverage>=0.75"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
