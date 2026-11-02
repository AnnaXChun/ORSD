#!/usr/bin/env bash
set -euo pipefail

# Coefficient-only comparison against the established beta=0.003 WebShop run.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export _RUBRIC_4GPU_OPD001_GEN0001=1

export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "WebShop coefficient comparison: additive OPD alpha=0.01; generator loss beta=0.001"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
