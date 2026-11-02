#!/usr/bin/env bash
set -euo pipefail

# Keep 5-8 rubric bundles, but assign each selected rubric's Spearman reward
# only to that rubric's sampled token span. The bundle-level experiment remains
# available through the selectedreward launcher for a matched comparison.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_OPD001_GEN0003_SPEARMAN04_ITEM_CREDIT=1

export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"
export RUBRIC_POOL_JUDGE_CONCURRENCY="${RUBRIC_POOL_JUDGE_CONCURRENCY:-4}"
export RUBRIC_POOL_JUDGE_RETRIES="${RUBRIC_POOL_JUDGE_RETRIES:-4}"
export RUBRIC_POOL_MIN_JUDGE_COVERAGE="${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-0.75}"

echo "Item-credit experiment: OPD=0.01; gen beta=0.003; Spearman>0.4; 5-8 rubric bundles; selected-item token credit"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
