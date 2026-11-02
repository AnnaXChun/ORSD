#!/usr/bin/env bash
set -euo pipefail

# Single-variable experiment over the strongest no-generator-loss OPD=0.01
# run: greedily remove rubrics with redundant cross-trajectory score rankings.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_POOL_ONLY_OPD001_BEHAVIOR_DEDUP=1

export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "Experiment: OPD alpha=0.01; generator loss disabled; behavior dedup enabled at Spearman>=0.9"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
