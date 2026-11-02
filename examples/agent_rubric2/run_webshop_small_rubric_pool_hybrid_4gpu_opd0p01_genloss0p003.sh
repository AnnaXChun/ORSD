#!/usr/bin/env bash
set -euo pipefail

# Lower-weight companion to the beta=0.01 continuation. All task, pool and OPD
# settings stay matched to the best pool-only OPD=0.01 run.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_OPD001_GEN0003=1

# The configured key is issued for the laboratory OpenAI-compatible endpoint.
export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "Conservative continuation: OPD alpha=0.01; rubric generator loss=True; beta=0.003"
echo "Compare rubric_generator/objective_abs_ratio_to_actor with the beta=0.01 run."
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
