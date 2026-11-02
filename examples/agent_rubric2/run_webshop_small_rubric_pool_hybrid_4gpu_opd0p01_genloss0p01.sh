#!/usr/bin/env bash
set -euo pipefail

# Conservative continuation of the best pool-only OPD=0.01 run: preserve all
# task/pool settings and enable the rubric-response PPO objective at beta=0.01.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_OPD001_GEN001=1

# The configured key is issued for the laboratory OpenAI-compatible endpoint.
export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "Continuation candidate: OPD alpha=0.01; rubric generator loss=True; beta=0.01"
echo "Monitor rubric_generator/objective_abs_ratio_to_actor; prefer <=1, with 0.1-0.5 as the conservative target band."
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
