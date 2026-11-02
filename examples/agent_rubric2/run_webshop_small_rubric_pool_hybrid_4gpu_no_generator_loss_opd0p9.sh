#!/usr/bin/env bash
set -euo pipefail

# Ablation: retain rubric generation, cloud judging, pool filtering and OPD;
# remove only the rubric-generator PPO objective and set OPD alpha to 0.9.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_POOL_ONLY_OPD09=1

# The configured key is issued for the laboratory OpenAI-compatible endpoint.
export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "Ablation: GPUs=${CUDA_VISIBLE_DEVICES}; rubric pool=True; generator loss=False; OPD alpha=0.9"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
