#!/usr/bin/env bash
set -euo pipefail

# Prompt-only comparison against the best early WebShop route. Keep OPD,
# generator loss, pool screening, judge settings, and training length intact.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PROJECT_NAME=verl_rubric_3
export _RUBRIC_4GPU_OPD001_GEN0003_SIMPLE_PROMPT=1

export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "WebShop prompt ablation: simple numbered rubric, one trajectory, no few-shot"
echo "Rubric count=5-8; OPD alpha=0.01; generator loss beta=0.003"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
