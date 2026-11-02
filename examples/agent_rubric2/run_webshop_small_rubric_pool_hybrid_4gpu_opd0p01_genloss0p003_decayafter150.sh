#!/usr/bin/env bash
set -euo pipefail

# Keep the best early-training WebShop configuration unchanged through step
# 150. Over steps 151--300, linearly decay OPD alpha 0.01 and generator-loss
# coefficient 0.003 to zero.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PROJECT_NAME=verl_rubric_3
export _RUBRIC_4GPU_OPD001_GEN0003_DECAY_AFTER150=1

# Use the same judge endpoint as the original best early-training run.
export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "WebShop continuation test: OPD alpha=0.01 and generator loss beta=0.003"
echo "Schedule: fixed through step 150, then both coefficients decay linearly to zero at step 300"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
