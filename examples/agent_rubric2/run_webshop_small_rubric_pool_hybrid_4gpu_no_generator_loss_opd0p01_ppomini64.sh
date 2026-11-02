#!/usr/bin/env bash
set -euo pipefail

# Ablation: retain rubric pool filtering and additive OPD, disable the rubric
# generator loss, keep OPD alpha at 0.01, and update all 64 rollout samples as
# one effective PPO mini-batch (configured as 8 prompts x 8 trajectories).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_POOL_ONLY_OPD001_PPOMINI64=1

# The configured key is issued for the laboratory OpenAI-compatible endpoint.
export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "Ablation: GPUs=${CUDA_VISIBLE_DEVICES}; rubric pool=True; generator loss=False; OPD alpha=0.01; configured PPO prompt mini-batch=8; effective rollout mini-batch=64"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
