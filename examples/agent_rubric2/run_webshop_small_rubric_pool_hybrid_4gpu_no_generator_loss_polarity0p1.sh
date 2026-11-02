#!/usr/bin/env bash
set -euo pipefail

# Ablation: retain rubric generation, cloud judging and pool filtering; disable
# the rubric-generator loss and redistribute GRPO credit with the bounded,
# sign-preserving polarity-gap rule at alpha=0.1.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_POOL_ONLY_POLARITY01=1

# The API key itself must remain in DASHSCOPE_API_KEY in the caller's shell.
export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"

echo "Ablation: GPUs=${CUDA_VISIBLE_DEVICES}; rubric pool=True; generator loss=False; OPD mode=polarity_gap; alpha=0.1; preserve trajectory mean=True"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
