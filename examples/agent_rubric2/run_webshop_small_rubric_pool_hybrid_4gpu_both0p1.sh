#!/usr/bin/env bash
set -euo pipefail

# Independent second experiment: GPUs 4-7, OPD=0.1, generator PPO=0.1.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Explicitly replace an inherited first-four-device selection.
export CUDA_VISIBLE_DEVICES=4,5,6,7
export _RUBRIC_4GPU_BOTH_01=1
# Start an independent local Ray instance; do not attach to the first run.
export RAY_ADDRESS=local
echo "Second experiment: GPUs=${CUDA_VISIBLE_DEVICES}, OPD=0.1, generator loss=0.1"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" \
  ++ray_kwargs.ray_init.address=local \
  ++ray_kwargs.ray_init.dashboard_port=0 \
  "$@"
