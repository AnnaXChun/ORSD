#!/usr/bin/env bash
set -euo pipefail

# Isolated rank-attenuation comparison. It reuses the same WebShop rubric pool,
# LOO generator loss, four-GPU profile, and post-step-150 decay as the additive
# comparison; only the actor token-credit rule and its own parameters differ.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export _RUBRIC_4GPU_CENTERED_OPD_LOO=1
export _RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP="${_RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP:-0}"
export RUBRIC_SCHEDULED_OPD_MODE=rank_attenuation
export RUBRIC_SCHEDULED_RANK_ATTENUATION=0.2
export TOKEN_CREDIT_RANK_CONFLICT_FRACTION=0.25
export TOKEN_CREDIT_RANK_MIN_GAP_RANGE=0.0001
export PROJECT_NAME=verl_rubric_3

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
