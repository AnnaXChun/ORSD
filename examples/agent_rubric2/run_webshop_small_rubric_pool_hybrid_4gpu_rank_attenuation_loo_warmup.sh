#!/usr/bin/env bash
set -euo pipefail

# Warmup counterpart to the rank-attenuation decay experiment. All settings
# are inherited from the same launcher; only the token-credit and generator-loss
# schedules switch to a 0->target linear warmup over the first 150 steps.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export _RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP=1

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu_rank_attenuation_loo.sh" "$@"
