#!/usr/bin/env bash
set -euo pipefail

# Warmup comparison for the restored additive OPD route: alpha rises from zero
# to 0.01 over the first 150 optimizer steps and then remains at 0.01.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export _RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP=1

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu_additiveopd_loo.sh" "$@"
