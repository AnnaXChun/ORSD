#!/usr/bin/env bash
set -euo pipefail

# Matched schedule comparison for the centered-OPD + LOO route. Both token
# credit and generator loss warm from zero to their full coefficients over the
# first 150 optimizer steps, then remain at full strength through step 300.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export _RUBRIC_4GPU_CENTERED_OPD_LOO=1
export _RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP=1
export PROJECT_NAME=verl_rubric_3

export RUBRIC_POOL_JUDGE_RETRIES="${RUBRIC_POOL_JUDGE_RETRIES:-4}"
export RUBRIC_POOL_MIN_JUDGE_COVERAGE="${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-0.75}"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
