#!/usr/bin/env bash
set -euo pipefail

# Optional WebShop route: trajectory-mean-preserving centered OPD plus
# continuous leave-one-out rubric-generator credit. ALFWorld uses this route
# only through its dedicated centeredopd_loo comparison launcher.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export _RUBRIC_4GPU_CENTERED_OPD_LOO=1
export PROJECT_NAME=verl_rubric_3

# Keep the already working partial-judge resilience profile. These settings
# affect request failure tolerance only; successful cells use the same scoring.
export RUBRIC_POOL_JUDGE_RETRIES="${RUBRIC_POOL_JUDGE_RETRIES:-4}"
export RUBRIC_POOL_MIN_JUDGE_COVERAGE="${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-0.75}"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
