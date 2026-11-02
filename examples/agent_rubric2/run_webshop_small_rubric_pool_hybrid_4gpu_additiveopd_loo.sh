#!/usr/bin/env bash
set -euo pipefail

# Main WebShop route: restore the previously stronger raw additive OPD while
# retaining the LOO generator reward and the post-step-150 decay schedule.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export _RUBRIC_4GPU_CENTERED_OPD_LOO=1
export _RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP="${_RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP:-0}"
export RUBRIC_SCHEDULED_OPD_MODE=additive_gap
export RUBRIC_SCHEDULED_OPD_ALPHA=0.01
export RUBRIC_SCHEDULED_GENERATOR_LOSS_COEF=0.001
export RUBRIC_SCHEDULED_GENERATOR_PROMPT_STYLE=simple_numbered
export RUBRIC_POOL_JUDGE_RETRIES="${RUBRIC_POOL_JUDGE_RETRIES:-4}"
export RUBRIC_POOL_MIN_JUDGE_COVERAGE="${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-0.75}"
# Keep model-scale comparisons on the 32K context supported by Qwen2.5-3B.
export MAX_MODEL_LENGTH=32768
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=30720
export TEACHER_MAX_PROMPT_LENGTH=32768
export PROJECT_NAME=verl_rubric_3

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
