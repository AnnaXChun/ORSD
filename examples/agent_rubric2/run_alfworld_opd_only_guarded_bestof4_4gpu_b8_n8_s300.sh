#!/usr/bin/env bash
set -euo pipefail

# One-variable extension of the successful ALFWorld OPD-only run. Candidate
# zero exactly follows the historical single-bundle generation route. Three
# additional samples may replace it only when their judged Spearman
# correlation is positive and exceeds candidate zero by at least 0.05.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUBRIC_GROUP_CANDIDATES=4
export RUBRIC_GROUP_SELECT_BY_CORRELATION=True
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED=True
export RUBRIC_GROUP_GUARDED_SELECTION=True
export RUBRIC_GROUP_MIN_SELECTED_CORRELATION=0.0
export RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT=0.05
# The successful historical OPD-only run used 16 environment workers. Keep
# that execution profile here unless the caller deliberately overrides it.
export ALFWORLD_4GPU_AGENT_LOOP_WORKERS="${ALFWORLD_4GPU_AGENT_LOOP_WORKERS:-16}"

export ALFWORLD_OPD_ONLY_EXPERIMENT_NAME="${ALFWORLD_GUARDED_BESTOF4_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_opd_only_guarded_bestof4_margin0p05_no_pool_4gpu_b8_n8_additiveopd0p01_official_reward_s300}"
export ALFWORLD_OPD_ONLY_OUTPUT_DIR="${ALFWORLD_GUARDED_BESTOF4_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME:-verl_rubric_3}/${ALFWORLD_OPD_ONLY_EXPERIMENT_NAME}}"

echo "ALFWorld OPD-only guarded best-of-4: candidate 0 is the historical baseline"
echo "Replacement rule: best Spearman > ${RUBRIC_GROUP_MIN_SELECTED_CORRELATION} and improvement >= ${RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT}"

exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_opd_only_no_pool_4gpu_b8_n8_s300.sh" "$@"
