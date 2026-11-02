#!/usr/bin/env bash
set -euo pipefail

# Control experiment for the successful ALFWorld OPD-only route. Generate four
# rubric candidates from the same prompt, but bypass cloud judging and
# correlation selection. The group-hint implementation consumes candidate
# zero only; if candidate zero is malformed, that group receives no rubric,
# exactly as in a single-candidate generation failure.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUBRIC_GROUP_CANDIDATES=4
export RUBRIC_GROUP_SELECT_BY_CORRELATION=False
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED=False
export RUBRIC_GROUP_FORCE_FIRST_CANDIDATE=True
export RUBRIC_GROUP_GUARDED_SELECTION=False
export RUBRIC_GROUP_MIN_SELECTED_CORRELATION=0.0
export RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT=0.0

# Match the execution profile recorded for the successful historical no-pool
# run. This affects environment concurrency, not the algorithm or data budget.
export ALFWORLD_4GPU_AGENT_LOOP_WORKERS="${ALFWORLD_4GPU_AGENT_LOOP_WORKERS:-16}"
export SAVE_FREQ=100

# The training comparison does not need another post-training full-dataset
# rubric sweep. Online rubric generation and OPD remain enabled.
export RUBRIC_SNAPSHOT_ENABLE=False

export ALFWORLD_OPD_ONLY_EXPERIMENT_NAME="${ALFWORLD_FIRST_OF4_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_opd_only_first_bundle_of4_no_pool_4gpu_b8_n8_additiveopd0p01_official_reward_s300}"
export ALFWORLD_OPD_ONLY_OUTPUT_DIR="${ALFWORLD_FIRST_OF4_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME:-verl_rubric_3}/${ALFWORLD_OPD_ONLY_EXPERIMENT_NAME}}"

echo "ALFWorld OPD-only first-bundle-of-4 control"
echo "Rubric candidates=4; judge=off; Spearman selection=off; use candidate zero only"
echo "Historical execution profile: agent_workers=${ALFWORLD_4GPU_AGENT_LOOP_WORKERS}"
echo "Checkpoint interval: every ${SAVE_FREQ} steps"

exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_opd_only_no_pool_4gpu_b8_n8_s300.sh" "$@"
