#!/usr/bin/env bash
set -euo pipefail

# Control against run_alfworld_opd_only_no_pool_4gpu_b8_n8_s300.sh.
# Inherit its worker count, saving, validation, snapshot and optimization
# settings unchanged. Only generate four candidates and consume candidate 0.
# Use identical external environment/config overrides for both runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUBRIC_GROUP_CANDIDATES=4
export RUBRIC_GROUP_FORCE_FIRST_CANDIDATE=True
export RUBRIC_GROUP_SELECT_BY_CORRELATION=False
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED=False
export RUBRIC_GROUP_GUARDED_SELECTION=False
export RUBRIC_GROUP_MIN_SELECTED_CORRELATION=0.0
export RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT=0.0

# Separate identity and outputs prevent resuming the baseline checkpoint.
export ALFWORLD_OPD_ONLY_EXPERIMENT_NAME="${ALFWORLD_STRICT_FIRST_OF4_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_opd_only_first_bundle_of4_strict_nopool_4gpu_b8_n8_additiveopd0p01_official_reward_s300}"
export ALFWORLD_OPD_ONLY_OUTPUT_DIR="${ALFWORLD_STRICT_FIRST_OF4_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME:-verl_rubric_3}/${ALFWORLD_OPD_ONLY_EXPERIMENT_NAME}}"

echo "Strict no-pool control: four samples, candidate zero only; judge=off"
echo "Workers, checkpoint interval, validation and snapshot: inherited from the no-pool baseline"

exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_opd_only_no_pool_4gpu_b8_n8_s300.sh" "$@"
