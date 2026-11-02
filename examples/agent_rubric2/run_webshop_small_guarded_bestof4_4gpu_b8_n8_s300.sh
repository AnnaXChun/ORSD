#!/usr/bin/env bash
set -euo pipefail

# Guarded-selection ablation for the current WebShop-small first-version run.
# Hardware, rollout, optimization, context, validation, checkpoint, and reward
# settings are inherited unchanged from run_webshop_small_first_version_4gpu.sh.
# This wrapper changes only rubric-bundle selection.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUBRIC_GROUP_CANDIDATES=4
export RUBRIC_GROUP_SELECT_BY_CORRELATION=True
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED=True
export RUBRIC_GROUP_GUARDED_SELECTION=True
export RUBRIC_GROUP_MIN_SELECTED_CORRELATION=0.0
export RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT=0.05

export WEBSHOP_FIRST_EXPERIMENT_NAME="${WEBSHOP_GUARDED_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_b8_n8_history0_traj4_guarded_bestof4_margin0p05_contrastgate_gapgate_temp0p7_hybrid_additiveopd0p01_nogenthink_4gpu_s300}"

echo "WebShop-small guarded selection: current 4-GPU, 300-step first-version protocol"
echo "Only rubric selection changes: candidates=${RUBRIC_GROUP_CANDIDATES}, guarded=${RUBRIC_GROUP_GUARDED_SELECTION}, minimum_correlation>${RUBRIC_GROUP_MIN_SELECTED_CORRELATION}, improvement>=${RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT}"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_first_version_4gpu.sh" "$@"
