#!/usr/bin/env bash
set -euo pipefail

# Unprivileged smoke-test condition: the fixed reference teacher sees only the
# same original prompt as the student. Force these values so an exported rubric
# setting from a previous shell cannot contaminate this control.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1_7b_additive_gap_001_grpo_no_rubric}"
export RUBRIC_PIPELINE_ENABLE="False"
export TOKEN_CREDIT_TEACHER_KEY="ref_log_prob"
export TOKEN_CREDIT_ACTIVE_MASK_KEY="null"
exec bash "${SCRIPT_DIR}/run_qwen3_1_7b_math.sh" "$@"
