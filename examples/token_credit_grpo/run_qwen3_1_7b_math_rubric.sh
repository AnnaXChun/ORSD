#!/usr/bin/env bash
set -euo pipefail

# Online actor-context condition. Rubrics are precomputed into the training
# parquet, while the current pre-update actor scores the same response under
# plain and rubric-augmented prompts. The fixed reference is used only by the
# standard GRPO KL loss.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_ROOT="${DATA_ROOT:-/workspace/verl_wsm/Data}"
export TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train_with_validated_rubrics.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1_7b_online_rubric_gap_001_grpo}"
export RUBRIC_PIPELINE_ENABLE="True"
export RUBRIC_PIPELINE_BACKEND="${RUBRIC_PIPELINE_BACKEND:-dataset}"
export TOKEN_CREDIT_TEACHER_KEY="rubric_actor_log_probs"
export TOKEN_CREDIT_ACTIVE_MASK_KEY="rubric_teacher_eligible"
export RUBRIC_FAIL_ON_ERROR="${RUBRIC_FAIL_ON_ERROR:-True}"
exec bash "${SCRIPT_DIR}/run_qwen3_1_7b_math.sh" "$@"
