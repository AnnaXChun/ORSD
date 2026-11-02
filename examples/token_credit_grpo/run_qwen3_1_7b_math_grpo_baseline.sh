#!/usr/bin/env bash
set -euo pipefail

# Strictly matched full-parameter GRPO baseline. Reuse the same model, data,
# rollout, optimizer, KL, validation, checkpoint, and resource settings as the
# token-credit experiment, then force every token-credit path off.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1_7b_grpo_baseline_matched}"
export RUBRIC_PIPELINE_ENABLE="False"
export TOKEN_CREDIT_TEACHER_KEY="ref_log_prob"
export TOKEN_CREDIT_ACTIVE_MASK_KEY="null"

# Put the disabling overrides after user-supplied arguments so this script
# cannot accidentally become a token-credit run through an outer environment.
exec bash "${SCRIPT_DIR}/run_qwen3_1_7b_math.sh" \
    "$@" \
    algorithm.token_credit.enable=False \
    algorithm.token_credit.rubric_pipeline.enable=False
