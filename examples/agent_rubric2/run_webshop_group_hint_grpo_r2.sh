#!/usr/bin/env bash
set -euo pipefail

# Fixed R2-reward entry point for the WebShop online rubric experiment.
# Assign these values unconditionally so stale exports from a previous GiGPO
# run cannot change the reward or the rubric outcome definition.
export WEBSHOP_REWARD_MODE=r2
export WEBSHOP_R2_REWARD_ENABLE=True
export WEBSHOP_R2_INVALID_ACTION_COEF=0.05
export WEBSHOP_R2_REPEAT_ACTION_COEF=0.025
export RUBRIC_OUTCOME_SCORE_SOURCE=training_reward
export RUBRIC_CORRECTNESS_THRESHOLD=0.6
export _WEBSHOP_MATCHED_ENTRY=0
export WEBSHOP_GROUP_HINT_EXPERIMENT_NAME="${WEBSHOP_R2_EXPERIMENT_NAME:-qwen3_1_7b_webshop_group_hint_r2_outcome_t0p6_aligned_grpo_r2}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_group_hint_grpo.sh" "$@"
