#!/usr/bin/env bash
set -euo pipefail

# Reward/batch ablation based on the history-4 WebShop configuration.  Relative
# to the R2 entry point, only reward_mode and train_batch_size are changed.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export WEBSHOP_REWARD_MODE=hybrid
export WEBSHOP_R2_REWARD_ENABLE=False
export WEBSHOP_R2_INVALID_ACTION_COEF=0.05
export WEBSHOP_R2_REPEAT_ACTION_COEF=0.025
export RUBRIC_OUTCOME_SCORE_SOURCE=training_reward
export RUBRIC_CORRECTNESS_THRESHOLD=0.6
export TRAIN_BATCH_SIZE=8

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_best_history4_r2.sh" "$@"
