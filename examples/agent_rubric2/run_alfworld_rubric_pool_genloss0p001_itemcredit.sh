#!/usr/bin/env bash
set -euo pipefail

# Fixed 300-step ALFWorld rubric-pool experiment: 8 prompts x 8 rollouts,
# additive OPD=0.01, and auxiliary generator loss beta=0.001.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"
export ALFWORLD_R2_REWARD_ENABLE=False
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export VAL_N="${VAL_N:-4}"
export TEST_FREQ="${TEST_FREQ:-50}"
export ALFWORLD_POOL_AUDIT_EVERY_N_STEPS=10
export REQUIRE_GROUP_OUTCOME_CONTRAST=True
export REQUIRE_OUTCOME_ALIGNED_GAP="${REQUIRE_OUTCOME_ALIGNED_GAP:-True}"
export MIN_OUTCOME_GAP_ALIGNMENT=0.0
export TOKEN_CREDIT_MODE=additive_gap
export TOKEN_CREDIT_ALPHA=0.01
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=True
export TOKEN_CREDIT_LINEAR_DECAY_START_STEP=150
export TOKEN_CREDIT_LINEAR_DECAY_FINAL_SCALE=0.0
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export ALFWORLD_POOL_GENERATOR_LOSS_COEF=0.001
export ALFWORLD_POOL_MIN_CORRELATION=0.4
export ALFWORLD_POOL_GENERATOR_REWARD_SOURCE=selected_loo_spearman_relu
export ALFWORLD_POOL_GENERATOR_SUPPORT_THRESHOLD=0.5
export ALFWORLD_POOL_GENERATOR_MIN_SATISFIED=2
export ALFWORLD_POOL_GENERATOR_MIN_UNSATISFIED=2
export ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_ENABLE=True
export ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_START_STEP=150
export ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_FINAL_SCALE=0.0
export ALFWORLD_POOL_GENERATOR_LINEAR_WARMUP_ENABLE=False
# Concise process-rubric generation: no demonstration messages, one numbered
# 5--8 item rubric grounded in the current task and trajectory.
export ALFWORLD_POOL_GENERATOR_PROMPT_STYLE=simple_numbered
export ALFWORLD_POOL_GENERATOR_INCLUDE_TRAJECTORY=True
export ALFWORLD_POOL_MIN_RUBRICS_PER_BUNDLE=5
export RUBRIC_TEMPERATURE=0.7
export RUBRIC_TOP_P=0.9
export RUBRIC_GENERATOR_THINKING="${RUBRIC_GENERATOR_THINKING:-False}"

case "${ALFWORLD_R2_REWARD_ENABLE:-True}" in
  True|true|TRUE|1|yes|YES|on|ON)
    _ALFWORLD_REWARD_SUFFIX="_v2"
    ;;
  *)
    _ALFWORLD_REWARD_SUFFIX="_official_reward"
    ;;
esac
export ALFWORLD_POOL_EXPERIMENT_NAME="${ALFWORLD_POOL_GEN0001_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_pool_4gpu_b8_n8_additiveopd0p01_decayafter150_genloss0p001_decayafter150_loo_spearman0p4_support2x2_simpleprompt_nofewshot_nogenthink_official_reward_s300}"

echo "ALFWorld coefficient comparison: rubric-pool item credit; generator loss beta=0.001"
echo "Training protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, additive OPD=${TOKEN_CREDIT_ALPHA}, steps=${MAX_STEPS}, official reward"
echo "Validation protocol: seen+unseen, rollouts/task=${VAL_N}, every=${TEST_FREQ} training steps"
echo "Schedules: OPD and generator loss stay fixed through step 150, then decay linearly to 0 at step 300"
echo "Rubric filtering: selected LOO Spearman-ReLU reward, Spearman>0.4, support>=2 satisfied and >=2 unsatisfied"
echo "Second outcome-gap gate: ${REQUIRE_OUTCOME_ALIGNED_GAP}"
echo "Rubric generator: prompt=simple_numbered, fewshot=off, items=5-8, temperature=${RUBRIC_TEMPERATURE}, thinking=${RUBRIC_GENERATOR_THINKING}"
echo "Expected W&B sentinel: rubric_pool/enabled=1 after the first completed training step"
exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_rubric_pool_genloss0p003_itemcredit.sh" \
  algorithm.token_credit.rubric_pipeline.enable=True \
  algorithm.token_credit.enable=True \
  algorithm.token_credit.mode=additive_gap \
  algorithm.token_credit.alpha=0.01 \
  algorithm.token_credit.linear_decay_enable=True \
  algorithm.token_credit.linear_decay_start_step=150 \
  algorithm.token_credit.linear_decay_final_scale=0.0 \
  algorithm.token_credit.linear_warmup_enable=False \
  algorithm.token_credit.rubric_pipeline.backend=actor_rollout_group_hints \
  algorithm.token_credit.rubric_pipeline.pool_enable=True \
  algorithm.token_credit.rubric_pipeline.pool_generator_prompt_style=simple_numbered \
  algorithm.token_credit.rubric_pipeline.pool_generator_include_trajectory=True \
  algorithm.token_credit.rubric_pipeline.pool_min_rubrics_per_bundle=5 \
  algorithm.token_credit.rubric_pipeline.pool_min_correlation=0.4 \
  algorithm.token_credit.rubric_pipeline.pool_generator_reward_source=selected_loo_spearman_relu \
  algorithm.token_credit.rubric_pipeline.pool_generator_support_threshold=0.5 \
  algorithm.token_credit.rubric_pipeline.pool_generator_min_satisfied=2 \
  algorithm.token_credit.rubric_pipeline.pool_generator_min_unsatisfied=2 \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=True \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_coef=0.001 \
  algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_enable=True \
  algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_start_step=150 \
  algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_final_scale=0.0 \
  algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_enable=False \
  trainer.total_training_steps=300 \
  "$@"
