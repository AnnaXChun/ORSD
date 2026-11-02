#!/usr/bin/env bash
set -euo pipefail

# ALFWorld comparison using the latest WebShop algorithm route. Environment,
# optimization, and pool semantics remain unchanged; the downstream four-GPU
# launcher supplies the matched auxiliary-token ceilings and higher rollout
# concurrency.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_NAME=verl_rubric_3
export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"

# Centered OPD: preserve the trajectory-level GRPO advantage while assigning
# relative token credit. Keep the first 150 optimizer steps flat, then decay.
export TOKEN_CREDIT_MODE=centered_opd
export TOKEN_CREDIT_ALPHA=0.1
export TOKEN_CREDIT_CENTERED_CLIP=2.0
export TOKEN_CREDIT_MULTIPLIER_MIN=0.5
export TOKEN_CREDIT_MULTIPLIER_MAX=1.5
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=True
export TOKEN_CREDIT_LINEAR_DECAY_START_STEP=150
export TOKEN_CREDIT_LINEAR_DECAY_FINAL_SCALE=0.0
export ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_ENABLE=True
export ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_START_STEP=150
export ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_FINAL_SCALE=0.0
export ALFWORLD_POOL_MIN_CORRELATION=0.4
export ALFWORLD_POOL_GENERATOR_REWARD_SOURCE=selected_loo_spearman_relu
export ALFWORLD_POOL_GENERATOR_SUPPORT_THRESHOLD=0.5
export ALFWORLD_POOL_GENERATOR_MIN_SATISFIED=2
export ALFWORLD_POOL_GENERATOR_MIN_UNSATISFIED=2
export ALFWORLD_POOL_RUBRICS_PER_PROMPT=8
export ALFWORLD_POOL_MAX_SELECTED=8
# Use the same score optimized by GRPO for rubric validation and the centered
# OPD group gate. The official binary success metric remains unchanged for
# evaluation. Override with strict_success to reproduce the previous route.
export ALFWORLD_POOL_OUTCOME_SCORE_SOURCE="${ALFWORLD_POOL_OUTCOME_SCORE_SOURCE:-training_reward}"
# Match the WebShop centered-OPD rubric-generator route. Keep every generator
# control explicit so legacy ALFWorld defaults or stale shell exports cannot
# change the comparison.
export RUBRIC_TEMPERATURE=0.7
export RUBRIC_TOP_P=0.9
export RUBRIC_GENERATOR_THINKING=False
export RUBRIC_MAX_TRAJECTORIES=4
export RUBRIC_MAX_TRAJECTORY_CHARS=6000
export RUBRIC_MAX_RUBRIC_CHARS=1600
export RUBRIC_MAX_ITEM_CHARS=280
export ALFWORLD_POOL_FULL_TRAJECTORY=True
export ALFWORLD_POOL_GENERATOR_INCLUDE_TRAJECTORY=True
export ALFWORLD_POOL_GENERATOR_PROMPT_STYLE=structured_json
export ALFWORLD_POOL_GENERATOR_MAX_INPUT_TOKENS=0
export ALFWORLD_POOL_GENERATOR_CONTEXT_RESERVE_TOKENS=256
export ALFWORLD_POOL_GENERATOR_MICRO_BATCH_SIZE=1
export ALFWORLD_POOL_GENERATOR_CLIP_RATIO=0.2
export ALFWORLD_POOL_GENERATOR_ITEM_CREDIT_ENABLE=True
export ALFWORLD_POOL_MIN_RUBRICS_PER_BUNDLE=5

case "${ALFWORLD_R2_REWARD_ENABLE:-True}" in
  True|true|TRUE|1|yes|YES|on|ON)
    REWARD_EXPERIMENT_SUFFIX="_v2"
    ;;
  *)
    REWARD_EXPERIMENT_SUFFIX="_official_reward"
    ;;
esac
export ALFWORLD_POOL_EXPERIMENT_NAME="${ALFWORLD_LATEST_POOL_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_pool_4gpu_b8_n8_centeredopd0p1_decayafter150_genloss0p003_decayafter150_loo_spearman0p4_support2x2_trainrewardgate_genmatch_r8s8_rt1024_w16_jt1536_jc8${REWARD_EXPERIMENT_SUFFIX}}"

echo "ALFWorld latest-algorithm comparison: centered OPD alpha=0.1; generator loss=0.003"
echo "Schedules: constant through step 150, then linear decay to the final training step"
echo "Generator reward: selected LOO ReLU(Spearman-0.4), support>=2 satisfied and >=2 unsatisfied"
echo "Rubric outcome and OPD gate score: ${ALFWORLD_POOL_OUTCOME_SCORE_SOURCE}"
echo "Rubric capacity: generated=8 per trajectory, selected<=8 (matched to WebShop)"
echo "Rubric generator sampling: temperature=${RUBRIC_TEMPERATURE}, effective_top_p=1.0, thinking=${RUBRIC_GENERATOR_THINKING}, prompt_style=${ALFWORLD_POOL_GENERATOR_PROMPT_STYLE}"

exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_rubric_pool_genloss0p003_itemcredit.sh" "$@"
