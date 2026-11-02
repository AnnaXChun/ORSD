#!/usr/bin/env bash
set -euo pipefail

# WebShop recommended configuration: preserve the official continuous score
# through R2 shaping, use the same score for rubric grouping, and retain four
# recent full reasoning/action turns.  Values are assigned unconditionally so
# stale exports from earlier GiGPO experiments cannot change this run.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Runtime and resources.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export N_GPUS=8
export WEBSHOP_ROOT="${WEBSHOP_ROOT:-${ROOT}/WebShop}"
export WEBSHOP_AGENT_LOOP_NUM_WORKERS=1
export WEBSHOP_DATALOADER_NUM_WORKERS=1
export USE_REWARD_LOOP=False
export ROLLOUT_TP=2
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.60

# R2 keeps the official continuous WebShop score and only subtracts small
# invalid/no-progress penalties.  Rubric grouping uses this exact training
# reward; 0.6 separates sufficiently useful and weak attempts.
export WEBSHOP_REWARD_MODE="${WEBSHOP_REWARD_MODE:-r2}"
export WEBSHOP_R2_REWARD_ENABLE="${WEBSHOP_R2_REWARD_ENABLE:-True}"
export WEBSHOP_R2_INVALID_ACTION_COEF="${WEBSHOP_R2_INVALID_ACTION_COEF:-0.05}"
export WEBSHOP_R2_REPEAT_ACTION_COEF="${WEBSHOP_R2_REPEAT_ACTION_COEF:-0.025}"
export RUBRIC_OUTCOME_SCORE_SOURCE="${RUBRIC_OUTCOME_SCORE_SOURCE:-training_reward}"
export RUBRIC_CORRECTNESS_THRESHOLD="${RUBRIC_CORRECTNESS_THRESHOLD:-0.6}"

# Online group-rubric token credit.  Non-contrastive and directionally
# inconsistent groups fall back to ordinary GRPO.
export _WEBSHOP_MATCHED_ENTRY="${_WEBSHOP_MATCHED_ENTRY:-0}"
export TOKEN_CREDIT_ENABLE=True
export RUBRIC_PIPELINE_ENABLE=True
export RUBRIC_BACKEND=actor_rollout_group_hints
export REQUIRE_GROUP_OUTCOME_CONTRAST=True
export OUTCOME_CONTRAST_EPSILON=1e-6
export REQUIRE_OUTCOME_ALIGNED_GAP=True
export MIN_OUTCOME_GAP_ALIGNMENT=0.0
export TOKEN_CREDIT_ALPHA=0.01

# Current-actor rubric generation and conditioned replay.
export ACTOR_THINKING=True
export RUBRIC_GENERATOR_THINKING=True
export RUBRIC_JUDGE_THINKING=True
export TEACHER_THINKING=True
export RUBRICS_PER_PROMPT=5
export RUBRIC_MAX_TRAJECTORIES=4
export RUBRIC_MAX_TRAJECTORY_CHARS=6000
export RUBRIC_MAX_RUBRIC_CHARS=1600
export RUBRIC_MAX_ITEM_CHARS=280
export RUBRIC_MAX_TOKENS=4096
export RUBRIC_TEMPERATURE=0.3
export RUBRIC_TOP_P=0.9
export TEACHER_MAX_PROMPT_LENGTH=40960

# Keep the fixed system/task/initial-observation prefix plus the latest four
# complete assistant/observation turns.  Full responses remain the policy
# outputs used by exact turn replay; past thinking outside the window is not
# visible to later turns.
export HISTORY_WINDOW=4
export HISTORY_FIXED_PREFIX_MESSAGES=3
export HISTORY_RESPONSE_MODE=full_response
export HISTORY_MIN_GENERATION_TOKENS=2048
export ACTOR_MAX_TOKENS_PER_TURN=2048
export MAX_ASSISTANT_TURNS=15
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=32768
export MAX_MODEL_LENGTH=40960
export CALCULATE_ROLLOUT_LOG_PROBS=True

# R2 defaults to four prompts x eight trajectories. Reward/batch ablation
# wrappers may override only TRAIN_BATCH_SIZE while leaving ROLLOUT_N fixed.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export ROLLOUT_N=8
export MICRO_BATCH_SIZE=1
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export LEARNING_RATE=1e-6
export KL_COEF=0.001

# Evaluation and checkpointing.
export VAL_N=4
export VAL_TEMPERATURE=0.6
export MAX_STEPS=1000
export TEST_FREQ=20
export SAVE_FREQ=25
export TOTAL_EPOCHS=20
export PROJECT_NAME=verl_agent_rubric2
export WEBSHOP_GROUP_HINT_EXPERIMENT_NAME="${WEBSHOP_GROUP_HINT_EXPERIMENT_NAME:-qwen3_1_7b_webshop_best_history4_r2_rubric_grpo}"

echo "Launching WebShop recommended config: history_window=4, reward=${WEBSHOP_REWARD_MODE}, train_batch_size=${TRAIN_BATCH_SIZE}, online rubric GRPO"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_group_hint_grpo.sh" "$@"
