#!/usr/bin/env bash
set -euo pipefail

# WebShop best-config candidate.  Every experimental choice is fixed here so
# the run is reproducible and cannot inherit stale settings from another shell.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Runtime and resources: one eight-GPU node. WebShop keeps one environment
# worker to avoid duplicating the product catalogue and Lucene index in RAM.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export N_GPUS=8
export WEBSHOP_ROOT="${WEBSHOP_ROOT:-${ROOT}/WebShop}"
export WEBSHOP_AGENT_LOOP_NUM_WORKERS=1
export WEBSHOP_DATALOADER_NUM_WORKERS=1
export USE_REWARD_LOOP=False
export ROLLOUT_TP=2
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.60

# GiGPO-style shaped training reward:
#   reward = 10 * strict_success - 0.1 * invalid_action_count
# Validation continues to report both official WebShop score and strict
# success, so the result remains comparable with the matched baseline.
export WEBSHOP_REWARD_MODE=gigpo
export WEBSHOP_GIGPO_SUCCESS_REWARD=10.0
export WEBSHOP_GIGPO_INVALID_ACTION_COEF=0.1
export WEBSHOP_R2_REWARD_ENABLE=False

# Online group-rubric token credit. Only mixed success/failure groups are used,
# and a rubric correction is retained only when its gap direction agrees with
# the observed outcome ordering; every rejected group falls back to pure GRPO.
export _WEBSHOP_MATCHED_ENTRY=0
export TOKEN_CREDIT_ENABLE=True
export RUBRIC_PIPELINE_ENABLE=True
export RUBRIC_BACKEND=actor_rollout_group_hints
export RUBRIC_OUTCOME_SCORE_SOURCE=strict_success
export RUBRIC_CORRECTNESS_THRESHOLD=0.5
export REQUIRE_GROUP_OUTCOME_CONTRAST=True
export OUTCOME_CONTRAST_EPSILON=1e-6
export REQUIRE_OUTCOME_ALIGNED_GAP=True
export MIN_OUTCOME_GAP_ALIGNMENT=0.0
export TOKEN_CREDIT_ALPHA=0.01

# One current-actor rubric is distilled from four reward-contrasting attempts.
# Thinking is enabled for the policy, rubric generator, and conditioned teacher
# pass. Generated thinking remains part of policy training and is retained for
# the two turns that remain visible through the history window.
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

# Agent context: keep the fixed system/task/initial-observation prefix and the
# latest two complete ACTION/observation turns. Full responses, including
# thinking tokens, are still saved for PPO/GRPO loss and replayed under the
# exact context used during rollout.
export HISTORY_WINDOW=2
export HISTORY_FIXED_PREFIX_MESSAGES=3
export HISTORY_RESPONSE_MODE=full_response
export HISTORY_MIN_GENERATION_TOKENS=2048
export ACTOR_MAX_TOKENS_PER_TURN=2048
export MAX_ASSISTANT_TURNS=15
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=32768
export MAX_MODEL_LENGTH=40960
export CALCULATE_ROLLOUT_LOG_PROBS=True

# GRPO optimization: four prompts x eight trajectories = 32 episodes/step.
# Turn replay uses micro-batch one to keep the two-context OPD pass stable.
export TRAIN_BATCH_SIZE=4
export ROLLOUT_N=8
export MICRO_BATCH_SIZE=1
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export LEARNING_RATE=1e-6
export KL_COEF=0.001

# Evaluation and checkpointing. Validation samples four trajectories per task.
export VAL_N=4
export VAL_TEMPERATURE=0.6
export MAX_STEPS=1000
export TEST_FREQ=20
export SAVE_FREQ=25
export TOTAL_EPOCHS=20
export PROJECT_NAME=verl_agent_rubric2
export WEBSHOP_GROUP_HINT_EXPERIMENT_NAME=qwen3_1_7b_webshop_best_history2_gigpo_rubric_grpo

echo "Launching WebShop best config: history_window=2, GiGPO reward, online rubric GRPO"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_group_hint_grpo.sh" "$@"
