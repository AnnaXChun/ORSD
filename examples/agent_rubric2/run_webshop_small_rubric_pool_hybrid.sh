#!/usr/bin/env bash
set -euo pipefail

# Milestone 02 candidate: keep WebShop-small + hybrid and add rubric pooling.
# BENCHMARK/rollout/context settings remain in the working shared launcher.
# Explicit assignments defeat stale reward/matched-mode exports from old runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export _WEBSHOP_MATCHED_ENTRY=0
export WEBSHOP_REWARD_MODE=hybrid
export WEBSHOP_R2_REWARD_ENABLE=False
export WEBSHOP_R2_INVALID_ACTION_COEF=0.05
export WEBSHOP_R2_REPEAT_ACTION_COEF=0.025
export RUBRIC_OUTCOME_SCORE_SOURCE=training_reward
export RUBRIC_CORRECTNESS_THRESHOLD=0.6
GENERATOR_LOSS_ENABLE="${RUBRIC_GENERATOR_LOSS_ENABLE:-True}"
GENERATOR_LOSS_COEF="${RUBRIC_GENERATOR_LOSS_COEF:-0.01}"
GENERATOR_ITEM_CREDIT_ENABLE="${RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE:-False}"
GENERATOR_INCLUDE_TRAJECTORY="${RUBRIC_POOL_GENERATOR_INCLUDE_TRAJECTORY:-True}"
GENERATOR_PROMPT_STYLE="${RUBRIC_POOL_GENERATOR_PROMPT_STYLE:-structured_json}"
GENERATOR_REWARD_SOURCE="${RUBRIC_POOL_GENERATOR_REWARD_SOURCE:-valid}"
BEHAVIOR_DEDUP_ENABLE="${RUBRIC_POOL_BEHAVIOR_DEDUP_ENABLE:-False}"
export WEBSHOP_GROUP_HINT_EXPERIMENT_NAME="${RUBRIC_POOL_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5}"
export PROJECT_NAME="${PROJECT_NAME:-verl_agent_rubric2}"
# Never accidentally resume the preceding baseline via a stale OUTPUT_DIR.
export OUTPUT_DIR="${RUBRIC_POOL_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${WEBSHOP_GROUP_HINT_EXPERIMENT_NAME}}"
export MAX_STEPS=300
export SAVE_FREQ=100
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export RUBRICS_PER_PROMPT="${RUBRICS_PER_PROMPT:-8}"
MIN_RUBRICS_PER_BUNDLE="${RUBRIC_POOL_MIN_RUBRICS_PER_BUNDLE:-5}"
export TOKEN_CREDIT_ALPHA="${TOKEN_CREDIT_ALPHA:-0.01}"
# Rubric responses need enough within-group exploration for relative generator
# rewards. This does not alter task-action rollout sampling.
export RUBRIC_TEMPERATURE="${RUBRIC_POOL_GENERATOR_TEMPERATURE:-0.7}"

# This is the rubric-pool entry point.  Force the pool on so a stale
# RUBRIC_POOL_ENABLE=False from a preceding matched/ablation run cannot make
# the launcher silently execute the legacy group-hint path.
export RUBRIC_POOL_ENABLE=True
POOL_ENABLE=True
JUDGE_MODEL="${RUBRIC_POOL_JUDGE_MODEL:-qwen3.5-plus}"
JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"
if [[ "${POOL_ENABLE}" == "True" || "${POOL_ENABLE}" == "true" ]]; then
  : "${DASHSCOPE_API_KEY:?Export DASHSCOPE_API_KEY for cloud rubric scoring}"
fi
if (( ROLLOUT_N < 2 )); then
  echo "Outcome-correlated rubric validation requires ROLLOUT_N >= 2 (recommended: 8)." >&2
  exit 1
fi
echo "Rubric pool: ${POOL_ENABLE}; actor generation per trajectory; cloud judge=${JUDGE_MODEL}; endpoint=${JUDGE_BASE_URL}"
echo "Hybrid reward fixed; rubric PPO=${GENERATOR_LOSS_ENABLE}, coef=${GENERATOR_LOSS_COEF}, item credit=${GENERATOR_ITEM_CREDIT_ENABLE} (effective only with pool)."
echo "Rubric generator conditioning: include trajectory=${GENERATOR_INCLUDE_TRAJECTORY}; reward source=${GENERATOR_REWARD_SOURCE}."
if [[ "${GENERATOR_PROMPT_STYLE}" == "simple_numbered" ]]; then
  echo "Rubric bundle: ${MIN_RUBRICS_PER_BUNDLE}-${RUBRICS_PER_PROMPT} criteria; prompt=${GENERATOR_PROMPT_STYLE}; few-shot=off."
else
  echo "Rubric bundle: ${MIN_RUBRICS_PER_BUNDLE}-${RUBRICS_PER_PROMPT} criteria; prompt=${GENERATOR_PROMPT_STYLE}; five-item few-shot=on."
fi
echo "Rubric generator sampling: temperature=${RUBRIC_TEMPERATURE}, top_p=1 while generator PPO is enabled."
echo "Rubric screening: score std>=${RUBRIC_POOL_MIN_SCORE_STD:-0.08}, score range>=${RUBRIC_POOL_MIN_SCORE_RANGE:-0.2}, Spearman>${RUBRIC_POOL_MIN_CORRELATION:-0.2}; generator reward source=${GENERATOR_REWARD_SOURCE}; same-task baseline + batch-standardized advantage with clipped PPO."
echo "Reference-copy rejection threshold: similarity<${RUBRIC_POOL_FEWSHOT_COPY_SIMILARITY:-0.82}."
echo "Behavior dedup: enabled=${BEHAVIOR_DEDUP_ENABLE}, max selected-rubric similarity=${RUBRIC_POOL_MAX_BEHAVIOR_SIMILARITY:-0.9}."
echo "Joint mode uses full-distribution rubric sampling (top_p=1) and fixed-size FSDP micro batches."
echo "Pool evidence: full trajectory=${RUBRIC_POOL_FULL_TRAJECTORY:-True}; generator input budget=${RUBRIC_POOL_GENERATOR_MAX_INPUT_TOKENS:-0} (0=auto); cloud evidence is not shortened to actor context."
echo "Pool logging: audit=${RUBRIC_POOL_AUDIT_MODE:-compact}, every ${RUBRIC_POOL_AUDIT_EVERY_N_STEPS:-1} step(s); judge evidence and per-request progress are disabled."
echo "Judge reliability: concurrency=${RUBRIC_POOL_JUDGE_CONCURRENCY:-8}, retries=${RUBRIC_POOL_JUDGE_RETRIES:-2}, minimum trajectory coverage=${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-1.0}."

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_group_hint_grpo.sh" \
  algorithm.token_credit.rubric_pipeline.model=current_actor \
  "algorithm.token_credit.rubric_pipeline.pool_enable=${POOL_ENABLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_full_trajectory=${RUBRIC_POOL_FULL_TRAJECTORY:-True}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_include_trajectory=${GENERATOR_INCLUDE_TRAJECTORY}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_prompt_style=${GENERATOR_PROMPT_STYLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_max_input_tokens=${RUBRIC_POOL_GENERATOR_MAX_INPUT_TOKENS:-0}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_context_reserve_tokens=${RUBRIC_POOL_CONTEXT_RESERVE_TOKENS:-256}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=${GENERATOR_LOSS_ENABLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_loss_coef=${GENERATOR_LOSS_COEF}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_micro_batch_size=${RUBRIC_GENERATOR_MICRO_BATCH_SIZE:-1}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_clip_ratio=${RUBRIC_GENERATOR_CLIP_RATIO:-0.2}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_item_credit_enable=${GENERATOR_ITEM_CREDIT_ENABLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_enable=${RUBRIC_GENERATOR_LINEAR_DECAY_ENABLE:-False}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_final_scale=${RUBRIC_GENERATOR_LINEAR_DECAY_FINAL_SCALE:-0.0}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_start_step=${RUBRIC_GENERATOR_LINEAR_DECAY_START_STEP:-0}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_enable=${RUBRIC_GENERATOR_LINEAR_WARMUP_ENABLE:-False}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_steps=${RUBRIC_GENERATOR_LINEAR_WARMUP_STEPS:-0}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_initial_scale=${RUBRIC_GENERATOR_LINEAR_WARMUP_INITIAL_SCALE:-0.0}" \
  "algorithm.token_credit.rubric_pipeline.pool_judge_model=${JUDGE_MODEL}" \
  "algorithm.token_credit.rubric_pipeline.pool_judge_base_url=${JUDGE_BASE_URL}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_api_key_env=DASHSCOPE_API_KEY \
  "algorithm.token_credit.rubric_pipeline.pool_judge_concurrency=${RUBRIC_POOL_JUDGE_CONCURRENCY:-8}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_timeout=90 \
  "algorithm.token_credit.rubric_pipeline.pool_judge_retries=${RUBRIC_POOL_JUDGE_RETRIES:-2}" \
  "algorithm.token_credit.rubric_pipeline.pool_judge_max_tokens=${RUBRIC_POOL_JUDGE_MAX_TOKENS:-2048}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_items_per_request=64 \
  "algorithm.token_credit.rubric_pipeline.pool_min_rubrics_per_bundle=${MIN_RUBRICS_PER_BUNDLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_min_correlation=${RUBRIC_POOL_MIN_CORRELATION:-0.2}" \
  "algorithm.token_credit.rubric_pipeline.pool_min_score_std=${RUBRIC_POOL_MIN_SCORE_STD:-0.08}" \
  "algorithm.token_credit.rubric_pipeline.pool_min_score_range=${RUBRIC_POOL_MIN_SCORE_RANGE:-0.2}" \
  "algorithm.token_credit.rubric_pipeline.pool_min_judge_coverage=${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-1.0}" \
  "algorithm.token_credit.rubric_pipeline.pool_behavior_dedup_enable=${BEHAVIOR_DEDUP_ENABLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_max_behavior_similarity=${RUBRIC_POOL_MAX_BEHAVIOR_SIMILARITY:-0.9}" \
  "algorithm.token_credit.rubric_pipeline.pool_fewshot_copy_similarity=${RUBRIC_POOL_FEWSHOT_COPY_SIMILARITY:-0.82}" \
  algorithm.token_credit.rubric_pipeline.pool_max_selected=8 \
  algorithm.token_credit.rubric_pipeline.pool_min_selected=1 \
  "algorithm.token_credit.rubric_pipeline.pool_generator_reward_source=${GENERATOR_REWARD_SOURCE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_support_threshold=${RUBRIC_POOL_GENERATOR_SUPPORT_THRESHOLD:-0.5}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_min_satisfied=${RUBRIC_POOL_GENERATOR_MIN_SATISFIED:-2}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_min_unsatisfied=${RUBRIC_POOL_GENERATOR_MIN_UNSATISFIED:-2}" \
  "algorithm.token_credit.rubric_pipeline.pool_audit_mode=${RUBRIC_POOL_AUDIT_MODE:-compact}" \
  "algorithm.token_credit.rubric_pipeline.pool_audit_every_n_steps=${RUBRIC_POOL_AUDIT_EVERY_N_STEPS:-1}" \
  "trainer.analysis_log_enable=${ANALYSIS_LOG_ENABLE:-True}" \
  "trainer.analysis_log_every_n_steps=${ANALYSIS_LOG_EVERY_N_STEPS:-1}" \
  "algorithm.token_credit.rubric_pipeline.pool_console_progress=${RUBRIC_POOL_CONSOLE_PROGRESS:-False}" \
  "$@"
