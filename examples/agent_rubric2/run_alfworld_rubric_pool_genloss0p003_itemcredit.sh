#!/usr/bin/env bash
set -euo pipefail

# ALFWorld comparison for the previously working group-hint launcher. The
# optimization protocol stays unchanged; rollout concurrency and auxiliary
# generation ceilings follow the four-GPU throughput profile.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Match the current WebShop four-GPU protocol: 8 prompt groups x 8 rollouts.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _ALFWORLD_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_ALFWORLD_VISIBLE_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export N_GPUS=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ALFWORLD_4GPU_VLLM_GPU_UTIL:-0.50}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ALFWORLD_4GPU_BATCHED_TOKENS:-32768}"
export ROLLOUT_MAX_NUM_SEQS="${ALFWORLD_4GPU_MAX_NUM_SEQS:-64}"
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
# Match WebShop's rubric proposal and teacher-context capacity.  The shared
# ALFWorld launcher otherwise defaults to five proposals per trajectory.
export RUBRICS_PER_PROMPT="${ALFWORLD_POOL_RUBRICS_PER_PROMPT:-8}"
POOL_MAX_SELECTED="${ALFWORLD_POOL_MAX_SELECTED:-8}"
# Each ALFWorld worker loads its own task/environment state. Eight workers keep
# useful request-level parallelism for 64 episodes without the host-memory
# spikes observed when sixteen workers initialize the 8,810-task index.
export AGENT_LOOP_NUM_WORKERS="${ALFWORLD_4GPU_AGENT_LOOP_WORKERS:-8}"
export DATALOADER_NUM_WORKERS="${ALFWORLD_4GPU_DATALOADER_WORKERS:-2}"
# Match the WebShop four-GPU auxiliary-generation limits.
export RUBRIC_MAX_TOKENS="${ALFWORLD_4GPU_RUBRIC_MAX_TOKENS:-1024}"

# Force the rubric-enabled branch even if this shell previously launched the
# matched-GRPO wrapper.
export _ALFWORLD_MATCHED_ENTRY=0
export ALFWORLD_R2_REWARD_ENABLE="${ALFWORLD_R2_REWARD_ENABLE:-True}"
export PROJECT_NAME="${PROJECT_NAME:-verl_agent_rubric2}"

case "${ALFWORLD_R2_REWARD_ENABLE}" in
  True|true|TRUE|1|yes|YES|on|ON)
    REWARD_EXPERIMENT_SUFFIX="_v2"
    ;;
  *)
    REWARD_EXPERIMENT_SUFFIX="_official_reward"
    ;;
esac

export ALFWORLD_GROUP_HINT_EXPERIMENT_NAME="${ALFWORLD_POOL_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_pool_4gpu_b8_n8_genloss0p003_itemcredit${REWARD_EXPERIMENT_SUFFIX}}"
export OUTPUT_DIR="${ALFWORLD_POOL_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${ALFWORLD_GROUP_HINT_EXPERIMENT_NAME}}"

: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY for the rubric-pool cloud judge}"
JUDGE_MODEL="${ALFWORLD_POOL_JUDGE_MODEL:-qwen3.5-plus}"
JUDGE_BASE_URL="${ALFWORLD_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"
JUDGE_CONCURRENCY="${ALFWORLD_POOL_JUDGE_CONCURRENCY:-8}"
JUDGE_MAX_TOKENS="${ALFWORLD_POOL_JUDGE_MAX_TOKENS:-1536}"
JUDGE_RETRIES="${ALFWORLD_POOL_JUDGE_RETRIES:-4}"
MIN_JUDGE_COVERAGE="${ALFWORLD_POOL_MIN_JUDGE_COVERAGE:-0.75}"
GENERATOR_LOSS_COEF="${ALFWORLD_POOL_GENERATOR_LOSS_COEF:-0.003}"
MIN_CORRELATION="${ALFWORLD_POOL_MIN_CORRELATION:-0.4}"
GENERATOR_REWARD_SOURCE="${ALFWORLD_POOL_GENERATOR_REWARD_SOURCE:-selected_loo_spearman_relu}"
GENERATOR_DECAY_ENABLE="${ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_ENABLE:-False}"
GENERATOR_DECAY_START_STEP="${ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_START_STEP:-0}"
GENERATOR_DECAY_FINAL_SCALE="${ALFWORLD_POOL_GENERATOR_LINEAR_DECAY_FINAL_SCALE:-0.0}"
GENERATOR_WARMUP_ENABLE="${ALFWORLD_POOL_GENERATOR_LINEAR_WARMUP_ENABLE:-False}"
GENERATOR_WARMUP_STEPS="${ALFWORLD_POOL_GENERATOR_LINEAR_WARMUP_STEPS:-0}"
GENERATOR_WARMUP_INITIAL_SCALE="${ALFWORLD_POOL_GENERATOR_LINEAR_WARMUP_INITIAL_SCALE:-0.0}"
GENERATOR_SUPPORT_THRESHOLD="${ALFWORLD_POOL_GENERATOR_SUPPORT_THRESHOLD:-0.5}"
GENERATOR_MIN_SATISFIED="${ALFWORLD_POOL_GENERATOR_MIN_SATISFIED:-2}"
GENERATOR_MIN_UNSATISFIED="${ALFWORLD_POOL_GENERATOR_MIN_UNSATISFIED:-2}"
OUTCOME_SCORE_SOURCE="${ALFWORLD_POOL_OUTCOME_SCORE_SOURCE:-strict_success}"
GENERATOR_INCLUDE_TRAJECTORY="${ALFWORLD_POOL_GENERATOR_INCLUDE_TRAJECTORY:-True}"
GENERATOR_PROMPT_STYLE="${ALFWORLD_POOL_GENERATOR_PROMPT_STYLE:-structured_json}"
GENERATOR_MAX_INPUT_TOKENS="${ALFWORLD_POOL_GENERATOR_MAX_INPUT_TOKENS:-0}"
GENERATOR_CONTEXT_RESERVE_TOKENS="${ALFWORLD_POOL_GENERATOR_CONTEXT_RESERVE_TOKENS:-256}"
GENERATOR_MICRO_BATCH_SIZE="${ALFWORLD_POOL_GENERATOR_MICRO_BATCH_SIZE:-1}"
GENERATOR_CLIP_RATIO="${ALFWORLD_POOL_GENERATOR_CLIP_RATIO:-0.2}"
GENERATOR_ITEM_CREDIT_ENABLE="${ALFWORLD_POOL_GENERATOR_ITEM_CREDIT_ENABLE:-True}"
POOL_FULL_TRAJECTORY="${ALFWORLD_POOL_FULL_TRAJECTORY:-True}"
MIN_RUBRICS_PER_BUNDLE="${ALFWORLD_POOL_MIN_RUBRICS_PER_BUNDLE:-5}"
POOL_AUDIT_EVERY_N_STEPS="${ALFWORLD_POOL_AUDIT_EVERY_N_STEPS:-1}"

echo "ALFWorld rubric-pool comparison: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, total=$((TRAIN_BATCH_SIZE * ROLLOUT_N)), max_steps=${MAX_STEPS}"
echo "Generator: rubrics/trajectory=${RUBRICS_PER_PROMPT}, genloss=${GENERATOR_LOSS_COEF}, item_credit=True, Spearman>${MIN_CORRELATION}, reward=${GENERATOR_REWARD_SOURCE}, selected<=${POOL_MAX_SELECTED}"
echo "Rubric outcome and OPD gate score: ${OUTCOME_SCORE_SOURCE}"
echo "Generator config: include_trajectory=${GENERATOR_INCLUDE_TRAJECTORY}, prompt_style=${GENERATOR_PROMPT_STYLE}, max_input_tokens=${GENERATOR_MAX_INPUT_TOKENS}, context_reserve=${GENERATOR_CONTEXT_RESERVE_TOKENS}, micro_batch=${GENERATOR_MICRO_BATCH_SIZE}, clip_ratio=${GENERATOR_CLIP_RATIO}, item_credit=${GENERATOR_ITEM_CREDIT_ENABLE}"
echo "Four-GPU profile: devices=${CUDA_VISIBLE_DEVICES}, TP=${ROLLOUT_TP}, vLLM_util=${ROLLOUT_GPU_MEMORY_UTILIZATION}, agent_workers=${AGENT_LOOP_NUM_WORKERS}"
echo "Auxiliary generation: rubric_max_tokens=${RUBRIC_MAX_TOKENS}, judge_max_tokens=${JUDGE_MAX_TOKENS}"
echo "Cloud judge: model=${JUDGE_MODEL}, endpoint=${JUDGE_BASE_URL}, concurrency=${JUDGE_CONCURRENCY}, retries=${JUDGE_RETRIES}, min_coverage=${MIN_JUDGE_COVERAGE}"
echo "Experiment: ${ALFWORLD_GROUP_HINT_EXPERIMENT_NAME}; checkpoints=${OUTPUT_DIR}"

exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_group_hint_grpo.sh" \
  algorithm.token_credit.rubric_pipeline.model=current_actor \
  algorithm.token_credit.rubric_pipeline.pool_enable=True \
  "algorithm.token_credit.rubric_pipeline.pool_full_trajectory=${POOL_FULL_TRAJECTORY}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_include_trajectory=${GENERATOR_INCLUDE_TRAJECTORY}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_prompt_style=${GENERATOR_PROMPT_STYLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_max_input_tokens=${GENERATOR_MAX_INPUT_TOKENS}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_context_reserve_tokens=${GENERATOR_CONTEXT_RESERVE_TOKENS}" \
  "algorithm.token_credit.rubric_pipeline.outcome_score_source=${OUTCOME_SCORE_SOURCE}" \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=True \
  "algorithm.token_credit.rubric_pipeline.pool_generator_loss_coef=${GENERATOR_LOSS_COEF}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_micro_batch_size=${GENERATOR_MICRO_BATCH_SIZE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_clip_ratio=${GENERATOR_CLIP_RATIO}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_item_credit_enable=${GENERATOR_ITEM_CREDIT_ENABLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_enable=${GENERATOR_DECAY_ENABLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_start_step=${GENERATOR_DECAY_START_STEP}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_final_scale=${GENERATOR_DECAY_FINAL_SCALE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_enable=${GENERATOR_WARMUP_ENABLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_steps=${GENERATOR_WARMUP_STEPS}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_initial_scale=${GENERATOR_WARMUP_INITIAL_SCALE}" \
  "algorithm.token_credit.rubric_pipeline.pool_judge_model=${JUDGE_MODEL}" \
  "algorithm.token_credit.rubric_pipeline.pool_judge_base_url=${JUDGE_BASE_URL}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_api_key_env=DASHSCOPE_API_KEY \
  "algorithm.token_credit.rubric_pipeline.pool_judge_concurrency=${JUDGE_CONCURRENCY}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_timeout=90 \
  "algorithm.token_credit.rubric_pipeline.pool_judge_retries=${JUDGE_RETRIES}" \
  "algorithm.token_credit.rubric_pipeline.pool_judge_max_tokens=${JUDGE_MAX_TOKENS}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_items_per_request=64 \
  "algorithm.token_credit.rubric_pipeline.pool_min_rubrics_per_bundle=${MIN_RUBRICS_PER_BUNDLE}" \
  "algorithm.token_credit.rubric_pipeline.pool_min_correlation=${MIN_CORRELATION}" \
  algorithm.token_credit.rubric_pipeline.pool_min_score_std=0.08 \
  algorithm.token_credit.rubric_pipeline.pool_min_score_range=0.2 \
  "algorithm.token_credit.rubric_pipeline.pool_min_judge_coverage=${MIN_JUDGE_COVERAGE}" \
  algorithm.token_credit.rubric_pipeline.pool_behavior_dedup_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_fewshot_copy_similarity=0.82 \
  "algorithm.token_credit.rubric_pipeline.pool_max_selected=${POOL_MAX_SELECTED}" \
  algorithm.token_credit.rubric_pipeline.pool_min_selected=1 \
  "algorithm.token_credit.rubric_pipeline.pool_generator_reward_source=${GENERATOR_REWARD_SOURCE}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_support_threshold=${GENERATOR_SUPPORT_THRESHOLD}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_min_satisfied=${GENERATOR_MIN_SATISFIED}" \
  "algorithm.token_credit.rubric_pipeline.pool_generator_min_unsatisfied=${GENERATOR_MIN_UNSATISFIED}" \
  algorithm.token_credit.rubric_pipeline.pool_audit_mode=compact \
  "algorithm.token_credit.rubric_pipeline.pool_audit_every_n_steps=${POOL_AUDIT_EVERY_N_STEPS}" \
  "trainer.analysis_log_enable=${ANALYSIS_LOG_ENABLE:-True}" \
  "trainer.analysis_log_every_n_steps=${ANALYSIS_LOG_EVERY_N_STEPS:-1}" \
  algorithm.token_credit.rubric_pipeline.pool_console_progress=False \
  "$@"
