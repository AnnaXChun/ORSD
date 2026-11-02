#!/usr/bin/env bash
set -euo pipefail

# Keep the same local W&B credential fallback as the math experiment entrypoint.
# An externally exported key still takes precedence.
export WANDB_API_KEY="${WANDB_API_KEY:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${BENCHMARK:?Set BENCHMARK to alfworld or webshop}"
: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to the validated rubric parquet}"
: "${VAL_PARQUET:?Set VAL_PARQUET to the official validation parquet}"
VAL_FILES="${VAL_FILES:-['${VAL_PARQUET}']}"

MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7B}"
# VERL's copy_to_local() rejects directory paths ending in '/'. Normalize the
# shared model path once so every ALFWorld/WebShop entry point accepts either
# spelling without changing the Hydra value or the experiment identity.
while [[ "${MODEL_PATH}" == */ && "${MODEL_PATH}" != "/" ]]; do
  MODEL_PATH="${MODEL_PATH%/}"
done
ACTOR_THINKING="${ACTOR_THINKING:-True}"
RUBRIC_GENERATOR_THINKING="${RUBRIC_GENERATOR_THINKING:-True}"
TEACHER_THINKING="${TEACHER_THINKING:-${ACTOR_THINKING}}"
RUBRIC_BACKEND="${RUBRIC_BACKEND:-actor_rollout}"
# Zero is the compatibility default: VERL keeps its original growing prompt
# and flatten-sequence PPO replay. A positive value enables bounded history
# plus exact per-turn replay for long-horizon environments.
HISTORY_WINDOW="${HISTORY_WINDOW:-0}"
HISTORY_FIXED_PREFIX_MESSAGES="${HISTORY_FIXED_PREFIX_MESSAGES:-3}"
HISTORY_RESPONSE_MODE="${HISTORY_RESPONSE_MODE:-action_only}"
# Reserve one full per-action generation budget. Oversized WebShop histories
# are shortened by whole action/observation turns, never by token slicing.
HISTORY_MIN_GENERATION_TOKENS="${HISTORY_MIN_GENERATION_TOKENS:-2048}"
ACTOR_MAX_TOKENS_PER_TURN="${ACTOR_MAX_TOKENS_PER_TURN:-null}"
TURN_REPLAY_MICRO_BATCH_SIZE="${TURN_REPLAY_MICRO_BATCH_SIZE:-2}"
# Agent environments keep substantial Python state (ALFWorld/TextWorld game
# objects or WebShop sessions) in every Ray process.  VERL defaults both the
# agent-loop pool and the PyTorch DataLoader to eight workers, which needlessly
# duplicates that state on a single node.  Four agent workers still serve all
# 32 rollouts asynchronously; two DataLoader workers are sufficient for these
# small parquet rows.
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
# ALFWorld and WebShop return their official environment reward directly in
# AgentLoopOutput.reward_score.  A colocated RewardLoopWorker per agent worker
# is therefore redundant and materially increases host-memory use.
USE_REWARD_LOOP="${USE_REWARD_LOOP:-False}"
# Long-horizon environment rollouts are far heavier than math completions.
# These defaults yield 8 tasks x 4 trajectories = 32 episodes per step.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PROJECT_NAME="${PROJECT_NAME:-verl_agent_rubric}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1_7b_${BENCHMARK}_rubric_grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
case "${BENCHMARK}" in
  alfworld)
    : "${ALFWORLD_CONFIG:?Set ALFWORLD_CONFIG to the official base_config.yaml}"
    : "${ALFWORLD_DATA:?Set ALFWORLD_DATA to the downloaded ALFWorld data directory}"
    INTERACTION_CONFIG="${ROOT}/examples/agent_rubric/config/alfworld_interaction_config.yaml"
    DEFAULT_MAX_RESPONSE_LENGTH=16384
    ;;
  webshop)
    : "${WEBSHOP_ROOT:?Set WEBSHOP_ROOT to the official princeton-nlp/WebShop checkout}"
    export WEBSHOP_PRODUCTS_JSON="${WEBSHOP_PRODUCTS_JSON:-}"
    INTERACTION_CONFIG="${ROOT}/examples/agent_rubric/config/webshop_interaction_config.yaml"
    DEFAULT_MAX_RESPONSE_LENGTH=32768
    ;;
  *)
    echo "BENCHMARK must be alfworld or webshop, got: ${BENCHMARK}" >&2
    exit 2
    ;;
esac
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${DEFAULT_MAX_RESPONSE_LENGTH}}"

if [[ "${RUBRIC_PIPELINE_ENABLE:-True}" == "False" || "${RUBRIC_PIPELINE_ENABLE:-True}" == "false" ]]; then
python3 - "${TRAIN_PARQUET}" <<'PY'
import sys

import pyarrow.parquet as pq

path = sys.argv[1]
required = {"prompt", "extra_info"}
columns = set(pq.read_schema(path).names)
missing = required.difference(columns)
if missing:
    raise SystemExit(f"Plain GRPO parquet is missing columns {sorted(missing)}: {path}")
print("Plain GRPO preflight passed: rubric pipeline, token credit, and generator loss are disabled")
PY
elif [[ "${RUBRIC_BACKEND}" == "dataset" ]]; then
python3 - "${TRAIN_PARQUET}" "${ALLOW_ACTOR_RUBRIC_INJECTION:-0}" "${TARGET_VALID_RATE:-0.60}" <<'PY'
import sys

import pyarrow.parquet as pq

path, allow_injection, target_valid_rate_text = sys.argv[1:]
target_valid_rate = float(target_valid_rate_text)
required = {"rubric", "rubric_valid", "rubric_selected_count", "extra_info"}
table = pq.read_table(path)
missing = required.difference(table.column_names)
if missing:
    raise SystemExit(f"Validated rubric parquet is missing columns {sorted(missing)}: {path}")
rows = table.select(sorted(required)).to_pylist()
valid = [row for row in rows if row["rubric_valid"]]
too_few = [row for row in valid if int(row["rubric_selected_count"] or 0) < 4]
invalid_nonempty = [row for row in rows if not row["rubric_valid"] and str(row["rubric"] or "").strip()]
injected = [
    index
    for index, row in enumerate(rows)
    if (row.get("extra_info") or {}).get("interaction_kwargs", {}).get("inject_rubric", False)
]
valid_rate = len(valid) / max(len(rows), 1)
if too_few or invalid_nonempty or valid_rate < target_valid_rate:
    raise SystemExit(
        f"Training parquet violates rubric gate: valid_rate={valid_rate:.2%}, "
        f"target={target_valid_rate:.2%}, valid_rows_fewer_than_4={len(too_few)}, "
        f"invalid_rows_with_nonempty_rubric={len(invalid_nonempty)}"
    )
if injected and allow_injection != "1":
    raise SystemExit(
        f"Found {len(injected)} rows that expose rubric to actor rollout. "
        "Regenerate with --no-inject-rubric, or set ALLOW_ACTOR_RUBRIC_INJECTION=1 only for that ablation."
    )
print(
    f"Validated rubric preflight passed: {len(valid)}/{len(rows)} ({valid_rate:.2%}) valid; "
    "valid tasks have >=4 rubrics, invalid tasks use pure-GRPO fallback, actor prompt remains plain"
)
PY
elif [[ "${RUBRIC_BACKEND}" == "actor_rollout" || "${RUBRIC_BACKEND}" == "actor_rollout_group_hints" ]]; then
python3 - "${TRAIN_PARQUET}" <<'PY'
import sys

import pyarrow.parquet as pq

path = sys.argv[1]
required = {"prompt", "extra_info"}
columns = set(pq.read_schema(path).names)
missing = required.difference(columns)
if missing:
    raise SystemExit(f"Online actor rubric parquet is missing columns {sorted(missing)}: {path}")
print("Online actor-rubric preflight passed: privileged guidance will be regenerated from each current rollout group")
PY
else
  echo "RUBRIC_BACKEND must be actor_rollout, actor_rollout_group_hints, or dataset; got: ${RUBRIC_BACKEND}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${ROOT}/logs"
echo "Host-memory settings: agent_loop_workers=${AGENT_LOOP_NUM_WORKERS}, dataloader_workers=${DATALOADER_NUM_WORKERS}, reward_loop=${USE_REWARD_LOOP}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# VERL hybrid vLLM sleep mode uses CuMemAllocator; expandable segments conflict
# with that pool and can make every rollout worker fail during initialization.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.token_credit.enable="${TOKEN_CREDIT_ENABLE:-True}" \
  algorithm.token_credit.mode="${TOKEN_CREDIT_MODE:-additive_gap}" \
  algorithm.token_credit.alpha="${TOKEN_CREDIT_ALPHA:-0.01}" \
  algorithm.token_credit.scale_quantile="${TOKEN_CREDIT_SCALE_QUANTILE:-0.5}" \
  algorithm.token_credit.min_scale="${TOKEN_CREDIT_MIN_SCALE:-0.0001}" \
  algorithm.token_credit.centered_clip="${TOKEN_CREDIT_CENTERED_CLIP:-2.0}" \
  algorithm.token_credit.centered_multiplier_min="${TOKEN_CREDIT_MULTIPLIER_MIN:-0.5}" \
  algorithm.token_credit.centered_multiplier_max="${TOKEN_CREDIT_MULTIPLIER_MAX:-1.5}" \
  algorithm.token_credit.rank_attenuation_strength="${TOKEN_CREDIT_RANK_ATTENUATION:-0.2}" \
  algorithm.token_credit.rank_conflict_fraction="${TOKEN_CREDIT_RANK_CONFLICT_FRACTION:-0.25}" \
  algorithm.token_credit.rank_min_gap_range="${TOKEN_CREDIT_RANK_MIN_GAP_RANGE:-0.0001}" \
  algorithm.token_credit.linear_decay_enable="${TOKEN_CREDIT_LINEAR_DECAY_ENABLE:-False}" \
  algorithm.token_credit.linear_decay_final_scale="${TOKEN_CREDIT_LINEAR_DECAY_FINAL_SCALE:-0.0}" \
  algorithm.token_credit.linear_decay_start_step="${TOKEN_CREDIT_LINEAR_DECAY_START_STEP:-0}" \
  algorithm.token_credit.linear_warmup_enable="${TOKEN_CREDIT_LINEAR_WARMUP_ENABLE:-False}" \
  algorithm.token_credit.linear_warmup_steps="${TOKEN_CREDIT_LINEAR_WARMUP_STEPS:-0}" \
  algorithm.token_credit.linear_warmup_initial_scale="${TOKEN_CREDIT_LINEAR_WARMUP_INITIAL_SCALE:-0.0}" \
  algorithm.token_credit.preserve_trajectory_mean="${TOKEN_CREDIT_PRESERVE_TRAJECTORY_MEAN:-True}" \
  algorithm.token_credit.teacher_log_prob_key="${TOKEN_CREDIT_TEACHER_LOG_PROB_KEY:-rubric_actor_log_probs}" \
  algorithm.token_credit.student_log_prob_key=old_log_probs \
  algorithm.token_credit.opsd_teacher_source="${OPSD_TEACHER_SOURCE:-current_actor}" \
  algorithm.token_credit.active_mask_key=rubric_teacher_eligible \
  algorithm.token_credit.rubric_pipeline.enable="${RUBRIC_PIPELINE_ENABLE:-True}" \
  algorithm.token_credit.rubric_pipeline.backend="${RUBRIC_BACKEND}" \
  algorithm.token_credit.rubric_pipeline.rubric_column=rubric \
  algorithm.token_credit.rubric_pipeline.valid_column=rubric_valid \
  algorithm.token_credit.rubric_pipeline.rubrics_per_prompt="${RUBRICS_PER_PROMPT:-4}" \
  algorithm.token_credit.rubric_pipeline.min_rubrics_per_prompt="${MIN_RUBRICS_PER_PROMPT:-2}" \
  algorithm.token_credit.rubric_pipeline.min_reliable_rubrics_per_prompt="${MIN_RELIABLE_RUBRICS_PER_PROMPT:-1}" \
  algorithm.token_credit.rubric_pipeline.reliability_threshold="${RUBRIC_RELIABILITY_THRESHOLD:-0.5}" \
  algorithm.token_credit.rubric_pipeline.correctness_threshold="${RUBRIC_CORRECTNESS_THRESHOLD:-0.5}" \
  algorithm.token_credit.rubric_pipeline.outcome_score_source="${RUBRIC_OUTCOME_SCORE_SOURCE:-strict_success}" \
  algorithm.token_credit.rubric_pipeline.require_group_outcome_contrast="${REQUIRE_GROUP_OUTCOME_CONTRAST:-True}" \
  algorithm.token_credit.rubric_pipeline.outcome_contrast_epsilon="${OUTCOME_CONTRAST_EPSILON:-1e-6}" \
  algorithm.token_credit.rubric_pipeline.require_outcome_aligned_gap="${REQUIRE_OUTCOME_ALIGNED_GAP:-False}" \
  algorithm.token_credit.rubric_pipeline.min_outcome_gap_alignment="${MIN_OUTCOME_GAP_ALIGNMENT:-0.0}" \
  algorithm.token_credit.rubric_pipeline.max_candidate_rubrics_per_group="${MAX_CANDIDATE_RUBRICS_PER_GROUP:-24}" \
  algorithm.token_credit.rubric_pipeline.max_rubric_chars="${RUBRIC_MAX_RUBRIC_CHARS:-4000}" \
  algorithm.token_credit.rubric_pipeline.max_rubric_item_chars="${RUBRIC_MAX_ITEM_CHARS:-280}" \
  algorithm.token_credit.rubric_pipeline.max_tokens="${RUBRIC_MAX_TOKENS:-2048}" \
  algorithm.token_credit.rubric_pipeline.temperature="${RUBRIC_TEMPERATURE:-0.2}" \
  algorithm.token_credit.rubric_pipeline.top_p="${RUBRIC_TOP_P:-0.95}" \
  algorithm.token_credit.rubric_pipeline.max_trajectories_per_prompt="${RUBRIC_MAX_TRAJECTORIES:-1}" \
  algorithm.token_credit.rubric_pipeline.max_trajectory_chars="${RUBRIC_MAX_TRAJECTORY_CHARS:-6000}" \
  algorithm.token_credit.rubric_pipeline.group_hint_candidate_count="${RUBRIC_GROUP_CANDIDATES:-1}" \
  algorithm.token_credit.rubric_pipeline.group_hint_select_by_correlation="${RUBRIC_GROUP_SELECT_BY_CORRELATION:-False}" \
  algorithm.token_credit.rubric_pipeline.group_hint_fallback_on_unranked="${RUBRIC_GROUP_FALLBACK_ON_UNRANKED:-False}" \
  algorithm.token_credit.rubric_pipeline.group_hint_force_first_candidate="${RUBRIC_GROUP_FORCE_FIRST_CANDIDATE:-False}" \
  algorithm.token_credit.rubric_pipeline.group_hint_evidence_order="${RUBRIC_GROUP_EVIDENCE_ORDER:-outcome_anchored_random}" \
  algorithm.token_credit.rubric_pipeline.group_hint_prompt_style="${RUBRIC_GROUP_PROMPT_STYLE:-process_rubric_5_to_k}" \
  algorithm.token_credit.rubric_pipeline.group_hint_guarded_selection="${RUBRIC_GROUP_GUARDED_SELECTION:-False}" \
  algorithm.token_credit.rubric_pipeline.group_hint_min_selected_correlation="${RUBRIC_GROUP_MIN_SELECTED_CORRELATION:-0.0}" \
  algorithm.token_credit.rubric_pipeline.group_hint_min_correlation_improvement="${RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT:-0.0}" \
  algorithm.token_credit.rubric_pipeline.snapshot_enable="${RUBRIC_SNAPSHOT_ENABLE:-False}" \
  algorithm.token_credit.rubric_pipeline.snapshot_step="${RUBRIC_SNAPSHOT_STEP:-${MAX_STEPS:-300}}" \
  algorithm.token_credit.rubric_pipeline.snapshot_batch_size="${RUBRIC_SNAPSHOT_BATCH_SIZE:-8}" \
  algorithm.token_credit.rubric_pipeline.snapshot_max_generation_attempts="${RUBRIC_SNAPSHOT_MAX_ATTEMPTS:-4}" \
  algorithm.token_credit.rubric_pipeline.snapshot_output_path="${RUBRIC_SNAPSHOT_OUTPUT_PATH:-null}" \
  algorithm.token_credit.rubric_pipeline.pool_enable="${RUBRIC_POOL_ENABLE:-False}" \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable="${RUBRIC_GENERATOR_LOSS_ENABLE:-False}" \
  algorithm.token_credit.rubric_pipeline.pool_generator_item_credit_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_judge_base_url="${RUBRIC_JUDGE_BASE_URL:-https://api.example.com/v1}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_model="${RUBRIC_JUDGE_MODEL:-null}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_api_key_env="${RUBRIC_JUDGE_API_KEY_ENV:-DASHSCOPE_API_KEY}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_timeout="${RUBRIC_JUDGE_TIMEOUT:-90}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_concurrency="${RUBRIC_JUDGE_CONCURRENCY:-8}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_retries="${RUBRIC_JUDGE_RETRIES:-4}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_max_tokens="${RUBRIC_JUDGE_MAX_TOKENS:-1536}" \
  algorithm.token_credit.rubric_pipeline.pool_min_judge_coverage="${RUBRIC_MIN_JUDGE_COVERAGE:-0.75}" \
  algorithm.token_credit.rubric_pipeline.generator_enable_thinking="${RUBRIC_GENERATOR_THINKING}" \
  algorithm.token_credit.rubric_pipeline.judge_enable_thinking="${RUBRIC_JUDGE_THINKING:-True}" \
  algorithm.token_credit.rubric_pipeline.judge_max_tokens="${RUBRIC_JUDGE_MAX_TOKENS:-1024}" \
  algorithm.token_credit.rubric_pipeline.judge_temperature="${RUBRIC_JUDGE_TEMPERATURE:-0.0}" \
  algorithm.token_credit.rubric_pipeline.teacher_enable_thinking="${TEACHER_THINKING}" \
  algorithm.token_credit.rubric_pipeline.teacher_max_prompt_length="${TEACHER_MAX_PROMPT_LENGTH:-${MAX_MODEL_LENGTH:-40960}}" \
  algorithm.token_credit.rubric_pipeline.fail_on_error=False \
  data.train_files="['${TRAIN_PARQUET}']" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking="${ACTOR_THINKING}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-2}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH:-40960}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-256}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}" \
  actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-2}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  actor_rollout_ref.rollout.calculate_log_probs="${CALCULATE_ROLLOUT_LOG_PROBS:-True}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-0.6}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-4}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path="${INTERACTION_CONFIG}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS:-null}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=null \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
  actor_rollout_ref.rollout.multi_turn.history_window="${HISTORY_WINDOW}" \
  actor_rollout_ref.rollout.multi_turn.history_fixed_prefix_messages="${HISTORY_FIXED_PREFIX_MESSAGES}" \
  actor_rollout_ref.rollout.multi_turn.history_response_mode="${HISTORY_RESPONSE_MODE}" \
  ++actor_rollout_ref.rollout.multi_turn.history_min_generation_tokens="${HISTORY_MIN_GENERATION_TOKENS}" \
  actor_rollout_ref.rollout.multi_turn.max_response_tokens_per_turn="${ACTOR_MAX_TOKENS_PER_TURN}" \
  actor_rollout_ref.rollout.multi_turn.turn_replay_micro_batch_size="${TURN_REPLAY_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-2}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward_model.enable=False \
  reward_model.use_reward_loop="${USE_REWARD_LOOP}" \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS:-8}" \
  trainer.nnodes=1 \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.save_freq="${SAVE_FREQ:-25}" \
  trainer.test_freq="${TEST_FREQ:-25}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-5}" \
  trainer.total_training_steps="${MAX_STEPS:-null}" \
  "$@" 2>&1 | tee -a "${ROOT}/logs/${EXPERIMENT_NAME}.log"
