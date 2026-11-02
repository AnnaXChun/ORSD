#!/usr/bin/env bash
set -euo pipefail

# Standalone ALFWorld Qwen3-1.7B first-version experiment.
# This file intentionally does not source or exec another launcher.  All
# training, rubric, rollout, validation, and snapshot parameters are written
# here so the experiment can be shared as a self-contained reference.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"
export MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7B}"

if [[ ! -f "${ALFWORLD_CONFIG}" ]]; then
  echo "ALFWORLD_CONFIG does not exist: ${ALFWORLD_CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${ALFWORLD_DATA}" ]]; then
  echo "ALFWORLD_DATA does not exist: ${ALFWORLD_DATA}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH%/}/config.json" ]]; then
  echo "Qwen3-1.7B config.json does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

python3 - "${MODEL_PATH%/}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
if config.get("model_type") != "qwen3":
    raise SystemExit(f"Expected a Qwen3 checkpoint, got model_type={config.get('model_type')!r}")
print(
    "Model preflight passed:"
    f" model_type={config.get('model_type')},"
    f" hidden_size={config.get('hidden_size')},"
    f" layers={config.get('num_hidden_layers')}"
)
PY

# Avoid stale scratch redirection from an earlier ALFWorld/Ray run.
unset ALFWORLD_TMPDIR RAY_TEMP_DIR TMPDIR TMP TEMP
export ALFWORLD_ENV_POOL_ENABLE=True
export RAY_INCLUDE_DASHBOARD=False
export ALFWORLD_R2_REWARD_ENABLE=False
export ALFWORLD_R2_SUCCESS_SCALE=10.0
export ALFWORLD_R2_INVALID_ACTION_COEF=0.1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#VISIBLE_GPUS[@]} != 4 )); then
  echo "This script requires exactly four visible GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

# Model and context.
MAX_MODEL_LENGTH=32768
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=16384
ACTOR_THINKING=True
RUBRIC_GENERATOR_THINKING=False
TEACHER_THINKING=True
TEACHER_MAX_PROMPT_LENGTH=4096

# ALFWorld training protocol: 8 prompts x 8 rollouts, 300 optimizer steps.
TRAIN_BATCH_SIZE=8
ROLLOUT_N=8
MAX_STEPS=300
TEST_FREQ=50
VAL_N=4
HISTORY_WINDOW=0
HISTORY_FIXED_PREFIX_MESSAGES=3
ACTOR_MAX_TOKENS_PER_TURN=null
TURN_REPLAY_MICRO_BATCH_SIZE=2
MICRO_BATCH_SIZE=2
AGENT_LOOP_NUM_WORKERS=4
DATALOADER_NUM_WORKERS=2
USE_REWARD_LOOP=False

# Four-GPU rollout profile.
N_GPUS=4
ROLLOUT_TP=1
ROLLOUT_GPU_MEMORY_UTILIZATION=0.60
ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
ROLLOUT_MAX_NUM_SEQS=256
ROLLOUT_ENABLE_CHUNKED_PREFILL=True
ROLLOUT_ENABLE_PREFIX_CACHING=True

# First-version rubric route: four complete bundles, select one intact bundle
# by bundle-level Spearman, and never mix criteria across bundles.
RUBRIC_PIPELINE_ENABLE=True
RUBRIC_BACKEND=actor_rollout_group_hints
RUBRICS_PER_PROMPT=8
MIN_RUBRICS_PER_PROMPT=5
RUBRIC_MAX_TRAJECTORIES=8
RUBRIC_GROUP_CANDIDATES=4
RUBRIC_GROUP_SELECT_BY_CORRELATION=True
RUBRIC_GROUP_FALLBACK_ON_UNRANKED=True
REQUIRE_GROUP_OUTCOME_CONTRAST=True
REQUIRE_OUTCOME_ALIGNED_GAP=False
MIN_OUTCOME_GAP_ALIGNMENT=0.0
RUBRIC_OUTCOME_SCORE_SOURCE=strict_success
RUBRIC_CORRECTNESS_THRESHOLD=0.5
RUBRIC_MAX_TRAJECTORY_CHARS=6000
RUBRIC_MAX_RUBRIC_CHARS=1600
RUBRIC_MAX_ITEM_CHARS=280
RUBRIC_MAX_TOKENS=4096
RUBRIC_TEMPERATURE=0.7
RUBRIC_TOP_P=0.9

# Additive OPD only.  The rubric pool and generator PPO objective are off.
TOKEN_CREDIT_ENABLE=True
TOKEN_CREDIT_MODE=additive_gap
TOKEN_CREDIT_ALPHA=0.01
TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
TOKEN_CREDIT_PRESERVE_TRAJECTORY_MEAN=True
RUBRIC_POOL_ENABLE=False
RUBRIC_GENERATOR_LOSS_ENABLE=False
RUBRIC_GENERATOR_LOSS_COEF=0.0

# Cloud judge used only to score the four candidate bundles.
RUBRIC_JUDGE_BASE_URL="${RUBRIC_JUDGE_BASE_URL:-https://api.example.com/v1}"
RUBRIC_JUDGE_MODEL="${RUBRIC_JUDGE_MODEL:-qwen3.5-plus}"
RUBRIC_JUDGE_API_KEY_ENV="${RUBRIC_JUDGE_API_KEY_ENV:-DASHSCOPE_API_KEY}"
RUBRIC_JUDGE_TIMEOUT=90
RUBRIC_JUDGE_CONCURRENCY=8
RUBRIC_JUDGE_RETRIES=4
RUBRIC_JUDGE_MAX_TOKENS=1536
RUBRIC_MIN_JUDGE_COVERAGE=0.75

# Export one unjudged complete bundle per training row after step 300.
RUBRIC_SNAPSHOT_ENABLE=True
RUBRIC_SNAPSHOT_STEP=300
RUBRIC_SNAPSHOT_BATCH_SIZE=8
RUBRIC_SNAPSHOT_MAX_ATTEMPTS=4

TRAIN_PARQUET="${TRAIN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_train.parquet}"
VAL_SEEN_PARQUET="${ALFWORLD_VAL_SEEN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_valid_seen.parquet}"
VAL_UNSEEN_PARQUET="${ALFWORLD_VAL_UNSEEN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_valid_unseen.parquet}"
VAL_FILES="['${VAL_SEEN_PARQUET}', '${VAL_UNSEEN_PARQUET}']"
for dataset_path in "${TRAIN_PARQUET}" "${VAL_SEEN_PARQUET}" "${VAL_UNSEEN_PARQUET}"; do
  if [[ ! -f "${dataset_path}" ]]; then
    echo "Dataset parquet does not exist: ${dataset_path}" >&2
    exit 2
  fi
done

python3 - "${TRAIN_PARQUET}" "${VAL_SEEN_PARQUET}" "${VAL_UNSEEN_PARQUET}" <<'PY'
import sys

import pyarrow.parquet as pq

for path in sys.argv[1:]:
    columns = set(pq.read_schema(path).names)
    missing = {"prompt", "extra_info"}.difference(columns)
    if missing:
        raise SystemExit(f"Parquet is missing {sorted(missing)}: {path}")
print("ALFWorld parquet preflight passed")
PY

PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1_7b_alfworld_b8_n8_bestof4_bundle_spearman_unrankedfallback_contrastgate_nogapgate_temp0p7_official01_additiveopd0p01_nogenthink_4gpu_s300}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "${OUTPUT_DIR}" "${ROOT}/logs"

echo "Standalone ALFWorld Qwen3-1.7B first-version experiment"
echo "Model: ${MODEL_PATH}"
echo "Devices: ${CUDA_VISIBLE_DEVICES}; TP=${ROLLOUT_TP}; vLLM util=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "Actor thinking=${ACTOR_THINKING}; rubric generator thinking=${RUBRIC_GENERATOR_THINKING}; teacher thinking=${TEACHER_THINKING}"
echo "Protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, total=$((TRAIN_BATCH_SIZE * ROLLOUT_N)), steps=${MAX_STEPS}"
echo "Rubrics: candidates=${RUBRIC_GROUP_CANDIDATES}, select_by_spearman=${RUBRIC_GROUP_SELECT_BY_CORRELATION}, temperature=${RUBRIC_TEMPERATURE}, top_p=${RUBRIC_TOP_P}, items=${MIN_RUBRICS_PER_PROMPT}-${RUBRICS_PER_PROMPT}"
echo "Gates: outcome_contrast=${REQUIRE_GROUP_OUTCOME_CONTRAST}, outcome_gap_alignment=${REQUIRE_OUTCOME_ALIGNED_GAP}, minimum_margin=${MIN_OUTCOME_GAP_ALIGNMENT}"
echo "OPD: mode=${TOKEN_CREDIT_MODE}, alpha=${TOKEN_CREDIT_ALPHA}, decay=${TOKEN_CREDIT_LINEAR_DECAY_ENABLE}, warmup=${TOKEN_CREDIT_LINEAR_WARMUP_ENABLE}"
echo "Pool=${RUBRIC_POOL_ENABLE}; generator_loss=${RUBRIC_GENERATOR_LOSS_ENABLE}; validation_every=${TEST_FREQ}; snapshot_attempts=${RUBRIC_SNAPSHOT_MAX_ATTEMPTS}"
echo "Experiment: ${EXPERIMENT_NAME}; checkpoints=${OUTPUT_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.token_credit.enable="${TOKEN_CREDIT_ENABLE}" \
  algorithm.token_credit.mode="${TOKEN_CREDIT_MODE}" \
  algorithm.token_credit.alpha="${TOKEN_CREDIT_ALPHA}" \
  algorithm.token_credit.scale_quantile=0.5 \
  algorithm.token_credit.min_scale=0.0001 \
  algorithm.token_credit.centered_clip=2.0 \
  algorithm.token_credit.centered_multiplier_min=0.5 \
  algorithm.token_credit.centered_multiplier_max=1.5 \
  algorithm.token_credit.rank_attenuation_strength=0.2 \
  algorithm.token_credit.rank_conflict_fraction=0.25 \
  algorithm.token_credit.rank_min_gap_range=0.0001 \
  algorithm.token_credit.linear_decay_enable="${TOKEN_CREDIT_LINEAR_DECAY_ENABLE}" \
  algorithm.token_credit.linear_decay_final_scale=0.0 \
  algorithm.token_credit.linear_decay_start_step=0 \
  algorithm.token_credit.linear_warmup_enable="${TOKEN_CREDIT_LINEAR_WARMUP_ENABLE}" \
  algorithm.token_credit.linear_warmup_steps=0 \
  algorithm.token_credit.linear_warmup_initial_scale=0.0 \
  algorithm.token_credit.preserve_trajectory_mean="${TOKEN_CREDIT_PRESERVE_TRAJECTORY_MEAN}" \
  algorithm.token_credit.teacher_log_prob_key=rubric_actor_log_probs \
  algorithm.token_credit.student_log_prob_key=old_log_probs \
  algorithm.token_credit.active_mask_key=rubric_teacher_eligible \
  algorithm.token_credit.rubric_pipeline.enable="${RUBRIC_PIPELINE_ENABLE}" \
  algorithm.token_credit.rubric_pipeline.backend="${RUBRIC_BACKEND}" \
  algorithm.token_credit.rubric_pipeline.rubrics_per_prompt="${RUBRICS_PER_PROMPT}" \
  algorithm.token_credit.rubric_pipeline.min_rubrics_per_prompt="${MIN_RUBRICS_PER_PROMPT}" \
  algorithm.token_credit.rubric_pipeline.min_reliable_rubrics_per_prompt=1 \
  algorithm.token_credit.rubric_pipeline.reliability_threshold=0.5 \
  algorithm.token_credit.rubric_pipeline.correctness_threshold="${RUBRIC_CORRECTNESS_THRESHOLD}" \
  algorithm.token_credit.rubric_pipeline.outcome_score_source="${RUBRIC_OUTCOME_SCORE_SOURCE}" \
  algorithm.token_credit.rubric_pipeline.require_group_outcome_contrast="${REQUIRE_GROUP_OUTCOME_CONTRAST}" \
  algorithm.token_credit.rubric_pipeline.outcome_contrast_epsilon=1e-6 \
  algorithm.token_credit.rubric_pipeline.require_outcome_aligned_gap="${REQUIRE_OUTCOME_ALIGNED_GAP}" \
  algorithm.token_credit.rubric_pipeline.min_outcome_gap_alignment="${MIN_OUTCOME_GAP_ALIGNMENT}" \
  algorithm.token_credit.rubric_pipeline.max_candidate_rubrics_per_group=24 \
  algorithm.token_credit.rubric_pipeline.max_rubric_chars="${RUBRIC_MAX_RUBRIC_CHARS}" \
  algorithm.token_credit.rubric_pipeline.max_rubric_item_chars="${RUBRIC_MAX_ITEM_CHARS}" \
  algorithm.token_credit.rubric_pipeline.max_tokens="${RUBRIC_MAX_TOKENS}" \
  algorithm.token_credit.rubric_pipeline.temperature="${RUBRIC_TEMPERATURE}" \
  algorithm.token_credit.rubric_pipeline.top_p="${RUBRIC_TOP_P}" \
  algorithm.token_credit.rubric_pipeline.max_trajectories_per_prompt="${RUBRIC_MAX_TRAJECTORIES}" \
  algorithm.token_credit.rubric_pipeline.max_trajectory_chars="${RUBRIC_MAX_TRAJECTORY_CHARS}" \
  algorithm.token_credit.rubric_pipeline.group_hint_candidate_count="${RUBRIC_GROUP_CANDIDATES}" \
  algorithm.token_credit.rubric_pipeline.group_hint_select_by_correlation="${RUBRIC_GROUP_SELECT_BY_CORRELATION}" \
  algorithm.token_credit.rubric_pipeline.group_hint_fallback_on_unranked="${RUBRIC_GROUP_FALLBACK_ON_UNRANKED}" \
  algorithm.token_credit.rubric_pipeline.snapshot_enable="${RUBRIC_SNAPSHOT_ENABLE}" \
  algorithm.token_credit.rubric_pipeline.snapshot_step="${RUBRIC_SNAPSHOT_STEP}" \
  algorithm.token_credit.rubric_pipeline.snapshot_batch_size="${RUBRIC_SNAPSHOT_BATCH_SIZE}" \
  algorithm.token_credit.rubric_pipeline.snapshot_max_generation_attempts="${RUBRIC_SNAPSHOT_MAX_ATTEMPTS}" \
  algorithm.token_credit.rubric_pipeline.snapshot_output_path=null \
  algorithm.token_credit.rubric_pipeline.pool_enable="${RUBRIC_POOL_ENABLE}" \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable="${RUBRIC_GENERATOR_LOSS_ENABLE}" \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_coef="${RUBRIC_GENERATOR_LOSS_COEF}" \
  algorithm.token_credit.rubric_pipeline.pool_generator_item_credit_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_judge_base_url="${RUBRIC_JUDGE_BASE_URL}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_model="${RUBRIC_JUDGE_MODEL}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_api_key_env="${RUBRIC_JUDGE_API_KEY_ENV}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_timeout="${RUBRIC_JUDGE_TIMEOUT}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_concurrency="${RUBRIC_JUDGE_CONCURRENCY}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_retries="${RUBRIC_JUDGE_RETRIES}" \
  algorithm.token_credit.rubric_pipeline.pool_judge_max_tokens="${RUBRIC_JUDGE_MAX_TOKENS}" \
  algorithm.token_credit.rubric_pipeline.pool_min_judge_coverage="${RUBRIC_MIN_JUDGE_COVERAGE}" \
  algorithm.token_credit.rubric_pipeline.generator_enable_thinking="${RUBRIC_GENERATOR_THINKING}" \
  algorithm.token_credit.rubric_pipeline.judge_enable_thinking=True \
  algorithm.token_credit.rubric_pipeline.judge_max_tokens=1024 \
  algorithm.token_credit.rubric_pipeline.judge_temperature=0.0 \
  algorithm.token_credit.rubric_pipeline.teacher_enable_thinking="${TEACHER_THINKING}" \
  algorithm.token_credit.rubric_pipeline.teacher_max_prompt_length="${TEACHER_MAX_PROMPT_LENGTH}" \
  algorithm.token_credit.rubric_pipeline.fail_on_error=False \
  data.train_files="['${TRAIN_PARQUET}']" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking="${ACTOR_THINKING}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path="${ROOT}/examples/agent_rubric/config/alfworld_interaction_config.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=null \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=null \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
  actor_rollout_ref.rollout.multi_turn.history_window="${HISTORY_WINDOW}" \
  actor_rollout_ref.rollout.multi_turn.history_fixed_prefix_messages="${HISTORY_FIXED_PREFIX_MESSAGES}" \
  actor_rollout_ref.rollout.multi_turn.history_response_mode=action_only \
  ++actor_rollout_ref.rollout.multi_turn.history_min_generation_tokens=2048 \
  actor_rollout_ref.rollout.multi_turn.max_response_tokens_per_turn="${ACTOR_MAX_TOKENS_PER_TURN}" \
  actor_rollout_ref.rollout.multi_turn.turn_replay_micro_batch_size="${TURN_REPLAY_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward_model.enable=False \
  reward_model.use_reward_loop="${USE_REWARD_LOOP}" \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.save_freq=25 \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs=5 \
  trainer.total_training_steps="${MAX_STEPS}" \
  ++ray_kwargs.ray_init.include_dashboard="${RAY_INCLUDE_DASHBOARD}" \
  2>&1 | tee -a "${ROOT}/logs/${EXPERIMENT_NAME}.log"
