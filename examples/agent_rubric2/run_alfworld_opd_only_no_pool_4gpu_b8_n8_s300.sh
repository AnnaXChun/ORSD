#!/usr/bin/env bash
set -euo pipefail

# ALFWorld OPD-only ablation. Match the current 8x8, four-GPU, 300-step
# official-reward protocol while using one group-level rubric solely for the
# teacher log-probability pass. Rubric pooling and generator optimization are
# disabled explicitly so stale shell exports cannot enable either route.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"
export ALFWORLD_R2_REWARD_ENABLE=False

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _ALFWORLD_OPD_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_ALFWORLD_OPD_VISIBLE_GPUS[@]} != 4 )); then
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
export AGENT_LOOP_NUM_WORKERS="${ALFWORLD_4GPU_AGENT_LOOP_WORKERS:-8}"
export DATALOADER_NUM_WORKERS="${ALFWORLD_4GPU_DATALOADER_WORKERS:-2}"
export MICRO_BATCH_SIZE=2
export TURN_REPLAY_MICRO_BATCH_SIZE=2
export HISTORY_WINDOW=0
export ACTOR_THINKING=True

# Generate one process-rubric block from each current rollout group, then use
# it only to compute rubric-conditioned teacher log probabilities for OPD.
export _ALFWORLD_MATCHED_ENTRY=0
export TOKEN_CREDIT_ENABLE=True
export TOKEN_CREDIT_MODE=additive_gap
export TOKEN_CREDIT_ALPHA=0.01
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export RUBRIC_PIPELINE_ENABLE=True
export RUBRIC_BACKEND=actor_rollout_group_hints
export RUBRIC_OUTCOME_SCORE_SOURCE=strict_success
export REQUIRE_GROUP_OUTCOME_CONTRAST=True
export REQUIRE_OUTCOME_ALIGNED_GAP=False
export RUBRICS_PER_PROMPT=8
export RUBRIC_MAX_TRAJECTORIES=4
export RUBRIC_GROUP_CANDIDATES="${RUBRIC_GROUP_CANDIDATES:-1}"
export RUBRIC_GROUP_SELECT_BY_CORRELATION="${RUBRIC_GROUP_SELECT_BY_CORRELATION:-False}"
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED="${RUBRIC_GROUP_FALLBACK_ON_UNRANKED:-False}"
export RUBRIC_GROUP_GUARDED_SELECTION="${RUBRIC_GROUP_GUARDED_SELECTION:-False}"
export RUBRIC_GROUP_MIN_SELECTED_CORRELATION="${RUBRIC_GROUP_MIN_SELECTED_CORRELATION:-0.0}"
export RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT="${RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT:-0.0}"
export RUBRIC_MAX_TRAJECTORY_CHARS=6000
export RUBRIC_MAX_RUBRIC_CHARS=1600
export RUBRIC_MAX_ITEM_CHARS=280
export RUBRIC_MAX_TOKENS=1024
export RUBRIC_TEMPERATURE=0.7
export RUBRIC_TOP_P=0.9
export RUBRIC_GENERATOR_THINKING=False
export TEACHER_THINKING=True
export TEACHER_MAX_PROMPT_LENGTH=40960

export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_LOSS_COEF=0.0
export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=False
export RUBRIC_GENERATOR_LINEAR_DECAY_ENABLE=False
export RUBRIC_GENERATOR_LINEAR_WARMUP_ENABLE=False

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export ALFWORLD_GROUP_HINT_EXPERIMENT_NAME="${ALFWORLD_OPD_ONLY_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_opd_only_no_pool_4gpu_b8_n8_additiveopd0p01_official_reward_s300}"
export OUTPUT_DIR="${ALFWORLD_OPD_ONLY_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${ALFWORLD_GROUP_HINT_EXPERIMENT_NAME}}"

echo "ALFWorld OPD-only: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, total=$((TRAIN_BATCH_SIZE * ROLLOUT_N)), steps=${MAX_STEPS}, official reward"
echo "Token credit: mode=${TOKEN_CREDIT_MODE}, alpha=${TOKEN_CREDIT_ALPHA}, schedule=fixed"
echo "Rubric route: candidates=${RUBRIC_GROUP_CANDIDATES}, correlation_selection=${RUBRIC_GROUP_SELECT_BY_CORRELATION}, guarded_selection=${RUBRIC_GROUP_GUARDED_SELECTION}; pool=off; generator loss=off"
echo "Rubric generator: criteria<=${RUBRICS_PER_PROMPT}, trajectories<=${RUBRIC_MAX_TRAJECTORIES}, temperature=${RUBRIC_TEMPERATURE}, thinking=${RUBRIC_GENERATOR_THINKING}"
echo "Four-GPU profile: devices=${CUDA_VISIBLE_DEVICES}, TP=${ROLLOUT_TP}, vLLM_util=${ROLLOUT_GPU_MEMORY_UTILIZATION}, agent_workers=${AGENT_LOOP_NUM_WORKERS}"
echo "Experiment: ${ALFWORLD_GROUP_HINT_EXPERIMENT_NAME}; checkpoints=${OUTPUT_DIR}"

exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_group_hint_grpo.sh" \
  algorithm.token_credit.enable=True \
  algorithm.token_credit.mode=additive_gap \
  algorithm.token_credit.alpha=0.01 \
  algorithm.token_credit.linear_decay_enable=False \
  algorithm.token_credit.linear_warmup_enable=False \
  algorithm.token_credit.rubric_pipeline.enable=True \
  algorithm.token_credit.rubric_pipeline.backend=actor_rollout_group_hints \
  algorithm.token_credit.rubric_pipeline.pool_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_coef=0.0 \
  algorithm.token_credit.rubric_pipeline.pool_generator_linear_decay_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_linear_warmup_enable=False \
  algorithm.token_credit.rubric_pipeline.outcome_score_source=strict_success \
  algorithm.token_credit.rubric_pipeline.require_group_outcome_contrast=True \
  algorithm.token_credit.rubric_pipeline.require_outcome_aligned_gap=False \
  trainer.total_training_steps=300 \
  "$@"
