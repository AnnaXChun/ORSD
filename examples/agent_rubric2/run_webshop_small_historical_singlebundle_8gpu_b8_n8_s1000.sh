#!/usr/bin/env bash
set -euo pipefail

# Reproduce the finished W&B milestone
# qwen3_1_7b_webshop_small_hybrid_bs8_rubric_grpo. Candidate-related
# variables use defaults so the guarded wrapper can change only selection.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _WEBSHOP_HISTORICAL_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_WEBSHOP_HISTORICAL_VISIBLE_GPUS[@]} != 8 )); then
  echo "Historical WebShop reproduction requires exactly eight visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

# Hardware and rollout configuration from W&B run lkqnfobj.
export N_GPUS=8
export ROLLOUT_TP=2
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.60
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
export ROLLOUT_MAX_NUM_SEQS=256
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True
export WEBSHOP_AGENT_LOOP_NUM_WORKERS=1
export WEBSHOP_DATALOADER_NUM_WORKERS=1

# Training, optimization, evaluation, and context budgets.
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=1000
export TOTAL_EPOCHS=5
export TEST_FREQ=10
export SAVE_FREQ=25
export MICRO_BATCH_SIZE=1
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export LEARNING_RATE=1e-6
export KL_COEF=0.001
export VAL_N=4
export VAL_TEMPERATURE=0.6
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=32768
export MAX_MODEL_LENGTH=40960
export MAX_ASSISTANT_TURNS=15
export HISTORY_WINDOW=0
export HISTORY_FIXED_PREFIX_MESSAGES=3
export HISTORY_RESPONSE_MODE=action_only
export HISTORY_MIN_GENERATION_TOKENS=2048
export ACTOR_MAX_TOKENS_PER_TURN=2048
export ACTOR_THINKING=True

# Exact hybrid reward and outcome labels used by the milestone.
export WEBSHOP_REWARD_MODE=hybrid
export WEBSHOP_R2_REWARD_ENABLE=False
export WEBSHOP_R2_INVALID_ACTION_COEF=0.05
export WEBSHOP_R2_REPEAT_ACTION_COEF=0.025
export RUBRIC_OUTCOME_SCORE_SOURCE=training_reward
export RUBRIC_CORRECTNESS_THRESHOLD=0.6

# Historical single-bundle online OPD route.
export _WEBSHOP_MATCHED_ENTRY=0
export TOKEN_CREDIT_ENABLE=True
export TOKEN_CREDIT_MODE=additive_gap
export TOKEN_CREDIT_ALPHA=0.01
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export RUBRIC_PIPELINE_ENABLE=True
export RUBRIC_BACKEND=actor_rollout_group_hints
export RUBRICS_PER_PROMPT=5
export MIN_RUBRICS_PER_PROMPT=2
export RUBRIC_MAX_TRAJECTORIES=4
export RUBRIC_GROUP_CANDIDATES="${RUBRIC_GROUP_CANDIDATES:-1}"
export RUBRIC_GROUP_SELECT_BY_CORRELATION="${RUBRIC_GROUP_SELECT_BY_CORRELATION:-False}"
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED="${RUBRIC_GROUP_FALLBACK_ON_UNRANKED:-False}"
export RUBRIC_GROUP_GUARDED_SELECTION="${RUBRIC_GROUP_GUARDED_SELECTION:-False}"
export RUBRIC_GROUP_MIN_SELECTED_CORRELATION="${RUBRIC_GROUP_MIN_SELECTED_CORRELATION:-0.0}"
export RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT="${RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT:-0.0}"
export RUBRIC_GROUP_EVIDENCE_ORDER=legacy_alternating_extremes
export RUBRIC_GROUP_PROMPT_STYLE=legacy_process_rubric_3_to_k
export REQUIRE_GROUP_OUTCOME_CONTRAST=True
export REQUIRE_OUTCOME_ALIGNED_GAP=True
export MIN_OUTCOME_GAP_ALIGNMENT=0.0

export RUBRIC_MAX_TRAJECTORY_CHARS=6000
export RUBRIC_MAX_RUBRIC_CHARS=1600
export RUBRIC_MAX_ITEM_CHARS=280
export RUBRIC_MAX_TOKENS=4096
export RUBRIC_TEMPERATURE=0.3
export RUBRIC_TOP_P=0.9
export RUBRIC_GENERATOR_THINKING=False
export TEACHER_THINKING=True
export TEACHER_MAX_PROMPT_LENGTH=40960

export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_LOSS_COEF=0.0
export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=False
export RUBRIC_SNAPSHOT_ENABLE=False

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export WEBSHOP_GROUP_HINT_EXPERIMENT_NAME="${WEBSHOP_HISTORICAL_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_historical_singlebundle_hybrid_additiveopd0p01_8gpu_b8_n8_s1000}"

echo "WebShop-small historical route: hybrid reward, single bundle, additive OPD=0.01"
echo "Protocol: GPUs=8, TP=2, prompts=8, rollouts/prompt=8, steps=1000, validation_every=10"
echo "Context: history_window=0, response_mode=action_only, max_turns=15"
echo "Rubric: candidates=${RUBRIC_GROUP_CANDIDATES}, trajectories=4, criteria=3-5, temperature=0.3, thinking=False"
echo "Gates: outcome_contrast=True, outcome_gap_alignment=True"
echo "Selection: enabled=${RUBRIC_GROUP_SELECT_BY_CORRELATION}, guarded=${RUBRIC_GROUP_GUARDED_SELECTION}"
echo "Experiment: ${WEBSHOP_GROUP_HINT_EXPERIMENT_NAME}"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_group_hint_grpo.sh" "$@"
