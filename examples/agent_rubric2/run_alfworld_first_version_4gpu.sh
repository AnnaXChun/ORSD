#!/usr/bin/env bash
set -euo pipefail

# Reproduce the first working ALFWorld training route while retaining the
# current validation implementation, including seen/unseen and task-type
# val-core rollups. Four-GPU execution is the only hardware adaptation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"
if [[ -n "${MODEL_PATH:-}" ]]; then
  python3 - "${MODEL_PATH%/}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    model_config = json.load(handle)
if model_config.get("model_type") != "qwen3":
    raise SystemExit(f"Qwen3-1.7B launcher received a different model: {model_config.get('model_type')!r}")
PY
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _ALFWORLD_FIRST_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_ALFWORLD_FIRST_VISIBLE_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export N_GPUS=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ALFWORLD_FIRST_VLLM_GPU_UTIL:-0.60}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
export ROLLOUT_MAX_NUM_SEQS=256
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True

# First-version task training protocol.
export _ALFWORLD_MATCHED_ENTRY=0
export ALFWORLD_R2_REWARD_ENABLE=False
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export AGENT_LOOP_NUM_WORKERS=4
export DATALOADER_NUM_WORKERS=2
export MICRO_BATCH_SIZE=2
export TURN_REPLAY_MICRO_BATCH_SIZE=2
export HISTORY_WINDOW=0
export HISTORY_FIXED_PREFIX_MESSAGES=3
# The historical launcher did not impose a separate per-turn cap.
export ACTOR_MAX_TOKENS_PER_TURN=null
export MAX_RESPONSE_LENGTH=16384
export SAVE_FREQ=25
export TEST_FREQ=50

# First-version group-hint rubric and additive OPD route.
export TOKEN_CREDIT_ENABLE=True
export TOKEN_CREDIT_MODE=additive_gap
export TOKEN_CREDIT_ALPHA=0.01
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
# Match the successful OPD-only eligibility logic: only contrastive rollout
# groups may generate guidance. The selected best-of-four bundle then supplies
# additive OPD directly, without the noisy second outcome-gap gate.
export REQUIRE_GROUP_OUTCOME_CONTRAST=True
export REQUIRE_OUTCOME_ALIGNED_GAP=False
export MIN_OUTCOME_GAP_ALIGNMENT=0.0
export RUBRIC_PIPELINE_ENABLE=True
export RUBRIC_BACKEND=actor_rollout_group_hints
export RUBRICS_PER_PROMPT=8
export MIN_RUBRICS_PER_PROMPT=5
export RUBRIC_MAX_TRAJECTORIES=8
export RUBRIC_GROUP_CANDIDATES=4
export RUBRIC_GROUP_SELECT_BY_CORRELATION=True
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED=True
export RUBRIC_SNAPSHOT_ENABLE=True
export RUBRIC_SNAPSHOT_STEP=300
export RUBRIC_SNAPSHOT_BATCH_SIZE=8
export RUBRIC_SNAPSHOT_MAX_ATTEMPTS=4
export RUBRIC_MAX_TRAJECTORY_CHARS=6000
export RUBRIC_MAX_RUBRIC_CHARS=1600
export RUBRIC_MAX_ITEM_CHARS=280
export RUBRIC_MAX_TOKENS=4096
export RUBRIC_TEMPERATURE=0.7
export RUBRIC_TOP_P=0.9
export RUBRIC_GENERATOR_THINKING=False
export ACTOR_THINKING=True
export TEACHER_THINKING=True
export TEACHER_MAX_PROMPT_LENGTH=4096

# The first route had no per-trajectory pool or auxiliary generator update.
# These exports also neutralize stale values from a preceding pool experiment.
export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_LOSS_COEF=0.0
export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=False

# Preserve current experiment organization and validation reporting while using
# the requested fixed training budget.
export MAX_STEPS=300
export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export ALFWORLD_GROUP_HINT_EXPERIMENT_NAME="${ALFWORLD_FIRST_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_b8_n8_bestof4_bundle_spearman_unrankedfallback_contrastgate_nogapgate_temp0p7_official01_additiveopd0p01_nogenthink_4gpu_s300}"
export OUTPUT_DIR="${ALFWORLD_FIRST_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${ALFWORLD_GROUP_HINT_EXPERIMENT_NAME}}"

echo "ALFWorld first-version route: official reward in {0,1}; aggregate group hints; additive OPD alpha=0.01"
echo "Protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, candidate bundles=${RUBRIC_GROUP_CANDIDATES}, items=${MIN_RUBRICS_PER_PROMPT}-${RUBRICS_PER_PROMPT}, generator_loss=off"
echo "Bundle selection: score each intact bundle and use the highest-Spearman bundle; no item mixing or rubric pool"
echo "Unranked groups: retain the first parsed complete bundle when Spearman is undefined or judging fails"
echo "Eligibility gates: outcome_contrast=${REQUIRE_GROUP_OUTCOME_CONTRAST}, outcome_gap_alignment=${REQUIRE_OUTCOME_ALIGNED_GAP}, minimum_margin=${MIN_OUTCOME_GAP_ALIGNMENT}"
echo "Training budget: ${MAX_STEPS} optimizer steps"
echo "Rubric generator: temperature=${RUBRIC_TEMPERATURE}, top_p=${RUBRIC_TOP_P}, thinking=${RUBRIC_GENERATOR_THINKING}, max_tokens=${RUBRIC_MAX_TOKENS}"
echo "Validation: current seen/unseen task-type val-core reporting is retained"
echo "OPSD snapshot: after final step 300, one unjudged complete rubric bundle per train task; saved with original parquet rows"
echo "Experiment: ${ALFWORLD_GROUP_HINT_EXPERIMENT_NAME}; checkpoints=${OUTPUT_DIR}"

exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_group_hint_grpo.sh" \
  algorithm.token_credit.rubric_pipeline.pool_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=False \
  algorithm.token_credit.rubric_pipeline.outcome_score_source=strict_success \
  "$@"
