#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Do not inherit the retired ALFWorld/Ray scratch redirection. The interaction
# layer bounds Fast Downward mappings by reusing native environments.
unset ALFWORLD_TMPDIR RAY_TEMP_DIR TMPDIR TMP TEMP
export ALFWORLD_ENV_POOL_ENABLE="${ALFWORLD_ENV_POOL_ENABLE:-True}"

# The Ray dashboard is not used by training. Some DSW images cannot start its
# MetricsHead subprocess, which otherwise aborts an otherwise healthy local
# Ray cluster during initialization.
export RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
echo "Ray dashboard enabled: ${RAY_INCLUDE_DASHBOARD}"
echo "ALFWorld native environment reuse: ${ALFWORLD_ENV_POOL_ENABLE}"

: "${ALFWORLD_CONFIG:?Set ALFWORLD_CONFIG to the official ALFWorld base_config.yaml}"
: "${ALFWORLD_DATA:?Set ALFWORLD_DATA to the downloaded ALFWorld data directory}"
if [[ ! -f "${ALFWORLD_CONFIG}" ]]; then
  echo "ALFWorld config does not exist: ${ALFWORLD_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${ALFWORLD_DATA}" ]]; then
  echo "ALFWorld data directory does not exist: ${ALFWORLD_DATA}" >&2
  exit 1
fi

# ALFWorld v2 training reward:
#   10 * official_success - 0.1 * invalid_action_count
# Validation ``acc``/``success`` remains the original binary ``won`` metric.
export ALFWORLD_R2_REWARD_ENABLE="${ALFWORLD_R2_REWARD_ENABLE:-False}"
export ALFWORLD_R2_SUCCESS_SCALE="${ALFWORLD_R2_SUCCESS_SCALE:-10.0}"
export ALFWORLD_R2_INVALID_ACTION_COEF="${ALFWORLD_R2_INVALID_ACTION_COEF:-0.1}"
echo "ALFWorld training reward: v2 enabled=${ALFWORLD_R2_REWARD_ENABLE}, success_scale=${ALFWORLD_R2_SUCCESS_SCALE}, invalid_coef=${ALFWORLD_R2_INVALID_ACTION_COEF}"

case "${ALFWORLD_R2_REWARD_ENABLE}" in
  True|true|TRUE|1|yes|YES|on|ON)
    REWARD_EXPERIMENT_SUFFIX="_v2"
    ;;
  *)
    REWARD_EXPERIMENT_SUFFIX="_official_reward"
    ;;
esac

# Use the same launch configuration for the rubric method and its matched GRPO
# control. The dedicated matched wrapper sets only this private mode flag.
if [[ "${_ALFWORLD_MATCHED_ENTRY:-0}" == "1" ]]; then
  ALFWORLD_EXPERIMENT_MODE=matched_grpo
else
  ALFWORLD_EXPERIMENT_MODE=group_hint
fi
case "${ALFWORLD_EXPERIMENT_MODE}" in
  group_hint)
    export TOKEN_CREDIT_ENABLE=True
    export RUBRIC_PIPELINE_ENABLE=True
    export RUBRIC_SNAPSHOT_ENABLE="${RUBRIC_SNAPSHOT_ENABLE:-True}"
    export EXPERIMENT_NAME="${ALFWORLD_GROUP_HINT_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_b8_n8_group_hint_bestof4_bundle_spearman_s300${REWARD_EXPERIMENT_SUFFIX}}"
    ;;
  matched_grpo)
    export TOKEN_CREDIT_ENABLE=False
    export RUBRIC_PIPELINE_ENABLE=False
    export RUBRIC_SNAPSHOT_ENABLE=False
    export EXPERIMENT_NAME="${ALFWORLD_MATCHED_EXPERIMENT_NAME:-qwen3_1_7b_alfworld_matched_grpo${REWARD_EXPERIMENT_SUFFIX}}"
    ;;
  *)
    echo "Internal error: unsupported ALFWorld experiment mode ${ALFWORLD_EXPERIMENT_MODE}" >&2
    exit 1
    ;;
esac

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    /workspace/checkpoints/qwen/Qwen3-1.7B \
    /workspace/checkpoints/Qwen3-1.7B \
    "${ROOT}/checkpoints/Qwen3-1.7B"; do
    if [[ -f "${candidate}/config.json" ]]; then
      MODEL_PATH="${candidate}"
      break
    fi
  done
fi
if [[ -z "${MODEL_PATH:-}" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Cannot find Qwen3-1.7B config.json. Set MODEL_PATH to the exact model directory." >&2
  exit 1
fi

export BENCHMARK=alfworld
export MODEL_PATH
export TRAIN_PARQUET="${TRAIN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_train.parquet}"
export ALFWORLD_VAL_SEEN_PARQUET="${ALFWORLD_VAL_SEEN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_valid_seen.parquet}"
export ALFWORLD_VAL_UNSEEN_PARQUET="${ALFWORLD_VAL_UNSEEN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_valid_unseen.parquet}"
# VAL_PARQUET remains populated for the common launcher's compatibility check;
# VAL_FILES is the actual Hydra list and always evaluates both official splits.
export VAL_PARQUET="${ALFWORLD_VAL_UNSEEN_PARQUET}"
export VAL_FILES="['${ALFWORLD_VAL_SEEN_PARQUET}', '${ALFWORLD_VAL_UNSEEN_PARQUET}']"
export RUBRIC_BACKEND=actor_rollout_group_hints
echo "ALFWorld validation splits: seen=${ALFWORLD_VAL_SEEN_PARQUET}, unseen=${ALFWORLD_VAL_UNSEEN_PARQUET}"
for dataset_path in "${TRAIN_PARQUET}" "${ALFWORLD_VAL_SEEN_PARQUET}" "${ALFWORLD_VAL_UNSEEN_PARQUET}"; do
  if [[ ! -f "${dataset_path}" ]]; then
    echo "Dataset parquet does not exist: ${dataset_path}" >&2
    exit 1
  fi
done

# Match the original ALFWorld sanity check in examples/agent_rubric:
# actor and rubric generation both use Qwen3 thinking, while history_window=0
# keeps VERL's original full multi-turn context instead of action-only projection.
export ROLLOUT_N="${ROLLOUT_N:-8}"
export VAL_N="${VAL_N:-4}"
export TEST_FREQ="${TEST_FREQ:-50}"
export RUBRICS_PER_PROMPT="${RUBRICS_PER_PROMPT:-8}"
export MIN_RUBRICS_PER_PROMPT="${MIN_RUBRICS_PER_PROMPT:-5}"
export RUBRIC_MAX_TRAJECTORIES="${RUBRIC_MAX_TRAJECTORIES:-${ROLLOUT_N}}"
export RUBRIC_GROUP_CANDIDATES="${RUBRIC_GROUP_CANDIDATES:-4}"
export RUBRIC_GROUP_SELECT_BY_CORRELATION="${RUBRIC_GROUP_SELECT_BY_CORRELATION:-True}"
export RUBRIC_SNAPSHOT_STEP="${RUBRIC_SNAPSHOT_STEP:-${MAX_STEPS:-300}}"
export RUBRIC_SNAPSHOT_BATCH_SIZE="${RUBRIC_SNAPSHOT_BATCH_SIZE:-8}"
export RUBRIC_SNAPSHOT_MAX_ATTEMPTS="${RUBRIC_SNAPSHOT_MAX_ATTEMPTS:-1}"
export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_JUDGE_MODEL="${RUBRIC_JUDGE_MODEL:-qwen3.5-plus}"
export RUBRIC_JUDGE_BASE_URL="${RUBRIC_JUDGE_BASE_URL:-https://api.example.com/v1}"
export RUBRIC_JUDGE_API_KEY_ENV="${RUBRIC_JUDGE_API_KEY_ENV:-DASHSCOPE_API_KEY}"
export RUBRIC_MIN_JUDGE_COVERAGE="${RUBRIC_MIN_JUDGE_COVERAGE:-0.75}"
export RUBRIC_MAX_TRAJECTORY_CHARS="${RUBRIC_MAX_TRAJECTORY_CHARS:-6000}"
export RUBRIC_MAX_RUBRIC_CHARS="${RUBRIC_MAX_RUBRIC_CHARS:-1600}"
export RUBRIC_MAX_ITEM_CHARS="${RUBRIC_MAX_ITEM_CHARS:-280}"
# Match the four-GPU WebShop rubric-generation ceiling. A valid 5--8 item JSON
# bundle is comfortably below this limit; the smaller cap prevents a malformed
# or thinking-heavy generation from occupying the rollout engine for 4096 tokens.
export RUBRIC_MAX_TOKENS="${RUBRIC_MAX_TOKENS:-4096}"
export RUBRIC_TEMPERATURE="${RUBRIC_TEMPERATURE:-0.3}"
export RUBRIC_TOP_P="${RUBRIC_TOP_P:-0.9}"
export RUBRIC_GENERATOR_THINKING="${RUBRIC_GENERATOR_THINKING:-True}"
export ACTOR_THINKING="${ACTOR_THINKING:-True}"
export TEACHER_THINKING="${TEACHER_THINKING:-True}"
export TEACHER_MAX_PROMPT_LENGTH="${TEACHER_MAX_PROMPT_LENGTH:-40960}"
export HISTORY_WINDOW="${HISTORY_WINDOW:-0}"
export HISTORY_FIXED_PREFIX_MESSAGES="${HISTORY_FIXED_PREFIX_MESSAGES:-3}"
export ACTOR_MAX_TOKENS_PER_TURN="${ACTOR_MAX_TOKENS_PER_TURN:-2048}"
export TURN_REPLAY_MICRO_BATCH_SIZE="${TURN_REPLAY_MICRO_BATCH_SIZE:-2}"
export TOKEN_CREDIT_ALPHA="${TOKEN_CREDIT_ALPHA:-0.01}"
export REQUIRE_OUTCOME_ALIGNED_GAP="${REQUIRE_OUTCOME_ALIGNED_GAP:-False}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export MAX_STEPS="${MAX_STEPS:-300}"
export PROJECT_NAME="${PROJECT_NAME:-verl_agent_rubric2}"

echo "Rubric route: aggregate all ${ROLLOUT_N} sibling trajectories; sample ${RUBRIC_GROUP_CANDIDATES} complete 5-8 item bundles"
echo "Bundle selection: highest bundle-level Spearman; no cross-bundle criterion mixing; no generator loss; no rubric pool"
echo "Teacher credit: additive OPD alpha=${TOKEN_CREDIT_ALPHA}; outcome-gap gate=${REQUIRE_OUTCOME_ALIGNED_GAP}"
echo "Training protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, total=$((TRAIN_BATCH_SIZE * ROLLOUT_N)), steps=${MAX_STEPS}"
echo "Full-train rubric snapshot: enabled=${RUBRIC_SNAPSHOT_ENABLE}, step=${RUBRIC_SNAPSHOT_STEP}, one complete bundle/task, no judging"

exec bash "${ROOT}/examples/agent_rubric/run_agent_grpo.sh" \
  "++ray_kwargs.ray_init.include_dashboard=${RAY_INCLUDE_DASHBOARD}" \
  "$@"
