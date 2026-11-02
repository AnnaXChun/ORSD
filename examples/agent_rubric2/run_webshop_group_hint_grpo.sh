#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${WEBSHOP_ROOT:?Set WEBSHOP_ROOT to the official princeton-nlp/WebShop checkout}"

# GiGPO WebShop reward (default):
#   10 * strict_success - 0.1 * invalid_action_count
# Alternative modes are retained for controlled ablations:
#   official: native continuous environment score
#   r2: official_score - alpha * invalid_rate - beta * repeat_rate
#   hybrid: 0.5 * official_score + 0.5 * strict_success
#           - alpha * invalid_rate - beta * repeat_rate
# Rubric grouping follows strict task success for official/GiGPO rewards. In
# R2 and hybrid modes it uses the same shaped training reward as GRPO and
# separates attempts at RUBRIC_CORRECTNESS_THRESHOLD (default 0.6).
export WEBSHOP_REWARD_MODE="${WEBSHOP_REWARD_MODE:-gigpo}"
export WEBSHOP_GIGPO_SUCCESS_REWARD="${WEBSHOP_GIGPO_SUCCESS_REWARD:-10.0}"
export WEBSHOP_GIGPO_INVALID_ACTION_COEF="${WEBSHOP_GIGPO_INVALID_ACTION_COEF:-0.1}"
export WEBSHOP_R2_REWARD_ENABLE="${WEBSHOP_R2_REWARD_ENABLE:-False}"
export WEBSHOP_R2_INVALID_ACTION_COEF="${WEBSHOP_R2_INVALID_ACTION_COEF:-0.05}"
export WEBSHOP_R2_REPEAT_ACTION_COEF="${WEBSHOP_R2_REPEAT_ACTION_COEF:-0.025}"

case "${WEBSHOP_REWARD_MODE}" in
  gigpo|gigpo_reward)
    REWARD_EXPERIMENT_SUFFIX="_gigpo_reward"
    echo "WebShop training reward: GiGPO (success=${WEBSHOP_GIGPO_SUCCESS_REWARD}, invalid_action=-${WEBSHOP_GIGPO_INVALID_ACTION_COEF})"
    ;;
  r2)
    REWARD_EXPERIMENT_SUFFIX="_r2"
    export RUBRIC_OUTCOME_SCORE_SOURCE="${RUBRIC_OUTCOME_SCORE_SOURCE:-training_reward}"
    export RUBRIC_CORRECTNESS_THRESHOLD="${RUBRIC_CORRECTNESS_THRESHOLD:-0.6}"
    echo "WebShop training reward: R2 (alpha=${WEBSHOP_R2_INVALID_ACTION_COEF}, beta=${WEBSHOP_R2_REPEAT_ACTION_COEF})"
    echo "WebShop rubric outcome: ${RUBRIC_OUTCOME_SCORE_SOURCE}, positive threshold=${RUBRIC_CORRECTNESS_THRESHOLD}"
    ;;
  hybrid|success_hybrid|hybrid_reward)
    REWARD_EXPERIMENT_SUFFIX="_hybrid_reward"
    export RUBRIC_OUTCOME_SCORE_SOURCE="${RUBRIC_OUTCOME_SCORE_SOURCE:-training_reward}"
    export RUBRIC_CORRECTNESS_THRESHOLD="${RUBRIC_CORRECTNESS_THRESHOLD:-0.6}"
    echo "WebShop training reward: hybrid (0.5*official + 0.5*success - ${WEBSHOP_R2_INVALID_ACTION_COEF}*invalid_rate - ${WEBSHOP_R2_REPEAT_ACTION_COEF}*repeat_rate)"
    echo "WebShop rubric outcome: ${RUBRIC_OUTCOME_SCORE_SOURCE}, positive threshold=${RUBRIC_CORRECTNESS_THRESHOLD}"
    ;;
  official|native|benchmark)
    REWARD_EXPERIMENT_SUFFIX="_official_reward"
    echo "WebShop training reward: official continuous environment score"
    ;;
  *)
    echo "WEBSHOP_REWARD_MODE must be one of: gigpo, r2, hybrid, official; got ${WEBSHOP_REWARD_MODE}" >&2
    exit 1
    ;;
esac

# Keep the group-hint and matched-GRPO entry points isolated.  Variables
# exported by a previous matched run must not silently disable rubric credit
# when this script is invoked directly in the same shell.
if [[ "${_WEBSHOP_MATCHED_ENTRY:-0}" == "1" ]]; then
  WEBSHOP_EXPERIMENT_MODE=matched_grpo
else
  WEBSHOP_EXPERIMENT_MODE=group_hint
fi
case "${WEBSHOP_EXPERIMENT_MODE}" in
  group_hint)
    export TOKEN_CREDIT_ENABLE=True
    export RUBRIC_PIPELINE_ENABLE=True
    if [[ "${WEBSHOP_REWARD_MODE}" == "r2" || "${WEBSHOP_REWARD_MODE}" == "hybrid" ]]; then
      threshold_tag="${RUBRIC_CORRECTNESS_THRESHOLD//./p}"
      default_group_hint_name="qwen3_1_7b_webshop_group_hint_bestof4_bundle_spearman_r2_t${threshold_tag}${REWARD_EXPERIMENT_SUFFIX}"
    else
      default_group_hint_name="qwen3_1_7b_webshop_group_hint_bestof4_bundle_spearman_strict_success${REWARD_EXPERIMENT_SUFFIX}"
    fi
    export EXPERIMENT_NAME="${WEBSHOP_GROUP_HINT_EXPERIMENT_NAME:-${default_group_hint_name}}"
    ;;
  matched_grpo)
    export TOKEN_CREDIT_ENABLE=False
    export RUBRIC_PIPELINE_ENABLE=False
    export EXPERIMENT_NAME="${WEBSHOP_MATCHED_EXPERIMENT_NAME:-qwen3_1_7b_webshop_matched_grpo${REWARD_EXPERIMENT_SUFFIX}}"
    ;;
  *)
    echo "Internal error: unsupported WebShop experiment mode ${WEBSHOP_EXPERIMENT_MODE}" >&2
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

export BENCHMARK=webshop
export MODEL_PATH
# Do not inherit generic TRAIN_PARQUET/VAL_PARQUET values from a preceding
# ALFWorld run in the same shell. Use benchmark-specific override names when a
# non-default WebShop dataset is intentional.
export TRAIN_PARQUET="${WEBSHOP_TRAIN_PARQUET:-${ROOT}/Data/agent_rubric/webshop2/webshop_train.parquet}"
export VAL_PARQUET="${WEBSHOP_VAL_PARQUET:-${ROOT}/Data/agent_rubric/webshop2/webshop_test.parquet}"
export VAL_FILES="['${VAL_PARQUET}']"
export RUBRIC_BACKEND=actor_rollout_group_hints
for dataset_path in "${TRAIN_PARQUET}" "${VAL_PARQUET}"; do
  if [[ ! -f "${dataset_path}" ]]; then
    echo "Dataset parquet does not exist: ${dataset_path}" >&2
    exit 1
  fi
  python3 - "${dataset_path}" <<'PY'
import json
import sys

import pyarrow.parquet as pq

path = sys.argv[1]
table = pq.read_table(path)
required = {"data_source", "agent_name", "prompt", "reward_model", "extra_info"}
missing = sorted(required.difference(table.column_names))
if missing:
    raise SystemExit(
        f"WebShop parquet has the old/incomplete schema: {path}; missing={missing}. "
        "Regenerate it with: python3 examples/agent_rubric2/prepare_webshop_data.py"
    )

if table.num_rows == 0:
    raise SystemExit(f"WebShop parquet is empty: {path}")

row = table.slice(0, 1).to_pylist()[0]
reward_model = row.get("reward_model") or {}
extra_info = row.get("extra_info") or {}
if isinstance(extra_info, str):
    extra_info = json.loads(extra_info)
interaction = extra_info.get("interaction_kwargs") or {}

errors = []
if row.get("data_source") != "webshop":
    errors.append("data_source must be 'webshop'")
if row.get("agent_name") != "tool_agent":
    errors.append("agent_name must be 'tool_agent'")
if not reward_model.get("ground_truth"):
    errors.append("reward_model.ground_truth is missing")
if interaction.get("name") != "webshop":
    errors.append("extra_info.interaction_kwargs.name must be 'webshop'")
if "goal_idx" not in interaction:
    errors.append("extra_info.interaction_kwargs.goal_idx is missing")
if interaction.get("use_interaction_reward") is not True:
    errors.append("extra_info.interaction_kwargs.use_interaction_reward must be true")
if errors:
    raise SystemExit(
        f"Invalid WebShop parquet row in {path}: " + "; ".join(errors) + ". "
        "Regenerate it with: python3 examples/agent_rubric2/prepare_webshop_data.py"
    )
PY
done

export ROLLOUT_N="${ROLLOUT_N:-8}"
export RUBRICS_PER_PROMPT="${RUBRICS_PER_PROMPT:-8}"
export MIN_RUBRICS_PER_PROMPT="${MIN_RUBRICS_PER_PROMPT:-5}"
export RUBRIC_MAX_TRAJECTORIES="${RUBRIC_MAX_TRAJECTORIES:-${ROLLOUT_N}}"
export RUBRIC_GROUP_CANDIDATES="${RUBRIC_GROUP_CANDIDATES:-4}"
export RUBRIC_GROUP_SELECT_BY_CORRELATION="${RUBRIC_GROUP_SELECT_BY_CORRELATION:-True}"
export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_JUDGE_MODEL="${RUBRIC_JUDGE_MODEL:-qwen3.5-plus}"
export RUBRIC_JUDGE_BASE_URL="${RUBRIC_JUDGE_BASE_URL:-https://api.example.com/v1}"
export RUBRIC_JUDGE_API_KEY_ENV="${RUBRIC_JUDGE_API_KEY_ENV:-DASHSCOPE_API_KEY}"
export RUBRIC_MIN_JUDGE_COVERAGE="${RUBRIC_MIN_JUDGE_COVERAGE:-0.75}"
export RUBRIC_MAX_TRAJECTORY_CHARS="${RUBRIC_MAX_TRAJECTORY_CHARS:-6000}"
export RUBRIC_MAX_RUBRIC_CHARS="${RUBRIC_MAX_RUBRIC_CHARS:-1600}"
export RUBRIC_MAX_ITEM_CHARS="${RUBRIC_MAX_ITEM_CHARS:-280}"
export RUBRIC_MAX_TOKENS="${RUBRIC_MAX_TOKENS:-4096}"
export RUBRIC_TEMPERATURE="${RUBRIC_TEMPERATURE:-0.3}"
export RUBRIC_TOP_P="${RUBRIC_TOP_P:-0.9}"
export RUBRIC_GENERATOR_THINKING="${RUBRIC_GENERATOR_THINKING:-False}"
export ACTOR_THINKING="${ACTOR_THINKING:-True}"
export TEACHER_THINKING="${TEACHER_THINKING:-True}"
export TEACHER_MAX_PROMPT_LENGTH="${TEACHER_MAX_PROMPT_LENGTH:-40960}"
export TOKEN_CREDIT_ALPHA="${TOKEN_CREDIT_ALPHA:-0.01}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
# WebShop's SimServer loads the full product catalogue and Lucene index inside
# each Ray AgentLoopWorker process.  The cache in WebshopInteraction is
# process-local, so four workers duplicate roughly 60 GiB of host memory.  A
# single worker can still schedule all rollout trajectories asynchronously and
# avoids this per-process duplication.  Benchmark-specific override names are
# used deliberately so a stale generic export from another run cannot restore
# the unsafe WebShop defaults.
export AGENT_LOOP_NUM_WORKERS="${WEBSHOP_AGENT_LOOP_NUM_WORKERS:-1}"
export DATALOADER_NUM_WORKERS="${WEBSHOP_DATALOADER_NUM_WORKERS:-1}"
# The compatibility default retains every full assistant response (including
# thinking) and every observation.  A dedicated experiment wrapper may enable
# the exact turn-replay path by setting a positive history window.
export HISTORY_WINDOW="${HISTORY_WINDOW:-0}"
export HISTORY_FIXED_PREFIX_MESSAGES="${HISTORY_FIXED_PREFIX_MESSAGES:-3}"
export HISTORY_MIN_GENERATION_TOKENS="${HISTORY_MIN_GENERATION_TOKENS:-2048}"
export ACTOR_MAX_TOKENS_PER_TURN="${ACTOR_MAX_TOKENS_PER_TURN:-2048}"
# Full-history WebShop trajectories are memory-heavy, so keep the actor micro
# batch at one. TURN_REPLAY_MICRO_BATCH_SIZE is inactive while HISTORY_WINDOW=0
# but remains available if the windowed path is explicitly re-enabled later.
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export TURN_REPLAY_MICRO_BATCH_SIZE="${TURN_REPLAY_MICRO_BATCH_SIZE:-1}"
export REQUIRE_GROUP_OUTCOME_CONTRAST="${REQUIRE_GROUP_OUTCOME_CONTRAST:-True}"
export REQUIRE_OUTCOME_ALIGNED_GAP="${REQUIRE_OUTCOME_ALIGNED_GAP:-False}"
export MIN_OUTCOME_GAP_ALIGNMENT="${MIN_OUTCOME_GAP_ALIGNMENT:-0.0}"
export CALCULATE_ROLLOUT_LOG_PROBS="${CALCULATE_ROLLOUT_LOG_PROBS:-True}"
# Match common WebShop agent-RL recipes: at most 15 environment actions. Each
# action generation is independently capped above, while the official WebShop
# continuous reward remains unchanged.
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-15}"
export MAX_STEPS="${MAX_STEPS:-50}"
export TEST_FREQ="${TEST_FREQ:-20}"
export PROJECT_NAME="${PROJECT_NAME:-verl_agent_rubric2}"

echo "Rubric route: rollout group=${ROLLOUT_N}; generator evidence<=${RUBRIC_MAX_TRAJECTORIES}; sample ${RUBRIC_GROUP_CANDIDATES} complete 5-8 item bundles"
if [[ "${RUBRIC_GROUP_SELECT_BY_CORRELATION}" == "True" || "${RUBRIC_GROUP_SELECT_BY_CORRELATION}" == "true" ]]; then
  echo "Bundle selection: highest bundle-level Spearman; no cross-bundle criterion mixing; no generator loss; no rubric pool"
else
  echo "Bundle selection: no judging; use the first parsed bundle (force candidate zero=${RUBRIC_GROUP_FORCE_FIRST_CANDIDATE:-False}); no generator loss; no rubric pool"
fi
echo "Teacher credit: additive OPD alpha=${TOKEN_CREDIT_ALPHA}; outcome-gap gate=${REQUIRE_OUTCOME_ALIGNED_GAP}"

exec bash "${ROOT}/examples/agent_rubric/run_agent_grpo.sh" "$@"
