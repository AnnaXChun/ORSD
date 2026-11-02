#!/usr/bin/env bash
set -euo pipefail

# Pure OPSD comparison on WebShop-small.
#
# VERL still performs the same online environment rollouts and validation as
# the matched GRPO run.  The only optimization change is that the actor uses
# the sampled-token reverse-KL advantage from a fixed per-task rubric:
#   stopgrad(log pi_ref(y | context, rubric) - log pi_old(y | context)).
# Environment rewards are logged but do not enter the actor objective.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPSD_RUBRIC_PARQUET:?Set OPSD_RUBRIC_PARQUET to the saved train parquet containing rubric columns}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _OPSD_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_OPSD_VISIBLE_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

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

export WEBSHOP_ROOT="${WEBSHOP_ROOT:-${ROOT}/WebShop}"
export WEBSHOP_USE_SMALL=True
export WEBSHOP_HUMAN_GOALS=False
export WEBSHOP_NUM_PRODUCTS=1000
export WEBSHOP_PRODUCTS_JSON="${WEBSHOP_ROOT}/data/items_shuffle_1000.json"
export WEBSHOP_TRAIN_PARQUET="${OPSD_RUBRIC_PARQUET}"
export WEBSHOP_VAL_PARQUET="${WEBSHOP_VAL_PARQUET:-${ROOT}/Data/agent_rubric/webshop_small/webshop_test.parquet}"

for required_path in \
  "${WEBSHOP_PRODUCTS_JSON}" \
  "${WEBSHOP_ROOT}/search_engine/indexes_1k" \
  "${WEBSHOP_TRAIN_PARQUET}" \
  "${WEBSHOP_VAL_PARQUET}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing WebShop-small/OPSD asset: ${required_path}" >&2
    exit 1
  fi
done

# Reject partial snapshots: pure OPSD deliberately has no GRPO fallback.
python3 - "${WEBSHOP_TRAIN_PARQUET}" <<'PY'
import json
import sys

import pyarrow.parquet as pq

path = sys.argv[1]
required = {
    "data_source", "agent_name", "prompt", "reward_model", "extra_info",
    "rubric", "rubric_valid", "rubric_selected_count",
}
table = pq.read_table(path)
missing = sorted(required.difference(table.column_names))
if missing:
    raise SystemExit(f"Saved-rubric parquet is missing columns {missing}: {path}")
if table.num_rows == 0:
    raise SystemExit(f"Saved-rubric parquet is empty: {path}")

rows = table.select(["data_source", "agent_name", "extra_info", "rubric", "rubric_valid", "rubric_selected_count"]).to_pylist()
bad = []
for index, row in enumerate(rows):
    extra = row["extra_info"] or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    interaction = extra.get("interaction_kwargs") or {}
    valid = bool(row["rubric_valid"])
    rubric = str(row["rubric"] or "").strip()
    count = int(row["rubric_selected_count"] or 0)
    if (
        row["data_source"] != "webshop"
        or row["agent_name"] != "tool_agent"
        or interaction.get("name") != "webshop"
        or interaction.get("use_small") is not True
        or interaction.get("human_goals") is not False
        or interaction.get("num_products") != 1000
        or interaction.get("inject_rubric", False)
        or not valid
        or not rubric
        or count < 5
    ):
        bad.append(index)
if bad:
    raise SystemExit(
        f"Pure OPSD requires every train task to have a hidden valid 5-8 item rubric; "
        f"invalid rows={len(bad)}, first={bad[:10]}"
    )
print(f"Pure OPSD rubric preflight passed: {len(rows)}/{len(rows)} tasks have hidden valid bundles")
PY

export BENCHMARK=webshop
export MODEL_PATH
export TRAIN_PARQUET="${WEBSHOP_TRAIN_PARQUET}"
export VAL_PARQUET="${WEBSHOP_VAL_PARQUET}"
export VAL_FILES="['${VAL_PARQUET}']"

# Matched data, rollout, optimizer, context, checkpoint, and validation budget.
export N_GPUS=4
export ROLLOUT_TP=1
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export TEST_FREQ=50
export SAVE_FREQ=50
export MICRO_BATCH_SIZE=1
export LEARNING_RATE=1e-6
export HISTORY_WINDOW=4
export HISTORY_RESPONSE_MODE=full_response
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export MAX_RESPONSE_LENGTH=32768
export MAX_MODEL_LENGTH=40960
export MAX_ASSISTANT_TURNS=15
export VAL_N=4
export VAL_TEMPERATURE=0.6

export ROLLOUT_GPU_MEMORY_UTILIZATION="${OPSD_VLLM_GPU_UTIL:-0.72}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export ROLLOUT_MAX_NUM_SEQS=64
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True
export WEBSHOP_AGENT_LOOP_NUM_WORKERS=1
export WEBSHOP_DATALOADER_NUM_WORKERS=1
export AGENT_LOOP_NUM_WORKERS=1
export DATALOADER_NUM_WORKERS=1

# Keep the benchmark reward identical for rollout diagnostics and validation.
# In opsd_reverse_kl it is intentionally excluded from actor advantages.
export WEBSHOP_REWARD_MODE=hybrid
export WEBSHOP_R2_INVALID_ACTION_COEF=0.05
export WEBSHOP_R2_REPEAT_ACTION_COEF=0.025

# Fixed saved rubrics; no online generation, judge, pool, or generator loss.
export TOKEN_CREDIT_ENABLE=True
export TOKEN_CREDIT_MODE=opsd_reverse_kl
export TOKEN_CREDIT_TEACHER_LOG_PROB_KEY=rubric_teacher_log_probs
export OPSD_TEACHER_SOURCE=reference_policy
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export RUBRIC_PIPELINE_ENABLE=True
export RUBRIC_BACKEND=dataset
export TARGET_VALID_RATE=1.0
export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=False
export RUBRIC_SNAPSHOT_ENABLE=False
export ACTOR_THINKING=True
export TEACHER_THINKING=True
export RUBRIC_GENERATOR_THINKING=False
export TEACHER_MAX_PROMPT_LENGTH=40960

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export EXPERIMENT_NAME="${OPSD_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_opsd_reversekl_savedrubric_b8_n8_history4_hybrid_4gpu_s300}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

echo "WebShop-small OPSD: VERL environment rollout + pure sampled-token reverse-KL actor objective"
echo "Protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, steps=${MAX_STEPS}, validation_every=${TEST_FREQ}"
echo "Objective: frozen-reference rubric log-prob minus current plain-context log-prob; GRPO advantage is diagnostic only"
echo "Rubrics: fixed dataset=${TRAIN_PARQUET}; online generation=off, pool=off, generator loss=off"
echo "Matched actor: lr=${LEARNING_RATE}, prompt mini-batch=${TRAIN_BATCH_SIZE}, micro-batch=${MICRO_BATCH_SIZE}, epochs=1, direct PG loss"
echo "Experiment: ${EXPERIMENT_NAME}"

exec bash "${ROOT}/examples/agent_rubric/run_agent_grpo.sh" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
  "$@"
