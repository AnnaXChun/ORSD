#!/usr/bin/env bash
set -euo pipefail

# Qwen3-4B ALFWorld OPSD comparison. VERL performs the same 8x8 online
# environment rollouts and validation as matched GRPO, while the actor update
# uses the fixed-rubric sampled-token reverse-KL objective.  As in OPSD's
# default self-distillation path, the rubric-conditioned teacher is the same
# pre-update actor rather than a separately resident frozen reference model.
# Environment GRPO advantages remain diagnostic and are not added to the OPSD
# actor objective.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ALFWORLD_OPSD_RUBRIC_PARQUET:?Set ALFWORLD_OPSD_RUBRIC_PARQUET to the saved ALFWorld train parquet containing rubric columns}"

export MODEL_PATH="${QWEN3_4B_MODEL_PATH:-/workspace/checkpoints/qwen/Qwen3-4B}"
python3 - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
actual = (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers"))
if actual != ("qwen3", 2560, 36):
    raise SystemExit(f"MODEL_PATH is not Qwen3-4B: {actual}")
PY

export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ROOT}/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/alfworld}"
export ALFWORLD_R2_REWARD_ENABLE=False
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _QWEN3_4B_OPSD_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_QWEN3_4B_OPSD_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export BENCHMARK=alfworld
export TRAIN_PARQUET="${ALFWORLD_OPSD_RUBRIC_PARQUET}"
export ALFWORLD_VAL_SEEN_PARQUET="${ALFWORLD_VAL_SEEN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_valid_seen.parquet}"
export ALFWORLD_VAL_UNSEEN_PARQUET="${ALFWORLD_VAL_UNSEEN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_valid_unseen.parquet}"
export VAL_PARQUET="${ALFWORLD_VAL_UNSEEN_PARQUET}"
export VAL_FILES="['${ALFWORLD_VAL_SEEN_PARQUET}', '${ALFWORLD_VAL_UNSEEN_PARQUET}']"

for path in "${ALFWORLD_CONFIG}" "${ALFWORLD_DATA}" "${TRAIN_PARQUET}" "${ALFWORLD_VAL_SEEN_PARQUET}" "${ALFWORLD_VAL_UNSEEN_PARQUET}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing Qwen3-4B ALFWorld/OPSD asset: ${path}" >&2
    exit 1
  fi
done

# Pure OPSD has no GRPO fallback, so every train task must carry one complete
# hidden rubric bundle produced by the earlier snapshot export.
python3 - "${TRAIN_PARQUET}" <<'PY'
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

rows = table.select(
    ["data_source", "agent_name", "extra_info", "rubric", "rubric_valid", "rubric_selected_count"]
).to_pylist()
bad = []
for index, row in enumerate(rows):
    extra = row["extra_info"] or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    interaction = extra.get("interaction_kwargs") or {}
    if (
        row["data_source"] != "alfworld"
        or row["agent_name"] != "tool_agent"
        or interaction.get("name") != "alfworld"
        or interaction.get("inject_rubric", False)
        or not bool(row["rubric_valid"])
        or not str(row["rubric"] or "").strip()
        or int(row["rubric_selected_count"] or 0) < 5
    ):
        bad.append(index)
if bad:
    raise SystemExit(
        "Qwen3-4B OPSD requires every ALFWorld train task to have a hidden valid "
        f"5-8 item rubric; invalid rows={len(bad)}, first={bad[:10]}"
    )
print(f"Qwen3-4B ALFWorld OPSD rubric preflight passed: {len(rows)}/{len(rows)} tasks")
PY

# Match the Qwen3-4B ALFWorld GRPO rollout, optimizer, context, validation, and
# four-GPU execution budget.
export N_GPUS=4
# Qwen3-4B OPSD performs an additional rubric-conditioned scoring forward on
# every sampled response.  Shard each vLLM replica over two GPUs so rollout
# weights and KV cache do not compete with a full TP=1 replica on every card.
export ROLLOUT_TP=2
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export TEST_FREQ=50
export SAVE_FREQ="${SAVE_FREQ:-50}"
export MICRO_BATCH_SIZE=1
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export LEARNING_RATE=1e-6
export HISTORY_WINDOW=0
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=16384
export MAX_MODEL_LENGTH=40960
export VAL_N=4
export VAL_TEMPERATURE=0.6
export AGENT_LOOP_NUM_WORKERS="${ALFWORLD_4GPU_AGENT_LOOP_WORKERS:-8}"
export DATALOADER_NUM_WORKERS="${ALFWORLD_4GPU_DATALOADER_WORKERS:-2}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${QWEN3_4B_OPSD_VLLM_GPU_UTIL:-0.15}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${QWEN3_4B_OPSD_BATCHED_TOKENS:-8192}"
export ROLLOUT_MAX_NUM_SEQS="${QWEN3_4B_OPSD_MAX_NUM_SEQS:-16}"
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True

# Saved dataset rubric + rubric-conditioned pre-update actor teacher.  This
# preserves OPSD self-distillation without allocating a second Qwen3-4B
# reference worker alongside the actor and vLLM rollout engine.  No online
# rubric generation, cloud judge, rubric pool, generator loss, or snapshot.
export TOKEN_CREDIT_ENABLE=True
export TOKEN_CREDIT_MODE=opsd_reverse_kl
export TOKEN_CREDIT_TEACHER_LOG_PROB_KEY=rubric_teacher_log_probs
export OPSD_TEACHER_SOURCE=current_actor
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
export EXPERIMENT_NAME="${QWEN3_4B_ALFWORLD_OPSD_EXPERIMENT_NAME:-qwen3_4b_alfworld_grpo_opsd_reversekl_currentactor_savedrubric_b8_n8_tp2_offload_memsafe_official01_4gpu_s300}"
export OUTPUT_DIR="${QWEN3_4B_ALFWORLD_OPSD_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

echo "Actor model: Qwen/Qwen3-4B (${MODEL_PATH})"
echo "ALFWorld GRPO+OPSD: online 8x8 rollout with fixed-rubric sampled-token reverse-KL update"
echo "Protocol: steps=${MAX_STEPS}, validation_every=${TEST_FREQ}, checkpoint_every=${SAVE_FREQ}, GPUs=${CUDA_VISIBLE_DEVICES}"
echo "Context: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, model=${MAX_MODEL_LENGTH}"
echo "Memory profile: rollout_TP=${ROLLOUT_TP}, vLLM_util=${ROLLOUT_GPU_MEMORY_UTILIZATION}, batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}, max_seqs=${ROLLOUT_MAX_NUM_SEQS}, micro_batch=${MICRO_BATCH_SIZE}, actor_offload=True"
echo "Rubrics: dataset=${TRAIN_PARQUET}; online generation=off; pool=off; generator loss=off"
echo "Teacher: source=${OPSD_TEACHER_SOURCE}; separate reference worker=off"
echo "Training memory: chunked LM head (torch fused backend, 512 tokens/chunk); entropy objective=off; rollout cache release=on"
echo "Optimizer memory: AdamW foreach=False, fused=False (per-tensor update)"
echo "Memory audit: ${OPSD_MEMORY_AUDIT:-1}; search training logs for [memory-audit]"
echo "Objective: rubric-conditioned pre-update actor log-prob minus plain-context actor log-prob; GRPO advantage is diagnostic only"
echo "Experiment: ${EXPERIMENT_NAME}; checkpoints=${OUTPUT_DIR}"

exec bash "${ROOT}/examples/agent_rubric/run_agent_grpo.sh" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_MEMORY_AUDIT='${OPSD_MEMORY_AUDIT:-1}'" \
  ++actor_rollout_ref.rollout.enable_sleep_mode=True \
  actor_rollout_ref.actor.ppo_epochs=1 \
  'actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false,fused:false}' \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.actor.calculate_entropy=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
  "$@"
