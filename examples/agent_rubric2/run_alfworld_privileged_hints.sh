#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${ALFWORLD_CONFIG:?Set ALFWORLD_CONFIG to the official base_config.yaml}"
: "${ALFWORLD_DATA:?Set ALFWORLD_DATA to the downloaded ALFWorld data directory}"

MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7B}"
INPUT_PARQUET="${INPUT_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_train.parquet}"
NUM_TASKS="${NUM_TASKS:-20}"
PROMPT_TAG="${PROMPT_TAG:-privileged_hints_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiment_outputs/agent_rubric/alfworld_prompt_debug_${NUM_TASKS}_${PROMPT_TAG}}"
BASELINE_PARQUET="${BASELINE_PARQUET:-${OUTPUT_DIR}/baseline_rollouts.parquet}"
VALIDATED_PARQUET="${VALIDATED_PARQUET:-${OUTPUT_DIR}/validated_rubrics.parquet}"
PRIVILEGED_PARQUET="${PRIVILEGED_PARQUET:-${OUTPUT_DIR}/rubric_rollouts.parquet}"
RUBRIC_JSON="${RUBRIC_JSON:-${OUTPUT_DIR}/generated_rubrics.json}"
START_STAGE="${START_STAGE:-1}"

if [[ ! "${START_STAGE}" =~ ^[1-4]$ ]]; then
  echo "START_STAGE must be one of 1, 2, 3, or 4; got ${START_STAGE}" >&2
  exit 2
fi
if [[ ! "${NUM_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_TASKS must be a positive integer; got ${NUM_TASKS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if (( START_STAGE <= 1 )); then
  echo "[1/4] Collecting baseline trajectories from ${NUM_TASKS} ALFWorld tasks"
  python3 "${ROOT}/examples/agent_rubric2/collect_rollout_trajectories.py" \
    --benchmark alfworld \
    --alfworld-config "${ALFWORLD_CONFIG}" \
    --alfworld-data "${ALFWORLD_DATA}" \
    --input-parquet "${INPUT_PARQUET}" \
    --output-parquet "${BASELINE_PARQUET}" \
    --model "${MODEL_PATH}" \
    --limit "${NUM_TASKS}" \
    --num-rollouts "${NUM_ROLLOUTS:-4}" \
    --episode-batch-size "${EPISODE_BATCH_SIZE:-8}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
    --max-model-len "${MAX_MODEL_LEN:-40960}" \
    --history-window "${HISTORY_WINDOW:-8}" \
    --max-action-tokens "${MAX_ACTION_TOKENS:-2048}" \
    --enable-thinking \
    --replace-existing
else
  [[ -f "${BASELINE_PARQUET}" ]] || {
    echo "Cannot resume at stage ${START_STAGE}: missing ${BASELINE_PARQUET}" >&2
    exit 1
  }
  echo "[1/4] Reusing ${BASELINE_PARQUET}"
fi

if (( START_STAGE <= 2 )); then
  echo "[2/4] Distilling task-level privileged hints from baseline rollout groups"
  python3 "${ROOT}/examples/agent_rubric2/generate_online_actor_rubrics_debug.py" \
    --input-parquet "${BASELINE_PARQUET}" \
    --output-parquet "${VALIDATED_PARQUET}" \
    --model "${MODEL_PATH}" \
    --limit "${NUM_TASKS}" \
    --rubrics-per-prompt "${RUBRICS_PER_PROMPT:-5}" \
    --max-trajectories "${RUBRIC_MAX_TRAJECTORIES:-4}" \
    --max-trajectory-chars "${RUBRIC_MAX_TRAJECTORY_CHARS:-6000}" \
    --max-rubric-chars "${RUBRIC_MAX_RUBRIC_CHARS:-1600}" \
    --max-item-chars "${RUBRIC_MAX_ITEM_CHARS:-280}" \
    --max-tokens "${RUBRIC_MAX_TOKENS:-4096}" \
    --temperature "${RUBRIC_TEMPERATURE:-0.3}" \
    --top-p "${RUBRIC_TOP_P:-0.9}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
    --max-model-len "${MAX_MODEL_LEN:-40960}" \
    --batch-size "${RUBRIC_BATCH_SIZE:-4}" \
    --no-enable-thinking
else
  [[ -f "${VALIDATED_PARQUET}" ]] || {
    echo "Cannot resume at stage ${START_STAGE}: missing ${VALIDATED_PARQUET}" >&2
    exit 1
  }
  echo "[2/4] Reusing ${VALIDATED_PARQUET}"
fi

echo "[rubric audit] Exporting thinking-free rubrics"
python3 "${ROOT}/examples/agent_rubric/export_rubrics_json.py" \
  --input-parquet "${VALIDATED_PARQUET}" \
  --output-json "${RUBRIC_JSON}"

if (( START_STAGE <= 3 )); then
  echo "[3/4] Re-running the same ${NUM_TASKS} tasks with privileged hints"
  STAGE3_EXTRA_ARGS=()
  if [[ "${DEBUG_FIRST_PROMPT:-0}" == "1" ]]; then
    STAGE3_EXTRA_ARGS+=(--debug-first-prompt)
  fi

  python3 "${ROOT}/examples/agent_rubric2/collect_rollout_trajectories.py" \
    --benchmark alfworld \
    --alfworld-config "${ALFWORLD_CONFIG}" \
    --alfworld-data "${ALFWORLD_DATA}" \
    --input-parquet "${VALIDATED_PARQUET}" \
    --output-parquet "${PRIVILEGED_PARQUET}" \
    --model "${MODEL_PATH}" \
    --limit "${NUM_TASKS}" \
    --num-rollouts "${NUM_ROLLOUTS:-4}" \
    --episode-batch-size "${EPISODE_BATCH_SIZE:-8}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
    --max-model-len "${MAX_MODEL_LEN:-40960}" \
    --max-action-tokens "${MAX_ACTION_TOKENS:-2048}" \
    --history-window "${HISTORY_WINDOW:-8}" \
    --enable-thinking \
    --inject-rubric \
    --replace-existing \
    "${STAGE3_EXTRA_ARGS[@]}"
else
  [[ -f "${PRIVILEGED_PARQUET}" ]] || {
    echo "Cannot resume at stage ${START_STAGE}: missing ${PRIVILEGED_PARQUET}" >&2
    exit 1
  }
  echo "[3/4] Reusing ${PRIVILEGED_PARQUET}"
fi

echo "[4/4] Comparing official environment outcomes"
python3 "${ROOT}/examples/agent_rubric2/compare_agent_rollouts.py" \
  --baseline-parquet "${BASELINE_PARQUET}" \
  --rubric-parquet "${PRIVILEGED_PARQUET}" \
  --tokenizer "${MODEL_PATH}" \
  --max-model-len "${MAX_MODEL_LEN:-40960}" \
  --max-action-tokens "${MAX_ACTION_TOKENS:-2048}" \
  --enable-thinking \
  --history-window "${HISTORY_WINDOW:-8}" \
  --output-json "${OUTPUT_DIR}/summary.json"
