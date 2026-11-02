#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${ALFWORLD_CONFIG:?Set ALFWORLD_CONFIG to the official base_config.yaml}"
: "${ALFWORLD_DATA:?Set ALFWORLD_DATA to the downloaded ALFWorld data directory}"

MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7B}"
INPUT_PARQUET="${INPUT_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_train.parquet}"
NUM_TASKS="${NUM_TASKS:-20}"
PROMPT_TAG="${PROMPT_TAG:-state_criteria_v6}"
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
# vLLM TP workers must not fork after the parent has imported/initialized
# PyTorch CUDA. Each stage is a separate Python process, so make the safe
# worker start method explicit for every stage.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if (( START_STAGE <= 1 )); then
  echo "[1/4] Collecting paired baseline trajectories from ${NUM_TASKS} ALFWorld tasks"
  python3 "${ROOT}/examples/agent_rubric/collect_rollout_trajectories.py" \
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
  echo "[2/4] Per-rollout rubric generation plus mixed-group item-matrix filtering"
  python3 "${ROOT}/examples/agent_rubric/generate_online_actor_rubrics_debug.py" \
    --input-parquet "${BASELINE_PARQUET}" \
    --output-parquet "${VALIDATED_PARQUET}" \
    --model "${MODEL_PATH}" \
    --limit "${NUM_TASKS}" \
    --rubrics-per-prompt "${RUBRICS_PER_PROMPT:-4}" \
    --min-rubrics "${MIN_RUBRICS_PER_PROMPT:-2}" \
    --min-reliable-rubrics "${MIN_RELIABLE_RUBRICS_PER_PROMPT:-1}" \
    --reliability-threshold "${RUBRIC_RELIABILITY_THRESHOLD:-0.5}" \
    --correctness-threshold "${RUBRIC_CORRECTNESS_THRESHOLD:-0.5}" \
    --max-candidate-rubrics "${MAX_CANDIDATE_RUBRICS_PER_GROUP:-24}" \
    --max-trajectories "${RUBRIC_MAX_TRAJECTORIES:-1}" \
    --max-trajectory-chars "${RUBRIC_MAX_TRAJECTORY_CHARS:-6000}" \
    --max-tokens "${RUBRIC_MAX_TOKENS:-2048}" \
    --judge-max-tokens "${RUBRIC_JUDGE_MAX_TOKENS:-1024}" \
    --judge-temperature "${RUBRIC_JUDGE_TEMPERATURE:-0.0}" \
    --temperature "${RUBRIC_TEMPERATURE:-0.2}" \
    --top-p "${RUBRIC_TOP_P:-0.95}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
    --max-model-len "${MAX_MODEL_LEN:-40960}" \
    --batch-size "${RUBRIC_BATCH_SIZE:-4}" \
    --enable-thinking
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
  echo "[3/4] Re-running the same ${NUM_TASKS} tasks with validated rubric privilege"
  python3 "${ROOT}/examples/agent_rubric/collect_rollout_trajectories.py" \
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
    --enable-thinking \
    --inject-rubric \
    --replace-existing
else
  [[ -f "${PRIVILEGED_PARQUET}" ]] || {
    echo "Cannot resume at stage ${START_STAGE}: missing ${PRIVILEGED_PARQUET}" >&2
    exit 1
  }
  echo "[3/4] Reusing ${PRIVILEGED_PARQUET}"
fi

echo "[4/4] Comparing official environment outcomes"
python3 "${ROOT}/examples/agent_rubric/compare_agent_rollouts.py" \
  --baseline-parquet "${BASELINE_PARQUET}" \
  --rubric-parquet "${PRIVILEGED_PARQUET}" \
  --tokenizer "${MODEL_PATH}" \
  --max-model-len "${MAX_MODEL_LEN:-40960}" \
  --max-action-tokens "${MAX_ACTION_TOKENS:-2048}" \
  --enable-thinking \
  --output-json "${OUTPUT_DIR}/summary.json"
