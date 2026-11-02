#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_PATH="${QWEN3_4B_MODEL_PATH:-/workspace/checkpoints/qwen/Qwen3-4B}"
python3 - "${MODEL_PATH}/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
actual = (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers"))
if actual != ("qwen3", 2560, 36):
    raise SystemExit(f"MODEL_PATH is not Qwen3-4B: {actual}")
PY

export ACTOR_THINKING=True
export RUBRIC_GENERATOR_THINKING=False
export TEACHER_THINKING=True
export MAX_MODEL_LENGTH=40960
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=16384
export MICRO_BATCH_SIZE=2
export TURN_REPLAY_MICRO_BATCH_SIZE=2
export TEACHER_MAX_PROMPT_LENGTH=40960
export ALFWORLD_4GPU_AGENT_LOOP_WORKERS="${ALFWORLD_4GPU_AGENT_LOOP_WORKERS:-8}"
export TEST_FREQ=50
export SAVE_FREQ=50
export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export ALFWORLD_MATCHED_EXPERIMENT_NAME="${QWEN3_4B_ALFWORLD_MATCHED_EXPERIMENT_NAME:-qwen3_4b_alfworld_grpo_4gpu_b8_n8_actor_thinking_official_reward_s300}"
export ALFWORLD_GRPO_OUTPUT_DIR="${QWEN3_4B_ALFWORLD_GRPO_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${ALFWORLD_MATCHED_EXPERIMENT_NAME}}"

echo "Actor model: Qwen/Qwen3-4B (${MODEL_PATH})"
echo "Thinking: actor=True; rubric generator and teacher are unused"
echo "Context: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, model=${MAX_MODEL_LENGTH}"
exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_matched_grpo_4gpu_b8_n8_s300.sh" "$@"
