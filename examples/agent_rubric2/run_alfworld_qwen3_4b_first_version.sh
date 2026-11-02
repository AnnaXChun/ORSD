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
export RUBRIC_SNAPSHOT_ENABLE=False
export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export ALFWORLD_GUARDED_BESTOF4_EXPERIMENT_NAME="${QWEN3_4B_ALFWORLD_FIRST_EXPERIMENT_NAME:-qwen3_4b_alfworld_opd_only_guarded_bestof4_margin0p05_no_pool_nosnapshot_4gpu_b8_n8_additiveopd0p01_actor_thinking_nogenthink_official_reward_s300}"
export ALFWORLD_GUARDED_BESTOF4_OUTPUT_DIR="${QWEN3_4B_ALFWORLD_FIRST_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${ALFWORLD_GUARDED_BESTOF4_EXPERIMENT_NAME}}"

echo "Actor model: Qwen/Qwen3-4B (${MODEL_PATH})"
echo "Thinking request: actor=True, rubric_generator=False, teacher=True"
echo "Thinking switch: actor reasoning enabled; rubric generation reasoning strictly disabled"
echo "Rubric snapshot: disabled for Qwen3-4B ALFWorld"
echo "Configuration source: matched to the Qwen3-1.7B ALFWorld guarded OPD-only launcher"
exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_opd_only_guarded_bestof4_4gpu_b8_n8_s300.sh" "$@"
