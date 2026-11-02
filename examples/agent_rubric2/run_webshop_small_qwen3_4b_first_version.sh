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
export RUBRIC_SNAPSHOT_ENABLE=True
export RUBRIC_SNAPSHOT_STEP=300
export RUBRIC_SNAPSHOT_BATCH_SIZE=8
export RUBRIC_SNAPSHOT_MAX_ATTEMPTS=4
export WEBSHOP_GUARDED_EXPERIMENT_NAME="${QWEN3_4B_WEBSHOP_FIRST_EXPERIMENT_NAME:-qwen3_4b_webshop_small_b8_n8_history0_traj4_guarded_bestof4_margin0p05_contrastgate_gapgate_temp0p7_hybrid_additiveopd0p01_actor_thinking_nogenthink_snapshot300_4gpu_s300}"

echo "Actor model: Qwen/Qwen3-4B (${MODEL_PATH})"
echo "Thinking request: actor=True, rubric_generator=False, teacher=True"
echo "Thinking switch: actor reasoning enabled; rubric generation reasoning strictly disabled"
echo "Rubric snapshot: enabled after step 300 for Qwen3-4B WebShop"
echo "Configuration source: matched to the Qwen3-1.7B WebShop guarded launcher"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_guarded_bestof4_4gpu_b8_n8_s300.sh" "$@"
