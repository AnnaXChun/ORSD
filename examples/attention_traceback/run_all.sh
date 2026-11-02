#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAGE="${1:-smoke}"
source "$ROOT/examples/attention_traceback/orsd_paths.sh"
echo "ORSD model: $MODEL_PATH"
echo "WebShop-small data: $WEBSHOP_PARQUET"
echo "ALFWorld data: $ALFWORLD_PARQUET"
echo "ALFWorld environment: config=$ALFWORLD_CONFIG; data=$ALFWORLD_DATA"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Missing Qwen3-1.7B checkpoint: $MODEL_PATH/config.json" >&2
  echo "ORSD defaults: /workspace/checkpoints/Qwen3-1.7B, /workspace/checkpoints/qwen/Qwen3-1.7B, $ROOT/checkpoints/Qwen3-1.7B" >&2
  echo "Override with MODEL_PATH or QWEN3_1_7B_MODEL_PATH only if your server uses a different location." >&2
  exit 2
fi
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/attention_traceback}"
INTERVENTION="${INTERVENTION:-mask_span}"
case "$STAGE" in
  smoke) TASKS=1; SIBLINGS=2; STAGE_ROOT="$RESULT_ROOT/smoke" ;;
  phase-a)
    TASKS="${TASKS:-50}"; SIBLINGS=4; STAGE_ROOT="$RESULT_ROOT"
    REPLAY_WORKERS="${REPLAY_WORKERS:-4}"
    python3 - "$RESULT_ROOT/smoke" "$MODEL_PATH" "$INTERVENTION" <<'PY'
import json, sys
from pathlib import Path
for name in ('webshop', 'alfworld', 'scienceworld'):
    row = json.loads((Path(sys.argv[1]) / name / 'smoke_pass.json').read_text())
    assert row['passed'] and row['intervention'] == sys.argv[3], name
    assert Path(row['model']).resolve() == Path(sys.argv[2]).resolve(), name
PY
    ;;
  *) echo 'Usage: bash examples/attention_traceback/run_all.sh smoke|phase-a' >&2; exit 2 ;;
esac
mkdir -p "$STAGE_ROOT"
bash examples/attention_traceback/server_environment.sh > "$STAGE_ROOT/server_environment.txt" 2>&1
if [[ "$STAGE" == smoke ]]; then
  python3 -m pytest -q tests/attention_traceback | tee "$STAGE_ROOT/server_tests.txt"
fi
for benchmark in webshop alfworld scienceworld; do
  parquet="$STAGE_ROOT/tasks/$benchmark.parquet"
  capture="$STAGE_ROOT/capture/$benchmark"
  if [[ ! -e "$parquet" ]]; then
    extra=()
    if [[ "$benchmark" == webshop ]]; then extra=(--input "$WEBSHOP_PARQUET"); fi
    if [[ "$benchmark" == alfworld ]]; then extra=(--input "$ALFWORLD_PARQUET"); fi
    python3 examples/attention_traceback/prepare_data.py --benchmark "$benchmark" \
      --tasks "$TASKS" --output "$parquet" "${extra[@]}"
  fi
  # An existing capture is always reused. Partial runs fail the exact group-size
  # check in replay instead of silently resampling the environment.
  if [[ ! -d "$capture" ]]; then
    BENCHMARK="$benchmark" TASK_PARQUET="$parquet" CAPTURE_DIR="$capture" \
      TRAJECTORIES_PER_TASK="$SIBLINGS" VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-1}" \
      bash examples/attention_traceback/collect.sh
  fi
  python3 -m verl.experimental.attention_traceback.run \
    --input "$capture" --output "$STAGE_ROOT/$benchmark" --model "$MODEL_PATH" \
    --stage "$STAGE" --smoke-dir "$RESULT_ROOT/smoke" \
    --workers "${REPLAY_WORKERS:-1}" \
    "attention_traceback.intervention=$INTERVENTION"
done
python3 examples/attention_traceback/compare.py --root "$STAGE_ROOT"
