#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/checkpoints/qwen}"
MODEL_PATH="${QWEN3_4B_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

mkdir -p "${MODEL_PATH}"
echo "Model: ${MODEL_ID}"
echo "Mirror: ${HF_ENDPOINT}"
echo "Destination: ${MODEL_PATH}"
if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
else
  echo "Install huggingface_hub first: pip install -U huggingface_hub" >&2
  exit 1
fi

python3 - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
config_path = model_path / "config.json"
tokenizer_path = model_path / "tokenizer_config.json"
if not config_path.is_file() or not tokenizer_path.is_file():
    raise SystemExit(f"Incomplete model metadata in {model_path}")
config = json.loads(config_path.read_text(encoding="utf-8"))
expected = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36}
mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if mismatches:
    raise SystemExit(f"Downloaded checkpoint is not Qwen3-4B: {mismatches}")
weights = sorted(model_path.glob("*.safetensors"))
if not weights or any(path.stat().st_size == 0 for path in weights):
    raise SystemExit(f"Missing or empty safetensors weights in {model_path}")
print(
    f"Download verified: model_type={config['model_type']}, hidden_size={config['hidden_size']}, "
    f"layers={config['num_hidden_layers']}, context={config.get('max_position_embeddings')}, "
    f"weight_shards={len(weights)}"
)
print(f"Model is ready at: {model_path}")
PY
