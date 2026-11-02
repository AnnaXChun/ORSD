#!/usr/bin/env bash
# Source-only path defaults shared by diagnostic launchers.
# Mirror agent_rubric2's Qwen3-1.7B ORSD launchers without sourcing training commands.
ATTENTION_ORSD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Explicit overrides win. Only autodetect when neither existing variable is set.
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -n "${QWEN3_1_7B_MODEL_PATH:-}" ]]; then
    MODEL_PATH="$QWEN3_1_7B_MODEL_PATH"
  else
    MODEL_PATH=/workspace/checkpoints/Qwen3-1.7B
    for attention_model_candidate in \
      /workspace/checkpoints/Qwen3-1.7B \
      /workspace/checkpoints/qwen/Qwen3-1.7B \
      "$ATTENTION_ORSD_ROOT/checkpoints/Qwen3-1.7B"; do
      if [[ -f "$attention_model_candidate/config.json" ]]; then
        MODEL_PATH="$attention_model_candidate"
        break
      fi
    done
    unset attention_model_candidate
  fi
fi
while [[ "$MODEL_PATH" == */ && "$MODEL_PATH" != / ]]; do
  MODEL_PATH="${MODEL_PATH%/}"
done
export MODEL_PATH

export WEBSHOP_ROOT="${WEBSHOP_ROOT:-$ATTENTION_ORSD_ROOT/WebShop}"
export WEBSHOP_SMALL_DATA_DIR="${WEBSHOP_SMALL_DATA_DIR:-$ATTENTION_ORSD_ROOT/Data/agent_rubric/webshop_small}"
export WEBSHOP_VAL_PARQUET="${WEBSHOP_VAL_PARQUET:-$WEBSHOP_SMALL_DATA_DIR/webshop_test.parquet}"
export WEBSHOP_PARQUET="${WEBSHOP_PARQUET:-$WEBSHOP_VAL_PARQUET}"

export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-$ATTENTION_ORSD_ROOT/alfworld/configs/base_config.yaml}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$ATTENTION_ORSD_ROOT/alfworld}"
export ALFWORLD_VAL_UNSEEN_PARQUET="${ALFWORLD_VAL_UNSEEN_PARQUET:-$ATTENTION_ORSD_ROOT/Data/agent_rubric/alfworld/alfworld_valid_unseen.parquet}"
export ALFWORLD_PARQUET="${ALFWORLD_PARQUET:-$ALFWORLD_VAL_UNSEEN_PARQUET}"
export RESULT_ROOT="${RESULT_ROOT:-$ATTENTION_ORSD_ROOT/results/attention_traceback}"
unset ATTENTION_ORSD_ROOT
