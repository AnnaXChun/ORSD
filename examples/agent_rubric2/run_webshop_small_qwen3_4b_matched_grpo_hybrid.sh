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
export MAX_RESPONSE_LENGTH=32768
export TEACHER_MAX_PROMPT_LENGTH=40960
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export TEST_FREQ=50
export SAVE_FREQ=50
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _QWEN3_4B_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_QWEN3_4B_VISIBLE_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export N_GPUS=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY_UTILIZATION="${QWEN3_4B_VLLM_GPU_UTIL:-0.72}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export ROLLOUT_MAX_NUM_SEQS=64
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True
export MICRO_BATCH_SIZE=1
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export HISTORY_WINDOW=0
export HISTORY_RESPONSE_MODE=action_only
export WEBSHOP_AGENT_LOOP_NUM_WORKERS=1
export WEBSHOP_DATALOADER_NUM_WORKERS=1
export WEBSHOP_REWARD_MODE=hybrid
export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export WEBSHOP_MATCHED_EXPERIMENT_NAME="${QWEN3_4B_WEBSHOP_MATCHED_EXPERIMENT_NAME:-qwen3_4b_webshop_small_matched_grpo_hybrid_4gpu_b8_n8_actor_thinking_s300}"

echo "Actor model: Qwen/Qwen3-4B (${MODEL_PATH})"
echo "Thinking: actor=True; rubric generator and teacher are unused"
echo "Matched protocol: prompts=8, rollouts/prompt=8, steps=300, validation/save every 50, history_window=0"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_matched_grpo.sh" "$@"
