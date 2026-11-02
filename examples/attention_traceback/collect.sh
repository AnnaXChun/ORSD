#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${BENCHMARK:?webshop, alfworld, or scienceworld}"
source "$ROOT/examples/attention_traceback/orsd_paths.sh"
: "${TASK_PARQUET:?Prepared diagnostic subset parquet}"
: "${CAPTURE_DIR:?Absolute empty output directory on server}"
if [[ -d "$CAPTURE_DIR" && -n "$(ls -A "$CAPTURE_DIR")" ]]; then
  echo "CAPTURE_DIR is not empty; reuse its cache for replay or choose a new directory" >&2
  exit 2
fi
PROMPT_LENGTH=2048
case "$BENCHMARK" in
  webshop)
    : "${WEBSHOP_ROOT:?Existing WebShop checkout}"
    export WEBSHOP_USE_SMALL=True WEBSHOP_HUMAN_GOALS=False WEBSHOP_NUM_PRODUCTS=1000
    export WEBSHOP_PRODUCTS_JSON="$WEBSHOP_ROOT/data/items_shuffle_1000.json"
    export WEBSHOP_REWARD_MODE="${WEBSHOP_REWARD_MODE:-hybrid}"
    INTERACTION_CONFIG="$ROOT/examples/agent_rubric/config/webshop_interaction_config.yaml"
    MAX_TURNS=15
    RESPONSE_LENGTH=32768
    ;;
  alfworld)
    : "${ALFWORLD_CONFIG:?Existing ALFWorld base_config.yaml}"
    : "${ALFWORLD_DATA:?Existing ALFWorld data directory}"
    export ALFWORLD_R2_REWARD_ENABLE=False
    INTERACTION_CONFIG="$ROOT/examples/agent_rubric/config/alfworld_interaction_config.yaml"
    MAX_TURNS=null
    RESPONSE_LENGTH=16384
    ;;
  scienceworld)
    if [[ "${TRAJECTORIES_PER_TASK:-2}" -ge 4 ]]; then
      # A protocol change needs its own real simulator smoke before Phase A.
      python3 - "${RESULT_ROOT:-$ROOT/results/attention_traceback}" <<'PY'
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1])
receipts = [root / 'smoke/scienceworld/smoke_pass.json',
            root / 'smoke_scienceworld_compact/smoke_pass.json']
for attempt in range(360):
    for path in receipts:
        if path.exists():
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if row.get('passed') and row.get('observation_styles') == ['compact']:
                print('ScienceWorld compact protocol smoke verified:', path, flush=True)
                sys.exit(0)
    if attempt == 0:
        print('Waiting for ScienceWorld compact protocol smoke; no Phase A rollout started.', flush=True)
    time.sleep(10)
raise RuntimeError('Run a ScienceWorld compact protocol smoke before Phase A')
PY
    fi
    # Reserve room for the task, public action templates and initial observation.
    PROMPT_LENGTH=8192
    INTERACTION_CONFIG="$ROOT/examples/attention_traceback/scienceworld_interaction.yaml"
    MAX_TURNS=null
    RESPONSE_LENGTH=32768
    ;;
  *) echo "Unknown benchmark: $BENCHMARK" >&2; exit 2 ;;
esac
# Keep the server's visible GPU assignment. No package installs or environment activation.
N_GPUS="${N_GPUS:-4}"
exec python3 -m verl.trainer.main_ppo --config-name attention_traceback \
  "attention_traceback.capture_dir=$CAPTURE_DIR" \
  "data.train_files=['$TASK_PARQUET']" "data.val_files=['$TASK_PARQUET']" \
  data.train_batch_size=1 "data.val_batch_size=${VALIDATION_BATCH_SIZE:-1}" data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=False data.truncation=error data.return_raw_chat=True \
  data.seed=1234 "data.max_prompt_length=$PROMPT_LENGTH" "data.max_response_length=$RESPONSE_LENGTH" \
  "+data.apply_chat_template_kwargs.enable_thinking=${ACTOR_THINKING:-True}" \
  "actor_rollout_ref.model.path=$MODEL_PATH" actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  "actor_rollout_ref.rollout.n=$N_GPUS" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1}" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}" \
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LENGTH:-40960}" \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$INTERACTION_CONFIG" \
  "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=null \
  actor_rollout_ref.rollout.multi_turn.history_window=0 \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  "actor_rollout_ref.rollout.val_kwargs.n=${TRAJECTORIES_PER_TASK:-2}" \
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.6}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.enable=False reward_model.use_reward_loop=False \
  trainer.val_only=True trainer.val_before_train=True trainer.resume_mode=disable \
  "trainer.n_gpus_per_node=$N_GPUS" trainer.nnodes=1 trainer.total_training_steps=1 \
  trainer.logger='[console]' trainer.save_freq=-1 trainer.test_freq=-1 \
  "trainer.default_local_dir=${CAPTURE_DIR}_runtime" \
  "$@"
