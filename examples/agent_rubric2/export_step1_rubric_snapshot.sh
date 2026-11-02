#!/usr/bin/env bash
# Keep the 300-step optimizer horizon; stop only after the step-1 full-data export.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENVIRONMENT="${1:?Usage: $0 alfworld|webshop qwen3_1_7b|qwen3_4b [Hydra overrides]}"
MODEL="${2:?Specify qwen3_1_7b or qwen3_4b}"
shift 2
case "${MODEL}" in
  qwen3_1_7b) export MODEL_PATH="${QWEN3_1_7B_MODEL_PATH:-/workspace/checkpoints/qwen/Qwen3-1.7B}" ;;
  qwen3_4b) export MODEL_PATH="${QWEN3_4B_MODEL_PATH:-/workspace/checkpoints/qwen/Qwen3-4B}" ;;
  *) echo "Unknown model: ${MODEL}" >&2; exit 2 ;;
esac
case "${ENVIRONMENT}" in
  alfworld) LAUNCHER=run_alfworld_opd_only_no_pool_4gpu_b8_n8_s300.sh ;;
  webshop) LAUNCHER="run_webshop_small_${MODEL}_opd_only_no_pool_4gpu.sh" ;;
  *) echo "Unknown environment: ${ENVIRONMENT}" >&2; exit 2 ;;
esac
python3 - "${MODEL_PATH}/config.json" "${MODEL}" <<'PYMODEL'
import json, sys
with open(sys.argv[1]) as source:
    config = json.load(source)
expected = (2048, 28) if sys.argv[2] == "qwen3_1_7b" else (2560, 36)
assert (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers")) == ("qwen3", *expected), "Wrong actor model config"
PYMODEL
NAME="${MODEL}_${ENVIRONMENT}_rubric_snapshot_step1"
OUTPUT="${STEP1_RUBRIC_EXPORT_DIR:-${ROOT}/experiment_outputs/rubric_snapshot_step1/${ENVIRONMENT}/${MODEL}}"
export ALFWORLD_OPD_ONLY_EXPERIMENT_NAME="${NAME}"
export ALFWORLD_OPD_ONLY_OUTPUT_DIR="${OUTPUT}"
export WEBSHOP_NOPOOL_EXPERIMENT_NAME="${NAME}"
export WEBSHOP_NOPOOL_OUTPUT_DIR="${OUTPUT}"
export RUBRIC_SNAPSHOT_STEP=1
export RUBRIC_SNAPSHOT_ENABLE=True
export WEBSHOP_NOPOOL_SNAPSHOT_ENABLE=True
export RUBRIC_GROUP_CANDIDATES=1
export RUBRIC_GROUP_SELECT_BY_CORRELATION=False
export RUBRIC_GROUP_GUARDED_SELECTION=False
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED=False
echo "Step-1 export: after first actor update; 8 tasks x 8 rollouts; original 300-step optimizer horizon"
echo "Then sweep ALL training tasks: one complete unjudged bundle/task; stop after export"
echo "Output: ${OUTPUT}/rubric_snapshots/train_step_1.parquet"
echo "Validation disabled only for this artifact-export job. Existing checkpoint/progress will resume."
exec bash "${ROOT}/examples/agent_rubric2/${LAUNCHER}" "$@" \
  trainer.total_training_steps=300 \
  trainer.save_freq=1 trainer.test_freq=-1 trainer.val_before_train=False \
  trainer.resume_mode=auto \
  "trainer.experiment_name=${NAME}" "trainer.default_local_dir=${OUTPUT}" \
  algorithm.token_credit.rubric_pipeline.backend=actor_rollout_group_hints \
  algorithm.token_credit.rubric_pipeline.pool_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=False \
  algorithm.token_credit.rubric_pipeline.group_hint_candidate_count=1 \
  algorithm.token_credit.rubric_pipeline.group_hint_select_by_correlation=False \
  algorithm.token_credit.rubric_pipeline.snapshot_enable=True \
  algorithm.token_credit.rubric_pipeline.snapshot_step=1 \
  ++algorithm.token_credit.rubric_pipeline.snapshot_stop_after_export=True \
  algorithm.token_credit.rubric_pipeline.snapshot_batch_size=8 \
  algorithm.token_credit.rubric_pipeline.snapshot_max_generation_attempts=4 \
  "algorithm.token_credit.rubric_pipeline.snapshot_output_path=${OUTPUT}/rubric_snapshots/train_step_1.parquet"
