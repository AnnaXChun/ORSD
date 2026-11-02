#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OFFLINE_RUBRIC_PARQUET:?Set OFFLINE_RUBRIC_PARQUET to the matching model and environment complete rubric snapshot parquet}"
OFFLINE_RUBRIC_EXPECTED_STEP="${OFFLINE_RUBRIC_EXPECTED_STEP:-1}"
python3 "${ROOT}/examples/agent_rubric2/check_offline_rubric_snapshot.py" \
  "${TRAIN_PARQUET:-${ROOT}/Data/agent_rubric/alfworld/alfworld_train.parquet}" "${OFFLINE_RUBRIC_PARQUET}" --expected-step "${OFFLINE_RUBRIC_EXPECTED_STEP}"
export MODEL_PATH="${QWEN3_4B_MODEL_PATH:-/workspace/checkpoints/qwen/Qwen3-4B}"
python3 - "${MODEL_PATH}/config.json" <<'PYMODEL'
import json, sys
with open(sys.argv[1]) as source:
    config = json.load(source)
assert (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers")) == ("qwen3", 2560, 36), "Wrong actor model config"
PYMODEL

export ALFWORLD_OPD_ONLY_EXPERIMENT_NAME="${ALFWORLD_QWEN3_4B_OFFLINE_EXPERIMENT_NAME:-qwen3_4b_alfworld_offline_snapshot_step${OFFLINE_RUBRIC_EXPECTED_STEP}_additiveopd0p01_b8_n8_4gpu_s300}"
export ALFWORLD_OPD_ONLY_OUTPUT_DIR="${ALFWORLD_QWEN3_4B_OFFLINE_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME:-verl_rubric_3}/${ALFWORLD_OPD_ONLY_EXPERIMENT_NAME}}"
export RUBRIC_SNAPSHOT_ENABLE=False
# Generic overrides were shared across four experiments; do not reuse them.
if [[ -n "${OFFLINE_RUBRIC_EXPERIMENT_NAME:-}" || -n "${OFFLINE_RUBRIC_OUTPUT_DIR:-}" ]]; then
  echo "Ignoring shared OFFLINE_RUBRIC_EXPERIMENT_NAME/OUTPUT_DIR; use the model/environment-specific overrides."
fi
export RUBRIC_GROUP_CANDIDATES=1
export RUBRIC_GROUP_SELECT_BY_CORRELATION=False
export RUBRIC_GROUP_FORCE_FIRST_CANDIDATE=False
export RUBRIC_GROUP_GUARDED_SELECTION=False
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED=False
export RUBRIC_GROUP_MIN_SELECTED_CORRELATION=0.0
export RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT=0.0

# Reuse the online launcher's training/validation protocol. Final Hydra overrides
# replace only the rubric source and disable the final export. No OPSD loss.
# Caller overrides come BEFORE the fixed-ablation overrides to prevent leakage.
echo "OFFLINE ABLATION: environment rollout remains online; GRPO + additive OPD=0.01"
echo "Fixed rubric snapshot: ${OFFLINE_RUBRIC_PARQUET}; no generator, judge, pool or generator loss"
echo "Base-launcher generation messages below describe inherited defaults; final Hydra backend is dataset"
exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_opd_only_no_pool_4gpu_b8_n8_s300.sh" "$@" \
  "data.train_files=['${OFFLINE_RUBRIC_PARQUET}']" \
  algorithm.token_credit.mode=additive_gap \
  algorithm.token_credit.alpha=0.01 \
  algorithm.token_credit.rubric_pipeline.backend=dataset \
  ++algorithm.token_credit.rubric_pipeline.dataset_match_online_gates=True \
  algorithm.token_credit.rubric_pipeline.snapshot_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_enable=False \
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=False \
  "trainer.experiment_name=${ALFWORLD_OPD_ONLY_EXPERIMENT_NAME}" \
  "trainer.default_local_dir=${ALFWORLD_OPD_ONLY_OUTPUT_DIR}"
