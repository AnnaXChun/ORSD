#!/usr/bin/env bash
set -euo pipefail

# Public-repository-matched WebShop split:
#   train = goals 500--end
#   eval  = goals 0--499
#
# All optimization, rollout, and rubric settings are inherited from the
# history-4 configuration. This reward ablation uses the success-aware hybrid
# reward and train_batch_size=8.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEBSHOP3_DATA_DIR="${WEBSHOP3_DATA_DIR:-${ROOT}/Data/agent_rubric/webshop3}"

export WEBSHOP_TRAIN_PARQUET="${WEBSHOP_TRAIN_PARQUET:-${WEBSHOP3_DATA_DIR}/webshop_train.parquet}"
export WEBSHOP_VAL_PARQUET="${WEBSHOP_VAL_PARQUET:-${WEBSHOP3_DATA_DIR}/webshop_test.parquet}"
export WEBSHOP_GROUP_HINT_EXPERIMENT_NAME="${WEBSHOP_GROUP_HINT_EXPERIMENT_NAME:-qwen3_1_7b_webshop3_best_history4_hybrid_bs8_rubric_grpo}"

for dataset_path in "${WEBSHOP_TRAIN_PARQUET}" "${WEBSHOP_VAL_PARQUET}"; do
  if [[ ! -f "${dataset_path}" ]]; then
    echo "WebShop3 dataset does not exist: ${dataset_path}" >&2
    echo "Generate it first with:" >&2
    echo "  WEBSHOP_ROOT=${WEBSHOP_ROOT:-${ROOT}/WebShop} python3 examples/agent_rubric2/prepare_webshop_data.py" >&2
    exit 1
  fi
done

echo "WebShop3 data: train=${WEBSHOP_TRAIN_PARQUET}, eval=${WEBSHOP_VAL_PARQUET}"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_best_history4_hybrid.sh" "$@"
