#!/usr/bin/env bash
set -euo pipefail

# WebShop-small protocol used by public agent-RL baselines:
#   catalog: 1000 products
#   goals: synthetic (human_goals=False)
#   eval: goals 0--499
#   train: goals 500--end
# All optimizer, rollout, reward, context, and rubric settings remain inherited
# from run_webshop_group_hint_grpo.sh so catalog size is the only forced change.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WEBSHOP_ROOT="${WEBSHOP_ROOT:-${ROOT}/WebShop}"
WEBSHOP_SMALL_DATA_DIR="${WEBSHOP_SMALL_DATA_DIR:-${ROOT}/Data/agent_rubric/webshop_small}"

export WEBSHOP_USE_SMALL=True
export WEBSHOP_HUMAN_GOALS=False
export WEBSHOP_NUM_PRODUCTS=1000
export WEBSHOP_PRODUCTS_JSON="${WEBSHOP_ROOT}/data/items_shuffle_1000.json"
export WEBSHOP_TRAIN_PARQUET="${WEBSHOP_TRAIN_PARQUET:-${WEBSHOP_SMALL_DATA_DIR}/webshop_train.parquet}"
export WEBSHOP_VAL_PARQUET="${WEBSHOP_VAL_PARQUET:-${WEBSHOP_SMALL_DATA_DIR}/webshop_test.parquet}"
export WEBSHOP_GROUP_HINT_EXPERIMENT_NAME="${WEBSHOP_GROUP_HINT_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_group_hint_grpo}"

required_paths=(
  "${WEBSHOP_ROOT}/data/items_shuffle_1000.json"
  "${WEBSHOP_ROOT}/data/items_ins_v2_1000.json"
  "${WEBSHOP_ROOT}/data/items_human_ins.json"
  "${WEBSHOP_ROOT}/search_engine/indexes_1k"
  "${WEBSHOP_TRAIN_PARQUET}"
  "${WEBSHOP_VAL_PARQUET}"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing WebShop-small asset: ${required_path}" >&2
    echo "Prepare it with:" >&2
    echo "  REBUILD_WEBSHOP_SMALL_INDEX=1 bash examples/agent_rubric2/setup_webshop_small.sh" >&2
    echo "  WEBSHOP_CATALOG_MODE=small python3 examples/agent_rubric2/prepare_webshop_data.py" >&2
    exit 1
  fi
done

python3 - "${WEBSHOP_TRAIN_PARQUET}" "${WEBSHOP_VAL_PARQUET}" <<'PY'
import json
import sys

import pyarrow.parquet as pq

for path in sys.argv[1:]:
    table = pq.read_table(path, columns=["extra_info"])
    if table.num_rows == 0:
        raise SystemExit(f"Empty WebShop-small parquet: {path}")
    extra_info = table.slice(0, 1).to_pylist()[0]["extra_info"] or {}
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    interaction = extra_info.get("interaction_kwargs") or {}
    expected = {
        "name": "webshop",
        "use_small": True,
        "human_goals": False,
        "num_products": 1000,
    }
    mismatches = {
        key: (interaction.get(key), value)
        for key, value in expected.items()
        if interaction.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            f"Parquet is not WebShop-small: {path}; mismatches={mismatches}. "
            "Regenerate with WEBSHOP_CATALOG_MODE=small."
        )
print("WebShop-small parquet preflight passed")
PY

echo "WebShop-small rubric GRPO: train=${WEBSHOP_TRAIN_PARQUET}, eval=${WEBSHOP_VAL_PARQUET}, index=indexes_1k"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_group_hint_grpo.sh" "$@"
