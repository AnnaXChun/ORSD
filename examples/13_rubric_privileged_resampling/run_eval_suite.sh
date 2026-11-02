#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a datasets <<<"${DATASETS:-aime24 aime25 math500 amc23}"

for dataset in "${datasets[@]}"; do
    echo
    echo "=============================================================================="
    echo "Experiment 13 dataset: ${dataset}"
    echo "=============================================================================="
    DATASET="${dataset}" bash "${SCRIPT_DIR}/run_dataset.sh" "$@"
done
