#!/usr/bin/env bash
set -euo pipefail

# Matched control: it keeps the reference forward pass and every training
# parameter identical, but alpha=0 makes the additive adjustment exactly zero.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1_7b_additive_gap_alpha0_control}"
export TOKEN_CREDIT_ALPHA="0.0"
exec bash "${SCRIPT_DIR}/run_qwen3_1_7b_math.sh" "$@"
