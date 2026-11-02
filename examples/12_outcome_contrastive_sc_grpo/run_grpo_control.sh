#!/usr/bin/env bash
set -euo pipefail

# Exact matched control for Experiment 12. Only token credit is disabled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-12_matched_grpo_control}"
exec bash "${SCRIPT_DIR}/run.sh" algorithm.token_credit.enable=False "$@"
