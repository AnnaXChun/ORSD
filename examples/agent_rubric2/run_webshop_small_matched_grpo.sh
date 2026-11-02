#!/usr/bin/env bash
set -euo pipefail

# Exact matched-GRPO control for run_webshop_small_group_hint_grpo.sh. Only the
# rubric pipeline and token-credit adjustment are disabled by the shared entry.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export _WEBSHOP_MATCHED_ENTRY=1
export WEBSHOP_MATCHED_EXPERIMENT_NAME="${WEBSHOP_MATCHED_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_matched_grpo}"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_group_hint_grpo.sh" "$@"
