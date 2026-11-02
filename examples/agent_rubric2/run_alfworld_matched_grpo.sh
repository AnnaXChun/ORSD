#!/usr/bin/env bash
set -euo pipefail

# Matched control for ALFWorld v2. All environment, reward, rollout, context,
# thinking, optimizer, and validation settings come from the group-hint entry;
# only rubric generation and token credit are disabled.
export _ALFWORLD_MATCHED_ENTRY=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/examples/agent_rubric2/run_alfworld_group_hint_grpo.sh" "$@"
