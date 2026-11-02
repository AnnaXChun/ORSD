#!/usr/bin/env bash
set -euo pipefail

# Fair control for the online group-hint experiment.  All rollout, context,
# thinking, batch, optimizer, validation, and episode-limit settings are
# inherited from run_webshop_group_hint_grpo.sh; the explicit mode switch makes
# that shared entry point disable only rubric token credit.
export _WEBSHOP_MATCHED_ENTRY=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_group_hint_grpo.sh" "$@"
