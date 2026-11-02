#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point.  The guarded comparison belongs to the current
# 4-GPU, 300-step first-version protocol; keep this old filename from silently
# launching the unrelated 8-GPU, 1000-step historical reproduction.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "This legacy filename now forwards to the matched 4-GPU, 300-step guarded launcher." >&2

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_guarded_bestof4_4gpu_b8_n8_s300.sh" "$@"
