#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Corrected coefficient experiment: ORSD, 4 bundles with judging; forwarding to run_webshop_small_qwen3_1_7b_orsd_opd0p001_4gpu.sh"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_qwen3_1_7b_orsd_opd0p001_4gpu.sh" "$@"
