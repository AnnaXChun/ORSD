#!/usr/bin/env bash
set -euo pipefail

# Milestone 02 ablation: keep trajectory rubric generation, cloud judging,
# pool selection, teacher OPD and every task-training setting unchanged, while
# disabling only the auxiliary PPO loss on generated rubric response tokens.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUBRIC_POOL_ENABLE=True
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_LOSS_COEF=0.0
export RUBRIC_POOL_EXPERIMENT_NAME="${RUBRIC_POOL_FILTER_ONLY_EXPERIMENT_NAME:-qwen3_1_7b_webshop_small_pool_filter_only_hybrid_v4}"

# Do not inherit a checkpoint directory from the joint-loss experiment. A
# caller can still provide an explicit filter-only directory through the
# dedicated variable below.
if [[ -n "${RUBRIC_POOL_FILTER_ONLY_OUTPUT_DIR:-}" ]]; then
  export RUBRIC_POOL_OUTPUT_DIR="${RUBRIC_POOL_FILTER_ONLY_OUTPUT_DIR}"
else
  unset RUBRIC_POOL_OUTPUT_DIR || true
fi

echo "Rubric Pool filter-only ablation: generator auxiliary loss disabled."
echo "Rubric generation, cloud judge, pool selection, teacher OPD and Hybrid reward remain enabled."
echo "Rubric sampling top_p=1.0 is fixed to match the archived joint-loss run."

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid.sh" \
  algorithm.token_credit.rubric_pipeline.top_p=1.0 \
  "$@"
