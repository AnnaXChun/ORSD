#!/usr/bin/env bash
set -euo pipefail

# Experiment 2: isolate trajectory conditioning. Match the best pool-only run,
# but generate each rubric bundle from the current task without its trajectory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_POOL_ONLY_OPD001_TASK_ONLY_GENERATOR_PARTIAL_JUDGE=1

export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"
export RUBRIC_POOL_JUDGE_CONCURRENCY="${RUBRIC_POOL_JUDGE_CONCURRENCY:-4}"
export RUBRIC_POOL_JUDGE_RETRIES="${RUBRIC_POOL_JUDGE_RETRIES:-4}"
export RUBRIC_POOL_MIN_JUDGE_COVERAGE="${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-0.75}"

echo "Experiment 2 restart: OPD=0.01; no genloss; Spearman>0.4; task-only generator; partial judge"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
