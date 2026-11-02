#!/usr/bin/env bash
set -euo pipefail

# Experiment 1: train the rubric generator conservatively, require stronger
# outcome discrimination, and credit only criteria used by the teacher replay.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export _RUBRIC_4GPU_OPD001_GEN0003_SPEARMAN04_SELECTED_PARTIAL_JUDGE=1
export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=False

export RUBRIC_POOL_JUDGE_BASE_URL="${RUBRIC_POOL_JUDGE_BASE_URL:-https://api.example.com/v1}"
export RUBRIC_POOL_JUDGE_CONCURRENCY="${RUBRIC_POOL_JUDGE_CONCURRENCY:-4}"
export RUBRIC_POOL_JUDGE_RETRIES="${RUBRIC_POOL_JUDGE_RETRIES:-4}"
export RUBRIC_POOL_MIN_JUDGE_COVERAGE="${RUBRIC_POOL_MIN_JUDGE_COVERAGE:-0.75}"

echo "Experiment 1 restart: OPD=0.01; gen beta=0.003; Spearman>0.4; selected-Spearman reward; partial judge"
exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu.sh" "$@"
