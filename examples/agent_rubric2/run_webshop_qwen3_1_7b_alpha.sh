#!/usr/bin/env bash
set -euo pipefail

# Keep the selected online control unchanged; override only additive OPD alpha
# and the run identity. No training starts in --dry-run mode.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
alpha="${1:?Usage: $0 0.001|0.1|1 --baseline pool|bundle [--dry-run] [Hydra overrides]}"
shift
case "${alpha}" in
  0.001|0.1|1) ;;
  *) echo "Expected alpha=0.001, 0.1, or 1; got ${alpha}" >&2; exit 2 ;;
esac

baseline="${WEBSHOP_ALPHA_BASELINE:-}"
dry_run=0
while (( $# )); do
  case "$1" in
    --baseline)
      baseline="${2:?--baseline requires pool or bundle}"
      shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
case "${baseline}" in
  pool)
    base_script="${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid_4gpu_no_generator_loss_opd0p01.sh"
    # The control prints these before its child launcher supplies the defaults.
    export RUBRIC_TEMPERATURE="${RUBRIC_TEMPERATURE:-0.7}"
    export RUBRIC_GENERATOR_THINKING="${RUBRIC_GENERATOR_THINKING:-False}" ;;
  bundle)
    base_script="${ROOT}/examples/agent_rubric2/run_webshop_small_first_version_4gpu.sh" ;;
  *)
    echo "Choose the main-experiment control: --baseline pool or --baseline bundle." >&2
    echo "pool = no_generator_loss_opd0p01; bundle = small_first_version_4gpu." >&2
    exit 2 ;;
esac

# Prevent selectors exported for another pool experiment from changing this run.
for variable in $(compgen -v _RUBRIC_4GPU_); do
  unset "${variable}"
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#visible_gpus[@]} != 4 )); then
  echo "Expected four visible GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export MODEL_PATH="${QWEN3_1_7B_MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7B}"
if [[ -z "${QWEN3_1_7B_MODEL_PATH:-}" && ! -f "${MODEL_PATH}/config.json" ]]; then
  for candidate in /workspace/checkpoints/qwen/Qwen3-1.7B "${ROOT}/checkpoints/Qwen3-1.7B"; do
    if [[ -f "${candidate}/config.json" ]]; then
      export MODEL_PATH="${candidate}"
      break
    fi
  done
fi

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
alpha_tag="${alpha//./p}"
run_name="qwen3_1_7b_webshop_small_${baseline}_alpha${alpha_tag}_4gpu_b8_n8_s300"
if [[ -n "${WEBSHOP_ALPHA_RUN_SUFFIX:-}" ]]; then
  run_name="${run_name}_${WEBSHOP_ALPHA_RUN_SUFFIX}"
fi
case "${run_name}" in
  *[!a-zA-Z0-9_.-]*) echo "Run suffix must contain only letters, digits, _, . or -." >&2; exit 2 ;;
esac
output_root="${WEBSHOP_ALPHA_OUTPUT_ROOT:-${ROOT}/checkpoints/${PROJECT_NAME}}"
export OUTPUT_DIR="${output_root}/${run_name}"
export RUBRIC_POOL_OUTPUT_DIR="${OUTPUT_DIR}"
export RUBRIC_POOL_EXPERIMENT_NAME_OVERRIDE="${run_name}"
export WEBSHOP_FIRST_EXPERIMENT_NAME="${run_name}"
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_LOSS_COEF=0.0
export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=False

# Put locked overrides last: shared launchers set alpha=0.01 in their environment.
# Hydra therefore receives the requested coefficient even when caller overrides
# or inherited shell variables contain an old alpha/model/output directory.
fixed_overrides=(
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  algorithm.token_credit.enable=True
  algorithm.token_credit.mode=additive_gap
  "algorithm.token_credit.alpha=${alpha}"
  algorithm.token_credit.linear_decay_enable=False
  algorithm.token_credit.linear_warmup_enable=False
  algorithm.token_credit.rubric_pipeline.pool_generator_loss_enable=False
  algorithm.token_credit.rubric_pipeline.pool_generator_item_credit_enable=False
  "trainer.experiment_name=${run_name}"
  "trainer.default_local_dir=${OUTPUT_DIR}"
)

echo "Alpha ablation: Qwen3-1.7B / WebShop-small / ${baseline} / alpha=${alpha}"
echo "Control: ${base_script}"
echo "Run: ${run_name}; checkpoints: ${OUTPUT_DIR}"
echo "Base-script messages retain alpha=0.01; the final Hydra override below is authoritative."
printf 'Command: bash %q' "${base_script}"
printf ' %q' "$@" "${fixed_overrides[@]}"
printf '\n'
if (( dry_run )); then
  echo "Preview only; model, datasets and API credentials have not been validated."
  exit 0
fi

python3 - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    config = json.load(source)
actual = (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers"))
if actual != ("qwen3", 2048, 28):
    raise SystemExit(f"Expected Qwen3-1.7B; got {actual} from {sys.argv[1]}")
PY

exec bash "${base_script}" "$@" "${fixed_overrides[@]}"
