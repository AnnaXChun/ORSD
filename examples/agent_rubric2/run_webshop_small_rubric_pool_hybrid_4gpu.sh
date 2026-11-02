#!/usr/bin/env bash
set -euo pipefail

# Four-A100-80GB execution profile for WebShop-small Rubric Pool runs.
# Dedicated wrappers select the algorithm and coefficients below.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Respect an explicit four-device selection; otherwise use the first four GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _RUBRIC_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_RUBRIC_VISIBLE_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

# Qwen3-1.7B fits on one 80GB GPU. TP=1 therefore keeps four independent
# rollout replicas on four GPUs, matching the replica count of the old 8-GPU
# TP=2 setup while avoiding unnecessary tensor-parallel communication.
export N_GPUS=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY_UTILIZATION="${FOUR_GPU_VLLM_GPU_UTIL:-0.72}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${FOUR_GPU_VLLM_BATCHED_TOKENS:-32768}"
export ROLLOUT_MAX_NUM_SEQS="${FOUR_GPU_VLLM_MAX_NUM_SEQS:-64}"
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True

# Preserve the experimental protocol and long-trajectory memory safety.
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MICRO_BATCH_SIZE=1
export TURN_REPLAY_MICRO_BATCH_SIZE=1
export RUBRIC_GENERATOR_MICRO_BATCH_SIZE=1
export TOKEN_CREDIT_MODE=additive_gap
export TOKEN_CREDIT_ALPHA=0.1
if [[ "${_RUBRIC_4GPU_BOTH_01:-0}" == "1" ]]; then
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.1
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD09:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.9
  export RUBRIC_GENERATOR_LOSS_ENABLE=False
  export RUBRIC_GENERATOR_LOSS_COEF=0.0
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=False
  export RUBRIC_GENERATOR_LOSS_COEF=0.0
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001_BEHAVIOR_DEDUP:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=False
  export RUBRIC_GENERATOR_LOSS_COEF=0.0
  export RUBRIC_POOL_BEHAVIOR_DEDUP_ENABLE=True
  export RUBRIC_POOL_MAX_BEHAVIOR_SIMILARITY=0.9
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN001:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.01
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.003
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0001:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.001
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_SIMPLE_PROMPT:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.003
  export RUBRIC_POOL_GENERATOR_PROMPT_STYLE=simple_numbered
  export RUBRIC_POOL_MIN_RUBRICS_PER_BUNDLE=5
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_DECAY_AFTER150:-0}" == "1" ]]; then
  # Preserve the best early-training route, then remove both auxiliary
  # contributions linearly over the second half of the 300-step run.
  export TOKEN_CREDIT_ALPHA=0.01
  export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=True
  export TOKEN_CREDIT_LINEAR_DECAY_START_STEP=150
  export TOKEN_CREDIT_LINEAR_DECAY_FINAL_SCALE=0.0
  export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.003
  export RUBRIC_GENERATOR_LINEAR_DECAY_ENABLE=True
  export RUBRIC_GENERATOR_LINEAR_DECAY_START_STEP=150
  export RUBRIC_GENERATOR_LINEAR_DECAY_FINAL_SCALE=0.0
  export RUBRIC_GENERATOR_LINEAR_WARMUP_ENABLE=False
  echo "OPD and generator-loss schedules: constant through step 150, then linear decay to 0 at step 300"
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_PARTIAL_JUDGE:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.003
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_SPEARMAN04_SELECTED:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.003
  export RUBRIC_POOL_MIN_CORRELATION=0.4
  export RUBRIC_POOL_GENERATOR_REWARD_SOURCE=selected_spearman_sum
  export RUBRIC_POOL_GENERATOR_INCLUDE_TRAJECTORY=True
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_SPEARMAN04_SELECTED_PARTIAL_JUDGE:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.003
  export RUBRIC_POOL_MIN_CORRELATION=0.4
  export RUBRIC_POOL_GENERATOR_REWARD_SOURCE=selected_spearman_sum
  export RUBRIC_POOL_GENERATOR_INCLUDE_TRAJECTORY=True
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_SPEARMAN04_ITEM_CREDIT:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF=0.003
  export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=True
  export RUBRIC_POOL_MIN_CORRELATION=0.4
  export RUBRIC_POOL_GENERATOR_REWARD_SOURCE=selected_spearman_sum
  export RUBRIC_POOL_GENERATOR_INCLUDE_TRAJECTORY=True
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001_TASK_ONLY_GENERATOR:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=False
  export RUBRIC_GENERATOR_LOSS_COEF=0.0
  export RUBRIC_POOL_MIN_CORRELATION=0.4
  export RUBRIC_POOL_GENERATOR_REWARD_SOURCE=valid
  export RUBRIC_POOL_GENERATOR_INCLUDE_TRAJECTORY=False
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001_TASK_ONLY_GENERATOR_PARTIAL_JUDGE:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=False
  export RUBRIC_GENERATOR_LOSS_COEF=0.0
  export RUBRIC_POOL_MIN_CORRELATION=0.4
  export RUBRIC_POOL_GENERATOR_REWARD_SOURCE=valid
  export RUBRIC_POOL_GENERATOR_INCLUDE_TRAJECTORY=False
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001_PPOMINI64:-0}" == "1" ]]; then
  export TOKEN_CREDIT_ALPHA=0.01
  export RUBRIC_GENERATOR_LOSS_ENABLE=False
  export RUBRIC_GENERATOR_LOSS_COEF=0.0
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_POLARITY01:-0}" == "1" ]]; then
  export TOKEN_CREDIT_MODE=polarity_gap
  export TOKEN_CREDIT_ALPHA=0.1
  export TOKEN_CREDIT_SCALE_QUANTILE=0.5
  export TOKEN_CREDIT_MIN_SCALE=0.0001
  export TOKEN_CREDIT_PRESERVE_TRAJECTORY_MEAN=True
  export RUBRIC_GENERATOR_LOSS_ENABLE=False
  export RUBRIC_GENERATOR_LOSS_COEF=0.0
fi
if [[ "${_RUBRIC_4GPU_CENTERED_OPD_LOO:-0}" == "1" ]]; then
  export PROJECT_NAME=verl_rubric_3
  export TOKEN_CREDIT_MODE="${RUBRIC_SCHEDULED_OPD_MODE:-centered_opd}"
  if [[ "${TOKEN_CREDIT_MODE}" == "additive_gap" ]]; then
    export TOKEN_CREDIT_ALPHA="${RUBRIC_SCHEDULED_OPD_ALPHA:-0.01}"
  elif [[ "${TOKEN_CREDIT_MODE}" == "rank_attenuation" ]]; then
    export TOKEN_CREDIT_RANK_ATTENUATION="${RUBRIC_SCHEDULED_RANK_ATTENUATION:-0.2}"
  else
    export TOKEN_CREDIT_ALPHA="${RUBRIC_SCHEDULED_OPD_ALPHA:-0.1}"
  fi
  export TOKEN_CREDIT_CENTERED_CLIP=2.0
  export TOKEN_CREDIT_MULTIPLIER_MIN=0.5
  export TOKEN_CREDIT_MULTIPLIER_MAX=1.5
  export RUBRIC_GENERATOR_LOSS_ENABLE=True
  export RUBRIC_GENERATOR_LOSS_COEF="${RUBRIC_SCHEDULED_GENERATOR_LOSS_COEF:-0.003}"
  export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=True
  # Canonical generator settings shared with the ALFWorld centered-OPD route.
  # Explicit assignments prevent stale shell values from changing a rerun.
  export RUBRIC_POOL_GENERATOR_TEMPERATURE=0.7
  export RUBRIC_TOP_P=0.9
  export RUBRIC_GENERATOR_THINKING=False
  export RUBRICS_PER_PROMPT=8
  export RUBRIC_MAX_TRAJECTORIES=4
  export RUBRIC_MAX_TRAJECTORY_CHARS=6000
  export RUBRIC_MAX_RUBRIC_CHARS=1600
  export RUBRIC_MAX_ITEM_CHARS=280
  export RUBRIC_POOL_FULL_TRAJECTORY=True
  export RUBRIC_POOL_GENERATOR_INCLUDE_TRAJECTORY=True
  export RUBRIC_POOL_GENERATOR_PROMPT_STYLE="${RUBRIC_SCHEDULED_GENERATOR_PROMPT_STYLE:-simple_numbered}"
  export RUBRIC_POOL_GENERATOR_MAX_INPUT_TOKENS=0
  export RUBRIC_POOL_CONTEXT_RESERVE_TOKENS=256
  export RUBRIC_GENERATOR_MICRO_BATCH_SIZE=1
  export RUBRIC_GENERATOR_CLIP_RATIO=0.2
  export RUBRIC_POOL_MIN_RUBRICS_PER_BUNDLE=5
  if [[ "${TOKEN_CREDIT_MODE}" == "rank_attenuation" ]]; then
    TOKEN_CREDIT_SCHEDULE_TARGET="${TOKEN_CREDIT_RANK_ATTENUATION}"
  else
    TOKEN_CREDIT_SCHEDULE_TARGET="${TOKEN_CREDIT_ALPHA}"
  fi
  if [[ "${_RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP:-0}" == "1" ]]; then
    export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
    export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=True
    export TOKEN_CREDIT_LINEAR_WARMUP_STEPS=150
    export TOKEN_CREDIT_LINEAR_WARMUP_INITIAL_SCALE=0.0
    export RUBRIC_GENERATOR_LINEAR_DECAY_ENABLE=False
    export RUBRIC_GENERATOR_LINEAR_WARMUP_ENABLE=True
    export RUBRIC_GENERATOR_LINEAR_WARMUP_STEPS=150
    export RUBRIC_GENERATOR_LINEAR_WARMUP_INITIAL_SCALE=0.0
    echo "${TOKEN_CREDIT_MODE} schedule: token-credit strength warms up 0->${TOKEN_CREDIT_SCHEDULE_TARGET} through step 150, then stays fixed; generator-loss scale warms up 0->1"
  else
    export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=True
    export TOKEN_CREDIT_LINEAR_DECAY_FINAL_SCALE=0.0
    export TOKEN_CREDIT_LINEAR_DECAY_START_STEP=150
    export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
    export RUBRIC_GENERATOR_LINEAR_DECAY_ENABLE=True
    export RUBRIC_GENERATOR_LINEAR_DECAY_FINAL_SCALE=0.0
    export RUBRIC_GENERATOR_LINEAR_DECAY_START_STEP=150
    export RUBRIC_GENERATOR_LINEAR_WARMUP_ENABLE=False
    echo "${TOKEN_CREDIT_MODE} schedule: token-credit strength stays at ${TOKEN_CREDIT_SCHEDULE_TARGET} through step 150, then decays to 0; generator-loss scale follows the same schedule"
  fi
  export RUBRIC_POOL_MIN_CORRELATION=0.4
  export RUBRIC_POOL_GENERATOR_REWARD_SOURCE=selected_loo_spearman_relu
  export RUBRIC_POOL_GENERATOR_SUPPORT_THRESHOLD=0.5
  export RUBRIC_POOL_GENERATOR_MIN_SATISFIED=2
  export RUBRIC_POOL_GENERATOR_MIN_UNSATISFIED=2
  export RUBRIC_POOL_GENERATOR_INCLUDE_TRAJECTORY=True
fi

# WebShop-small has only 1,000 products, so four environment workers do not
# duplicate the very large full-catalogue state that caused earlier host OOMs.
export WEBSHOP_AGENT_LOOP_NUM_WORKERS="${FOUR_GPU_AGENT_LOOP_WORKERS:-4}"
export WEBSHOP_DATALOADER_NUM_WORKERS="${FOUR_GPU_DATALOADER_WORKERS:-2}"

# Correct rubric JSON is short. Lower output ceilings remove pathological long
# generations without reducing the 5-8 criterion bundle requested by prompt.
export RUBRIC_MAX_TOKENS="${FOUR_GPU_RUBRIC_MAX_TOKENS:-1024}"
export RUBRIC_POOL_JUDGE_MAX_TOKENS="${FOUR_GPU_JUDGE_MAX_TOKENS:-1536}"
export RUBRIC_POOL_JUDGE_CONCURRENCY="${RUBRIC_POOL_JUDGE_CONCURRENCY:-${FOUR_GPU_JUDGE_CONCURRENCY:-8}}"

# Evaluation and audit I/O do not affect gradients. Run them less often so the
# wall-clock profile reflects training rather than 500-task validation or JSON.
export TEST_FREQ="${FOUR_GPU_TEST_FREQ:-20}"
export RUBRIC_POOL_AUDIT_EVERY_N_STEPS="${FOUR_GPU_AUDIT_EVERY_N_STEPS:-10}"

# Isolate this coefficient experiment from stale names/checkpoint directories
# exported for earlier pool runs. Re-running this entry resumes its own run.
export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_opd0p1
if [[ "${_RUBRIC_4GPU_BOTH_01:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME="${RUBRIC_POOL_EXPERIMENT_NAME}_gen0p1"
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD09:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_no_genloss_opd0p9
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_no_genloss_opd0p01
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001_BEHAVIOR_DEDUP:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_v7_4gpu_no_genloss_opd0p01_behavior_dedup0p9
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN001:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_opd0p01_genloss0p01
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_opd0p01_genloss0p003
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0001:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_opd0p01_genloss0p001
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_SIMPLE_PROMPT:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_simpleprompt_nofewshot_4gpu_opd0p01_genloss0p003
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_DECAY_AFTER150:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_opd0p01_decayafter150_genloss0p003_decayafter150
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_PARTIAL_JUDGE:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_v8_4gpu_opd0p01_genloss0p003_partialjudge_cov0p75
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_SPEARMAN04_SELECTED:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_v6_4gpu_opd0p01_genloss0p003_spearman0p4_selectedspearman
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_SPEARMAN04_SELECTED_PARTIAL_JUDGE:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_v8_4gpu_opd0p01_genloss0p003_spearman0p4_selectedspearman_partialjudge_cov0p75
fi
if [[ "${_RUBRIC_4GPU_OPD001_GEN0003_SPEARMAN04_ITEM_CREDIT:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_v9_4gpu_opd0p01_genloss0p003_spearman0p4_itemcredit_partialjudge_cov0p75
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001_TASK_ONLY_GENERATOR:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_v6_4gpu_no_genloss_opd0p01_task_only_generator_spearman0p4
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001_TASK_ONLY_GENERATOR_PARTIAL_JUDGE:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_v8_4gpu_no_genloss_opd0p01_task_only_generator_spearman0p4_partialjudge_cov0p75
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_OPD001_PPOMINI64:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_no_genloss_opd0p01_effective_ppomini64
fi
if [[ "${_RUBRIC_4GPU_POOL_ONLY_POLARITY01:-0}" == "1" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME=qwen3_1_7b_webshop_small_pool_validfrac_anticopy_v5_4gpu_no_genloss_polarity0p1
fi
if [[ "${_RUBRIC_4GPU_CENTERED_OPD_LOO:-0}" == "1" ]]; then
  _RUBRIC_MODEL_TAG="${MODEL_EXPERIMENT_TAG:-qwen3_1_7b}"
  _RUBRIC_GENLOSS_TAG="${RUBRIC_GENERATOR_LOSS_COEF//./p}"
  if [[ "${TOKEN_CREDIT_MODE}" == "rank_attenuation" && "${_RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP:-0}" == "1" ]]; then
    export RUBRIC_POOL_EXPERIMENT_NAME="${_RUBRIC_MODEL_TAG}_webshop_small_pool_4gpu_rankatten0p2_q0p25_warmup150_gen${_RUBRIC_GENLOSS_TAG}_warmup150_loo_spearman0p4_support2x2"
  elif [[ "${TOKEN_CREDIT_MODE}" == "rank_attenuation" ]]; then
    export RUBRIC_POOL_EXPERIMENT_NAME="${_RUBRIC_MODEL_TAG}_webshop_small_pool_4gpu_rankatten0p2_q0p25_decayafter150_gen${_RUBRIC_GENLOSS_TAG}_decayafter150_loo_spearman0p4_support2x2"
  elif [[ "${TOKEN_CREDIT_MODE}" == "additive_gap" && "${_RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP:-0}" == "1" ]]; then
    export RUBRIC_POOL_EXPERIMENT_NAME="${_RUBRIC_MODEL_TAG}_webshop_small_pool_4gpu_additiveopd0p01_warmup150_gen${_RUBRIC_GENLOSS_TAG}_warmup150_loo_spearman0p4_support2x2"
  elif [[ "${TOKEN_CREDIT_MODE}" == "additive_gap" ]]; then
    export RUBRIC_POOL_EXPERIMENT_NAME="${_RUBRIC_MODEL_TAG}_webshop_small_pool_4gpu_additiveopd0p01_decayafter150_gen${_RUBRIC_GENLOSS_TAG}_decayafter150_loo_spearman0p4_support2x2"
  elif [[ "${_RUBRIC_4GPU_CENTERED_OPD_LOO_WARMUP:-0}" == "1" ]]; then
    export RUBRIC_POOL_EXPERIMENT_NAME="${_RUBRIC_MODEL_TAG}_webshop_small_pool_4gpu_centeredopd0p1_warmup150_gen${_RUBRIC_GENLOSS_TAG}_warmup150_loo_spearman0p4_support2x2"
  else
    export RUBRIC_POOL_EXPERIMENT_NAME="${_RUBRIC_MODEL_TAG}_webshop_small_pool_4gpu_centeredopd0p1_decayafter150_gen${_RUBRIC_GENLOSS_TAG}_decayafter150_loo_spearman0p4_support2x2"
  fi
fi
if [[ -n "${RUBRIC_POOL_EXPERIMENT_NAME_OVERRIDE:-}" ]]; then
  export RUBRIC_POOL_EXPERIMENT_NAME="${RUBRIC_POOL_EXPERIMENT_NAME_OVERRIDE}"
fi
export RUBRIC_POOL_OUTPUT_DIR="${RUBRIC_POOL_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME:-verl_agent_rubric2}/${RUBRIC_POOL_EXPERIMENT_NAME}}"

if [[ "${TOKEN_CREDIT_MODE}" == "rank_attenuation" ]]; then
  TOKEN_CREDIT_SUMMARY="attenuation=${TOKEN_CREDIT_RANK_ATTENUATION:-0.2}, conflict_fraction=${TOKEN_CREDIT_RANK_CONFLICT_FRACTION:-0.25}, min_gap_range=${TOKEN_CREDIT_RANK_MIN_GAP_RANGE:-0.0001}"
else
  TOKEN_CREDIT_SUMMARY="alpha=${TOKEN_CREDIT_ALPHA}"
fi
echo "Experiment: ${RUBRIC_POOL_EXPERIMENT_NAME}; OPD mode=${TOKEN_CREDIT_MODE}, ${TOKEN_CREDIT_SUMMARY}; generator loss=${RUBRIC_GENERATOR_LOSS_ENABLE:-True}, coef=${RUBRIC_GENERATOR_LOSS_COEF:-0.01}; PPO prompt mini-batch=${TRAIN_BATCH_SIZE}, effective rollout mini-batch=$((TRAIN_BATCH_SIZE * ROLLOUT_N)); checkpoints=${RUBRIC_POOL_OUTPUT_DIR}"
echo "Four-GPU profile: devices=${CUDA_VISIBLE_DEVICES}, TP=${ROLLOUT_TP}, vLLM util=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "vLLM batching: max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}, max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
echo "Auxiliary generation: rubric_max_tokens=${RUBRIC_MAX_TOKENS}, judge_max_tokens=${RUBRIC_POOL_JUDGE_MAX_TOKENS}, rubric_temperature=${RUBRIC_POOL_GENERATOR_TEMPERATURE:-${RUBRIC_TEMPERATURE}}, generator_thinking=${RUBRIC_GENERATOR_THINKING}"
echo "Throughput controls: agent_workers=${WEBSHOP_AGENT_LOOP_NUM_WORKERS}, test_freq=${TEST_FREQ}, audit_every=${RUBRIC_POOL_AUDIT_EVERY_N_STEPS}"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_rubric_pool_hybrid.sh" "$@"
