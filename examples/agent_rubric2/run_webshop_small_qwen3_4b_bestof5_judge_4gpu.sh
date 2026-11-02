#!/usr/bin/env bash
set -euo pipefail

# Five-bundle judge selection, matched to the Qwen3-4B single-bundle run. All experiment settings are declared here;
# only environment/data preflight and the common trainer command are reused.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_PATH="${QWEN3_4B_MODEL_PATH:-/workspace/checkpoints/qwen/Qwen3-4B}"
python3 - "${MODEL_PATH}/config.json" <<'PYMODEL'
import json, sys
with open(sys.argv[1], encoding="utf-8") as source:
    config = json.load(source)
actual = (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers"))
if actual != ("qwen3", 2560, 36):
    raise SystemExit(f"Expected Qwen3-4B; got {actual} from {sys.argv[1]}")
PYMODEL

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _WEBSHOP_FIRST_VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#_WEBSHOP_FIRST_VISIBLE_GPUS[@]} != 4 )); then
  echo "This launcher requires exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export N_GPUS=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY_UTILIZATION="${WEBSHOP_FIRST_VLLM_GPU_UTIL:-0.72}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export ROLLOUT_MAX_NUM_SEQS=64
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True

# Matched training protocol.
export _WEBSHOP_MATCHED_ENTRY=0
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export MAX_STEPS=300
export TEST_FREQ=50
export SAVE_FREQ=100
export LEARNING_RATE=1e-6
export KL_COEF=0.001
export VAL_N=4
export VAL_TEMPERATURE=0.6
export MAX_ASSISTANT_TURNS=15
export ACTOR_MAX_TOKENS_PER_TURN=2048
export USE_REWARD_LOOP=False
export CALCULATE_ROLLOUT_LOG_PROBS=True
export MICRO_BATCH_SIZE=1
export TURN_REPLAY_MICRO_BATCH_SIZE=1

# Lock the historical Qwen3-1.7B context budget.  Group-hint generation uses
# MAX_MODEL_LENGTH - RUBRIC_MAX_TOKENS for its fitted input prompt, matching
# the strongest earlier WebShop run: 40960 - 4096 = 36864 input tokens.
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=32768
export MAX_MODEL_LENGTH=40960

# Disable bounded turn replay and use the historical full-history actor path.
export HISTORY_WINDOW=0
export HISTORY_RESPONSE_MODE=action_only

# WebShop-specific environment/reward settings. A single agent worker avoids
# loading the full search index into several host processes.
export WEBSHOP_REWARD_MODE=hybrid
export WEBSHOP_R2_INVALID_ACTION_COEF=0.05
export WEBSHOP_R2_REPEAT_ACTION_COEF=0.025
export RUBRIC_OUTCOME_SCORE_SOURCE=training_reward
export RUBRIC_CORRECTNESS_THRESHOLD=0.6
export WEBSHOP_AGENT_LOOP_NUM_WORKERS=1
export WEBSHOP_DATALOADER_NUM_WORKERS=1

# Aggregate first-version rubric route.
export TOKEN_CREDIT_ENABLE=True
export TOKEN_CREDIT_MODE=additive_gap
export TOKEN_CREDIT_ALPHA=0.01
export TOKEN_CREDIT_LINEAR_DECAY_ENABLE=False
export TOKEN_CREDIT_LINEAR_WARMUP_ENABLE=False
export RUBRIC_PIPELINE_ENABLE=True
export RUBRIC_BACKEND=actor_rollout_group_hints
export RUBRICS_PER_PROMPT=8
export MIN_RUBRICS_PER_PROMPT=5
export RUBRIC_MAX_TRAJECTORIES=4
export RUBRIC_GROUP_CANDIDATES=5
export RUBRIC_GROUP_SELECT_BY_CORRELATION=True
export RUBRIC_GROUP_FALLBACK_ON_UNRANKED=True
export RUBRIC_GROUP_FORCE_FIRST_CANDIDATE=False
export RUBRIC_GROUP_GUARDED_SELECTION=False
export RUBRIC_GROUP_MIN_SELECTED_CORRELATION=0.0
export RUBRIC_GROUP_MIN_CORRELATION_IMPROVEMENT=0.0
export RUBRIC_GROUP_EVIDENCE_ORDER=outcome_anchored_random
export RUBRIC_GROUP_PROMPT_STYLE=process_rubric_5_to_k
export REQUIRE_GROUP_OUTCOME_CONTRAST=True
export REQUIRE_OUTCOME_ALIGNED_GAP=True
export MIN_OUTCOME_GAP_ALIGNMENT=0.0

export RUBRIC_MAX_TRAJECTORY_CHARS=6000
export RUBRIC_MAX_RUBRIC_CHARS=1600
export RUBRIC_MAX_ITEM_CHARS=280
export RUBRIC_MAX_TOKENS=4096
export RUBRIC_MAX_INPUT_TOKENS=$((MAX_MODEL_LENGTH - RUBRIC_MAX_TOKENS))
export RUBRIC_TEMPERATURE=0.7
export RUBRIC_TOP_P=0.9
export RUBRIC_GENERATOR_THINKING=False
export ACTOR_THINKING=True
export TEACHER_THINKING=True
export TEACHER_MAX_PROMPT_LENGTH=40960

if (( RUBRIC_MAX_INPUT_TOKENS != 36864 )); then
  echo "Unexpected rubric input budget: ${RUBRIC_MAX_INPUT_TOKENS}; expected 36864" >&2
  exit 2
fi

# No criterion pool and no generator update.
export RUBRIC_POOL_ENABLE=False
export RUBRIC_GENERATOR_LOSS_ENABLE=False
export RUBRIC_GENERATOR_LOSS_COEF=0.0
export RUBRIC_GENERATOR_ITEM_CREDIT_ENABLE=False

# Preserve the model-specific snapshot protocol; it is separate from online training.
# Use WEBSHOP_BESTOF5_SNAPSHOT_ENABLE=False to disable the final export.
export RUBRIC_SNAPSHOT_ENABLE="${WEBSHOP_BESTOF5_SNAPSHOT_ENABLE:-True}"
export RUBRIC_SNAPSHOT_STEP=300
export RUBRIC_SNAPSHOT_BATCH_SIZE=8
export RUBRIC_SNAPSHOT_MAX_ATTEMPTS=4

export PROJECT_NAME="${PROJECT_NAME:-verl_rubric_3}"
export WEBSHOP_GROUP_HINT_EXPERIMENT_NAME="${WEBSHOP_BESTOF5_EXPERIMENT_NAME:-qwen3_4b_webshop_small_b8_n8_history0_traj4_bestof5_bundle_spearman_unrankedfallback_nopool_contrastgate_gapgate_temp0p7_hybrid_additiveopd0p01_nogenthink_4gpu_s300}"
export OUTPUT_DIR="${WEBSHOP_BESTOF5_OUTPUT_DIR:-${ROOT}/checkpoints/${PROJECT_NAME}/${WEBSHOP_GROUP_HINT_EXPERIMENT_NAME}}"

echo "WebShop-small Qwen3-4B five-bundle judge route: hybrid reward; aggregate group hints; additive OPD alpha=0.01"
echo "Hybrid reward: 0.5*official + 0.5*success - 0.05*invalid_rate - 0.025*repeat_rate"
echo "Protocol: prompts=${TRAIN_BATCH_SIZE}, rollouts/prompt=${ROLLOUT_N}, candidate bundles=${RUBRIC_GROUP_CANDIDATES}, items=${MIN_RUBRICS_PER_PROMPT}-${RUBRICS_PER_PROMPT}, generator_loss=off"
echo "Rubric evidence: ${RUBRIC_MAX_TRAJECTORIES} of ${ROLLOUT_N} trajectories (highest reward, lowest reward, then seeded random remainder)"
echo "Bundle selection: highest bundle-level Spearman; judge=on; no cross-bundle mixing; guarded selection=off"
echo "If judging fails or correlation is undefined: retain the first parsed complete bundle; existing OPD gates still apply"
echo "Judge: model=${RUBRIC_JUDGE_MODEL:-qwen3.5-plus}, endpoint=${RUBRIC_JUDGE_BASE_URL:-https://api.example.com/v1}, key_env=${RUBRIC_JUDGE_API_KEY_ENV:-DASHSCOPE_API_KEY}, concurrency=${RUBRIC_JUDGE_CONCURRENCY:-8}, retries=${RUBRIC_JUDGE_RETRIES:-4}"
echo "Eligibility gates: outcome_contrast=${REQUIRE_GROUP_OUTCOME_CONTRAST}, outcome_gap_alignment=${REQUIRE_OUTCOME_ALIGNED_GAP}, minimum_margin=${MIN_OUTCOME_GAP_ALIGNMENT}"
echo "Rubric generator: temperature=${RUBRIC_TEMPERATURE}, top_p=${RUBRIC_TOP_P}, thinking=${RUBRIC_GENERATOR_THINKING}, max_tokens=${RUBRIC_MAX_TOKENS}"
echo "Rubric token budget: input<=${RUBRIC_MAX_INPUT_TOKENS}, output<=${RUBRIC_MAX_TOKENS}, model_context=${MAX_MODEL_LENGTH}"
echo "Training: steps=${MAX_STEPS}, validation_every=${TEST_FREQ}, save_every=${SAVE_FREQ}, history_window=${HISTORY_WINDOW}, history_response=${HISTORY_RESPONSE_MODE}, pool=off, OPD decay=off"
echo "Rubric snapshot: enabled=${RUBRIC_SNAPSHOT_ENABLE}, step=${RUBRIC_SNAPSHOT_STEP}"
echo "Experiment: ${WEBSHOP_GROUP_HINT_EXPERIMENT_NAME}"

exec bash "${ROOT}/examples/agent_rubric2/run_webshop_small_group_hint_grpo.sh" "$@"
