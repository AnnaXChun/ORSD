#!/usr/bin/env bash
set -euo pipefail

# Experiment 12: Outcome-Contrastive SC-GRPO.
# The current actor compares verified-correct and verified-incorrect rollout
# references with full-vocabulary JS, then gates the ordinary GRPO advantage.
VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/verl_wsm/Data}"
MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/Qwen3-1.7b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${VERL_ROOT}/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${VERL_ROOT}/logs}"
PROJECT_NAME="${PROJECT_NAME:-verl_math}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-12_outcome_contrastive_sc_grpo}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.parquet}"
AIME24_FILE="${AIME24_FILE:-${DATA_ROOT}/aime_2024.parquet}"
AIME25_FILE="${AIME25_FILE:-${DATA_ROOT}/aime25_test.parquet}"
MATH500_FILE="${MATH500_FILE:-${DATA_ROOT}/math500_test.parquet}"
AMC23_FILE="${AMC23_FILE:-${DATA_ROOT}/amc23_test.parquet}"

for required_path in \
    "${TRAIN_FILE}" \
    "${AIME24_FILE}" \
    "${AIME25_FILE}" \
    "${MATH500_FILE}" \
    "${AMC23_FILE}" \
    "${MODEL_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export NCCL_DEBUG="WARN"
export TOKENIZERS_PARALLELISM="false"
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
export HYDRA_FULL_ERROR="1"

TRAIN_FILES="['${TRAIN_FILE}']"
VAL_FILES="['${AIME24_FILE}','${AIME25_FILE}','${MATH500_FILE}','${AMC23_FILE}']"
LOG_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}.log"

cd "${VERL_ROOT}"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.token_credit.enable=True \
    algorithm.token_credit.mode=outcome_contrastive_js \
    algorithm.token_credit.scale_quantile="${OC_SCALE_QUANTILE:-0.75}" \
    algorithm.token_credit.min_scale="${OC_MIN_SCALE:-0.0001}" \
    algorithm.token_credit.outcome_success_threshold="${OC_SUCCESS_THRESHOLD:-0.5}" \
    algorithm.token_credit.outcome_min_correct="${OC_MIN_CORRECT:-2}" \
    algorithm.token_credit.outcome_teacher_max_prompt_length="${OC_TEACHER_MAX_PROMPT_LENGTH:-10240}" \
    algorithm.token_credit.outcome_teacher_enable_thinking="${OC_TEACHER_THINKING:-False}" \
    algorithm.token_credit.outcome_distribution_temperature="${OC_DISTRIBUTION_TEMPERATURE:-1.0}" \
    algorithm.token_credit.outcome_js_chunk_size="${OC_JS_CHUNK_SIZE:-128}" \
    algorithm.token_credit.outcome_micro_batch_size_per_gpu="${OC_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    algorithm.token_credit.rubric_pipeline.enable=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-8192}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS:-1}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS:-True}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${NUM_ROLLOUTS:-8}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-1.0}" \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH:-20480}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-2}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-0.6}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-1.0}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER:-['console','wandb']}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.default_local_dir="${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.save_freq="${SAVE_FREQ:-25}" \
    trainer.test_freq="${TEST_FREQ:-25}" \
    trainer.total_training_steps="${MAX_STEPS:-250}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-20}" \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
