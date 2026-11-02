# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

train_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangkongcheng/data/math/train.parquet
math500_test_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangkongcheng/data/math/math500_test.parquet
amc23_test_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangkongcheng/data/math/amc23_test.parquet
aime25_test_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangkongcheng/data/math/aime25_test.parquet
train_files="['$train_path']"
test_files="['$math500_test_path', '$amc23_test_path', '$aime25_test_path']"

trainer_project_name='verl_math'
trainer_experiment_name='qwen3_17b_grpo_math'

LOG_DATA_HOME='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtai/users/zhangkongcheng/logs'
export TENSORBOARD_DIR="${LOG_DATA_HOME}/tensorboard_dir/${trainer_project_name}/${trainer_experiment_name}"

log_file="${LOG_DATA_HOME}/qwen3-17b.log"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtai/users/zhangkongcheng/model/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${trainer_project_name} \
    trainer.experiment_name=${trainer_experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    2>&1 | tee "$log_file"
