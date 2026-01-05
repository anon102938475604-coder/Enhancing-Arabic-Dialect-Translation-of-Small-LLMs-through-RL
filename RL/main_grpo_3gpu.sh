#!/bin/bash

set -x

# Memory monitoring function
monitor_memory() {
    echo "=== Memory Usage ==="
    free -h
    echo "=== GPU Memory ==="
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    echo "=================="
}

model_path=Qwen/Qwen2.5-3B
train_file_path=data/train/parquet/madar_dialect_en.parquet
test_file_path=data/test/parquet/madar_dialect_en.parquet
comet_model_path="" #set your metric ckpt
comet_free_model_path="" #set your metric ckpt

### Step 2: RL Training (3 A100 80G GPU version)
export WANDB_API_KEY="wandb_api_key" # set your wandb api key

export VLLM_ATTENTION_BACKEND=XFORMERS
datetime=$(date +"%Y%m%d%H%M%S")
echo $datetime

# Balanced parameters for 3 A100 80G GPU setup
# Note: train_batch_size must be divisible by 3 (number of GPUs)
train_batch_size=3
rollout_num=3
comet_rm=False
comet_free_rm=False 
reward_metric=BLEU
use_think_length_reward=True
exp_name=3GPU_bs@${train_batch_size}_n@${rollout_num}_comet_rm@${comet_rm}_cometfree_rm@${comet_free_rm}_reward_metric@${reward_metric}_@${datetime}

# Monitor memory before starting
monitor_memory

# Use 3 GPUs
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file_path} \
    data.val_files=${test_file_path} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=64 \
    data.max_prompt_length=384 \
    data.max_response_length=768 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    comet_model.enable=False \
    comet_model.use_rm=${comet_rm} \
    comet_model.use_valid=True \
    comet_model.ckpt_path=${comet_model_path} \
    comet_free_model.enable=False \
    comet_free_model.use_rm=${comet_free_rm} \
    comet_free_model.use_valid=True \
    comet_free_model.ckpt_path=${comet_free_model_path} \
    algorithm.reward_type='continuous' \
    algorithm.reward_continuous_scale=100 \
    algorithm.reward_metric=${reward_metric} \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.check_think=True \
    trainer.val_before_train=False \
    trainer.logger=['wandb'] \
    trainer.project_name='RL' \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${exp_name} \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=1000 \
    trainer.test_freq=200 \
    trainer.total_epochs=1 $@ 2>&1 | tee output.log

