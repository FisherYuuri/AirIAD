#!/bin/bash
# AiRIAD SSGiGPO training script — 4B model
set -x

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

CONFIG_PATH="${ROOT_DIR}/examples/sglang_multiturn/config"
TOOL_CONFIG="${ROOT_DIR}/examples/sglang_multiturn/config/tool_config/airiad_tool_config.yaml"
REWARD_PATH="${ROOT_DIR}/verl/utils/reward_score/airiad_reward.py"

DATA_DIR="/path/to/verl/data"
MODEL_PATH=${MODEL_PATH:-"/path/to/model"}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='airiad_multiturn_ssgigpo_4b' \
    algorithm.adv_estimator=ssgigpo \
    algorithm.use_kl_in_reward=False \
    ++algorithm.ssgigpo.step_advantage_w=1.0 \
    ++algorithm.ssgigpo.iou_threshold=0.5 \
    ++algorithm.ssgigpo.mode=mean_norm \
    data.train_files=${DATA_DIR}/train_mix.parquet \
    data.val_files=${DATA_DIR}/test_os.parquet \
    data.val_max_samples=10 \
    data.train_batch_size=64 \
    data.max_prompt_length=3072 \
    data.max_response_length=8096 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=airiad_tool_agent \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG} \
    custom_reward_function.path=${REWARD_PATH} \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.default_local_dir='/path/to/checkpoint/qwen3vl_4b_airiad_ssgigpo' \
    trainer.project_name='verl_ssgigpo_iad' \
    trainer.experiment_name='qwen3vl_4b_airiad_ssgigpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.val_before_train=false
