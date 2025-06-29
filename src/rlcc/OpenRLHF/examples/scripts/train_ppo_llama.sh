# Notice: Need to configure the model paths, data paths, API key, and base URL.

set -x
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain ./models/user_simulator_model \
   --save_path ./checkpoint_usp/output_ppo \
   --ckpt_path ./checkpoint_usp/ckpt_ppo \
   --max_ckpt_num 2 \
   --load_checkpoint \
   --save_hf_ckpt \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps -1 
   --micro_train_batch_size 2 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 32 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --generate_max_len 512 \
   --repetition_penalty 1.2 \
   --zero_stage 3 \
   --adam_offload \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data ../../dataset/train.jsonl \
   --input_key conv \
   --apply_chat_template \
   --flash_attn \
   --max_samples 100000 \
   --normalize_reward \
   --load_checkpoint \
   --use_wandb True \
   --model_name gpt-4o-mini \
   --api_key sk-xxx \
   --api_base https://openai_url \
   --max_turns 10 \
   --freezing_actor_steps 1 \
   --n_samples_per_prompt 1 \
   --remote_rm_url http://127.0.0.1:5000/get_reward/
    
EOF



if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands 2>&1 | tee test.log
fi
