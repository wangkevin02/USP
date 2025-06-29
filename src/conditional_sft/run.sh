#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Log file setup
LOG_FILE="training_log.txt"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
mkdir -p logs
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    echo -e "${2}${1}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "${LOG_FILE}"
}

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    log_message "CUDA is not available. Please check your GPU installation." "${RED}"
    exit 1
fi

# Main parameters
MODEL_PATH="Your/Path/to/Llama-3-8B" # TODO: Fill your path to llama3-8B-Base model
DATA_PATH="./train_data/train.jsonl"
EVAL_DATA_PATH="./eval_data/eval.jsonl"
DS_CONFIG="./config/ds_config_z3_offload.json"
OUTPUT_DIR="./save/Llama-3-8B_sft/full"
mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3


# Log training start
log_message "Starting training with the following configuration:" "${BLUE}"
log_message "- Model: $MODEL_PATH" "${BLUE}"
log_message "- Data: $DATA_PATH" "${BLUE}"
log_message "- Output: $OUTPUT_DIR" "${BLUE}"
log_message "- GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${BLUE}"




# Main training command
log_message "Launching training..." "${GREEN}"

accelerate launch \
     --config_file  ./config/default_config.yaml \
    ./train.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --eval_data_path "$EVAL_DATA_PATH" \
        --bf16 True \
        --do_train True \
        --use_flash_attention True \
        --trust_remote_code True \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs 3 \
        --eval_strategy "steps" \
        --eval_steps 50 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --save_strategy "steps" \
        --save_steps 50 \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.1 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --model_max_length 8192 \
        --gradient_checkpointing True \
        --load_best_model_at_end True \
        --deepspeed "$DS_CONFIG" 2>&1 | tee -a "${LOG_FILE}"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    log_message "Training completed successfully!" "${GREEN}"
else
    log_message "Training failed with error code $?" "${RED}"
fi