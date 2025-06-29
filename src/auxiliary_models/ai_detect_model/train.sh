#!/bin/bash
#
# Longformer Classifier Training Script
# This script sets up the environment and runs the training process for a Longformer classifier
# using Hugging Face Transformers and Accelerate for distributed training.

# ===== Environment Setup =====

# Add current directory to Python path
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Set visible GPU devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ===== Constants and Configuration =====
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME="longformer_classifier_output"
OUTPUT_DIR="${MODEL_NAME}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# ===== Helper Functions =====

# Function to log information with timestamp
log_info() {
    local message="$1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $message" | tee -a "${LOG_FILE}"
}

# Function to capture system information
log_system_info() {
    {
        echo "===== Training Environment ====="
        echo "Training start time: $(date)"
        echo ""
        echo "===== GPU Information ====="
        nvidia-smi
        echo ""
        echo "===== Python Environment ====="
        python --version
        pip list | grep "torch\|transformers\|wandb"
        echo "=============================="
    } >> "${LOG_FILE}"
}

# Function to run the training process
run_training() {
    local model_name="$1"
    local train_path="$2"
    local eval_path="$3"

    log_info "Starting training process..."
    
    accelerate launch \
        --config_file  ./config/default_config.yaml \
        ./train.py \
        --model_name_or_path "${model_name}" \
        --num_labels 2 \
        --train_file_path "${train_path}" \
        --eval_file_path "${eval_path}" \
        --display_num_examples 10 \
        --do_train True \
        --trust_remote_code True \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "steps" \
        --eval_steps 1 \
        --save_strategy "steps" \
        --save_steps 1 \
        --save_total_limit 3 \
        --load_best_model_at_end True \
        --metric_for_best_model f1_score \
        --learning_rate 2e-5 \
        --weight_decay 0.1 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --seed 42 \
        --logging_dir "${LOG_DIR}" \
        --logging_steps 100 \
        --bf16 true \
        --model_max_length 4096 2>&1 | tee -a "${LOG_FILE}"
        
    log_info "Training completed"
}

# ===== Main Execution =====

# Log system information
log_system_info

# Run the training with parameters
# Parameters:
# 1. Model name/path
# 2. Training data path
# 3. Evaluation data path
# 4. Number of epochs
run_training "allenai/longformer-base-4096" "./train_data/train.jsonl" "./eval_data/eval.jsonl" 

# ===== Training Complete =====
log_info "Script execution completed successfully"