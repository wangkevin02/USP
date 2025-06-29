# Notice: Need to deploy three GPU models concurrently—the profile generator, SimCSE, and the AI detection model. The total GPU memory requirement is approximately 35 GB.

export CUDA_VISIBLE_DEVICES=1
python ./remote_rm.py \
    --profile_predicter_model_path ./models/profile_predictor \
    --simcse_model_path  ./models/simcse \
    --ai_detector_model_path ./models/ai_detect_model \
    --max_len 4096 \
    --port 5000 \
    --bf16 \
    --profile_generation_batch_size 16 \
    --profile_importance_ratio 0.8 \
    --ai_detect_max_len 4096 \
    --ai_detect_batch_size 128

