# Auxiliary Models Implementation

## AI Detection Model

### Features

- Treats user utterances within a dialogue as human-labeled data and assistant utterances as AI-labeled data.
- Supports multi-GPU training.

### Usage

1. Adjust the configuration, including model weight paths in [`train.sh`](./ai_detect_model/train.sh) and multi-GPU settings in the `config` directory.

2. Start training:

   ```bash
   ./train.sh
   ```

------

## Profile Generation Model

### Features

- Takes a dialogue as input and outputs user profile descriptions.
- Concatenates the dialogue into a single text block serving as the user input, enabling the model to learn to generate profiles conditioned on both the instruction and the user dialogue.

### Usage

1. Use [`transfer_data_format.py`](./profile_predictor/transfer_data_format.py) to convert OpenAI-style dialogues into instruction-following format.
2. Train with [LLaMA-Factory↗](https://github.com/hiyouga/LLaMA-Factory) using the transformed data.





