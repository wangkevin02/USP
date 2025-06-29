# Conditional SFT

## Features

- Unlike vanilla LLMs trained on assistant responses, this approach optimizes the **user-side utterance** loss.
- Supports **multi-turn training** by computing the loss directly at each user utterance, avoiding redundant forward passes.

## Usage

1. Adjust the configuration, including **model weight paths** in [`run.sh`](./run.sh) and **multi-GPU settings** in the `config` directory.

2. Launch training:

   ```bash
   ./run.sh
   ```