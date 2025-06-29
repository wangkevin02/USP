## RLCC Design Overview

**Key Characteristics**

- RLCC performs reinforcement learning at the **dialogue level**.
- The **core modification** is in [`experience_maker.py`](./OpenRLHF/openrlhf/trainer/ppo_utils/experience_maker.py), extending standard single-turn optimization to multi-turn interactions.
- During loss computation, dialogue-level rewards are redistributed to <u>individual turns</u> for gradient updates.

**Implementation Approach**

- Standard PPO sampling is extended to generate complete multi-turn dialogues between the user simulator and GPT, iterating until reaching `max_turn` or `max_length`.

- Each dialogue is segmented into a batch of samples, where each sample represents a subsequence of the conversation:

  ```
  Turn 1: x1 + y1
  Turn 2: x1 + y1 + x2 + y2
  ...
  ```

  For each turn *k*, the preceding tokens are treated as context (excluded from loss), and the action loss is computed solely for *yₖ*, effectively reducing optimization to the single-turn level.

- **Reward Design:**

  - *Dialogue-level consistency*: The simulated dialogue is used to extract a profile representation, which is compared with the target profile to compute consistency rewards for factual and subjective alignment.
  - *Utterance-level authenticity*: Each generated utterance is evaluated with an AI detection loss.

**Training and Deployment**

- Start the reward model backend and prepare the following models:

  - `profile_predictor_model`: extracts profile representations from simulated dialogues.
  - `simcse_model`: computes consistency rewards based on the extracted and target profiles.
  - `ai_detector_model`: assesses utterance-level authenticity.

  ```bash
  ./run_remote_rm.sh
  ```

- Please complete the basic path configurations in [`train_ppo_llama.sh`](OpenRLHF\examples\scripts\train_ppo_llama.sh), including:

  - The path to the model checkpoint obtained from Conditional SFT training.
  - The dataset path for PPO training.
  - API credentials and endpoint configurations for interacting with the GPT API during training.
  
- Launch training:

  ```bash
  cd ./examples/scripts
  ./train_ppo_llama.sh
  ```

**Notes**

- *Reward request failures*: Verify that the reward service IP and ports match between the deployment and `train_ppo_llama.sh`.
- *Multi-GPU limitations*: Multi-GPU training is <span style="background-color: #f8d7da; color: #721c24; padding: 4px; border-radius: 4px;"> currently unsupported</span>. Due to variable turn counts arising from differing response lengths in sampling, **batches across devices may have inconsistent shapes**, preventing synchronization. Optimization is ongoing.



