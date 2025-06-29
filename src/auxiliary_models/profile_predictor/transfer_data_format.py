# Import necessary libraries
from dataclasses import dataclass
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define configuration class
@dataclass
class ProfileConfig:
    """Configuration for profile generation"""
    system_prompt: str = "You are an expert in creating user profile descriptions based on dialogue analysis."
    instruction: str = "Analyze the user utterances marked by [User] to generate a comprehensive and descriptive user profile."
    max_length: int = 4096

# Helper function to extract user messages
def extract_user_messages(messages: List[Dict[str, str]]) -> List[str]:
    """Extract user messages from a message list."""
    return [msg["content"] for msg in messages if msg["role"] == "user"]

# Prepare messages for model input
def prepare_messages(utterances: List[str], config: ProfileConfig, profile: str) -> str:
    """Prepare messages for model input with optimized formatting."""
    user_prompt = "".join(f"[User]: {u}\n---\n" for u in utterances)
    formatted_msg = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": f"{config.instruction}\n{user_prompt}"},
        {"role": "assistant", "content": f"{profile}"}
    ]
    return formatted_msg


openai_format_msg = {"profile": "<example_profile>", "conversation": [{"role": "system", "content": "system_prompt"}, {"role": "user", "content": "utterance_1"}, {"role": "assistant", "content": "xxx"}, {"role": "user", "content": "utterance_2"}, {"role": "assistant", "content": "xxx"}]}

user_utterances = extract_user_messages(openai_format_msg["conversation"])
profile = openai_format_msg["profile"]
formatted_msg = prepare_messages(user_utterances, ProfileConfig(), profile)
print(formatted_msg)

