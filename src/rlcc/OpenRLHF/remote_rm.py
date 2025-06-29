import argparse
import re

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from openrlhf.models.ai_detect import AIDetector
from openrlhf.models.profile_predictor import ProfileGenerator
from typing import List, Dict, Optional

logger = init_logger(__name__)
app = FastAPI()

parser = argparse.ArgumentParser()
# Reward Model
parser.add_argument("--profile_predicter_model_path", type=str, default="openrlhf/models/profile_predicter_model")
parser.add_argument("--simcse_model_path", type=str, default="openrlhf/models/simcse_model")
parser.add_argument("--ai_detector_model_path", type=str, default="openrlhf/models/ai_detector_model")
parser.add_argument("--max_len", type=int, default=4096)

# Server
parser.add_argument("--port", type=int, default=8080, help="Port number for the server")
parser.add_argument("--host", type=str, default="127.0.0.1",help="IP for the server")

# Performance
parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
parser.add_argument("--profile_generation_batch_size", type=int, default= 4)
parser.add_argument("--ai_detect_max_len", type=int, default=4096)
parser.add_argument("--ai_detect_batch_size", type=int, default=16)
parser.add_argument("--profile_importance_ratio", type=float, default= 0.5)

args = parser.parse_args()

# # server


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text

def extract_user_messages(messages: List[Dict[str, str]]) -> List[str]:
    """
    从消息列表中提取用户消息。
    
    Args:
        messages (List[Dict[str, str]]): 消息列表
        
    Returns:
        List[str]: 用户消息列表
    """
    return [msg["content"] for msg in messages if msg["role"] == "user"]


def extract_profile_from_text(text):
    """
    Extract profile content from input text
    
    Args:
        text (str): Input text containing profile information
        
    Returns:
        str: Extracted profile content, returns None if not found or in case of error
    """
    pattern = r"your profile is:\s+(.*?)(?=\s+You can say anything you want)"
    
    try:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            profile = match.group(1).strip()
            # Remove trailing double dots if present
            if profile.endswith(".."):
                profile = profile[:-1]
            return profile
        return None
    except Exception as e:
        print(f"Error occurred during extraction: {e}")
        return None

def extract_profile_from_system_messages(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Extract profile content from system messages
    
    Args:
        messages (List[Dict[str, str]]): System messages
    
    Returns:
        Optional[str]: Extracted profile content, returns None if not found
    """
    for msg in messages:
        if msg["role"] == "system":
            profile = extract_profile_from_text(msg["content"])
            if profile:
                return profile
    return None




class RewardModelProxy:
    def __init__(self, args):
        self.profile_generator = ProfileGenerator(
            model_path=args.profile_predicter_model_path,
            simcse_model_path=args.simcse_model_path,
            bf16=args.bf16,
            use_flash_attention=args.flash_attn,
            batch_size=args.profile_generation_batch_size
        )
        self.ai_detector = AIDetector(
            model_name=args.ai_detector_model_path,
            max_length=args.ai_detect_max_len,
            batch_size=args.ai_detect_batch_size
        )
        self.profile_importance_ratio = args.profile_importance_ratio
        print("Reward model proxy initialized with profile importance ratio:", self.profile_importance_ratio)

    def get_reward(self, queries:List[List[Dict]]):
        if not queries:
            return []
        # if only one query(i.e. one dialogue example), which is default setting
        if isinstance(queries[0], dict):
            queries = [queries]
        # [batch_size(num_dialogue_examples), num_turns_per_dialogue]
        golden_profile = [extract_profile_from_system_messages(query) for query in queries]
        # [batch_size(num_dialogue_examples), num_turns_per_dialogue]
        user_messages = [extract_user_messages(query) for query in queries]
        input_batch_size = len(user_messages)
        # [batch_size(num_dialogue_examples), num_turns_per_dialogue]
        turns_per_sample = [len(messages) for messages in user_messages]
        # Get rewards
        # ai_detect_rewards [batch_size(num_dialogue_examples), num_turns_per_dialogue]
        ai_detect_rewards = self.ai_detector(user_messages)
        # profile_rewards = IQ_reward + EQ_reward
        profile_rewards = self.profile_generator(golden_profile, user_messages)

        combined_rewards = []
        for i, num_turns in enumerate(turns_per_sample):
            # For each dialogue example, repeat the profile reward num_turns times to assign to each turn
            sample_profile_reward = profile_rewards[i].repeat(num_turns).to(ai_detect_rewards[i].device) 
            combined_reward = sample_profile_reward * self.profile_importance_ratio + ai_detect_rewards[i] * (1 - self.profile_importance_ratio) * 2
            combined_rewards.append(combined_reward.detach().cpu().numpy().tolist())
        return combined_rewards



reward_model = RewardModelProxy(args)

        
@app.post("/get_reward/")
async def get_reward(request: Request):
    data = await request.json()
    queries = data.get("query")
    print(f"Received queries: {queries}")
    rewards = reward_model.get_reward(queries)
    result = {"rewards": rewards}
    logger.info(f"Sent JSON: {result}")
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run("remote_rm:app", host=args.host, port=args.port, log_level="info", reload=False)
