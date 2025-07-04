import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import aiohttp
import asyncio
import ray
import torch
import torch.nn as nn
from tqdm import tqdm
import re
import os
from torch import distributed as dist

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    def __getitem__(self, idx: int) -> "Samples":
        """
        Access a single sample by index.
        Returns a new Samples object containing data for the specified index.
        """
        return Samples(
            sequences=self.sequences[idx],
            attention_mask=self.attention_mask[idx] if self.attention_mask is not None else None,
            action_mask=self.action_mask[idx] if self.action_mask is not None else None,
            num_actions=self.num_actions[idx] if isinstance(self.num_actions, torch.Tensor) else self.num_actions,
            packed_seq_lens=self.packed_seq_lens[idx] if self.packed_seq_lens is not None else None,
            response_length=self.response_length[idx],
            total_length=self.total_length[idx]
        )
    

class UserSimulatorDialogueState:
    def __init__(self, history: List[Dict[str, str]], idx: int = None):
        self.history = history            # 对话历史
        self.turn_count = 0              # 轮次计数
        self.is_active = True            # 对话是否活跃
        self.has_error = False           # 是否发生错误
        self.samples = []                # 存储该对话的所有样本
        self.idx = idx


    def update(self, 
              turn_msg: List[Dict[str, str]], 
              turn_addition: bool = False,
              turn_sample: Optional[Samples] = None):
        self.history.extend(turn_msg)
        if turn_sample is not None:
            self.samples.append(turn_sample)
        if turn_addition:
            self.turn_count += 1

    def is_completed(self, max_turns: int) -> bool:
        """检查对话是否已完成"""
        return (self.turn_count >= max_turns or 
                not self.is_active or 
                self.has_error)

    def get_samples(self, seq_pad_value: int = None, mask_pad_value: int = None) -> Samples:
        """获取对话的所有样本"""
        return self.merge_samples(self.samples, seq_pad_value, mask_pad_value)
    
    @staticmethod
    def merge_samples(samples: List["Samples"], seq_pad_value: int = None, mask_pad_value: int = None) -> Samples:
        """
            We treat every whole dialogue as a batch of sample, where each turn is one of the batch.
            For example: a dialogue with 8 turn means the batch has 8 example, each example is the subsequence of the entire dialogue:
                # turn 1: [context1, response1]
                # turn 2: [context1, response1, context2, response2]
                # turn n: [context1, response1, context2, response2, ..., context n, response n]
            We pad the context of response k to the same length, and only calculate the action mask of response k.
        """
        # batch_size 个 [seq_len] 的张量
        rank = dist.get_rank()
        # there exists cases when turn k < turn k-1, since there exists 2 padding situations
        # For example: 
        # turn 1, shape: [2, 200]:
        #   [seq 1]: system prompt: 100 tokens, user response: 10 tokens
        #   [seq 2]: system prompt: 50 tokens, user response: 100 tokens
        # turn 2, shape: [2, 190]:
        #   [seq 1]: system prompt: 120 tokens, user response: 10 tokens
        #   [seq 2]: system prompt: 170 tokens, user response: 20 tokens
        # Since the sequences are added in order, no need to re-order the sequences
        sequences_list = [s.sequences for s in samples]
        
        # batch_size 个 [seq_len] 的张量
        attention_mask_list = [s.attention_mask for s in samples if s.attention_mask is not None]
        # batch_size 个 [num_actions] 的张量
        action_mask_list = [s.action_mask for s in samples if s.action_mask is not None]
        # batch_size 个 张量
        num_actions_list = [s.num_actions if isinstance(s.num_actions, torch.Tensor) else torch.tensor(s.num_actions) for s in samples]
        # batch_size 个 张量
        response_length_list = [s.response_length for s in samples]
        # batch_size 个 张量
        total_length_list = [s.total_length for s in samples]

        # turn 0 as base turn
        min_seq_length = len(sequences_list[0])
        max_seq_length = len(sequences_list[-1])
        # dim = 0 -> [batch_size]        
        context_token_len_list = [len(sequences_list[i][:-num_action]) for i, num_action in enumerate(num_actions_list)]
        max_context_token_len = max(context_token_len_list)
        for i, context_token_len in enumerate(context_token_len_list):
            # left pad to ensure context is pad to the same length
            off_set_between_generations = max_context_token_len - context_token_len
            sequences_list[i] = torch.cat(
                [
                    torch.full((off_set_between_generations,), 
                            seq_pad_value, 
                            dtype=torch.long,  
                            device=sequences_list[i].device),
                    sequences_list[i]
                ]
            )
            attention_mask_list[i] = torch.cat(
                [
                    torch.full((off_set_between_generations,), 
                            mask_pad_value,
                            dtype=torch.bool,  
                            device=attention_mask_list[i].device),
                    attention_mask_list[i]
                ]
            )

        max_num_actions = max(num_actions_list)   
                   

        # 使用 pad_sequence 进行填补
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            sequences_list, batch_first=True, padding_value=seq_pad_value
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=mask_pad_value
        ) if attention_mask_list else None
        padded_action_mask = torch.nn.utils.rnn.pad_sequence(
            action_mask_list, batch_first=True, padding_value=mask_pad_value
        ) if action_mask_list else None



        # dim = 0 -> [batch_size]
        merged_response_lengths = torch.stack(response_length_list)
        merged_total_lengths = torch.stack(total_length_list)

        # 构造合并后的 Samples
        return Samples(
            sequences=padded_sequences,
            attention_mask=padded_attention_mask,
            action_mask=padded_action_mask,
            num_actions=max_num_actions, 
            packed_seq_lens=None,
            response_length=merged_response_lengths,
            total_length=merged_total_lengths,
        )
    def __repr__(self):
        return f"UserSimulatorDialogueState(idx={self.idx}, turn_count={self.turn_count}, is_active={self.is_active}, has_error={self.has_error}, history={self.history})"
  
        

class OpenAIDialogue:
    """OpenAI API Dialogue Manager"""
    def __init__(self, 
                api_key: str, 
                model_name: str = "gpt-4o", 
                api_base: str = "https://api.openai.com"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.session = None

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def generate_response(self, 
                              dialogue: List[Dict], 
                              max_length: int = 4096) -> Dict:
        try:
            await self.init_session()

            # filter system prompt since the profile is user's profile
            messages = [msg for msg in dialogue if msg.get("role") != "system"]
            
            async with self.session.post(
                f"{self.api_base}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_length,
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"API Error: {await response.text()}")
                    
                result = await response.json()
                return {
                    "role": "assistant",
                    "content": result["choices"][0]["message"]["content"]
                }
        except Exception as e:
            raise e

    async def __aenter__(self):
        await self.init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()




class BatchDialogueProcessor:
    def __init__(self, 
                 batch_simulators: List[UserSimulatorDialogueState],
                 tokenizer,
                 max_turns: int,
                 max_length: int):
        self.simulators = batch_simulators
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_length = max_length
        self.batch_size = len(batch_simulators)
        self.active_indices = set(range(self.batch_size))
        
    def get_valid_dialogues(self) -> Tuple[List[List[Dict]], List[int]]:
        valid_dialogues = []
        valid_indices = []
        
        completed = set()

        cur_valid_indices = list(self.active_indices)
        cur_valid_simulators = [self.simulators[idx] for idx in cur_valid_indices]
        valid_history = [simulator.history for simulator in cur_valid_simulators]
        valid_tokenized_dialogue = self.tokenizer.apply_chat_template(
            valid_history,
            tokenize=False,
            add_generation_prompt=True
        )
        for org_idx, tokenized_dialogue in zip(cur_valid_indices, valid_tokenized_dialogue):
            tokenized_ids = self.tokenizer.encode(tokenized_dialogue, add_special_tokens=False)
            if len(tokenized_ids) >= self.max_length or self.simulators[org_idx].is_completed(self.max_turns):
                self.simulators[org_idx].is_active = False
                completed.add(org_idx)
            else:
                valid_dialogues.append(tokenized_dialogue)
                valid_indices.append(org_idx)
            
        self.active_indices -= completed
        return valid_dialogues, valid_indices
    

    def update_dialogue_states(self,
                             valid_indices: List[int],
                             responses: List[str],
                             turn_samples: Optional[Samples] = None):
        for sample_idx, (response, orig_idx) in enumerate(zip(responses, valid_indices)):
            simulator = self.simulators[orig_idx]
            turn_sample = turn_samples[sample_idx] if turn_samples is not None else None
            simulator.update(
                [{"role": "user", "content": response}],
                turn_sample=turn_sample,
                turn_addition=True
            )
    
    def update_gpt_responses(self, 
                           gpt_instances: List[OpenAIDialogue], 
                           gpt_responses: List[Union[Dict, Exception]],
                           active_gpt_indices: List[int]):
        for (idx, _), response in zip(active_gpt_indices, gpt_responses):
            if isinstance(response, Exception):
                self.simulators[idx].has_error = True
                print(f"Error in US({self.simulators[idx]}) between GPT response for idx {idx}: {response}")
                self.active_indices.remove(idx)
            else:
                self.simulators[idx].update([response])
    
    @property
    def is_active(self) -> bool:
        return len(self.active_indices) > 0
    
    def get_all_samples(self, seq_pad_value: int = None, mask_pad_value: int = None) -> List[Samples]:
        return [simulator.get_samples(
            seq_pad_value=seq_pad_value,
            mask_pad_value=mask_pad_value) for simulator in self.simulators]
    
    def get_all_history(self) -> List[List[Dict]]:
        return [simulator.history for simulator in self.simulators]


async def process_gpt_batch(
    batch_processor: 'BatchDialogueProcessor',
    batch_gpt: List[OpenAIDialogue],
    rank: int
) -> List[Dict]:
    active_gpt_indices = [
        (idx, gpt) 
        for idx, gpt in enumerate(batch_gpt) 
        if idx in batch_processor.active_indices
    ]
    
    try:
        await asyncio.gather(
            *[gpt.init_session() for _, gpt in active_gpt_indices],
            return_exceptions=True
        )
        
        responses = await asyncio.gather(
            *[gpt.generate_response(batch_processor.simulators[idx].history)
              for idx, gpt in active_gpt_indices],
            return_exceptions=True
        )
        
        return responses
        
    except Exception as e:
        print(f"[Rank {rank}] Error in batch processing: {e}")
        raise
    finally:
        await asyncio.gather(
            *[gpt.close_session() for _, gpt in active_gpt_indices],
            return_exceptions=True
        )

    

class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        rank = dist.get_rank()
        # generate responses
        samples_list, dialogue_history_list = self.generate_samples(all_prompts, **generate_kwargs)
        torch.distributed.barrier(device_ids=[rank])
        experiences = []
        for samples, dialogue_history in tqdm(
            zip(samples_list, dialogue_history_list),
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
            total=len(samples_list),
        ):
            experiences.append(self.make_experience(samples,history=dialogue_history).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences
    
    @staticmethod
    def extract_system_messages_from_template(template: List[str]) -> List[str]:
        pattern = re.compile(r'<\|start_header_id\|>system<\|end_header_id\|>(.*?)<\|eot_id\|>', re.DOTALL)
        
        def extract_messages():
            for msg in template:
                if match := pattern.search(msg):
                    yield match.group(1).strip()
        
        return list(extract_messages())

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """Multiturn Simulated Dialogue Generation"""
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        
        # Extract only system prompt as profile
        all_prompts = self.extract_system_messages_from_template(all_prompts)
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        user_simulators = [UserSimulatorDialogueState([{"role": "system", "content": prompt}], idx=i)
                        for i, prompt in enumerate(all_prompts)]
        
        all_samples = [] 
        all_history = []
        rank = dist.get_rank()    
        if rank == 0:
            print(f"[Rank {rank}] max_turns: {args.max_turns}, prompt_max_len: {args.prompt_max_len}")
            print(f"[Rank {rank}] len(user_simulators): {len(user_simulators)}")
        for batch_start in range(0, len(user_simulators), args.micro_rollout_batch_size):
            batch_end = batch_start + args.micro_rollout_batch_size
            try:
                batch_processor = BatchDialogueProcessor(
                    user_simulators[batch_start:batch_end],
                    self.tokenizer,
                    args.max_turns,
                    args.prompt_max_len
                )
                print(f"[Rank {rank}] Processing {batch_start}-{batch_end}")
                print(f"[Rank {rank}] batch_processor: {batch_processor.simulators}")
                
                batch_gpt = [OpenAIDialogue(args.api_key, args.model_name, args.api_base) 
                            for _ in range(batch_end - batch_start)]
                turn_idx = 0
                
                while batch_processor.is_active:
                    print(f"[Rank {rank}]-[Turn {turn_idx}] Active indices: {batch_processor.active_indices}")
                    valid_dialogues, valid_indices = batch_processor.get_valid_dialogues()
                    print(f"[Rank {rank}]-[Turn {turn_idx}] Valid dialogues: {valid_dialogues}")
                    if not valid_dialogues:
                        break
                    valid_inputs = self.tokenize_fn(valid_dialogues, args.prompt_max_len, device="cuda")
                    sequences, attention_mask, action_mask = self.actor.generate(**valid_inputs, **generate_kwargs)                    
                    turn_samples = Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        # [batch_size, num_generated_tokens]
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        # [batch_size]
                        response_length=action_mask.float().sum(dim=-1),
                        # [batch_size]
                        total_length=attention_mask.float().sum(dim=-1),
                    )
                    
                    valid_batch_generated_ids = [
                        output_ids[len(input_ids):] 
                        for input_ids, output_ids in zip(valid_inputs["input_ids"], sequences)
                    ]
                    valid_batch_responses = self.tokenizer.batch_decode(
                        valid_batch_generated_ids, 
                        skip_special_tokens=True
                    )
                    
                    batch_processor.update_dialogue_states(
                        valid_indices,
                        valid_batch_responses,
                        turn_samples
                    )                    
                    active_gpt_indices = [
                        (idx, gpt) 
                        for idx, gpt in enumerate(batch_gpt) 
                        if idx in batch_processor.active_indices
                    ]
                    
                    loop = asyncio.get_event_loop()
                    gpt_responses = loop.run_until_complete(process_gpt_batch(
                        batch_processor,
                        batch_gpt,
                        rank
                    ))
                    
                    print(f"[Rank {rank}]-[Turn {turn_idx}] GPT responses: {gpt_responses}")
                    batch_processor.update_gpt_responses(
                        batch_gpt,
                        gpt_responses,
                        active_gpt_indices
                    )
                    turn_idx += 1
                
                # dialogue-level completion, [micro_rollout_batch_size]
                all_samples.extend(batch_processor.get_all_samples(seq_pad_value=self.tokenizer.pad_token_id, mask_pad_value=0))
                all_history.extend(batch_processor.get_all_history())
                print(f"[Rank {rank}] Finished {batch_start}-{batch_end}")
                
            except torch.cuda.OutOfMemoryError:
                print(f"[Rank {rank}] OOM error in batch {batch_start}-{batch_end}, skipping to next batch")
                
            except Exception as e:
                print(f"[Rank {rank}] Error processing batch {batch_start}-{batch_end}: {e}")
            finally:
                del batch_processor
                torch.cuda.empty_cache()

                
        return all_samples, all_history

    @torch.no_grad()
    def make_experience(self, samples: Samples, history: List[Dict]=None) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        rank = dist.get_rank()
        valid_actions = sequences[:, -num_actions:] 
        for i in range(len(sequences)):
            mask = action_mask[i]
            tokens = valid_actions[i, mask]
            print(f"[Rank {rank}] valid actions-{i}: {self.tokenizer.decode(tokens.tolist())}")

        # ## TODO
        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            r = remote_rm_fn(self.remote_rm_url, queries=history).to(device=sequences.device)
            if len(r) == 1:
                r = r[0]
            print(f"[Rank {rank}] Remote rewards: {r}")
        else:
            # local RM
            r = self.reward_model(queries=history).to(device=sequences.device)


        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()
        ret_exp = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )
        return ret_exp

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for RLOO
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        return self._generate_vllm(all_prompts, **generate_kwargs)

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        else:
            # remote RM
            if not self.packing_samples:
                queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
            else:
                sequences_list = []
                offset = 0
                tokens_list = sequences_cpu.tolist()[0]
                for length in packed_seq_lens:
                    sequences_list.append(tokens_list[offset : offset + length])
                    offset += length
                queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

            for rm in self.remote_rm_url:
                r = remote_rm_fn_ray.remote(rm, queries=queries)
                r_refs.append(r)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = dist.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None