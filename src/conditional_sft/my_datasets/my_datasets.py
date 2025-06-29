import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Sequence, List
import jsonlines
import os
import torch
import transformers
from torch.utils.data import Dataset
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
from config.constants import *
from typing import Union
from tqdm import tqdm
from datasets import load_dataset
from datasets import Dataset as HFDataset
from functools import partial
import torch.distributed as dist

def read_jsonl_file(jsonl_path: str) -> List[dict]:
    """
    Reads a JSONL file and returns its contents as a list of dictionaries.
    
    Args:
    - jsonl_path (str): Path to the JSONL file.
    
    Returns:
    - List[dict]: A list of JSON objects read from the JSONL file.
    """
    data = []
    try:
        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                data.append(obj)
    except Exception as e:
        # If reading the file fails, print an error message.
        print(f"Error reading JSONL file {jsonl_path}: {e}")
    return data

def read_and_merge_jsonl(jsonl_paths: List[str]) -> List[dict]:
    """
    Reads multiple JSONL files and merges their contents into a single list.
    
    Args:
    - jsonl_paths (List[str]): List of paths to JSONL files.
    
    Returns:
    - List[dict]: A merged list containing all JSON objects from the input files.
    """

    merged_data = []
    # Iterate over each JSONL file path and read its contents.
    for jsonl_path in jsonl_paths:
        merged_data.extend(read_jsonl_file(jsonl_path))  # Use extend to add all elements from each file.

    return merged_data

def read_jsonl_from_dir(dir_path: str) -> List[dict]:
    """
    Reads all JSONL files from a directory and merges their contents.
    
    Args:
    - dir_path (str): Path to the directory containing JSONL files.
    
    Returns:
    - List[dict]: A merged list containing all JSON objects from the JSONL files in the directory.
    """
    if os.path.isdir(dir_path):
        # Get the list of all JSONL files in the specified directory.
        jsonl_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jsonl')]
        return read_and_merge_jsonl(jsonl_paths)
    else:
        json_paths = [dir_path]
        return read_and_merge_jsonl(json_paths)



class Constants:
    """
    A class to hold constants for token IDs used in the preprocessing.
    """
    def __init__(self, tokenizer):
        self.BEGIN_OF_TEXT_ID = tokenizer(BEGIN_OF_TEXT_TOKEN, add_special_tokens=False).input_ids
        self.START_HEADER_ID = tokenizer(START_HEADER_TOKEN, add_special_tokens=False).input_ids
        self.END_HEADER_ID = tokenizer(END_HEADER_TOKEN, add_special_tokens=False).input_ids
        self.EOT_ID = tokenizer(EOT_TOKEN, add_special_tokens=False).input_ids
        self.NL_TOKENS = tokenizer(NL_TOKENS, add_special_tokens=False).input_ids
        self.SYSTEM = tokenizer(SYSTEM_TOKEN, add_special_tokens=False).input_ids
        self.USER = tokenizer(USER_TOKEN, add_special_tokens=False).input_ids
        self.ASSISTANT = tokenizer(ASSISTANT_TOKEN, add_special_tokens=False).input_ids
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.PAD_TOKEN_ID = tokenizer.pad_token_id
        self.IGNORE_TOKEN_ID = IGNORE_TOKEN_ID
        self.CHAT_TEMPLATE = CHAT_TEMPLATE


def role_template(role_id, start_header_id, end_header_id, nl_tokens):
    """
    Creates a role template for a message.
    """
    return start_header_id + role_id + end_header_id + nl_tokens


def message_template(tokenizer, message, eot_id):
    """
    Encodes a message and appends the end-of-text token.
    """
    message = message.strip()
    return tokenizer(message, add_special_tokens=False).input_ids + eot_id


def print_target_text(tokenizer, target, IGNORE_TOKEN_ID):
    """
    Prints non-ignored parts of the target text.
    """
    i = 0
    while i < len(target):
        if target[i] != IGNORE_TOKEN_ID:
            j = i
            while j < len(target) and target[j] != IGNORE_TOKEN_ID:
                j += 1
            print("*" * 20)
            print("target train text:", tokenizer.decode(target[i:j]))
            print("*" * 20)
            i = j
        i += 1

def print_the_same(tokenizer, input_ids, target, IGNORE_TOKEN_ID):
    """
    Prints the same parts of the input and target text.
    """
    i = 0
    while i < len(target):
        if target[i] == input_ids[i]:
            j = i
            while j < len(target) and target[j] == input_ids[j]:
                j += 1
            print("*" * 20)
            print("the same text:", tokenizer.decode(input_ids[i:j]))
            print("*" * 20)
            i = j
        i += 1




def preprocess_single_conversation(
        source,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "",
        sft_user: bool = True
) -> Dict:
    """
    Preprocesses the source messages into input IDs and targets.

    Args:
    - sources: The source messages.
    - tokenizer: The tokenizer used for encoding messages.
    - max_len: The maximum length of each sequence.
    - system_message: Optional system message to prepend.
    - sft_user: Flag for training mode.

    Returns:
    - A dictionary containing input IDs, labels, and attention masks.
    """
    source = source["data"]
    constants = Constants(tokenizer)

    # Apply prompt templates
    input_ids, targets, attention_masks = [], [], []
    input_id, target = constants.BEGIN_OF_TEXT_ID.copy(), []

    # Add system message if available
    if system_message:
        system = role_template(constants.SYSTEM, constants.START_HEADER_ID, constants.END_HEADER_ID, constants.NL_TOKENS)
        input_id += system + message_template(tokenizer, system_message, constants.EOT_ID)
    target += [IGNORE_TOKEN_ID] * len(input_id)

    # Iterate through user/assistant messages
    for j, sentence in enumerate(source):
        role = sentence["role"]
        value = sentence["content"]

        if role == 'user':
            role_id = role_template(constants.USER, constants.START_HEADER_ID, constants.END_HEADER_ID, constants.NL_TOKENS)
            _input_id = role_id + message_template(tokenizer, value, constants.EOT_ID)
            _target = ([IGNORE_TOKEN_ID] * len(role_id) + message_template(tokenizer, value, constants.EOT_ID)) if sft_user else [IGNORE_TOKEN_ID] * len(_input_id)
        elif role == 'assistant':
            role_id = role_template(constants.ASSISTANT, constants.START_HEADER_ID, constants.END_HEADER_ID, constants.NL_TOKENS)
            _input_id = role_id + message_template(tokenizer, value, constants.EOT_ID)
            _target = [IGNORE_TOKEN_ID] * len(_input_id) if sft_user else ([IGNORE_TOKEN_ID] * len(role_id) + message_template(tokenizer, value, constants.EOT_ID))
        elif role == 'system':
            role_id = role_template(constants.SYSTEM, constants.START_HEADER_ID, constants.END_HEADER_ID, constants.NL_TOKENS)
            _input_id = role_id + message_template(tokenizer, value, constants.EOT_ID)
            _target = [IGNORE_TOKEN_ID] * len(_input_id)
        else:
            raise NotImplementedError("Role type not supported: " + role)

        target += _target
        input_id += _input_id
    input_id = tokenizer.apply_chat_template(source)
    # check with golden input_id to ensure the target is correct
    assert len(input_id) == len(target), f"input_id: {len(input_id)}, target: {len(target)} should be the same"

    # Padding the sequences to max_len
    input_id_length = len(input_id)
    mask = [1] * input_id_length
        
    input_ids = input_id[:max_len]
    targets = target[:max_len]
    attention_masks = mask[:max_len]


    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_masks, # eos_token_id is used as padding token, thus we cannot use input_ids.ne(tokenizer.pad_token_id)
    )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, sft_user: bool = False, max_length: int = 4096) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazyInstructionDataset if data_args.lazy_load else  InstructionDataset
    train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path,sft_user=sft_user, num_processors=data_args.num_processors, max_length=max_length)

    if data_args.eval_data_path:
        eval_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.eval_data_path, num_processors=data_args.num_processors)
    else:
        eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

class InstructionDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, sft_user: bool = True, num_processors: int = 16, max_length: int = 4096):
        super(InstructionDataset, self).__init__()
        logging.info("Loading data...")
        list_data_dict = read_jsonl_from_dir(data_path)
        data_dict = {"data": list_data_dict}
        dataset = HFDataset.from_dict(data_dict)
        self.tokenizer = tokenizer
        preprocess_fn = partial(preprocess_single_conversation, tokenizer=tokenizer, max_len=max_length, sft_user=sft_user) 
        processed_dataset = dataset.map(
            preprocess_fn, remove_columns=dataset.column_names, num_proc=num_processors
        )
        self.input_ids = processed_dataset["input_ids"]
        self.labels = processed_dataset["labels"]
        self.attention_mask = processed_dataset["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i: Union[int, slice, List[int]]) -> Dict[str, torch.Tensor]:
            if isinstance(i, int):  # 单个索引
                return dict(
                    input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    attention_mask=self.attention_mask[i],
                )
            elif isinstance(i, slice):  # 切片
                return dict(
                    input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    attention_mask=self.attention_mask[i],
                )
            elif isinstance(i, list):  # 列表索引
                return dict(
                    input_ids=[self.input_ids[idx] for idx in i],
                    labels=[self.labels[idx] for idx in i],
                    attention_mask=[self.attention_mask[idx] for idx in i],
                )
            else:
                raise TypeError(f"Unsupported index type: {type(i)}")

class LazyInstructionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, cache_size: int = 100, sft_user: bool = False, max_length: int = 4096):
        super(LazyInstructionDataset, self).__init__()
        logging.info("Loading raw data...")
        self.tokenizer = tokenizer
        list_data_dict = read_jsonl_from_dir(data_path)
        self.raw_data = {"data": list_data_dict}
        self.cache_size = cache_size
        self.cached_data_dict = {}
        self.cached_indices = []
        self.sft_user = sft_user
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data["data"])

    def __getitem__(self, i: Union[int, slice, List[int]]) -> Dict[str, torch.Tensor]:
        if isinstance(i, int):
            if i in self.cached_indices:
                return self.cached_data_dict[i]
            else:
                return self.process_and_cache_data(i)
        elif isinstance(i, slice):
            # 处理切片
            indices = range(*i.indices(len(self)))
            return [self.__getitem__(idx) for idx in indices]
        elif isinstance(i, list):
            # 处理列表索引
            return [self.__getitem__(idx) for idx in i]
        else:
            raise TypeError(f"Unsupported index type: {type(i)}")

    def process_and_cache_data(self, i: int) -> Dict[str, torch.Tensor]:
        # 处理单个数据样本
        data = {"data": [self.raw_data["data"][i]]}
        data_dict = preprocess_single_conversation(data, self.tokenizer, max_len=self.max_length, sft_user=self.sft_user)
        
        # 获取单个样本的数据
        processed_data = {
            "input_ids": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "attention_mask": data_dict["attention_mask"]
        }

        # 缓存数据
        self.cache_data(i, processed_data)
        return processed_data

    def cache_data(self, i: int, data_dict: Dict[str, torch.Tensor]):
        if len(self.cached_indices) >= self.cache_size:
            # 移除最早缓存的数据（LRU策略）
            lru_index = self.cached_indices.pop(0)
            del self.cached_data_dict[lru_index]

        self.cached_indices.append(i)
        self.cached_data_dict[i] = data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in instances]
        labels = [torch.tensor(instance["labels"], dtype=torch.long) for instance in instances]
        attention_mask = [torch.tensor(instance["attention_mask"], dtype=torch.bool) for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )
    

def tensor2list(tensor):
    return tensor.detach().cpu().tolist()
    