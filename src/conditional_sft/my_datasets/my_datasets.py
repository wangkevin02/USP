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
    


if __name__ == "__main__":
    # 测试数据读取功能
    @dataclass
    class DataArguments:
        data_path: str = field(default=None, metadata={"help": "训练数据路径"})
        lazy_load: bool = field(default=False, metadata={"help": "懒加载数据集"})
        eval_data_path: str = field(default=None, metadata={"help": "评估数据路径"})
        num_processors: int = field(default=8, metadata={"help": "处理器数量"})

    data_args = DataArguments(data_path="../train_data/train.jsonl")
    instruct_path = "/home/export/base/sc100164/sc100164/online1/workspace/git/models/Llama-3-8B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(instruct_path)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE
    print(f"tokenizer.chat_template: {tokenizer.chat_template}")
    print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    
    # 获取原始数据
    list_data_dict = read_jsonl_from_dir(data_args.data_path)
    
    # 创建手动处理的数据集
    dataset_dict = make_supervised_data_module(tokenizer, data_args, sft_user=False)
    train_dataset = dataset_dict['train_dataset']
    collator = dataset_dict['data_collator']
    
    print(f"\n数据集大小: {len(train_dataset)}")
    print("-" * 50)
    
    # 测试样本数量
    num_samples = min(1000, len(train_dataset))  # 限制测试样本数量
    
    for i in range(num_samples):
        print(f"\n{'='*20} 样本 {i} {'='*20}")
        
        # 获取原始对话数据
        raw_data = list_data_dict[i]
        print(f"原始数据: {raw_data}")
        
        # 获取手动处理后的数据
        manual_processed = train_dataset[i]
        manual_input_ids = manual_processed['input_ids']
        manual_labels = manual_processed['labels']
        manual_attention_mask = manual_processed['attention_mask']
        print_the_same(tokenizer, manual_input_ids, manual_labels, IGNORE_TOKEN_ID)
        
        # 计算有效长度（去除padding）
        valid_len = len([x for x in manual_attention_mask if x == 1])
        manual_valid_input_ids = manual_input_ids[:valid_len]
        
        print(f"手动处理后长度: {len(manual_input_ids)} (有效长度: {valid_len})")
        print(f"手动处理解码: {repr(tokenizer.decode(manual_valid_input_ids))}")
        
        # 使用tokenizer的chat_template直接处理
        try:
            # 确保数据格式正确
            if isinstance(raw_data, list) and all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in raw_data):
                chat_template_input_ids = tokenizer.apply_chat_template(
                    raw_data, 
                    tokenize=True, 
                    add_generation_prompt=False,
                    return_tensors="pt"
                ).squeeze().tolist()
                
                print(f"chat_template处理长度: {len(chat_template_input_ids)}")
                print(f"templat解码: {repr(tokenizer.decode(chat_template_input_ids))}")
                
                # 比较两种方式的结果
                if manual_valid_input_ids == chat_template_input_ids:
                    print("✅ 手动处理与chat_template结果一致")
                else:
                    print("❌ 手动处理与chat_template结果不一致")
                    print(f"手动处理input_ids: {manual_valid_input_ids}")
                    print(f"chat_template input_ids: {chat_template_input_ids}")
                    
                    # 找出差异位置
                    min_len = min(len(manual_valid_input_ids), len(chat_template_input_ids))
                    for j in range(min_len):
                        if manual_valid_input_ids[j] != chat_template_input_ids[j]:
                            print(f"第一个差异位置: {j}")
                            print(f"手动处理token: {manual_valid_input_ids[j]} -> '{tokenizer.decode([manual_valid_input_ids[j]])}'")
                            print(f"chat_template token: {chat_template_input_ids[j]} -> '{tokenizer.decode([chat_template_input_ids[j]])}'")
                            break
                    
                    if len(manual_valid_input_ids) != len(chat_template_input_ids):
                        print(f"长度差异: 手动处理={len(manual_valid_input_ids)}, chat_template={len(chat_template_input_ids)}")
                        
            else:
                print("⚠️ 原始数据格式不符合chat_template要求，跳过比较")
                
        except Exception as e:
            print(f"⚠️ chat_template处理失败: {e}")
        
        # 显示标签信息（用于调试训练目标）
        valid_labels = [label for label in manual_labels[:valid_len] if label != IGNORE_TOKEN_ID]
        if valid_labels:
            print(f"训练标签部分: {tokenizer.decode(valid_labels)}")
        else:
            print("没有训练标签（全部被忽略）")
        
        print("-" * 50)
        

    # 批处理测试
    print(f"\n{'='*20} 批处理测试 {'='*20}")
    batch_size = 5
    data_samples = [train_dataset[i] for i in range(min(batch_size, len(train_dataset)))]
    processed_batch = collator(data_samples)
    
    print(f"批处理后shape:")
    print(f"input_ids: {processed_batch['input_ids'].shape}")
    print(f"labels: {processed_batch['labels'].shape}")
    print(f"attention_mask: {processed_batch['attention_mask'].shape}")
    
    # 验证批处理后的第一个样本
    batch_input_ids = processed_batch['input_ids'][0]
    batch_attention_mask = processed_batch['attention_mask'][0]
    batch_valid_len = batch_attention_mask.sum().item()
    batch_valid_input_ids = batch_input_ids[:batch_valid_len].tolist()
    
    # 与原始单个样本比较
    original_sample = train_dataset[0]
    original_valid_len = len([x for x in original_sample['attention_mask'] if x == 1])
    original_valid_input_ids = original_sample['input_ids'][:original_valid_len]
    
    if batch_valid_input_ids == original_valid_input_ids:
        print("✅ 批处理后数据与原始数据一致")
    else:
        print("❌ 批处理后数据与原始数据不一致")
        print(f"原始: {original_valid_input_ids}")
        print(f"批处理: {batch_valid_input_ids}")