from typing import Optional, Tuple
import jsonlines
import os
from typing import List
import torch
import transformers
import logging
from config.constants import *

from transformers import TrainerCallback, Trainer, TrainingArguments
import torch


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
    # Get the list of all JSONL files in the specified directory.
    jsonl_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jsonl')]
    return read_and_merge_jsonl(jsonl_paths)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    trainable_params_ls = []
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_ls.append(name)
    print("=" * 80)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    print(f"Trainable parameters({len(trainable_params_ls)}): {trainable_params_ls}")
    print("=" * 80)
    del trainable_params_ls




class CustomTrainingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
    
    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        """
        在每个训练步骤结束后调用
        Args:
            - args: 训练参数
            - state: 训练状态
            - control: 训练控制流
            - kwargs: 额外的关键字参数，通常包含 model, inputs, outputs 等
        """
        trainer = kwargs.get('trainer')
        
        # 获取最近一个批次的输出
        if hasattr(trainer, 'last_batch'):
            inputs, labels = trainer.last_batch
            predictions = trainer.model(inputs).logits
            
            # 根据任务类型进行分析
            if len(labels.shape) == 1:  # 分类任务
                pred_classes = torch.argmax(predictions, dim=1)
                accuracy = torch.mean((pred_classes == labels).float())
                
                # 每隔N步记录
                if state.global_step % 100 == 0:
                    print(f"Step {state.global_step}:")
                    print(f"Predictions: {pred_classes[:5]}")
                    print(f"Ground Truth: {labels[:5]}")
                    print(f"Batch Accuracy: {accuracy.item()}")
            
            elif len(labels.shape) == 2:  # 序列标注或其他复杂任务
                # 根据具体任务实现不同的评估逻辑
                pass
        
        return control