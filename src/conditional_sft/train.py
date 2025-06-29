from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import (
    Trainer, 
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from transformers import BitsAndBytesConfig
from utils import safe_save_model_for_hf_trainer, print_trainable_parameters
from my_datasets.my_datasets import make_supervised_data_module
import pathlib
from deepspeed import DeepSpeedConfig
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from accelerate import Accelerator
from rich.box import ROUNDED
from rich.markup import escape
import re
from rich.panel import Panel
from config.constants import *



# Configure logging with rich
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def is_local_main_process():
    """检查是否为主进程"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True
    except:
        return True

@dataclass
class ModelArguments:
    """Arguments related to the model configuration."""
    model_name_or_path: str = field(
        default="/home/wangkuang/workshop/git/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="/home/wangkuang/workshop/git/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Pretrained tokenizer name or path if different from model_name"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention for faster training"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )

@dataclass
class DataArguments:
    """Arguments related to data loading and preprocessing."""
    data_path: str = field(
        default= "/mntcephfs/home_data/wangkuang/User_Simulator/dataset/train_data_split",
        metadata={"help": "Path to the training data directory"}
    )
    eval_data_path: Optional[str] = field(
        default= "/mntcephfs/home_data/wangkuang/User_Simulator/dataset/eval_data_split",
        metadata={"help": "Path to the evaluation data directory"}
    )
    lazy_load : bool = field(
        default=False,
        metadata={"help": "Whether to lazy load the dataset"}
    )
    num_processors: int = field(
        default=16,
        metadata={"help": "Number of processors to use"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Extended training arguments with DeepSpeed and Flash Attention support."""
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    deepspeed: str = field(
        default="./config/ds_config_z3_offload.json",
        metadata={"help": "Path to DeepSpeed config file"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing for memory efficiency"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 mixed precision training"}
    )



# Define the function to count trainable parameters
def count_ds_params(model, include_all_params=False):
    """
    Count the trainable parameters in the model. If include_all_params=True,
    it also counts all parameters (trainable + non-trainable).
    """
    if include_all_params:
        # Count all parameters (trainable + non-trainable)
        total_params = sum(p.numel() + getattr(p, 'ds_numel', 0) for p in model.parameters())
    else:
        # Count only trainable parameters
        total_params = sum(p.numel() + getattr(p, 'ds_numel', 0) for p in model.parameters() if p.requires_grad)
    return total_params

def print_model_info(model):
    """Print detailed model information using rich."""
    table = Table(title="Model Information")
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="magenta")
    
    
    total_params = count_ds_params(model, include_all_params=True)  # count all parameters
    trainable_params = count_ds_params(model, include_all_params=False)  # count only trainable parameters
    print("Total Parameters: ", total_params)
    print("Trainable Parameters: ", trainable_params)
    
 # Create a table for rich output
    table = Table(title="Model Information")
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Add rows to the table
    table.add_row("Model Type", model.__class__.__name__)
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")

    # Calculate and print parameter efficiency
    if total_params == 0:
        total_params = 1  # To avoid division by zero
    parameter_efficiency = (trainable_params / total_params) * 100
    table.add_row("Parameter Efficiency", f"{parameter_efficiency:.2f}%")
    
    # Print the table to the console
    console.print(table)



def print_dataset_examples(dataset, tokenizer, num_examples=100):
    """Print example inputs and targets from the dataset with separate tables.
    Handles special characters and markup safely.
    
    Args:
        dataset: The dataset to print examples from
        tokenizer: Tokenizer for decoding
        num_examples: Number of examples to print
    """
    if not is_local_main_process():
        return
    
    console = Console(width=150)
    
    def clean_text(text):
        """Clean and escape text to avoid markup errors.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text safe for Rich printing
        """
        # Escape any Rich markup characters
        text = escape(text)
        
        # Replace potentially problematic characters
        text = text.replace('\x00', '')  # Remove null characters
        
        # Handle other special characters if needed
        text = ''.join(char if ord(char) < 65536 else '?' for char in text)
        
        return text
        
    separator = "─" * 130
    for idx in range(min(num_examples, len(dataset))):
        try:
            example = dataset[idx]
            
            # Create individual table for each example
            table = Table(
                title=f"Example {idx}",
                show_header=True,
                header_style="bold magenta",
                box=ROUNDED,
                width=150
            )
            
            # Configure columns
            table.add_column("Field", style="cyan", width=20)
            table.add_column("Content", style="green", width=130)
            
            try:
                # Decode input text
                input_text = tokenizer.decode(
                    example['input_ids'],
                    skip_special_tokens=False
                )
                input_text = clean_text(input_text)
                
                # Get target text
                target_ids = [
                    token if token == example['input_ids'][i] else None
                    for i, token in enumerate(example['labels'])
                ]
                target_ids = list(filter(None, target_ids))
                target_text = tokenizer.decode(
                    target_ids,
                    skip_special_tokens=False
                )
                target_text = clean_text(target_text)
                
                # Add rows
                table.add_row(
                    "Input",
                    Panel(
                        input_text,
                        border_style="blue",
                        padding=(1, 0)
                    )
                )

                table.add_row("", separator, style="dim")

                table.add_row(
                    "Target",
                    Panel(
                        target_text,
                        border_style="green",
                        padding=(1, 0)
                    )
                )
                
                # Print with spacing
                console.print()
                console.print(table)
                console.print()
                
            except Exception as e:
                console.print(f"[red]Error processing example {idx}: {str(e)}[/red]")
                continue
                
        except Exception as e:
            console.print(f"[red]Fatal error at example {idx}: {str(e)}[/red]")
            break


def load_model_and_tokenizer(model_args, training_args):
    """Load and configure the model and tokenizer with Flash Attention support."""
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
        print(f"before tokenizer.pad_token: {tokenizer.pad_token}")
        print(f"before tokenizer.eos_token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            tokenizer.chat_template = CHAT_TEMPLATE
        
        # Configure model loading
        model_config = {
            "pretrained_model_name_or_path": model_args.model_name_or_path,
            "trust_remote_code": model_args.trust_remote_code,
            "use_cache": not training_args.gradient_checkpointing,
            "torch_dtype": torch.bfloat16 if training_args.bf16 else torch.float32,
            "output_loading_info": True,
        }
        
        # Add Flash Attention if enabled
        if model_args.use_flash_attention:
            model_config["attn_implementation"] = "flash_attention_2"
            # model_config["flash_attention"] = True
            
        model = AutoModelForCausalLM.from_pretrained(**model_config)
        if isinstance(model, tuple):
            model = model[0] 
        model.train()
        # Enable gradient checkpointing if specified
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {str(e)}")
        raise

def prepare_data(tokenizer, data_args, max_length: int = 4096):
    """Prepare and validate the dataset."""
    try:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, max_length=max_length)
        
        # Print dataset statistics
        train_size = len(data_module["train_dataset"])
        if data_args.eval_data_path:
            eval_size = len(data_module["eval_dataset"])
        
        if data_args.lazy_load:
            console.print(f"[bold green]Loaded dataset with lazy loading[/bold green]")
        console.print(f"[bold green]Dataset Statistics:[/bold green]")
        console.print(f"Training samples: {train_size:,}")
        console.print(f"Evaluation samples: {eval_size:,}")
        
        return data_module
    
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train():
    """Main training function with improved monitoring and logging."""
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        DataArguments, 
        TrainingArguments, 
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Print configuration
    if is_local_main_process():
        console.print("[bold blue]Starting training with configuration:[/bold blue]")
        console.print(model_args)
        console.print(training_args)
        console.print(data_args)

    # # Initialize the Accelerator
    # accelerator = Accelerator()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    # Prepare data
    data_module = prepare_data(tokenizer, data_args, max_length=training_args.model_max_length)
    if is_local_main_process():
        print_dataset_examples(data_module["train_dataset"], tokenizer, num_examples=100)
        print_model_info(model)
        try:
            console.print("[bold yellow]Trainable Parameters using internal API:[/bold yellow]")
            model.print_trainable_parameters()
        except:
            pass
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        **data_module
    )


    # trainer = accelerator.prepare(Trainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     **data_module
    # ))
    
    # Training
    try:
        checkpoint_dir = pathlib.Path(training_args.output_dir)
        if list(checkpoint_dir.glob("checkpoint-*")):
            if is_local_main_process():
                console.print("[yellow]Resuming from checkpoint...[/yellow]")
            trainer.train(resume_from_checkpoint=True)
        else:
            if is_local_main_process():
                console.print("[green]Starting training from scratch...[/green]")
            trainer.train()


        # After training, access the path of the best checkpoint like this
        try:
            best_ckpt_path = trainer.state.best_model_checkpoint
            if is_local_main_process():
                console.print(f"[bold green]Best checkpoint path: {best_ckpt_path}[/bold green]")
        except:
            console.print("[red]Failed to get best checkpoint path[/red]")

        # Save final model
        trainer.save_state()
        best_model_dir = os.path.join(training_args.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        try:
            console.print(f"[bold green]Saving best model to {best_model_dir}[/bold green]")
            trainer.save_model(best_model_dir)
        except Exception as e:
            print("Failed to save model , due to ", e)
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=best_model_dir)
        if is_local_main_process():
            console.print("[bold green]Training completed successfully![/bold green]")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train()