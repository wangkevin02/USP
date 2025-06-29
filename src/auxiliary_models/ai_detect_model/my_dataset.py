import json
import torch
from torch.utils.data import Dataset
from transformers import LongformerTokenizer
from datasets import Dataset as HFDataset
from typing import List, Dict, Union, Tuple
import pandas as pd
from pathlib import Path
import jsonlines


class DialogueClassificationDataset:
    """
    A class for processing dialogue data for classification tasks using Longformer.
    Handles loading, preprocessing, and encoding of dialogue data from JSONL files.
    """
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
    ):
        """
        Initialize the dataset processor with tokenizer and configuration.
        
        Args:
            model_name: Pretrained model name for the tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.label_map = {"user": 0, "assistant": 1}
    
    def read_jsonl(self, file_path: Union[str, Path]) -> List[Dict]:
            """
            Read and process JSONL file containing dialogues.
            
            Args:
                file_path: Path to the JSONL file
                
            Returns:
                List of dictionaries with content and corresponding labels
            """
            # First read the raw data from JSONL
            json_data = self._read_from_jsonl(file_path)
            
            # Process each dialogue entry
            data = []
            for item in json_data:
                try:
                    content = item.get("content", "").strip()
                    speaker = item.get("role", "").strip().lower()
                    
                    # Only include valid entries with recognized speakers
                    if content and speaker in self.label_map:
                        data.append({
                            "content": content,
                            "label": self.label_map[speaker]
                        })
                except json.JSONDecodeError:
                    continue  
            return data


    def prepare_dataset(self, file_path: Union[str, Path], num_proc: int = 16) -> HFDataset:
        """
        Prepare a HuggingFace Dataset from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            num_proc: Number of processes for parallel processing
            
        Returns:
            HuggingFace Dataset with encoded inputs ready for model training
        """
        # Read and process the raw data
        raw_data = self.read_jsonl(file_path)
        
        # Convert to DataFrame and then create a HuggingFace Dataset
        df = pd.DataFrame(raw_data)
        dataset = HFDataset.from_pandas(df)
        
        # Preprocess the dataset (tokenization and encoding)
        encoded_dataset = dataset.map(
            self._encode_text,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing texts"
        )
        
        # Set the format for PyTorch compatibility
        encoded_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )
        
        return encoded_dataset

    def _encode_text(self, examples: Dict) -> Dict:
        """
        Encode text data into model input format.
        
        Args:
            examples: Batch of examples to encode
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        # Encode the text with the tokenizer
        encodings = self.tokenizer(
            examples['content'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None  # Return lists instead of tensors
        )
        
        # Add labels to the encoded inputs
        encodings['labels'] = examples['label']
        
        return encodings

    @staticmethod
    def _read_from_jsonl(file_path: Union[str, Path]) -> List[Dict]:
        """
        Read data from a JSONL file and flatten nested conversations.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of flattened dialogue entries
        """
        data = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                data.append(obj)
        
        # Handle nested conversation structure if present
        if data and isinstance(data[0], dict) and data[0].get("conversation", None):
            data = [item["conversation"] for item in data]
        
        # Flatten nested conversations
        flat_data = []
        for item in data:
            flat_data.extend(item)
        
        return flat_data

    def make_dataset(self, train_file_path: str, eval_file_path: str) -> Dict[str, HFDataset]:
        """
        Create train and validation datasets from JSONL files.
        
        Args:
            train_file_path: Path to training data JSONL file
            eval_file_path: Path to evaluation data JSONL file
            
        Returns:
            Dictionary containing train and validation datasets
        """
        train_dataset = self.prepare_dataset(train_file_path)
        val_dataset = self.prepare_dataset(eval_file_path)
        return {"train_dataset": train_dataset, "val_dataset": val_dataset}

def get_dataset_stats(dataset: HFDataset) -> Dict:
    """
    Calculate statistics for a dataset.
    
    Args:
        dataset: HuggingFace dataset to analyze
        
    Returns:
        Dictionary with various statistics about the dataset
    """
    stats = {
        "total_samples": len(dataset),
        "max_length": max(len(x) for x in dataset['input_ids']),
        "avg_length": sum(len(x) for x in dataset['input_ids']) / len(dataset),
        "label_distribution": dict(pd.Series(dataset['labels']).value_counts().items())
    }
    return stats
