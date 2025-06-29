import torch
from transformers import (
    LongformerTokenizer, 
    LongformerForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import logging
import numpy as np
from datasets import Dataset
from typing import Dict, List, Union, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from my_dataset import DialogueClassificationDataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.columns import Columns
from rich import print as rprint
import random
import pathlib
import transformers

# Initialize Rich console for pretty output
console = Console()


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Calculate F1 score for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method, supports 'macro'
        
    Returns:
        float: F1 score value
    """
    def compute_class_metrics(true_positive: int, false_positive: int, false_negative: int) -> Tuple[float, float, float]:
        """
        Calculate precision, recall and f1 for a single class.
        
        Args:
            true_positive: Number of true positives
            false_positive: Number of false positives
            false_negative: Number of false negatives
            
        Returns:
            Tuple containing precision, recall, and f1 values
        """
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of y_true and y_pred must be the same")
        
    # Get all unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Calculate metrics for each class
    f1_scores = []
    for cls in classes:
        # Calculate confusion matrix elements
        true_positive = np.sum((y_true == cls) & (y_pred == cls))
        false_positive = np.sum((y_true != cls) & (y_pred == cls))
        false_negative = np.sum((y_true == cls) & (y_pred != cls))
        
        # Calculate metrics for this class
        _, _, f1 = compute_class_metrics(true_positive, false_positive, false_negative)
        f1_scores.append(f1)
    
    if average == 'macro':
        return np.mean(f1_scores)
    else:
        raise ValueError(f"Unsupported average type: {average}")

def compute_metrics(pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        pred: Prediction object from Trainer
        
    Returns:
        Dictionary with accuracy and F1 score
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = np.mean(labels == preds)
    f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'f1_score': f1
    }



@dataclass
class ModelArguments:
    """
    Arguments related to model configuration.
    """
    model_name_or_path: str = field(
        default="allenai/longformer-base-4096",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum length of input sequences"}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of classes for classification task"}
    )



@dataclass
class DataArguments:
    """
    Arguments related to data loading and preprocessing.
    """
    train_file_path: str = field(
        default="./train_data/train.jsonl",
        metadata={"help": "Path to the training data file"}
    )
    eval_file_path: str = field(
        default="./eval_data/eval.jsonl",
        metadata={"help": "Path to the evaluation data file"}
    )
    display_num_examples: int = field(
        default=2,
        metadata={"help": "Number of examples to display per split"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Extended training arguments with additional optimization options.
    """
    optim: str = field(default="adamw_torch")


def create_rich_table(title: str, headers: List[str], style: str = "bold magenta") -> Table:
    """
    Create a Rich table with standardized formatting.
    
    Args:
        title: Title for the table
        headers: List of column headers
        style: Style for the header
        
    Returns:
        Configured Rich Table object
    """
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    table = Table(show_header=True, header_style=style)
    for header in headers:
        table.add_column(header)
    return table


def display_arguments(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    """
    Display all configuration arguments using Rich library.
    
    Args:
        model_args: Model configuration arguments
        data_args: Data configuration arguments
        training_args: Training configuration arguments
    """
    # Display model arguments
    model_table = create_rich_table("Model Parameters", ["Parameter", "Value"])
    for field_name, field_value in model_args.__dict__.items():
        model_table.add_row(field_name, str(field_value))
    console.print(model_table)

    # Display data arguments
    data_table = create_rich_table("Data Parameters", ["Parameter", "Value"])
    for field_name, field_value in data_args.__dict__.items():
        data_table.add_row(field_name, str(field_value))
    console.print(data_table)

    # Display training arguments
    training_table = create_rich_table("Training Parameters", ["Parameter", "Value"])
    for field_name, field_value in training_args.__dict__.items():
        if not field_name.startswith('_'):
            training_table.add_row(field_name, str(field_value))
    console.print(training_table)


def display_dataset_stats(dataset: Dict[str, Dataset]):
    """
    Display dataset statistics using Rich library.
    
    Args:
        dataset: Dictionary containing dataset splits or a single dataset
    """
    stats_table = create_rich_table("Dataset Statistics", ["Dataset", "Sample Count", "Label Distribution", "Proportion"])
    
    # Handle both single dataset and dictionary of datasets
    if not isinstance(dataset, dict):
        dataset = {"dataset": dataset}
    
    for split_name, split_dataset in dataset.items():
        label_counts = np.unique(split_dataset['labels'], return_counts=True)
        label_distribution = dict(zip(*label_counts))
        proportion = {k: v / len(split_dataset) for k, v in label_distribution.items()}
        
        stats_table.add_row(
            split_name,
            str(len(split_dataset)),
            str(label_distribution),
            str(proportion)
        )
    
    console.print(stats_table)

def display_dataset_examples(dataset: Dict[str, Dataset], num_examples: int = 2, tokenizer: Optional[LongformerTokenizer] = None):
    """
    Display examples from the dataset using Rich library.
    
    Args:
        dataset: Dictionary containing dataset splits or a single dataset
        num_examples: Number of examples to display per split
        tokenizer: Tokenizer to decode text from input IDs
    """
    console.print(f"="*100)
    console.print(f"\n[bold green]Displaying {num_examples} examples per split[/bold green]")
    console.print("\n[bold green]Dataset Examples[/bold green]")

    # Handle both single dataset and dictionary of datasets
    if not isinstance(dataset, dict):
        dataset = {"dataset": dataset}
    
    for split_name, split_dataset in dataset.items():
        console.print(f"\n[cyan]{split_name} examples:[/cyan]")
        # Randomly sample examples
        indices = random.sample(range(len(split_dataset)), min(num_examples, len(split_dataset)))
        
        for idx in indices:
            input_ids = split_dataset['input_ids'][idx]
            labels = split_dataset['labels'][idx]
            attention_mask = split_dataset['attention_mask'][idx]
            
            # Decode text from input IDs
            decoded_text = tokenizer.decode(input_ids[:sum(attention_mask)]) if tokenizer else "Tokenizer not provided"
            
            # Display in a panel
            panel = Panel(
                f"Text: {decoded_text}\nLabel: {labels}",
                title=f"Example {idx}",
                border_style="blue"
            )
            console.print(panel)

    console.print(f"="*100)

class LongformerClassifier:
    """
    A classifier based on the Longformer architecture for sequence classification tasks.
    """
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        num_labels: int = 2,
        max_length: int = 4096
    ):
        """
        Initialize the Longformer classifier.
        
        Args:
            model_name: Pretrained model name or path
            num_labels: Number of classification labels
            max_length: Maximum sequence length to handle
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Load model and tokenizer with status display
        with console.status("[bold green]Loading model and tokenizer..."):
            self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
            self.model = LongformerForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        console.print("[bold green]✓[/bold green] Model and tokenizer loaded")
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        training_args: TrainingArguments,
        callbacks: Optional[List] = None
    ) -> Trainer:
        """
        Train the Longformer classifier.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            training_args: Training configuration
            callbacks: Optional list of training callbacks
            
        Returns:
            Trained Trainer instance
        """
        # Set default callbacks if none provided
        if callbacks is None:
            callbacks = []
            
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        # Check for existing checkpoints and resume if available
        checkpoint_dir = pathlib.Path(training_args.output_dir)
        if list(checkpoint_dir.glob("checkpoint-*")):
            console.print("[yellow]Resuming from checkpoint...[/yellow]")
            trainer.train(resume_from_checkpoint=True)
        else:
            console.print("[green]Starting training from scratch...[/green]")
            trainer.train()

        console.print("[bold green]✓[/bold green] Model training completed")
        
        # Save the best model
        with console.status("[bold green]Saving model..."):
            trainer.save_model(training_args.output_dir)
            self.tokenizer.save_pretrained(training_args.output_dir)
        console.print("[bold green]✓[/bold green] Model saved")
        
        return trainer


def prepare_and_train_model(
    model_args: ModelArguments, 
    data_args: DataArguments, 
    training_args: TrainingArguments
) -> Tuple[LongformerClassifier, Dict[str, Dataset], Trainer]:
    """
    Prepare datasets and train the model.
    
    Args:
        model_args: Model configuration arguments
        data_args: Data configuration arguments
        training_args: Training configuration arguments
        
    Returns:
        Tuple containing the classifier, datasets, and trainer
    """
    # Initialize classifier
    classifier = LongformerClassifier(
        model_name=model_args.model_name_or_path,
        num_labels=model_args.num_labels,
        max_length=model_args.model_max_length
    )
    
    # Prepare datasets
    dataset_obj = DialogueClassificationDataset(
        model_name=classifier.model_name, 
        max_length=classifier.max_length
    )
    
    datasets = dataset_obj.make_dataset(
        train_file_path=data_args.train_file_path, 
        eval_file_path=data_args.eval_file_path
    )
    
    # Display dataset information
    display_dataset_stats(datasets)
    display_dataset_examples(datasets, tokenizer=classifier.tokenizer, num_examples=data_args.display_num_examples)
    
    # Train model
    trainer = classifier.train(
        train_dataset=datasets["train_dataset"],
        val_dataset=datasets["val_dataset"],
        training_args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    return classifier, datasets, trainer

def main():
    """
    Main function to parse arguments and run the training pipeline.
    """
    # Parse command line arguments
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        DataArguments, 
        TrainingArguments, 
    ))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Display parsed arguments
    display_arguments(model_args, data_args, training_args)
    
    # Prepare data and train model
    classifier, datasets, trainer = prepare_and_train_model(
        model_args, data_args, training_args
    )

if __name__ == "__main__":
    main()