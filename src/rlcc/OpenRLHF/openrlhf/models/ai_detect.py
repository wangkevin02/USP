from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
from typing import Dict, List
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import nn


from typing import List, Dict
import torch
import torch.nn.functional as F
from transformers import LongformerTokenizer, LongformerForSequenceClassification

class AIDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
        num_labels: int = 2,
        device_map: str = "auto",
        batch_size: int = 32
    ):
        super(AIDetector, self).__init__()
        self.max_length = max_length        
        # Initialize tokenizer and model
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            device_map=device_map
        )
        
        # Set model to evaluation mode and configure tokenizer
        self.model.eval()
        self.tokenizer.padding_side = "right"  # Set padding once during initialization
        self.batch_size = batch_size
        
        # Enable torch inference optimizations
        torch.backends.cudnn.benchmark = True
        
    @torch.no_grad()  # Disable gradient computation for inference
    def get_probability(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Get classification probabilities for input texts."""
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]
            
        # Batch process texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.model.device)
        
        # Get model outputs
        outputs = self.model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }

    @torch.no_grad()
    def calculate_reward(
        self,
        generated_utterances: List[List[str]]
    ) -> torch.Tensor:
        """
        Calculate reward based on probability of class 0 for generated texts.
        
        Args:
            generated_utterances: 二维列表，形状为[batch_size, num_turns]
            
        Returns:
            torch.Tensor: 重组后的奖励值，与输入的二维结构对应
        """
        batch_size = len(generated_utterances)
        # [turn1, turn2, turn3, ...]   [batch_size,]     
        turns_per_sample = [len(utterance) for utterance in generated_utterances]
        
        flattened_utterances = [
            utterance
            for sample in generated_utterances
            for utterance in sample
        ]
        
        all_rewards = []
        for i in range(0, len(flattened_utterances), self.batch_size):
            batch = flattened_utterances[i:i + self.batch_size]
            results = self.get_probability(batch)
            batch_rewards = results['probabilities'][:, 0]  # 获取class 0的概率
            all_rewards.append(batch_rewards)
        
        # [batch_size * num_turns]
        flattened_rewards = torch.cat(all_rewards)
        
        reconstructed_rewards = []
        start_idx = 0
        for num_turns in turns_per_sample:
            # different number of turns for each sample
            sample_rewards = flattened_rewards[start_idx:start_idx + num_turns]
            reconstructed_rewards.append(sample_rewards)
            start_idx += num_turns
        # [batch_size, num_turns]   
        return reconstructed_rewards

    def __call__(self, texts: List[str]) -> torch.Tensor:
        """Convenience method to calculate rewards directly."""
        return self.calculate_reward(texts)


# 示例使用
if __name__ == "__main__":
    classifier = AIDetector(
        model_name="/home/wangkuang/workshop/git/User_Simulator/AI_detect/longformer_classifier_output/checkpoint-13000",
        max_length=4096,
        num_labels=2
    )
    
    target_text = ["This is a human-written sentence.","hello there","write a sentence for me"]
    generated_text = ["How can I help you?","what can you do for me?","Here’s the corrected and refined explanation in English: Corrected Approach to Solve the Problem The array A is first decreasing and then increasing, and we need to accurately locate the" ]
    result = classifier.calculate_loss(target_text, generated_text)
    print("Loss calculation result:")
    print(f"Loss: {result}")