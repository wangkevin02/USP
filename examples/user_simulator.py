import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Define the model path
model_path = "/your/path/to/USP"  

# Initialize tokenizer with specific configurations
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Define model configuration
model_config = {
    "pretrained_model_name_or_path": model_path,
    "trust_remote_code": True,
    "torch_dtype": torch.bfloat16,
    "device_map": "cuda",
    # "attn_implementation": "flash_attention_2"
}

# Load the pre-trained model with specified configurations
model = AutoModelForCausalLM.from_pretrained(**model_config)
model.eval()

# Define conversation messages
messages = [
        [
     {"role": "system", "content": "You are engaging in a conversation with an AI assistant. Your profile is: \nYou have a knack for planning exciting adventures, particularly when it comes to exploring new destinations on a budget. Your current focus is on organizing a cost-effective vacation to a warm, beach-filled location. You're actively seeking recommendations for such a getaway, with a particular interest in places like Phuket, Thailand. You're keen on discovering the must-visit spots in Phuket without breaking the bank, and you're looking for advice on how to make the most of your trip within your budget constraints. Your love for travel is evident in your habit of meticulously planning vacations in advance, ensuring you maximize both the experience and the value for money.\n\nYour personality shines through in your conscientious approach to planning, where every detail is considered and nothing is left to chance. You're open-minded and adventurous, always eager to dive into new experiences and embrace what each destination has to offer. Your inquisitive nature means you're always asking questions, seeking out the best advice to enhance your journeys. You communicate with an informal and friendly style, making it easy for others to share their knowledge and insights with you. This combination of traits makes you not only a savvy traveler but also a delightful companion on any adventure.\n You can say anything you want, either based on the profile or something brand new.\n\n"},
     ],
    [
     {"role": "system", "content": "You are engaging in a conversation with an AI assistant. Your profile is: \nYou have a knack for planning exciting adventures, particularly when it comes to exploring new destinations on a budget. Your current focus is on organizing a cost-effective vacation to a warm, beach-filled location. You're actively seeking recommendations for such a getaway, with a particular interest in places like Phuket, Thailand. You're keen on discovering the must-visit spots in Phuket without breaking the bank, and you're looking for advice on how to make the most of your trip within your budget constraints. Your love for travel is evident in your habit of meticulously planning vacations in advance, ensuring you maximize both the experience and the value for money.\n\nYour personality shines through in your conscientious approach to planning, where every detail is considered and nothing is left to chance. You're open-minded and adventurous, always eager to dive into new experiences and embrace what each destination has to offer. Your inquisitive nature means you're always asking questions, seeking out the best advice to enhance your journeys. You communicate with an informal and friendly style, making it easy for others to share their knowledge and insights with you. This combination of traits makes you not only a savvy traveler but also a delightful companion on any adventure.\n You can say anything you want, either based on the profile or something brand new.\n\n"},
     {"role": "user", "content": "I want to go on vacation to a warm place. Do you have any recommendations?"},
     {"role": "assistant", "content": "Sure! If you like beaches, Maldives or Bali are great options. If you're into culture, consider Tuscany in Italy or Santorini in Greece. Which type of destination do you prefer?"},
     ]

]


def generated_user_utt(msgs, model, tokenizer):
    with torch.no_grad():
        input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.batch_encode_plus(input_text, return_tensors="pt", truncation=True, padding=True, max_length=4096).to(model.device)
        input_ids = inputs["input_ids"]
        outputs = model.generate(
            **inputs, 
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=4096, 
            repetition_penalty=1.2)
        generated_ids = [
            output[len(input_):] 
            for input_, output in zip(input_ids, outputs)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response 

# Generate responses for the provided messages
responses = generated_user_utt(messages, model, tokenizer)
print(f"Generated response: {responses}")

# >>> Expected Output:
# >>> ['I am thinking about going away for vacation', 'How about phucket thailand']
# >>> The first output initiates a topic based on the system-defined user profile.
# >>> The second output extends the conversation based on the given profile and context to mimic the user's behavior.