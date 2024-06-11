from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import json

# Load the pre-trained TinyLlama model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare dataset
with open("E:/cods/aiwifu/wifuwork.json", encoding="utf-8") as f:
    data = json.load(f)

# Custom dataset class
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, conversations, tokenizer, max_length=128):
        self.inputs = []
        self.labels = []
        self.max_length = max_length
        for conversation in conversations:
            for response in conversation["responses"]:
                input_text = str(conversation["input"])
                response_text = str(response)

                # Tokenize input and response texts with padding and truncation
                input_encodings = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
                response_encodings = tokenizer(response_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)

                self.inputs.append(input_encodings)
                self.labels.append(response_encodings)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"].squeeze(),
            "attention_mask": self.inputs[idx]["attention_mask"].squeeze(),
            "labels": self.labels[idx]["input_ids"].squeeze()
        }

# Create dataset and data loader
dataset = ChatDataset(data["conversations"], tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=data_collator)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    no_cuda=True,  # Ensure training happens on CPU
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_model_tinyllama")
tokenizer.save_pretrained("./fine_tuned_model_tinyllama")

# Generate responses
def generate_response(prompt, tokenizer, model, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# Test the model
def test_model(model, tokenizer):
    # Example prompt
    prompt = "Hey bro, should I go watch KonoSuba? I've already watched it 10 times."
    
    # Generate response
    response = generate_response(prompt, tokenizer, model)
    
    # Print the prompt and generated response
    print("Prompt:", prompt)
    print("Generated response:", response)

if __name__ == "__main__":
    # Load fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model_tinyllama")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model_tinyllama")
    
    # Test the model
    test_model(model, tokenizer)
