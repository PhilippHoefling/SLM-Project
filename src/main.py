
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load your short jokes dataset
dataset_path = "C:/Users/phili/DataspellProjects/SLM-Project/data/shortjokes.csv"

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# Load dataset and tokenize
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=dataset_path,
    block_size=128  # Adjust as needed
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=1,
    save_steps=1_000,
    save_total_limit=2,
    logging_steps=100,
    logging_dir="./logs",
    evaluation_strategy="steps",
    eval_steps=1_000,
    report_to="tensorboard",
    fp16=False,  # Enable mixed precision training if your GPU supports it
    gradient_accumulation_steps=1,
    dataloader_num_workers=4,
    load_best_model_at_end=True,
)

# Instantiate Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("models/finetuned")
#%%
torch.cuda.empty_cache()
#%%
