import pandas as pd
from datasets import  Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


# for i in range(1,4):

# Initializing the model and the tokenizer
print("Loading model..")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained("openai-community/gpt2", num_labels=2)
model.config.pad_token_id = model.config.eos_token_id

# Reading the data
print("Loading data..")
df_train = pd.read_csv(f"datasets/dataset4_train.csv")
train_dataset = Dataset.from_pandas(df_train, preserve_index = False)
tokenized_train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], padding="max_length", max_length=128, truncation=True), batched=True)

# Creating HF Trainer
training_args = TrainingArguments(
    output_dir=f"savings/gpt2-savings-dataset4", 
    evaluation_strategy="no",
    per_device_train_batch_size=32)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset
)

# Train the model
print("Training..")
trainer.train()

# Save the model and the tokenizer
print("Saving model")
trainer.save_model(f'savings/models/gpt2-dataset4')
tokenizer.save_pretrained(f'savings/models/gpt2-dataset4')
