import pandas as pd
import re
import emoji
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import  Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def evaluate_model(model, tokenizer, test_dataset, batch_size=32, device='cpu'):
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    # Iterate over the test dataset in batches
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch['text'].to_list(), 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Move tensors to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get the true labels
        labels = torch.tensor(batch['labels'].to_list()).to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            # Assuming the model returns logits
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')  # Change 'average' as needed
    recall = recall_score(all_labels, all_preds, average='weighted')  # Change 'average' as needed
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Change 'average' as needed
    
    return accuracy, precision, recall, f1

for i in range(1,4):

    # Initializing the model and the tokenizer
    print("Loading model..")
    tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("xlnet/xlnet-base-cased", num_labels=3)

    # Reading the data
    print("Loading data..")
    df_train = pd.read_csv(f"dataset{i}_train.csv")
    train_dataset = Dataset.from_pandas(df_train, preserve_index = False)
    tokenized_train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], padding="max_length", max_length=128, truncation=True), batched=True)

    # Creating HF Trainer
    training_args = TrainingArguments(
        output_dir=f"savings/bert-savings-dataset{i}", 
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8)

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
    trainer.save_model(f'savings/models/bert-dataset{i}')
    tokenizer.save_pretrained(f'savings/models/bert-dataset{i}')

    print("Calculating metrics on test set..")
    df_test = pd.read_csv(f"dataset{i}_train.csv")
    test_dataset = Dataset.from_pandas(df_test, preserve_index = False)
    tokenized_test_dataset = test_dataset.map(lambda x: tokenizer(x["text"], padding="max_length", max_length=128, truncation=True), batched=True)
    accuracy, precision, recall, f1 = evaluate_model(model, tokenizer, df_test,)

    

