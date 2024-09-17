import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm  # For progress bar
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Assuming you have a PyTorch model, tokenizer, and a DataFrame 'df' with columns 'inputs' and 'labels'
# Replace these with your actual model, tokenizer, and DataFrame
def evaluate_model_from_df(model, tokenizer, df, batch_size=32, device='cpu'):
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    # Iterate over the DataFrame in batches
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch['text'].tolist(),  # Tokenizer expects a list of texts
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get the true labels
        labels = torch.tensor(batch['labels'].tolist()).to(device)
        
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
    
    return all_preds, accuracy, precision, recall, f1


for i in range(1,4):

    # Initializing the model and the tokenizer
    print("Loading model..")

    # tokenizer_bert = AutoTokenizer.from_pretrained(f"savings/models/bert-dataset{i}")
    # model_bert = AutoModelForSequenceClassification.from_pretrained(f"savings/models/bert-dataset{i}", num_labels=3)

    print(f"Loading GPT2 trained on 'dataset{i}_train.csv'")
    tokenizer_gpt2 = AutoTokenizer.from_pretrained(f"savings/models/gpt2-dataset{i}")
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
    model_gpt2 = AutoModelForSequenceClassification.from_pretrained(f"savings/models/gpt2-dataset{i}", num_labels=3)
    model_gpt2.config.pad_token_id = model_gpt2.config.eos_token_id

    # tokenizer_xlnet = AutoTokenizer.from_pretrained(f"savings/models/xlnet-dataset{i}")
    # model_xlnet = AutoModelForSequenceClassification.from_pretrained(f"savings/models/xlnet-dataset{i}", num_labels=3)

    df = pd.read_csv(f"dataset{i}_test.csv")

    predictions_gpt2, accuracy_gpt2, precision_gpt2, recall_gpt2, f1_gpt2 = evaluate_model_from_df(model_gpt2, tokenizer_gpt2, df)

    print(f"Metrics for GPT2 on 'dataset{i}_test.csv'")
    print("Acc:", accuracy_gpt2)
    print("precision:", precision_gpt2)
    print("recall:", recall_gpt2)
    print("f1:", f1_gpt2)