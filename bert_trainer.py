import pandas as pd
from datasets import  Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


for i in range(1,4):

    # Initializing the model and the tokenizer
    print("Loading model..")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=3)

    # Reading the data
    print("Loading data..")
    df_train = pd.read_csv(f"datasets/dataset{i}_train.csv")
    train_dataset = Dataset.from_pandas(df_train, preserve_index = False)
    tokenized_train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], padding="max_length", max_length=128, truncation=True), batched=True)

    # Creating HF Trainer
    training_args = TrainingArguments(
        output_dir=f"savings/bert-savings-dataset{i}", 
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
    trainer.save_model(f'savings/models/bert-dataset{i}')
    tokenizer.save_pretrained(f'savings/models/bert-dataset{i}')
