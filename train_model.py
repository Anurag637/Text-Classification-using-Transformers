import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from model_utils import get_tokenizer, encode_data

# Load dataset
df = pd.read_csv("dataset.csv")
df = df[df['Text Content'] != "No content"]  # remove empty entries

# Binary classification as placeholder (replace with your task)
df["label"] = 0  # Or use some target values if applicable

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Huggingface Dataset
train_dataset = Dataset.from_pandas(train_df[['Text Content', 'label']])
val_dataset = Dataset.from_pandas(val_df[['Text Content', 'label']])

# Tokenization
tokenizer = get_tokenizer()
def tokenize_fn(batch):
    return tokenizer(batch['Text Content'], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save model
model.save_pretrained("model/trained_model")
tokenizer.save_pretrained("model/trained_model")
print("✅ Model trained and saved.")
