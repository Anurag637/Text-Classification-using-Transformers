# app/streamlit_app.py

import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

# Use pretrained Hugging Face model directly, or change path to your local model if trained
model_path = "bert-base-uncased"  # or "./model/checkpoints" if saved locally

# Load model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

st.title("Text Classification using Transformers")

user_input = st.text_area("Enter text to classify", "")

if st.button("Classify") and user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    st.write(f"**Prediction:** Class {prediction}")
    st.write(f"**Confidence:** {confidence:.2%}")
