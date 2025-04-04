from transformers import AutoTokenizer

def get_tokenizer(model_name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

def encode_data(texts, tokenizer, max_len=128):
    return tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
