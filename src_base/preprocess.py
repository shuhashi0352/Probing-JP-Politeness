import torch
import yaml
from transformers import AutoTokenizer
import numpy as np

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_tokenizer(cfg, train, dev, test, text, label, df):

    tokenizer = cfg["tokenizer"]
    truncation = tokenizer["truncation"]
    return_tensors = tokenizer["return_tensors"]
    LineTokenizer = tokenizer["name"]
    trust_remote_code = tokenizer["trust_remote_code"]
    padding = tokenizer["padding_strategy"]


    dtype = torch.long

    tok = AutoTokenizer.from_pretrained(LineTokenizer, trust_remote_code=trust_remote_code)
    sentence_lengths = [len(tok.tokenize(sent)) for sent in df[text].dropna()]
    max_padding_length = int(np.percentile(sentence_lengths, 95))

    train_enc = tok(
        list(train[text].dropna()),
        padding=padding,
        truncation=truncation, 
        max_length=max_padding_length, 
        return_tensors=return_tensors)

    dev_enc = tok(
        list(dev[text].dropna()), 
        padding=padding, 
        truncation=truncation, 
        max_length=max_padding_length, 
        return_tensors=return_tensors)  

    test_enc = tok(
        list(test[text].dropna()), 
        padding=padding, 
        truncation=truncation, 
        max_length=max_padding_length, 
        return_tensors=return_tensors)  
    
    # Pytorch expects 4 classes RANGED FROM 0 to 3
    # NOT 1 to 4 as labeled in the dataset
    # So -1 for every label
    train_labels = torch.tensor(list(train[label]), dtype=dtype) - 1
    dev_labels = torch.tensor(list(dev[label]), dtype=dtype) - 1
    test_labels = torch.tensor(list(test[label]), dtype=dtype) - 1

    return train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels