import torch
import yaml
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_tokenizer(cfg, train_df, dev_df, test_df):
    tok = cfg["tokenizer"]

    truncation = tok["truncation"]
    return_tensors = tok["return_tensors"]
    model_name = tok["name"]
    trust_remote_code = tok["trust_remote_code"]
    padding = tok["padding_strategy"]
    max_length = tok["max_length"]

    text_col = cfg["data"]["text_col"]
    label_col = cfg["data"]["label_id_col"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    for name, split in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        if split[text_col].isna().any():
            raise ValueError(f"{name} split contains missing text values.")
        if split[label_col].isna().any():
            raise ValueError(f"{name} split contains missing label values.")

    train_enc = tokenizer(
        list(train_df[text_col]),
        padding=padding,
        truncation=truncation, 
        max_length=max_length, 
        return_tensors=return_tensors)

    dev_enc = tokenizer(
        list(dev_df[text_col]), 
        padding=padding, 
        truncation=truncation, 
        max_length=max_length, 
        return_tensors=return_tensors)  

    test_enc = tokenizer(
        list(test_df[text_col]), 
        padding=padding, 
        truncation=truncation, 
        max_length=max_length, 
        return_tensors=return_tensors)  
    
    train_labels = torch.tensor(train_df[label_col].tolist(), dtype=torch.long)
    dev_labels = torch.tensor(dev_df[label_col].tolist(), dtype=torch.long)
    test_labels = torch.tensor(test_df[label_col].tolist(), dtype=torch.long)

    print(train_enc)
    print(train_labels.shape)

    return tokenizer, train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels