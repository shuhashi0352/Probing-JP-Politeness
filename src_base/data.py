import requests
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def pull_data(cfg):
    """
    Pull and save data from Keico corpus to a .csv file
    """

    url = cfg["data"]["url"]
    file_name = url.split('/')[-1]

    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)

    if not os.path.exists(file_path):
        print('Downloading data...')
        with requests.get(url) as r:
            with open(file_path, 'wb') as f:
                f.write(r.content)
        
        print('Download successful')
    
    else:
        print('Data already exists. Skip downloading...')
    
    return file_path

def read_data(file_path):
    return pd.read_csv(file_path)

def split_df(cfg, df):

    text = cfg["data"]["text_col"]
    label = cfg["data"]["label_col"]
    train_size = cfg["experiment"]["train_size"]
    ratio_dev_test = cfg["experiment"]["ratio_dev_test"]
    seed = cfg["experiment"]["seed"]

    # Raise Error if data lacks any required columns
    missing = [c for c in [text, label] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    train, test = train_test_split(df, train_size=train_size, random_state=seed)
    test, dev = train_test_split(test, train_size=ratio_dev_test, random_state=seed)

    return train, dev, test, text, label
    
