import requests
import os
import pandas as pd

def pull_data():
    """
    Pull and save data from Keico corpus to a .csv file
    """

    url = "https://raw.githubusercontent.com/Liumx2020/KeiCO-corpus/main/keico_corpus(forLREC)-OldVersion.csv"
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
    
