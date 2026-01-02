import os
import sys
import yaml
import tarfile
import urllib.request
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from pull_data import pull_data, read_data
from preprocess import build_tokenizer
from amlp.train_amlp import train_amlp
from amlp.eval_amlp import eval_amlp

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def download_file(url, dst_path):
    # if Path exists and not empty
    if dst_path.exists() and dst_path.stat().st_size > 0:
        print("The model already exists. Skip downloading...")
        return

    create_dir(dst_path.parent)
    print(f"Downloading {url} to {dst_path}...")

    # Save the content
    urllib.request.urlretrieve(url, dst_path)
    print("Download complete")


def extract_tar_bz2(archive_path, extract_to):
    create_dir(extract_to)
    print(f"Extracting {archive_path} in {extract_to}...")
    # Decompress the archive
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete")


def prepare_model(cfg):
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    url = model_cfg["download_url"]

    """
    1. Download the archive
    """
    # Create the directory for the archive to be downloaded
    out_dir = Path(cfg["experiment"]["output_dir"])
    create_dir(out_dir)

    # aMLP-base-ja: https://nama.ne.jp/models/aMLP-base-ja.tar.bz2
    archive_path = out_dir / f"{model_name}.tar.bz2"
    download_file(url, archive_path)

    """
    2. Objectify the archive and allocate it in an appropriate field
    """
    # Create the path for the model
    model_dir = Path(model_cfg["model_dir"]).resolve()

    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model directory already exists: {model_dir}")
        return

    # The path for where to decompress the archive
    extract_to = model_dir.parent
    extract_tar_bz2(archive_path, extract_to)

    if not model_dir.exists():
        raise FileNotFoundError("Expected model directory not found. Check aMLP-base-ja.tar.bz2 and config.yaml")

def confirm_split_df(cfg, df):

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

    return train, dev, test

def run_amlp(cfg):
    """
    This function orchestrate the training and evaluation process.
    1. Split the data
    2. Build the tokenizer
    3. Train the model
    4. Evaluate the outcome
    """
    
    #1. Pull and read the dataset to split it into train/dev/test
    file_path = pull_data()
    df = read_data(file_path)
    train_df, dev_df, test_df = confirm_split_df(cfg, df)

    #2. Build the tokenizer
    tokenizer = build_tokenizer()

    #3. Train the model
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    model_ckpt = train_amlp(cfg, train_df, dev_df, out_dir)

    #4. Evaluate the outcome
    # metrics = eval_amlp(cfg, tokenizer, model_ckpt, test_df, out_dir)

    # return metrics



def main():
    # Avoid creating a path whose parent becomes the current directory
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"
    cfg = load_yaml(CONFIG_PATH)

    # resolve() avoids creating a path whose parent becomes the current directory
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    create_dir(out_dir)

    # Set up the model
    prepare_model(cfg)

    # Train and evaluate the model
    run_amlp(cfg)

if __name__ == "__main__":
    main()
