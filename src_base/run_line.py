from pathlib import Path
import yaml
from data import pull_data, read_data, split_df
from preprocess import build_tokenizer
from line_distil_bert.train_line import prepare_model, train
from line_distil_bert.eval_line import dev, test

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def create_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def run_line(cfg):
    """
    This function orchestrate the training and evaluation process.
    1. Split the data
    2. Build the tokenizer
    3. Train the model
    4. Evaluate the outcome
    """
    
    #1. Pull and read the dataset to split it into train/dev/test
    file_path = pull_data(cfg)
    df = read_data(file_path)
    train_df, dev_df, test_df, text, label = split_df(cfg, df)

    #2. Build the tokenizer
    train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels = build_tokenizer(cfg, train_df, dev_df, test_df, text, label, df)

    train_dl, dev_dl, test_dl, model = prepare_model(cfg, train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels)

    train(cfg, train_dl, model)

    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()

    dev(dev_dl, model)

    test(test_dl, model, out_dir)

if __name__ == "__main__":
    # Avoid creating a path whose parent becomes the current directory
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"
    cfg = load_yaml(CONFIG_PATH)

    # resolve() avoids creating a path whose parent becomes the current directory
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    create_dir(out_dir)

    run_line(cfg)