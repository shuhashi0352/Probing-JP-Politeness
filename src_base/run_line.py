from pathlib import Path
import yaml
from data import pull_data, read_data, split_df
from preprocess import build_tokenizer
from line_distil_bert.train_line import prepare_model, train
from line_distil_bert.eval_line import dev, test
from line_distil_bert.checkpoint import inspect_checkpoint
from probing.extract_cls import extract_cls_by_layer
from probing.visual import line_graph, heatmap, compare_ft_vs_probe_bar
from probing.probe_dev import layerwise_logreg_scores
from probing.probe_test_bestLayer import train_trdev_probe_and_eval_test

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
    3. Train the model (Save/Load the checkpoint)
    4. Make the checkpoint visualized
    5. (Optionally) Evaluate the outcome using dev
    6, Get the test scores for comparison
    7. Extract hidden states from train(optional) and dev
    8. Probing using logistic regression (Get the macro-f1 for every layer) to get the best layer
    9. Visualize the scores layerwise
    10. Test the score on the best layer selected in 8.
    11. Compare the score to the fine-tune test score
    """
    
    #1. Pull and read the dataset to split it into train/dev/test
    file_path = pull_data(cfg)
    df = read_data(file_path)
    train_df, dev_df, test_df, text, label = split_df(cfg, df)

    #2. Build the tokenizer
    train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels = build_tokenizer(cfg, train_df, dev_df, test_df, text, label, df)

    train_dl, dev_dl, test_dl, model, device = prepare_model(cfg, train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels)

    #3. For fine-tuning
    train(cfg, train_dl, model, device)

    #4. Checkpoint inspection
    inspect_checkpoint()

    #5. Evaluation (optional)
    # dev(dev_dl, model, device)

    #6. Test for fine-tune (optional for probing)
    # However, you can compare the score on this test to the one with proving
    # If the score on finetune is nearly equal to the one on probing,
    # it implies that the information on politeness is quite explicit at the best layer even in a linear fashion
    accuracy_ft, macro_f1_ft = test(test_dl, model, out_dir)

    #7. Hidden state extraction
    X_train_layers, y_train = extract_cls_by_layer(train_dl, model, device, desc="Extract train")
    X_dev_layers, y_dev = extract_cls_by_layer(dev_dl, model, device, desc="Extract dev")

    #8. Layerwise probing
    dev_f1_macro_by_layer, best_layer, best_f1_macro = layerwise_logreg_scores(X_train_layers, y_train, X_dev_layers, y_dev)
    print(f"[Layerwise probing] Best layer = {best_layer} (dev macro-F1 = {best_f1_macro:.3f})")

    #9. Visualize the score
    # line_graph(dev_f1_macro_by_layer)
    # heatmap(dev_f1_macro_by_layer)

    #10. Test on the best layer
    X_test_layers, y_test = extract_cls_by_layer(test_dl, model, device, desc="Extract test")
    probe, accuracy_pr, macro_f1_pr = train_trdev_probe_and_eval_test(X_train_layers, y_train, X_test_layers, y_test, out_dir, best_layer=best_layer, C=1.0)

    #11. Compare the probing score (best layer) to the finetune test score (all layers passed)
    compare_ft_vs_probe_bar(accuracy_ft, macro_f1_ft, accuracy_pr, macro_f1_pr)

if __name__ == "__main__":
    # Avoid creating a path whose parent becomes the current directory
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"
    cfg = load_yaml(CONFIG_PATH)

    # resolve() avoids creating a path whose parent becomes the current directory
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    create_dir(out_dir)

    run_line(cfg)