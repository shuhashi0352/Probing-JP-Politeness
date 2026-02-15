from pathlib import Path
import yaml
from data import pull_data, read_data, split_df, split_donor_receiver_df
from preprocess import build_tokenizer
from line_distil_bert.train_line import prepare_model, make_dataloader, train
from line_distil_bert.eval_line import dev, test
from line_distil_bert.checkpoint import inspect_checkpoint
from probing.extract_cls import extract_cls_by_layer, extract_cls_at_layer
from probing.visual import line_graph, heatmap, compare_ft_vs_probe_bar, plot_transition_heatmap_from_json
from probing.probe_dev import layerwise_logreg_scores
from probing.probe_test_bestLayer import train_trdev_probe_and_eval_test
from probing.utils import get_encoder_layer_module
from probing.patching import causal_cls_patching

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
    all_df = read_data(file_path)
    train_df, dev_df, test_df, text, label = split_df(cfg, all_df)

    #2. Build the tokenizer
    train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels = build_tokenizer(cfg, train_df, dev_df, test_df, text, label, all_df)

    train_dl, dev_dl, test_dl, model, device, model_num_layers = prepare_model(cfg, train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels)

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
    dev_f1_macro_by_layer, best_layer, best_f1_macro = layerwise_logreg_scores(X_train_layers, y_train, X_dev_layers, y_dev, C=0.1)
    print(f"\n[Layerwise probing] Best layer = {best_layer} (dev macro-F1 = {best_f1_macro:.3f})")

    #9. Visualize the score
    line_graph(dev_f1_macro_by_layer, out_path=out_dir / "dev_f1_by_layer_line.png", title=f"Dev Macro F1 by Layer")
    heatmap(dev_f1_macro_by_layer, out_path=out_dir / "dev_f1_by_layer_heatmap.png", title=f"Dev Macro F1 by Layer")

    #10. Test on the best layer
    X_test_layers, y_test = extract_cls_by_layer(test_dl, model, device, desc="Extract test")
    probe, accuracy_pr, macro_f1_pr = train_trdev_probe_and_eval_test(X_train_layers, y_train, X_test_layers, y_test, out_dir, best_layer=best_layer, C=0.1)

    #11. Compare the probing score (best layer) to the finetune test score (all layers passed)
    compare_ft_vs_probe_bar(accuracy_ft, macro_f1_ft, accuracy_pr, macro_f1_pr, out_path=out_dir / "ft_vs_probe_bar.png")

    """
    Causality Test
    """

    print("\nCAUSALITY TEST - PATCHING\n")

    train_donor_df, train_receiver_df = split_donor_receiver_df(train_df, label, donor_label=1, receiver_label=4)
    dev_donor_df, dev_receiver_df = split_donor_receiver_df(dev_df, label, donor_label=1, receiver_label=4)
    test_donor_df, test_receiver_df = split_donor_receiver_df(test_df, label, donor_label=1, receiver_label=4)

    print("Loading donor and receiver...")

    _, donor_dev_enc, donor_test_enc, _, donor_dev_labels, donor_test_labels = build_tokenizer(cfg, train_donor_df, dev_donor_df, test_donor_df, text, label, all_df)
    _, receiver_dev_enc, receiver_test_enc, _, receiver_dev_labels, receiver_test_labels = build_tokenizer(cfg, train_receiver_df, dev_receiver_df, test_receiver_df, text, label, all_df)

    donor_dev_dl = make_dataloader(donor_dev_enc, donor_dev_labels, cfg)
    receiver_dev_dl = make_dataloader(receiver_dev_enc, receiver_dev_labels, cfg)

    encoder_layer_idx = best_layer - 1
    layer_module = get_encoder_layer_module(model, layer_idx=encoder_layer_idx)

    patch_results_dev = causal_cls_patching(model, receiver_dev_dl, layer_module, device, out_dir, donor_dl=donor_dev_dl, mode="paired", hs_index=best_layer, target_class_idx=0, out_path="patching_results_dev.json", random_donor=False, seed=42, data="dev")

    plot_transition_heatmap_from_json(json_path=out_dir / "patching_results_dev.json", out_path=out_dir / "transition_heatmap_counts_dev.png", class_names=["1(polite)", "2", "3", "4(casual)"], normalize=None, name="Dev")

    donor_test_dl = make_dataloader(donor_test_enc, donor_test_labels, cfg)
    receiver_test_dl = make_dataloader(receiver_test_enc, receiver_test_labels, cfg)

    patch_results_test = causal_cls_patching(model, receiver_test_dl, layer_module, device, out_dir, donor_dl=donor_test_dl, mode="paired", hs_index=best_layer, target_class_idx=0, out_path="patching_results_test.json", random_donor=False, seed=42, data="test")
    plot_transition_heatmap_from_json(json_path=out_dir / "patching_results_test.json", out_path=out_dir / "transition_heatmap_counts_test.png", class_names=["1(polite)", "2", "3", "4(casual)"], normalize=None, name="Test")

    """
    Controls for patching
    1. Self-patch: donor_batch = receiver_batch
    2. Random patch: shuffle donor CLS within label=0
    3. Wrong-layer patch: patching 0 to 3 for all the layers
    """
    # 1. Self-patch
    print("Control - Self-patch")
    self_train_donor_df, self_train_receiver_df = split_donor_receiver_df(train_df, label, donor_label=1, receiver_label=1)
    self_dev_donor_df, self_dev_receiver_df = split_donor_receiver_df(dev_df, label, donor_label=1, receiver_label=1)
    self_test_donor_df, self_test_receiver_df = split_donor_receiver_df(test_df, label, donor_label=1, receiver_label=1)

    print("Loading donor and receiver...")
    _, _, self_donor_test_enc, _, _, self_donor_test_labels = build_tokenizer(cfg, self_train_donor_df, self_dev_donor_df, self_test_donor_df, text, label, all_df)
    _, _, self_receiver_test_enc, _, _, self_receiver_test_labels = build_tokenizer(cfg, self_train_receiver_df, self_dev_receiver_df, self_test_receiver_df, text, label, all_df)

    self_receiver_test_dl = make_dataloader(self_receiver_test_enc, self_receiver_test_labels, cfg)

    self_patch_results_test = causal_cls_patching(model, self_receiver_test_dl, layer_module, device, out_dir, donor_dl=None, mode="self", hs_index=best_layer, target_class_idx=0, out_path="self_patching_results_test.json", random_donor=False, seed=42, data="control(self)")
    plot_transition_heatmap_from_json(json_path=out_dir / "self_patching_results_test.json", out_path=out_dir / "self_transition_heatmap_counts_test.png", class_names=["1(polite)", "2", "3", "4(casual)"], normalize=None, name="Self-patch")

    # Random Patch
    print("Control - Random patch")
    print("Data already set up")
    random_patch_results_test = causal_cls_patching(model, receiver_test_dl, layer_module, device, out_dir, donor_dl=donor_test_dl, mode="paired", hs_index=best_layer, target_class_idx=0, out_path="random_patching_results_test.json", random_donor=True, seed=42, data="control(random)")
    plot_transition_heatmap_from_json(json_path=out_dir / "random_patching_results_test.json", out_path=out_dir / "random_transition_heatmap_counts_test.png", class_names=["1(polite)", "2", "3", "4(casual)"], normalize=None, name="Random Patch")

    # Wrong-layer patch
    print("\nControl - Wrong patch")
    count = 0
    for layer_idx in range(model_num_layers):
        layer_module = get_encoder_layer_module(model, layer_idx=layer_idx)
        hs_idx = layer_idx + 1
        patch_results = causal_cls_patching(model, receiver_test_dl, layer_module, device, out_dir, donor_dl=donor_test_dl, mode="paired", hs_index=hs_idx, target_class_idx=0, out_path=f"patching_results_test_layer{layer_idx}.json", random_donor=False, seed=42, data="control(wrong-layer)")
        plot_transition_heatmap_from_json(json_path=out_dir / f"patching_results_test_layer{layer_idx}.json", out_path=out_dir / f"transition_heatmap_counts_test_layer{layer_idx}.png", class_names=["1(polite)", "2", "3", "4(casual)"], normalize=None, name=f"Wrong Patch at layer {count + 1}")
        count += 1


if __name__ == "__main__":
    # Avoid creating a path whose parent becomes the current directory
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"
    cfg = load_yaml(CONFIG_PATH)

    # resolve() avoids creating a path whose parent becomes the current directory
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    create_dir(out_dir)

    run_line(cfg)
