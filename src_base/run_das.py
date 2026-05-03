from pathlib import Path
import yaml
import json
from data import split_data, prepare_model, make_das_dataloaders
from preprocess import build_tokenizer
from probing.extract_hs import run_extraction
from probing.probe_dev import layerwise_logreg_scores
from line_distil_bert.das import get_distilbert_layer_module, train_das, eval_das
from line_distil_bert.train_das_probe import StandardizedLinearProbe, train_pooled_vector_probe, eval_pooled_vector_probe, apply_standardizer

# For finetuning (if needed)
from line_distil_bert.train_line import train
from line_distil_bert.eval_line import dev, test

# Visualization
from probing.visual import load_das_layer_results, generate_das_visualizations


def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def create_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def run_das(cfg):
    train_df, dev_df, test_df, label2id = split_data(cfg)

    tokenizer, train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels = build_tokenizer(cfg, train_df, dev_df, test_df)
    train_dataloader, dev_dataloader, test_dataloader, model, device, model_num_layers = prepare_model(cfg, train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels)

    train(cfg, train_dataloader, model, device, dev_dl=dev_dataloader)

    (x_train_layers, x_train_layers_full, train_masks, y_train, train_context_masks, train_quote_masks, x_train_cq_layers) = run_extraction(train_dataloader, model, device, train_df["text"].tolist(), train_enc, tokenizer, desc="Train hidden states")
    (x_dev_layers, x_dev_layers_full, dev_masks,y_dev, dev_context_masks, dev_quote_masks, x_dev_cq_layers) = run_extraction(dev_dataloader, model, device, dev_df["text"].tolist(), dev_enc, tokenizer, desc="Dev hidden states")
    (x_test_layers, x_test_layers_full, test_masks, y_test, test_context_masks, test_quote_masks, x_test_cq_layers) = run_extraction(test_dataloader, model, device, test_df["text"].tolist(), test_enc, tokenizer, desc="Test hidden states")

    probes_cq, dev_scores_cq, best_layer_cq, best_score_cq, best_probe_cq = layerwise_logreg_scores(x_train_cq_layers, y_train, x_dev_cq_layers, y_dev, C=0.1, seed=cfg["experiment"]["seed"])
    print("best_laye_q: ", best_layer_cq, "best_score_q: ", best_score_cq)
    print(dev_scores_cq)

    # hidden_size = model.config.dim * 2

    # torch_probe = StandardizedLinearProbe(hidden_size, num_labels=2)
    # torch_probe, probe_mean, probe_std = train_pooled_vector_probe(cfg, torch_probe, x_train_cq_layers[best_layer_cq], y_train, device)
    # dev_probe_results = eval_pooled_vector_probe(cfg, torch_probe, x_dev_cq_layers[best_layer_cq], y_dev, device, probe_mean, probe_std)
    # print("PyTorch pooled-vector probe dev:", dev_probe_results["accuracy"])

    # ##### DAS #####

    train_receiver_dl, train_donor_dl, train_receiver_df, train_donor_df = make_das_dataloaders(train_df, tokenizer, batch_size=cfg["task"]["batch_size"], max_length=cfg["tokenizer"]["max_length"])

    dev_receiver_dl, dev_donor_dl, dev_receiver_df, dev_donor_df = make_das_dataloaders(dev_df, tokenizer, batch_size=cfg["task"]["batch_size"], max_length=cfg["tokenizer"]["max_length"])

    # layer_module = get_distilbert_layer_module(model, best_layer_cq)

    # das = train_das(cfg, model, torch_probe, probe_mean, probe_std, train_receiver_dl, train_donor_dl, layer_module, best_layer_cq, hidden_size, device)
    # dev_das_results = eval_das(cfg, model, torch_probe, probe_mean, probe_std, dev_receiver_dl, dev_donor_dl, layer_module, best_layer_cq, das, device)
    # print(dev_das_results)

    all_results = []

    for hs_index in range(1, model_num_layers + 1):
        print(f"\n===== DAS layer {hs_index} =====")

        hidden_size = model.config.dim * 2

        torch_probe = StandardizedLinearProbe(hidden_size, num_labels=2)

        torch_probe, probe_mean, probe_std = train_pooled_vector_probe(
            cfg,
            torch_probe,
            x_train_cq_layers[hs_index],
            y_train,
            device,
            x_dev=x_dev_cq_layers[hs_index],
            y_dev=y_dev,
        )

        dev_probe_results = eval_pooled_vector_probe(
            cfg,
            torch_probe,
            x_dev_cq_layers[hs_index],
            y_dev,
            device,
            probe_mean,
            probe_std,
        )

        layer_module = get_distilbert_layer_module(model, hs_index)

        das = train_das(
            cfg,
            model,
            torch_probe,
            probe_mean,
            probe_std,
            train_receiver_dl,
            train_donor_dl,
            layer_module,
            hs_index,
            hidden_size,
            device,
        )

        dev_das_results = eval_das(
            cfg,
            model,
            torch_probe,
            probe_mean,
            probe_std,
            dev_receiver_dl,
            dev_donor_dl,
            layer_module,
            hs_index,
            das,
            device,
        )

        all_results.append({
            "hs_index": hs_index,
            "sklearn_probe_f1": float(dev_scores_cq[hs_index]),
            "torch_probe_acc": float(dev_probe_results["accuracy"]),
            "das_target_acc": float(dev_das_results["patched_target_accuracy"]),
            "das_flip_rate": float(dev_das_results["flip_rate"]),
            "transition_counts": dev_das_results["transition_counts"],
        })

    out_path = Path(cfg["data"]["das_out_dir"]) / "das_all_layers_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    result_path = cfg["data"]["das_out_dir"]
    generate_das_visualizations(all_results, out_dir=f"{result_path}/figures")


if __name__ == "__main__":
    # Avoid creating a path whose parent becomes the current directory
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"
    cfg = load_yaml(CONFIG_PATH)

    # resolve() avoids creating a path whose parent becomes the current directory
    out_dir = Path(cfg["data"]["das_out_dir"]).resolve()
    create_dir(out_dir)

    run_das(cfg)