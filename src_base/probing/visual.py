from pathlib import Path
import json
import csv

import matplotlib.pyplot as plt
import numpy as np


def load_das_layer_results(json_path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_das_layer_results_csv(all_results, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "hs_index",
        "sklearn_probe_f1",
        "torch_probe_acc",
        "das_target_acc",
        "das_flip_rate",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in all_results:
            writer.writerow({
                "hs_index": row["hs_index"],
                "sklearn_probe_f1": row.get("sklearn_probe_f1"),
                "torch_probe_acc": row.get("torch_probe_acc"),
                "das_target_acc": row.get("das_target_acc"),
                "das_flip_rate": row.get("das_flip_rate"),
            })


def plot_layerwise_probe_scores(all_results, out_path):
    """
    Plot sklearn probe F1 and PyTorch probe accuracy across layers.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [r["hs_index"] for r in all_results]
    sklearn_scores = [r["sklearn_probe_f1"] for r in all_results]
    torch_scores = [r["torch_probe_acc"] for r in all_results]

    plt.figure(figsize=(8, 5))
    plt.plot(layers, sklearn_scores, marker="o", label="sklearn probe F1")
    plt.plot(layers, torch_scores, marker="o", label="PyTorch probe accuracy")
    plt.xlabel("Hidden-state index")
    plt.ylabel("Score")
    plt.title("Layerwise Probe Performance")
    plt.xticks(layers)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_layerwise_das_scores(all_results, out_path):
    """
    Plot DAS target accuracy and flip rate across layers.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [r["hs_index"] for r in all_results]
    target_acc = [r["das_target_acc"] for r in all_results]
    flip_rate = [r["das_flip_rate"] for r in all_results]

    plt.figure(figsize=(8, 5))
    plt.plot(layers, target_acc, marker="o", label="DAS target accuracy")
    plt.plot(layers, flip_rate, marker="o", label="DAS flip rate")
    plt.xlabel("Hidden-state index")
    plt.ylabel("Score")
    plt.title("Layerwise DAS Intervention Results")
    plt.xticks(layers)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_probe_vs_das_target(all_results, out_path):
    """
    Compare PyTorch probe accuracy and DAS target accuracy.
    This is useful because DAS is optimized against the PyTorch probe, not sklearn.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [r["hs_index"] for r in all_results]
    torch_scores = [r["torch_probe_acc"] for r in all_results]
    das_scores = [r["das_target_acc"] for r in all_results]

    plt.figure(figsize=(8, 5))
    plt.plot(layers, torch_scores, marker="o", label="PyTorch probe accuracy")
    plt.plot(layers, das_scores, marker="o", label="DAS target accuracy")
    plt.xlabel("Hidden-state index")
    plt.ylabel("Score")
    plt.title("Probe Readout vs DAS Intervention Accuracy")
    plt.xticks(layers)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_das_transition_heatmap(
    transition_counts,
    out_path,
    title="DAS Transition Counts",
    class_names=("unnatural", "natural"),
    normalize=None,
):
    """
    transition_counts format:
        [[0->0, 0->1],
         [1->0, 1->1]]

    normalize:
        None   -> raw counts
        "row"  -> row-normalized
        "all"  -> all-count normalized
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mat = np.array(transition_counts, dtype=float)

    if normalize == "row":
        denom = mat.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        display_mat = mat / denom
    elif normalize == "all":
        denom = mat.sum()
        display_mat = mat / denom if denom > 0 else mat
    elif normalize is None:
        display_mat = mat
    else:
        raise ValueError("normalize must be None, 'row', or 'all'.")

    plt.figure(figsize=(5, 4))
    plt.imshow(display_mat)
    plt.colorbar()

    plt.xticks(range(len(class_names)), [f"patched {c}" for c in class_names], rotation=30, ha="right")
    plt.yticks(range(len(class_names)), [f"base {c}" for c in class_names])

    for i in range(display_mat.shape[0]):
        for j in range(display_mat.shape[1]):
            if normalize is None:
                text = str(int(display_mat[i, j]))
            else:
                text = f"{display_mat[i, j]:.2f}"
            plt.text(j, i, text, ha="center", va="center")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def get_best_layer_result(all_results, key="das_target_acc"):
    if len(all_results) == 0:
        raise ValueError("all_results is empty.")

    return max(all_results, key=lambda r: r[key])


def plot_best_layer_transition_heatmaps(all_results, out_dir, key="das_target_acc"):
    """
    Save raw and row-normalized transition heatmaps for the best DAS layer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = get_best_layer_result(all_results, key=key)
    hs_index = best["hs_index"]
    transition_counts = best["transition_counts"]

    plot_das_transition_heatmap(
        transition_counts,
        out_dir / f"best_layer{hs_index}_transition_counts.png",
        title=f"Best DAS Layer {hs_index}: Transition Counts",
        normalize=None,
    )

    plot_das_transition_heatmap(
        transition_counts,
        out_dir / f"best_layer{hs_index}_transition_row_normalized.png",
        title=f"Best DAS Layer {hs_index}: Row-Normalized Transitions",
        normalize="row",
    )


def plot_all_layer_transition_heatmaps(all_results, out_dir, normalize=None):
    """
    Save one transition heatmap per layer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "counts" if normalize is None else f"{normalize}_normalized"

    for result in all_results:
        hs_index = result["hs_index"]
        transition_counts = result["transition_counts"]

        plot_das_transition_heatmap(
            transition_counts,
            out_dir / f"layer{hs_index}_transition_{suffix}.png",
            title=f"Layer {hs_index}: DAS Transition {suffix}",
            normalize=normalize,
        )


def write_best_layer_summary(all_results, out_path, key="das_target_acc"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = get_best_layer_result(all_results, key=key)

    lines = []
    lines.append("# Best DAS Layer Summary")
    lines.append("")
    lines.append(f"- selection_key: `{key}`")
    lines.append(f"- hs_index: `{best['hs_index']}`")
    lines.append(f"- sklearn_probe_f1: `{best.get('sklearn_probe_f1'):.4f}`")
    lines.append(f"- torch_probe_acc: `{best.get('torch_probe_acc'):.4f}`")
    lines.append(f"- das_target_acc: `{best.get('das_target_acc'):.4f}`")
    lines.append(f"- das_flip_rate: `{best.get('das_flip_rate'):.4f}`")
    lines.append("")
    lines.append("## Transition Counts")
    lines.append("")
    lines.append("Rows = base prediction, columns = patched prediction.")
    lines.append("")
    lines.append("```text")
    lines.append(str(best["transition_counts"]))
    lines.append("```")
    lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_das_visualizations(all_results, out_dir):
    """
    Main convenience function.

    Expected all_results item format:
        {
          "hs_index": 1,
          "sklearn_probe_f1": 0.65,
          "torch_probe_acc": 0.64,
          "das_target_acc": 0.60,
          "das_flip_rate": 0.44,
          "transition_counts": [[...], [...]]
        }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_das_layer_results_csv(
        all_results,
        out_dir / "das_all_layers_results.csv",
    )

    plot_layerwise_probe_scores(
        all_results,
        out_dir / "layerwise_probe_scores.png",
    )

    plot_layerwise_das_scores(
        all_results,
        out_dir / "layerwise_das_scores.png",
    )

    plot_probe_vs_das_target(
        all_results,
        out_dir / "probe_vs_das_target_accuracy.png",
    )

    plot_best_layer_transition_heatmaps(
        all_results,
        out_dir,
        key="das_target_acc",
    )

    plot_all_layer_transition_heatmaps(
        all_results,
        out_dir / "all_layer_transition_counts",
        normalize=None,
    )

    plot_all_layer_transition_heatmaps(
        all_results,
        out_dir / "all_layer_transition_row_normalized",
        normalize="row",
    )

    write_best_layer_summary(
        all_results,
        out_dir / "best_layer_summary.md",
        key="das_target_acc",
    )


# ------------------------------------------------------------
# Add this to run_das.py after saving all_results
# ------------------------------------------------------------

"""
from probing.visual_das import generate_das_visualizations

viz_dir = Path(cfg["data"]["das_out_dir"]) / "figures"

generate_das_visualizations(
    all_results=all_results,
    out_dir=viz_dir,
)
"""


# ------------------------------------------------------------
# Example for your current result file
# ------------------------------------------------------------

"""
from probing.visual_das import load_das_layer_results, generate_das_visualizations

all_results = load_das_layer_results("results_das/das_all_layers_results.json")

generate_das_visualizations(
    all_results=all_results,
    out_dir="results_das/figures",
)
"""