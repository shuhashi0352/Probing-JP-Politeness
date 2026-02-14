import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def line_graph(dev_f1_macro, out_path=None, title="Layerwise Probing Performance"):  # dev_f1_macro: macro-F1 per layer
    print("\nGenerating line graph...\n")
    layers = list(range(len(dev_f1_macro)))
    plt.figure()
    plt.plot(layers, dev_f1_macro)
    plt.xlabel("Layer")
    plt.ylabel("Dev Macro-F1")
    plt.title(title)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()

def heatmap(dev_f1_macro, out_path=None, title="Layerwise Probing Heatmap"):
    print("\nGenerating heatmap...\n")
    scores = np.array(dev_f1_macro)[None, :]  # (1, num_layers)
    plt.figure()
    plt.imshow(scores, aspect="auto")
    plt.yticks([0], ["macro-F1"])
    plt.xticks(range(len(dev_f1_macro)), range(len(dev_f1_macro)))
    plt.xlabel("Layer")
    plt.title(title)
    plt.colorbar()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()

import matplotlib.pyplot as plt
import numpy as np

def compare_ft_vs_probe_bar(
    accuracy_ft, macro_f1_ft,
    accuracy_pr, macro_f1_pr,
    title="Fine-tune vs Probe (Test Metrics)",
    out_path=None,
):
    """
    One figure, grouped bars:
      x-axis = metrics (Accuracy, Macro-F1)
      bars = Fine-tune vs Probe
    """

    print("\nGenerating bar graph...\n")

    # Cast to plain floats (avoid numpy scalar quirks)
    accuracy_ft = float(accuracy_ft)
    macro_f1_ft = float(macro_f1_ft)
    accuracy_pr = float(accuracy_pr)
    macro_f1_pr = float(macro_f1_pr)

    metrics = ["Accuracy", "Macro-F1"]
    ft_vals = [accuracy_ft, macro_f1_ft]
    pr_vals = [accuracy_pr, macro_f1_pr]

    x = np.arange(len(metrics))      # [0, 1]
    width = 0.35

    fig, ax = plt.subplots()
    bars_ft = ax.bar(x - width/2, ft_vals, width, label="Fine-tune")
    bars_pr = ax.bar(x + width/2, pr_vals, width, label="Probe")

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Value labels on bars
    for bars in (bars_ft, bars_pr):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom"
            )

    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()

def plot_transition_heatmap(
    transition_counts,
    out_path=None,
    class_names=None,
    normalize=None,   # None | "row" | "all"
    title=None,
    annotate=True,
    dpi=300,
    name=None
):
    """
    Plot 4x4 transition matrix heatmap for patching results.

    Args:
        transition_counts: 4x4 list (or np.array) where [i][j] = # (base=i -> patched=j)
        out_path: str|Path|None. If provided, saves figure to this path.
        class_names: list of 4 strings for axis tick labels. If None, uses ["0","1","2","3"].
        normalize: None (counts), "row" (row-wise %), "all" (global %)
        title: figure title. If None, auto title based on normalize.
        annotate: whether to print values in each cell.
        dpi: save dpi.

    Returns:
        fig, ax
    """
    M = np.array(transition_counts, dtype=float)
    if M.shape != (4, 4):
        raise ValueError(f"transition_counts must be 4x4, got {M.shape}")

    # normalization
    disp = M.copy()
    if normalize == "row":
        row_sums = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, row_sums, out=np.zeros_like(disp), where=row_sums != 0) * 100.0
        value_fmt = "{:.1f}%"
        cbar_label = "Row-normalized (%)"
        default_title = "Transition matrix (row-normalized)"
    elif normalize == "all":
        total = disp.sum()
        disp = (disp / total * 100.0) if total != 0 else disp
        value_fmt = "{:.1f}%"
        cbar_label = "Global (%)"
        default_title = "Transition matrix (global-normalized)"
    elif normalize is None:
        value_fmt = "{:.0f}"
        cbar_label = "Count"
        default_title = f"Transition matrix (counts) - {name}"
    else:
        raise ValueError("normalize must be None, 'row', or 'all'")

    if class_names is None:
        class_names = [str(i) for i in range(4)]
    if len(class_names) != 4:
        raise ValueError("class_names must have length 4")

    if title is None:
        title = default_title

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(disp, interpolation="nearest", aspect="equal")

    # axes
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Patched prediction")
    ax.set_ylabel("Baseline prediction")
    ax.set_title(title)

    # grid lines (subtle)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    # annotations
    if annotate:
        for i in range(4):
            for j in range(4):
                ax.text(j, i, value_fmt.format(disp[i, j]), ha="center", va="center")

    fig.tight_layout()

    # save
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_transition_heatmap_from_json(
    json_path,
    out_path=None,
    class_names=None,
    normalize=None,
    title=None,
    annotate=True,
    dpi=300,
    name=None
):
    """
    Convenience wrapper: read patching_results.json and plot its transition_counts.
    """

    print("\nCreating heatmap...\n")
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "transition_counts" not in data:
        raise KeyError("transition_counts not found in json")

    return plot_transition_heatmap(
        transition_counts=data["transition_counts"],
        out_path=out_path,
        class_names=class_names,
        normalize=normalize,
        title=title,
        annotate=annotate,
        dpi=dpi,
        name=name
    )