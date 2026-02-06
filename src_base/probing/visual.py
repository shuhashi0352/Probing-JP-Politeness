import numpy as np
import matplotlib.pyplot as plt

def line_graph(dev_f1_macro): # dev_f1_macro: macro-F1 per layer
    layers = list(range(len(dev_f1_macro)))
    plt.figure()
    plt.plot(layers, dev_f1_macro)
    plt.xlabel("Layer")
    plt.ylabel("Dev Macro-F1")
    plt.title("Layerwise Probing Performance")
    plt.show()

def heatmap(dev_f1_macro):
    scores = np.array(dev_f1_macro)[None, :]  # (1, num_layers)
    plt.figure()
    plt.imshow(scores, aspect="auto")
    plt.yticks([0], ["macro-F1"])
    plt.xticks(range(len(dev_f1_macro)), range(len(dev_f1_macro)))
    plt.xlabel("Layer")
    plt.title("Layerwise Probing Heatmap")
    plt.colorbar()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def compare_ft_vs_probe_bar(
    accuracy_ft, macro_f1_ft,
    accuracy_pr, macro_f1_pr,
    title="Fine-tune vs Probe (Test Metrics)"
):
    """
    One figure, grouped bars:
      x-axis = metrics (Accuracy, Macro-F1)
      bars = Fine-tune vs Probe
    """

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
    plt.show()