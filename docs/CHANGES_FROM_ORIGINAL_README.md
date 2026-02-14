# Code Changes Compared to Original Repo

This document describes **exactly** what code was added or modified compared to the original [Probing-JP-Politeness](https://github.com/shuhashi0352/Probing-JP-Politeness) repository.

**Reference baseline:** `Probing-JP-Politeness-reference/` (fresh clone from GitHub, unmodified).

---

## Verification: Files that differ (clean vs reference)

| File / folder | Status |
|---------------|--------|
| `src_base/run_line.py` | **Modified** in clean |
| `src_base/probing/visual.py` | **Modified** in clean |
| `results_lineDistilBERT/` | **Only in clean** (output dir; contains this README) |

**Unchanged:** All other files are identical between clean and reference, including:
- `probing/probe_dev.py`, `probe_test_bestLayer.py`, `extract_cls.py`, `patching.py`, `utils.py`
- `data.py`, `preprocess.py`, `line_distil_bert/` (train_line, eval_line, checkpoint)
- Causality test logic, StandardScaler in probes, etc.

---

## Summary

The only additions are:

1. **C sweep** for the logistic regression probe: try C ∈ {0.01, 0.1, 1, 10, 100}, pick best by dev macro F1, use best C for test evaluation.
2. **Saving plots to files** instead of (or in addition to) displaying with `plt.show()`.
3. **Fix for undefined `out_dir`** in `run_line()` (it was used but not defined in the original).

No other functions or logic were changed.

---

## 1. `src_base/run_line.py`

### 1a. Import

```diff
 from pathlib import Path
+import json
 import yaml
```

### 1b. Add `out_dir` at start of `run_line()` (fix for undefined variable)

```diff
 def run_line(cfg):
     """
     ...
     """
+   out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
+   create_dir(out_dir)
+
     #1. Pull and read the dataset...
     file_path = pull_data(cfg)
```

### 1c. Replace single probing call with C sweep

**Original:**
```python
    #8. Layerwise probing
    dev_f1_macro_by_layer, best_layer, best_f1_macro = layerwise_logreg_scores(X_train_layers, y_train, X_dev_layers, y_dev)
    print(f"\n[Layerwise probing] Best layer = {best_layer} (dev macro-F1 = {best_f1_macro:.3f})")

    #9. Visualize the score
    line_graph(dev_f1_macro_by_layer)
    # heatmap(dev_f1_macro_by_layer)   # was commented out

    #10. Test on the best layer
    X_test_layers, y_test = extract_cls_by_layer(test_dl, model, device, desc="Extract test")
    probe, accuracy_pr, macro_f1_pr = train_trdev_probe_and_eval_test(X_train_layers, y_train, X_test_layers, y_test, out_dir, best_layer=best_layer, C=1.0)

    #11. Compare
    compare_ft_vs_probe_bar(accuracy_ft, macro_f1_ft, accuracy_pr, macro_f1_pr)
```

**Modified:**
```python
    #8. Layerwise probing with C sweep
    C_VALUES = [0.01, 0.1, 1, 10, 100]
    seed = cfg["experiment"].get("seed", 42)
    dev_results = []
    for C in C_VALUES:
        dev_f1_macro_by_layer, best_layer, best_f1_macro = layerwise_logreg_scores(
            X_train_layers, y_train, X_dev_layers, y_dev, C=C
        )
        print(f"C={C}: best_layer={best_layer}, dev macro-F1={best_f1_macro:.4f}")
        dev_results.append((C, best_layer, best_f1_macro))

    print("\n--- Dev results per C ---")
    for C, bl, f1 in dev_results:
        print(f"  C={C}: best_layer={bl}, dev_f1_macro={f1:.4f}")

    best_C, best_layer, _ = max(dev_results, key=lambda x: x[2])
    dev_f1_macro_by_layer, _, _ = layerwise_logreg_scores(
        X_train_layers, y_train, X_dev_layers, y_dev, C=best_C
    )
    print(f"\n[Layerwise probing] Best C={best_C}, best layer={best_layer}")

    c_sweep_path = out_dir / "C_sweep_dev_results.json"
    with c_sweep_path.open("w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "per_C": [{"C": C, "best_layer": int(bl), "dev_f1_macro": float(f1)} for C, bl, f1 in dev_results],
            "best_C": best_C,
            "best_layer": int(best_layer),
        }, f, indent=2)
    print(f"Saved: {c_sweep_path}")

    #9. Visualize the score
    line_graph(dev_f1_macro_by_layer, out_path=out_dir / "dev_f1_by_layer_line.png", title=f"Dev Macro F1 by Layer (C={best_C})")
    heatmap(dev_f1_macro_by_layer, out_path=out_dir / "dev_f1_by_layer_heatmap.png", title=f"Dev Macro F1 by Layer (C={best_C})")

    #10. Test on the best layer
    X_test_layers, y_test = extract_cls_by_layer(test_dl, model, device, desc="Extract test")
    probe, accuracy_pr, macro_f1_pr = train_trdev_probe_and_eval_test(
        X_train_layers, y_train, X_test_layers, y_test, out_dir, best_layer=best_layer, C=best_C
    )

    #11. Compare
    compare_ft_vs_probe_bar(accuracy_ft, macro_f1_ft, accuracy_pr, macro_f1_pr, out_path=out_dir / "ft_vs_probe_bar.png")
```

Here are the differences:
- C sweep loop over `[0.01, 0.1, 1, 10, 100]`
- Best C chosen by max dev macro F1
- `C_sweep_dev_results.json` saved
- `line_graph` and `heatmap` called with `out_path` and `title` (heatmap was previously commented out)
- `train_trdev_probe_and_eval_test` uses `C=best_C` instead of `C=1.0`
- `compare_ft_vs_probe_bar` called with `out_path`

---

## 2. `src_base/probing/visual.py`

### 2a. `line_graph` — add optional save to file

**Original:**
```python
def line_graph(dev_f1_macro):
    ...
    plt.title("Layerwise Probing Performance")
    plt.show()
```

**Modified:**
```python
def line_graph(dev_f1_macro, out_path=None, title="Layerwise Probing Performance"):
    ...
    plt.title(title)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()
```

### 2b. `heatmap` — save to file

**Original:**
```python
def heatmap(dev_f1_macro):
    ...
    plt.title("Layerwise Probing Heatmap")
    plt.colorbar()
    plt.show()
```

**Modified:**
```python
def heatmap(dev_f1_macro, out_path=None, title="Layerwise Probing Heatmap"):
    ...
    plt.title(title)
    plt.colorbar()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()
```

### 2c. `compare_ft_vs_probe_bar` — save to file

**Original:**
```python
def compare_ft_vs_probe_bar(
    accuracy_ft, macro_f1_ft,
    accuracy_pr, macro_f1_pr,
    title="Fine-tune vs Probe (Test Metrics)"
):
    ...
    fig.tight_layout()
    plt.show()
```

**Modified:**
```python
def compare_ft_vs_probe_bar(
    accuracy_ft, macro_f1_ft,
    accuracy_pr, macro_f1_pr,
    title="Fine-tune vs Probe (Test Metrics)",
    out_path=None,
):
    ...
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()
```

---

## New output files

| File | Description |
|------|-------------|
| `C_sweep_dev_results.json` | Dev results per C and selected best C |
| `dev_f1_by_layer_line.png` | Line plot (was displayed only, now saved) |
| `dev_f1_by_layer_heatmap.png` | Heatmap (was commented out, now saved) |
| `ft_vs_probe_bar.png` | Bar chart (was displayed only, now saved) |

`probing_results.json` now uses the best C from the sweep instead of fixed `C=1.0`.

---

## Unchanged from original

- **Causality test** (patching, self-patch, random-patch, wrong-layer) — identical in clean and reference
- **Probe implementation** — `StandardScaler` + `LogisticRegression` in `probing/probe_dev.py` and `probe_test_bestLayer.py`
- **Best-layer selection** — same logic (`argmax` over dev macro F1 per layer); only the C value is now swept
