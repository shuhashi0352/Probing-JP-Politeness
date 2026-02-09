import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import json


def train_trdev_probe_and_eval_test(
    X_train_layers, y_train,
    X_test_layers, y_test,
    out_dir,
    best_layer,
    C=1.0,
    max_iter=2000
):
    """
    Re-train probes with train+dev using logistic regression.
    Return the accuracy and f1 score
    """
    print("[Probe] Evaluating on test ...")

    X_tr = X_train_layers[best_layer]
    y_tr = y_train

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            C=C,
        )
    )
    probe.fit(X_tr, y_tr)

    pred_test = probe.predict(X_test_layers[best_layer])
    acc_test = accuracy_score(y_test, pred_test)
    f1_test = f1_score(y_test, pred_test, average="macro")

    results = {
        "best_layer": int(best_layer),
        "C": float(C),
        "max_iter": int(max_iter),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_accuracy": float(acc_test),
        "test_macro_f1": float(f1_test),
    }


    out_path = out_dir / "probing_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return probe, acc_test, f1_test