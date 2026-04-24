import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def train_trdev_probe_and_eval_test(
    X_train_layers, y_train,
    X_dev_layers, y_dev,
    X_test_layers, y_test,
    out_dir,
    best_layer,
    C=0.1,
    max_iter=2000,
    seed=42,
):
    """
    After selecting best_layer on dev, re-train the probe on train+dev
    and evaluate once on test.
    """
    print("\n[Probe] Evaluating on test...\n")

    X_trdev = np.concatenate(
        [X_train_layers[best_layer], X_dev_layers[best_layer]],
        axis=0,
    )
    y_trdev = np.concatenate([y_train, y_dev], axis=0)

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            C=C,
            random_state=seed,
        )
    )

    probe.fit(X_trdev, y_trdev)

    pred_test = probe.predict(X_test_layers[best_layer])
    acc_test = accuracy_score(y_test, pred_test)
    f1_test = f1_score(y_test, pred_test, average="macro")

    results = {
        "best_layer": int(best_layer),
        "C": float(C),
        "max_iter": int(max_iter),
        "n_train_dev": int(len(y_trdev)),
        "n_test": int(len(y_test)),
        "test_accuracy": float(acc_test),
        "test_macro_f1": float(f1_test),
    }

    out_path = out_dir / "probing_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return probe, acc_test, f1_test