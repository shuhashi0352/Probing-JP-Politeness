import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

def layerwise_logreg_scores(X_train_layers, y_train, X_dev_layers, y_dev, C=1.0, desc="Layerwise probes"):
    print("\nSelecting the best layer..\n")
    dev_f1_macro = []
    for l in tqdm(range(len(X_train_layers)), desc=desc, unit="layer"):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                multi_class="multinomial",
                max_iter=2000,
                C=C,
            )
        )
        clf.fit(X_train_layers[l], y_train)
        pred = clf.predict(X_dev_layers[l])
        dev_f1_macro.append(f1_score(y_dev, pred, average="macro"))

    dev_f1_macro_by_layer = np.array(dev_f1_macro) # shape: (num_layers,)

    best_layer = int(np.argmax(dev_f1_macro_by_layer))
    best_score = float(dev_f1_macro_by_layer[best_layer])

    return dev_f1_macro_by_layer, best_layer, best_score