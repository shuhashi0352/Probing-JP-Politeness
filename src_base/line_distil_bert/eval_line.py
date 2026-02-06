from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score
import json

def dev(dev_dl, model, device):
    # set the model to evaluating mode
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dev_dl, desc="Dev Per Batch", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items()}
            # outputs = model(**batch)
            outputs = model(**batch)

            # Convert the tensor to the numpy array for the sake of scikit-learn
            preds = torch.argmax(outputs.logits, dim=1).numpy()
            labels = batch["labels"].numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")

def test(test_dl, model, out_dir):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Test Per Batch", unit="batch"):
            batch = {key: val for key, val in batch.items()}
            outputs = model(**batch)

            preds = torch.argmax(outputs.logits, dim=1)
            labels = batch["labels"]

            # Json module doesn't understand Numpy integer (int64)
            # Converts them to python int
            all_preds.extend([int(x) for x in preds.detach().cpu().tolist()])
            all_labels.extend([int(x) for x in labels.detach().cpu().tolist()])

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")

    results = {
        "metrics": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
        },
        "n_examples": int(len(all_labels)),
        "y_true": all_labels,
        "y_pred": all_preds,
    }

    out_path = out_dir / "finetune_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")

    return accuracy, f1_macro