import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class StandardizedLinearProbe(nn.Module):
    def __init__(self, hidden_size, num_labels=2):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.classifier(x)


def fit_standardizer(X_train, eps=1e-8):
    if not torch.is_tensor(X_train):
        X_train = torch.tensor(X_train, dtype=torch.float32)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp(min=eps)

    return mean, std


def apply_standardizer(X, mean, std):
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    return (X - mean) / std

def mean_pool_hidden_torch(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    return summed / lengths


def train_pooled_vector_probe(
    cfg,
    probe,
    x_train,
    y_train,
    device,
    x_dev=None,
    y_dev=None,
):
    """
    Train a linear probe on pooled context+quote vectors.

    Prints train/dev loss and accuracy for each epoch if dev data is provided.

    x_train: np.ndarray, shape (N, D)
    y_train: np.ndarray, shape (N,)
    x_dev:   optional np.ndarray, shape (N_dev, D)
    y_dev:   optional np.ndarray, shape (N_dev,)
    """

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    probe = probe.to(device)

    # Standardize using train statistics only
    train_mean = x_train.mean(axis=0, keepdims=True)
    train_std = x_train.std(axis=0, keepdims=True)
    train_std = np.where(train_std == 0, 1.0, train_std)

    x_train_std = (x_train - train_mean) / train_std

    x_train_tensor = torch.tensor(x_train_std, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    batch_size = cfg["das"].get("probe_batch_size", cfg["task"]["batch_size"])
    epochs = cfg["das"].get("probe_epochs", 20)
    lr = cfg["das"].get("probe_lr", 1e-3)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    if x_dev is not None and y_dev is not None:
        x_dev_std = (x_dev - train_mean) / train_std
        x_dev_tensor = torch.tensor(x_dev_std, dtype=torch.float32)
        y_dev_tensor = torch.tensor(y_dev, dtype=torch.long)

        dev_dataset = TensorDataset(x_dev_tensor, y_dev_tensor)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        dev_loader = None

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        probe.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = probe(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            batch_size_actual = xb.size(0)
            total_train_loss += loss.item() * batch_size_actual

            preds = logits.argmax(dim=-1)
            total_train_correct += (preds == yb).sum().item()
            total_train_examples += batch_size_actual

        avg_train_loss = total_train_loss / total_train_examples
        train_acc = total_train_correct / total_train_examples

        if dev_loader is not None:
            probe.eval()

            total_dev_loss = 0.0
            total_dev_correct = 0
            total_dev_examples = 0

            with torch.no_grad():
                for xb, yb in dev_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    logits = probe(xb)
                    loss = criterion(logits, yb)

                    batch_size_actual = xb.size(0)
                    total_dev_loss += loss.item() * batch_size_actual

                    preds = logits.argmax(dim=-1)
                    total_dev_correct += (preds == yb).sum().item()
                    total_dev_examples += batch_size_actual

            avg_dev_loss = total_dev_loss / total_dev_examples
            dev_acc = total_dev_correct / total_dev_examples

            print(
                f"[Probe epoch {epoch:03d}/{epochs}] "
                f"train_loss={avg_train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"dev_loss={avg_dev_loss:.4f} "
                f"dev_acc={dev_acc:.4f}",
                flush=True,
            )

        else:
            print(
                f"[Probe epoch {epoch:03d}/{epochs}] "
                f"train_loss={avg_train_loss:.4f} "
                f"train_acc={train_acc:.4f}",
                flush=True,
            )

    return probe, train_mean, train_std


def eval_pooled_vector_probe(
    cfg,
    probe,
    X,
    y,
    device,
    mean,
    std,
):
    batch_size = cfg["tr_probe"]["batch_size"]

    X = apply_standardizer(X, mean, std)
    y = torch.tensor(y, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    probe.eval()
    total_correct = 0
    total_n = 0

    preds = []
    golds = []

    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)

            logits = probe(x)
            pred = logits.argmax(dim=1)

            total_correct += (pred == labels).sum().item()
            total_n += labels.size(0)

            preds.append(pred.cpu())
            golds.append(labels.cpu())

    return {
        "accuracy": total_correct / total_n,
        "preds": torch.cat(preds),
        "golds": torch.cat(golds),
    }