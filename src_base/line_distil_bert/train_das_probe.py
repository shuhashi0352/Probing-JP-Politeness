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
    X_train,
    y_train,
    device,
    mean=None,
    std=None,
):
    lr = cfg["tr_probe"]["lr"]
    epochs = cfg["tr_probe"]["epochs"]
    batch_size = cfg["tr_probe"]["batch_size"]

    if mean is None or std is None:
        mean, std = fit_standardizer(X_train)

    X_train = apply_standardizer(X_train, mean, std)
    y_train = torch.tensor(y_train, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    probe = probe.to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
        weight_decay=cfg["tr_probe"]["weight_decay"],
    )

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for x, labels in tqdm(dataloader, desc=f"Probe epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = probe(x)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (pred == labels).sum().item()
            total_n += labels.size(0)

        print(
            f"epoch={epoch+1} "
            f"loss={total_loss / total_n:.4f} "
            f"acc={total_correct / total_n:.4f}"
        )

    return probe, mean, std


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