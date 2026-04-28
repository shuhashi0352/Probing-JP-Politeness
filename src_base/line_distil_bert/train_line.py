import yaml
from transformers import get_scheduler
from tqdm import tqdm
from torch.optim import AdamW
import torch
from pathlib import Path
    
def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train(cfg, train_dl, model, device, dev_dl=None):
    import torch
    from tqdm import tqdm

    model.to(device)

    epochs = cfg["task"].get("epochs", 3)
    lr = cfg["task"].get("lr", 2e-5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0

        pbar = tqdm(train_dl, desc=f"Classifier train epoch {epoch}/{epochs}", unit="batch")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            batch_size = batch["labels"].size(0)

            total_train_loss += loss.item() * batch_size
            total_train_examples += batch_size

            preds = logits.argmax(dim=-1)
            total_train_correct += (preds == batch["labels"]).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })

        avg_train_loss = total_train_loss / total_train_examples
        train_acc = total_train_correct / total_train_examples

        if dev_dl is not None:
            model.eval()

            total_dev_loss = 0.0
            total_dev_correct = 0
            total_dev_examples = 0

            with torch.no_grad():
                for batch in tqdm(dev_dl, desc=f"Classifier dev epoch {epoch}/{epochs}", unit="batch"):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    outputs = model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits

                    batch_size = batch["labels"].size(0)

                    total_dev_loss += loss.item() * batch_size
                    total_dev_examples += batch_size

                    preds = logits.argmax(dim=-1)
                    total_dev_correct += (preds == batch["labels"]).sum().item()

            avg_dev_loss = total_dev_loss / total_dev_examples
            dev_acc = total_dev_correct / total_dev_examples

            print(
                f"[Classifier epoch {epoch:03d}/{epochs}] "
                f"train_loss={avg_train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"dev_loss={avg_dev_loss:.4f} "
                f"dev_acc={dev_acc:.4f}",
                flush=True,
            )

        else:
            print(
                f"[Classifier epoch {epoch:03d}/{epochs}] "
                f"train_loss={avg_train_loss:.4f} "
                f"train_acc={train_acc:.4f}",
                flush=True,
            )

    return model