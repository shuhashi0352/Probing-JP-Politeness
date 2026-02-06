import torch
from tqdm import tqdm

def extract_cls_by_layer(dataloader, model, device, desc="Extract CLS by layer"):
    model.eval()
    all_layer_cls = None
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch", total=len(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch["labels"].cpu().numpy()
            all_labels.append(labels)

            batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}

            outputs = model(**batch_no_labels, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # tuple (L, B, T, H)

            if all_layer_cls is None:
                all_layer_cls = [[] for _ in range(len(hidden_states))]

            for l, h in enumerate(hidden_states):
                cls_vec = h[:, 0, :]              # (B, H)
                all_layer_cls[l].append(cls_vec.cpu())

    X_layers = [torch.cat(chunks, dim=0).cpu().numpy() for chunks in all_layer_cls]  # list of (N,H)
    y = torch.from_numpy(__import__("numpy").concatenate(all_labels, axis=0)).numpy()
    return X_layers, y