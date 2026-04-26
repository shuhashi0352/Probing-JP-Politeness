import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from pathlib import Path
from line_distil_bert.train_das_probe import mean_pool_hidden_torch

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class DASSubspace(nn.Module):
    def __init__(self, hidden_size, k):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k

        # Trainable rotation matrix
        self.R_raw = nn.Parameter(torch.eye(hidden_size))

    def orthogonal_R(self):
        # QR keeps R approximately orthogonal
        Q, _ = torch.linalg.qr(self.R_raw)
        return Q
    
    def mix(self, receiver_vec, donor_vec):
        R = self.orthogonal_R()

        z_r = receiver_vec @ R
        z_d = donor_vec @ R

        z_mix = torch.cat(
            [
                z_d[..., :self.k],
                z_r[..., self.k:],
            ],
            dim=-1,
        )

        mixed = z_mix @ R.T
        return mixed
    
def run_with_das_patch_return_hidden(
    model,
    receiver_no_labels,
    donor_no_labels,
    layer_module,
    hs_index,
    das_module,
):
    with torch.no_grad():
        donor_out = model(
            **donor_no_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        donor_hidden = donor_out.hidden_states[hs_index].detach()  # (B, T, H)

    patched_hidden_container = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output  # (B, T, H)

        # all-token DAS mix
        mixed_hs = das_module.mix(
            hs,
            donor_hidden.to(hs.device),
        )  # (B, T, H)

        # keep padding positions unchanged
        attn = receiver_no_labels["attention_mask"].to(hs.device).unsqueeze(-1).float()
        patched_hs = mixed_hs * attn + hs * (1.0 - attn)

        patched_hidden_container["hidden"] = patched_hs

        if isinstance(output, tuple):
            return (patched_hs,) + output[1:]
        return patched_hs

    handle = layer_module.register_forward_hook(hook)
    try:
        _ = model(**receiver_no_labels, return_dict=True)
    finally:
        handle.remove()

    return patched_hidden_container["hidden"]

def train_das(cfg, model, torch_probe, probe_mean, probe_std, receiver_dl, donor_dl, layer_module, hs_index, hidden_size, device):
    """
    Train DAS rotation so that patched receiver predicts donor/counterfactual label.

    Assumption:
    - receiver_dl and donor_dl are aligned pairwise
    - donor labels are the target labels after intervention
    """

    k = cfg["das"]["k"]
    lr = cfg["das"]["lr"]
    epochs = cfg["das"]["epochs"]

    das = DASSubspace(hidden_size=hidden_size, k=k).to(device)
    optimizer = torch.optim.AdamW(das.parameters(), lr=lr)

    model.eval()
    torch_probe.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in torch_probe.parameters():
        p.requires_grad = False

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        donor_iter = iter(donor_dl)

        for receiver_batch in tqdm(receiver_dl, desc=f"DAS epoch {epoch+1}/{epochs}"):
            try:
                donor_batch = next(donor_iter)
            except StopIteration:
                donor_iter = iter(donor_dl)
                donor_batch = next(donor_iter)

            receiver_batch = {k: v.to(device) for k, v in receiver_batch.items()}
            donor_batch = {k: v.to(device) for k, v in donor_batch.items()}

            receiver_no_labels = {
                k: v for k, v in receiver_batch.items()
                if k not in ["labels", "offset_mapping"]
            }
            donor_no_labels = {
                k: v for k, v in donor_batch.items()
                if k not in ["labels", "offset_mapping"]
            }

            target_labels = donor_batch["labels"]

            B = min(receiver_no_labels["input_ids"].size(0), donor_no_labels["input_ids"].size(0))

            receiver_no_labels = {k: v[:B] for k, v in receiver_no_labels.items()}
            donor_no_labels = {k: v[:B] for k, v in donor_no_labels.items()}
            target_labels = target_labels[:B]

            optimizer.zero_grad()

            patched_hidden = run_with_das_patch_return_hidden(model, receiver_no_labels, donor_no_labels, layer_module, hs_index, das)

            patched_vec = mean_pool_hidden_torch(patched_hidden, receiver_no_labels["attention_mask"])
            probe_mean = probe_mean.to(device)
            probe_std = probe_std.to(device)
            patched_vec = (patched_vec - probe_mean) / probe_std
            logits = torch_probe(patched_vec)
            loss = F.cross_entropy(logits, target_labels)

            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            total_correct += (pred == target_labels).sum().item()
            total_loss += loss.item() * B
            total_n += B

        print(
            f"epoch={epoch+1} "
            f"loss={total_loss / max(total_n, 1):.4f} "
            f"acc={total_correct / max(total_n, 1):.4f}"
        )

    return das

def get_hs_at_layer(model, batch_no_labels, hs_index):

    out = model(**batch_no_labels, output_hidden_states=True, return_dict=True)

    return out.hidden_states[hs_index]

def eval_das(cfg, model, torch_probe, probe_mean, probe_std, receiver_dl, donor_dl, layer_module, hs_index, das, device):
    out_dir = Path(cfg["data"]["das_out_dir"])
    out_name = cfg["data"]["das_out_dev_name"]

    model.eval()
    torch_probe.eval()
    das.eval()

    total_n = 0
    total_correct = 0
    flip_count = 0

    num_labels = torch_probe.classifier.out_features
    transition_counts = torch.zeros((num_labels, num_labels), dtype=torch.long)

    donor_iter = iter(donor_dl)

    with torch.no_grad():
        for receiver_batch in tqdm(receiver_dl, desc="Evaluating DAS"):
            try:
                donor_batch = next(donor_iter)
            except StopIteration:
                donor_iter = iter(donor_dl)
                donor_batch = next(donor_iter)

            receiver_batch = {k: v.to(device) for k, v in receiver_batch.items()}
            donor_batch = {k: v.to(device) for k, v in donor_batch.items()}

            receiver_no_labels = {
                k: v for k, v in receiver_batch.items()
                if k not in ["labels", "offset_mapping"]
            }
            donor_no_labels = {
                k: v for k, v in donor_batch.items()
                if k not in ["labels", "offset_mapping"]
            }

            target_labels = donor_batch["labels"]

            B = min(receiver_no_labels["input_ids"].size(0), donor_no_labels["input_ids"].size(0))

            receiver_no_labels = {k: v[:B] for k, v in receiver_no_labels.items()}
            donor_no_labels = {k: v[:B] for k, v in donor_no_labels.items()}
            target_labels = target_labels[:B]

            base_hidden = get_hs_at_layer(model, receiver_no_labels, hs_index)
            base_vec = mean_pool_hidden_torch(base_hidden, receiver_no_labels["attention_mask"])
            probe_mean = probe_mean.to(device)
            probe_std = probe_std.to(device)
            base_vec = (base_vec - probe_mean) / probe_std
            base_logits = torch_probe(base_vec)
            base_pred = base_logits.argmax(dim=1)

            patched_hidden = run_with_das_patch_return_hidden(model, receiver_no_labels, donor_no_labels, layer_module, hs_index, das)
            patched_vec = mean_pool_hidden_torch(patched_hidden, receiver_no_labels["attention_mask"])
            patched_vec = (patched_vec - probe_mean) / probe_std
            patched_logits = torch_probe(patched_vec)
            patched_pred = patched_logits.argmax(dim=1)

            total_correct += (patched_pred == target_labels).sum().item()
            flip_count += (patched_pred != base_pred).sum().item()
            total_n += B

            pair_index = base_pred.cpu() * num_labels + patched_pred.cpu()
            transition_counts += torch.bincount(
                pair_index,
                minlength=num_labels * num_labels,
            ).view(num_labels, num_labels)

    results = {
        "n": int(total_n),
        "patched_target_accuracy": float(total_correct / max(total_n, 1)),
        "flip_rate": float(flip_count / max(total_n, 1)),
        "transition_counts": transition_counts.tolist(),
        "hs_index": int(hs_index),
        "k": int(das.k),
    }

    if out_dir is not None:
        out_path = out_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results

def get_distilbert_layer_module(model, layer_idx):
    """
    layer_idx here should correspond to hidden_states index.

    hidden_states[0] = embeddings
    hidden_states[1] = transformer layer 0 output
    hidden_states[2] = transformer layer 1 output
    ...
    """
    if layer_idx == 0:
        raise ValueError("DAS patching at embedding layer not implemented here.")

    return model.distilbert.transformer.layer[layer_idx - 1]

def run_das(model, device, best_layer, train_receiver_dl, train_donor_dl, dev_receiver_dl, dev_donor_dl):
    hs_index = best_layer  # e.g. 2
    layer_module = get_distilbert_layer_module(model, hs_index)

    hidden_size = model.config.dim  # DistilBERT hidden size

    das = train_das(
        model=model,
        receiver_dl=train_receiver_dl,
        donor_dl=train_donor_dl,
        layer_module=layer_module,
        hs_index=hs_index,
        hidden_size=hidden_size,
        k=8,
        device=device,
        lr=1e-3,
        epochs=10,
    )

    dev_results = eval_das(
        model=model,
        receiver_dl=dev_receiver_dl,
        donor_dl=dev_donor_dl,
        layer_module=layer_module,
        hs_index=hs_index,
        das=das,
        device=device,
    )