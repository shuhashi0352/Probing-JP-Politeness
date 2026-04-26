import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from pathlib import Path
from probing.extract_hs import context_quote_rep_torch

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
    
def run_with_das_on_context_quote_reps(receiver_rep, donor_rep, das):
    return das.mix(receiver_rep, donor_rep)

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

    das = DASSubspace(hidden_size=model.config.dim * 2, k=k).to(device)
    optimizer = torch.optim.AdamW(das.parameters(), lr=lr)

    model.eval()
    torch_probe.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in torch_probe.parameters():
        p.requires_grad = False

    probe_mean = probe_mean.to(device)
    probe_std = probe_std.to(device)

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
                if k not in ["labels", "offset_mapping", "context_mask", "quote_mask"]
            }
            donor_no_labels = {
                k: v for k, v in donor_batch.items()
                if k not in ["labels", "offset_mapping", "context_mask", "quote_mask"]
            }

            target_labels = donor_batch["labels"]

            B = min(receiver_no_labels["input_ids"].size(0), donor_no_labels["input_ids"].size(0))

            receiver_no_labels = {k: v[:B] for k, v in receiver_no_labels.items()}
            donor_no_labels = {k: v[:B] for k, v in donor_no_labels.items()}
            target_labels = target_labels[:B]

            optimizer.zero_grad()

            receiver_hidden = get_hs_at_layer(model, receiver_no_labels, hs_index)
            donor_hidden = get_hs_at_layer(model, donor_no_labels, hs_index)

            receiver_context_mask = receiver_batch["context_mask"][:B]
            receiver_quote_mask = receiver_batch["quote_mask"][:B]
            donor_context_mask = donor_batch["context_mask"][:B]
            donor_quote_mask = donor_batch["quote_mask"][:B]

            receiver_rep = context_quote_rep_torch(receiver_hidden, receiver_context_mask, receiver_quote_mask)
            donor_rep = context_quote_rep_torch(donor_hidden, donor_context_mask, donor_quote_mask)

            patched_rep = das.mix(receiver_rep, donor_rep)
            patched_rep = (patched_rep - probe_mean) / probe_std
            logits = torch_probe(patched_rep)
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

    probe_mean = probe_mean.to(device)
    probe_std = probe_std.to(device)

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
                if k not in ["labels", "offset_mapping", "context_mask", "quote_mask"]
            }
            donor_no_labels = {
                k: v for k, v in donor_batch.items()
                if k not in ["labels", "offset_mapping", "context_mask", "quote_mask"]
            }

            target_labels = donor_batch["labels"]

            B = min(receiver_no_labels["input_ids"].size(0), donor_no_labels["input_ids"].size(0))

            receiver_no_labels = {k: v[:B] for k, v in receiver_no_labels.items()}
            donor_no_labels = {k: v[:B] for k, v in donor_no_labels.items()}
            target_labels = target_labels[:B]

            receiver_context_mask = receiver_batch["context_mask"][:B]
            receiver_quote_mask = receiver_batch["quote_mask"][:B]
            donor_context_mask = donor_batch["context_mask"][:B]
            donor_quote_mask = donor_batch["quote_mask"][:B]

            receiver_hidden = get_hs_at_layer(model, receiver_no_labels, hs_index)
            donor_hidden = get_hs_at_layer(model, donor_no_labels, hs_index)

            receiver_rep = context_quote_rep_torch(receiver_hidden, receiver_context_mask, receiver_quote_mask)
            donor_rep = context_quote_rep_torch(donor_hidden, donor_context_mask, donor_quote_mask)

            base_logits = torch_probe((receiver_rep - probe_mean) / probe_std)
            base_pred = base_logits.argmax(dim=1)

            patched_rep = das.mix(receiver_rep, donor_rep)

            patched_logits = torch_probe((patched_rep - probe_mean) / probe_std)
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