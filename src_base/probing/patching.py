import torch
from tqdm import tqdm
import json

def run_with_cls_patched(model, receiver_batch_no_labels, layer_module, donor_cls):
    """
    Run the model once, but patch CLS at the hooked layer to donor_cls.
    donor_cls should be shape (B, H) and on the right device (or moved inside hook).
    """
    def patch_cls_hook(module, inputs, output):
        # output is usually (B, T, H), but some modules may return tuples.
        if isinstance(output, tuple):
            hs = output[0]
            patched = hs.clone()
            patched[:, 0, :] = donor_cls.to(patched.device)
            return (patched,) + output[1:]

        patched = output.clone()
        patched[:, 0, :] = donor_cls.to(patched.device)
        return patched

    handle = layer_module.register_forward_hook(patch_cls_hook)
    try:
        out = model(**receiver_batch_no_labels, return_dict=True)
    finally:
        handle.remove()  # always remove, even if something fails
    return out

def causal_cls_patching_dev(model, donor_dl, receiver_dl, layer_module, best_layer, device, out_dir, target_class_idx=0):
    """
    Return:
    "n_receiver_instances": the total number of instances in receiver_dl
    "avg_delta_target_logit": how much logit increased through patching
    "flip_to_target_rate": How much of the predictions on the baseline changed after patching
    "base_pred_counts": the list showing how many times each label (0-3) is predicted BEFORE patching (baseline)
    "patched_pred_counts": the list showing how many times each label (0-3) is predicted AFTER patching (baseline)
    "transition_counts": the showing how many times each transition heppened (e.g., 3 -> 0, 3 -> 1, 3 -> 2, 3 -> 3)
    """
    model.eval()

    n_total = 0
    n_flip_to_target = 0
    delta_target_logit_sum = 0.0

    donor_iter = iter(donor_dl)

    for receiver_batch in tqdm(receiver_dl, desc="CLS patching (dev receiver batches)"):
        receiver_batch = {k: v.to(device) for k, v in receiver_batch.items()}
        receiver_no_labels = {k: v for k, v in receiver_batch.items() if k != "labels"}

        try:
            donor_batch = next(donor_iter)
        except StopIteration:
            donor_iter = iter(donor_dl)
            donor_batch = next(donor_iter)

        donor_batch = {k: v.to(device) for k, v in donor_batch.items()}
        donor_no_labels = {k: v for k, v in donor_batch.items() if k != "labels"}

        # batch size align
        B = min(receiver_no_labels["input_ids"].size(0), donor_no_labels["input_ids"].size(0))
        receiver_no_labels = {k: v[:B] for k, v in receiver_no_labels.items()}
        donor_no_labels = {k: v[:B] for k, v in donor_no_labels.items()}

        with torch.no_grad():
            # Baseline receiver output
            base_out = model(**receiver_no_labels, return_dict=True)
            base_logits = base_out.logits  # (B, num_classes)
            base_pred = torch.argmax(base_logits, dim=1)

            # Donor CLS at best_layer
            donor_out = model(**donor_no_labels, output_hidden_states=True, return_dict=True)
            donor_cls = donor_out.hidden_states[best_layer][:, 0, :].detach()  # (B, H)

        # Patch hook
        def patch_cls_hook(module, inputs, output):
            # DistilBERT layer output is usually a Tensor (B,T,H) or a tuple
            if isinstance(output, tuple):
                hs = output[0]
                patched = hs.clone()
                patched[:, 0, :] = donor_cls.to(patched.device)
                return (patched,) + output[1:]
            else:
                patched = output.clone()
                patched[:, 0, :] = donor_cls.to(patched.device)
                return patched

        handle = layer_module.register_forward_hook(patch_cls_hook)
        try:
            with torch.no_grad():
                patched_out = model(**receiver_no_labels, return_dict=True)
                patched_logits = patched_out.logits
                patched_pred = torch.argmax(patched_logits, dim=1)
        finally:
            handle.remove()

        # metrics
        delta_target = (patched_logits[:, target_class_idx] - base_logits[:, target_class_idx]).cpu()
        delta_target_logit_sum += float(delta_target.sum().item())

        flip_to_target = ((base_pred != target_class_idx) & (patched_pred == target_class_idx)).sum().item()
        n_flip_to_target += int(flip_to_target)

        base_pred_counts += torch.bincount(base_pred.detach().cpu(), minlength=4)
        patched_pred_counts += torch.bincount(patched_pred.detach().cpu(), minlength=4)

        pair_index = (base_pred.detach().cpu() * 4 + patched_pred.detach().cpu())
        transition_counts += torch.bincount(pair_index, minlength=4 * 4).view(4, 4)

        n_total += B

    results = {
        "n_receiver_instances": int(n_total),
        "avg_delta_target_logit": float(delta_target_logit_sum / max(n_total, 1)),
        "flip_to_target_rate": float(n_flip_to_target / max(n_total, 1)),
        "base_pred_counts": base_pred_counts.tolist(), 
        "patched_pred_counts": patched_pred_counts.tolist(), 
        "transition_counts": transition_counts.tolist()
    }

    out_path = out_dir / "patching_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results