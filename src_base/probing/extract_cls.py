import numpy as np
import torch
from tqdm import tqdm


def extract_hs_by_layer(dataloader, model, device, desc="Extract hidden states"):
    print("\nExtracting full hidden states by layer...\n")

    model.eval()
    all_layer_hidden = None
    all_attention_masks = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch", total=len(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch["labels"].cpu()
            attention_mask = batch["attention_mask"].cpu()

            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

            batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}

            outputs = model(
                **batch_no_labels,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            if all_layer_hidden is None:
                all_layer_hidden = [[] for _ in range(len(hidden_states))]

            for layer_idx, h in enumerate(hidden_states):
                # h: (B, T, H)
                all_layer_hidden[layer_idx].append(h.cpu())

    x_layers = [
        torch.cat(layer_chunks, dim=0).numpy()
        for layer_chunks in all_layer_hidden
    ]

    attention_masks = torch.cat(all_attention_masks, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    return x_layers, attention_masks, y

def mean_pool_hs(hidden, attention_mask):
    """
    hidden: (N, T, H)
    attention_mask: (N, T)
    return: (N, H)
    """
    mask = attention_mask[:, :, None]
    summed = (hidden * mask).sum(axis=1)
    lengths = mask.sum(axis=1).clip(min=1)

    return summed / lengths

def run_extraction(dataloader, model, device):
    x_train_layers_full, train_masks, y_train = extract_hs_by_layer(dataloader, model, device, desc="Train hidden states")
    x_train_layers = [mean_pool_hs(h, train_masks)for h in x_train_layers_full]

    return x_train_layers
