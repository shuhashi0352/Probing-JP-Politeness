import numpy as np
import torch
from tqdm import tqdm


def extract_hs_by_layer(dataloader, model, device, desc="Extract hidden states"):
    print(f"\n{desc}...\n")

    model.eval()
    all_layer_hidden = None
    all_attention_masks = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch", total=len(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            all_labels.append(batch["labels"].cpu())
            all_attention_masks.append(batch["attention_mask"].cpu())

            batch_no_labels = {
                k: v for k, v in batch.items()
                if k not in ["labels", "offset_mapping"]
            }

            outputs = model(
                **batch_no_labels,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            if all_layer_hidden is None:
                all_layer_hidden = [[] for _ in range(len(hidden_states))]

            for layer_idx, h in enumerate(hidden_states):
                all_layer_hidden[layer_idx].append(h.cpu())

    x_layers_full = [
        torch.cat(chunks, dim=0).numpy()
        for chunks in all_layer_hidden
    ]

    attention_masks = torch.cat(all_attention_masks, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    return x_layers_full, attention_masks, y


def mean_pool_np(hidden, mask):
    """
    hidden: (N, T, H)
    mask:   (N, T)
    return: (N, H)
    """
    mask = mask[:, :, None]
    summed = (hidden * mask).sum(axis=1)
    lengths = mask.sum(axis=1).clip(min=1)
    return summed / lengths


def split_context_quote_text(text):
    """
    Example:
    依頼人は弁護士に「ここで待ってください。」と言った。

    context_text = 依頼人は弁護士に
    quote_text   = ここで待ってください。
    """
    start = text.find("「")
    end = text.find("」")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Invalid quote span: {text}")

    context_text = text[:start]
    quote_text = text[start + 1:end]

    return context_text, quote_text


def find_subsequence(full_ids, sub_ids):
    if len(sub_ids) == 0:
        return None

    n = len(sub_ids)
    for i in range(len(full_ids) - n + 1):
        if full_ids[i:i + n] == sub_ids:
            return i

    return None


def build_context_quote_masks(texts, encodings, tokenizer):
    """
    Builds two masks:
      context_mask: tokens before 「
      quote_mask:   tokens inside 「...」

    This avoids absolute token-position alignment.
    """
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if hasattr(input_ids, "cpu"):
        input_ids = input_ids.cpu().numpy()
    if hasattr(attention_mask, "cpu"):
        attention_mask = attention_mask.cpu().numpy()

    context_masks = []
    quote_masks = []

    for text, ids, attn in zip(texts, input_ids, attention_mask):
        context_text, quote_text = split_context_quote_text(text)

        context_ids = tokenizer(
            context_text,
            add_special_tokens=False,
        )["input_ids"]

        quote_ids = tokenizer(
            quote_text,
            add_special_tokens=False,
        )["input_ids"]

        real_len = int(attn.sum())
        full_ids = ids[:real_len].tolist()

        context_start = find_subsequence(full_ids, context_ids)
        quote_start = find_subsequence(full_ids, quote_ids)

        if context_start is None:
            raise ValueError(
                "Could not find context token span.\n"
                f"text: {text}\n"
                f"context_text: {context_text}\n"
                f"context_ids: {context_ids}\n"
                f"tokens: {tokenizer.convert_ids_to_tokens(full_ids)}"
            )

        if quote_start is None:
            raise ValueError(
                "Could not find quote token span.\n"
                f"text: {text}\n"
                f"quote_text: {quote_text}\n"
                f"quote_ids: {quote_ids}\n"
                f"tokens: {tokenizer.convert_ids_to_tokens(full_ids)}"
            )

        context_mask = np.zeros_like(ids)
        quote_mask = np.zeros_like(ids)

        context_mask[context_start:context_start + len(context_ids)] = 1
        quote_mask[quote_start:quote_start + len(quote_ids)] = 1

        context_masks.append(context_mask)
        quote_masks.append(quote_mask)

    return np.stack(context_masks), np.stack(quote_masks)


def span_pool_np(hidden, span_mask):
    """
    hidden:    (N, T, H)
    span_mask: (N, T)
    return:    (N, H)
    """
    mask = span_mask[:, :, None]
    summed = (hidden * mask).sum(axis=1)
    lengths = mask.sum(axis=1).clip(min=1)
    return summed / lengths


def context_quote_rep_np(hidden, context_mask, quote_mask):
    """
    hidden:       (N, T, H)
    context_mask: (N, T)
    quote_mask:   (N, T)

    return:       (N, 2H)
    """
    context_vec = span_pool_np(hidden, context_mask)
    quote_vec = span_pool_np(hidden, quote_mask)

    return np.concatenate([context_vec, quote_vec], axis=-1)


def span_pool_torch(hidden, span_mask):
    """
    hidden:    (B, T, H)
    span_mask: (B, T)
    return:    (B, H)
    """
    mask = span_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1.0)

    return summed / lengths


def context_quote_rep_torch(hidden, context_mask, quote_mask):
    """
    hidden:       (B, T, H)
    context_mask: (B, T)
    quote_mask:   (B, T)

    return:       (B, 2H)
    """
    context_vec = span_pool_torch(hidden, context_mask)
    quote_vec = span_pool_torch(hidden, quote_mask)

    return torch.cat([context_vec, quote_vec], dim=-1)


def run_extraction(dataloader, model, device, texts, encodings, tokenizer, desc):
    """
    Memory-efficient extraction.

    Instead of storing full hidden states:
        list[(N, T, H)]

    this stores only context+quote pooled vectors:
        list[(N, 2H)]

    Returns the same tuple shape expected by run_das.py.
    """

    print(f"\n{desc}...\n", flush=True)

    model.eval()

    context_masks_np, quote_masks_np = build_context_quote_masks(
        texts=texts,
        encodings=encodings,
        tokenizer=tokenizer,
    )

    all_layer_cq_chunks = None
    all_attention_masks = []
    all_labels = []

    start = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch", total=len(dataloader)):
            batch_size = batch["input_ids"].size(0)
            end = start + batch_size

            batch_context_masks = torch.tensor(
                context_masks_np[start:end],
                dtype=torch.float32,
                device=device,
            )
            batch_quote_masks = torch.tensor(
                quote_masks_np[start:end],
                dtype=torch.float32,
                device=device,
            )

            batch = {k: v.to(device) for k, v in batch.items()}

            all_labels.append(batch["labels"].detach().cpu())
            all_attention_masks.append(batch["attention_mask"].detach().cpu())

            batch_no_labels = {
                k: v for k, v in batch.items()
                if k not in ["labels", "offset_mapping"]
            }

            outputs = model(
                **batch_no_labels,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            if all_layer_cq_chunks is None:
                all_layer_cq_chunks = [[] for _ in range(len(hidden_states))]

            for layer_idx, h in enumerate(hidden_states):
                cq = context_quote_rep_torch(
                    h,
                    batch_context_masks,
                    batch_quote_masks,
                )
                all_layer_cq_chunks[layer_idx].append(cq.detach().cpu())

            start = end

    print(f"{desc}: finished batch-level context+quote pooling", flush=True)

    x_layers_context_quote = [
        torch.cat(chunks, dim=0).numpy()
        for chunks in all_layer_cq_chunks
    ]

    attention_masks = torch.cat(all_attention_masks, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    print(f"{desc}: returning pooled hidden states", flush=True)

    return (
        None,
        None,
        attention_masks,
        y,
        context_masks_np,
        quote_masks_np,
        x_layers_context_quote,
    )