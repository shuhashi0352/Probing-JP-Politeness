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

            labels = batch["labels"].cpu()
            attention_mask = batch["attention_mask"].cpu()

            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

            batch_no_labels = {k: v for k, v in batch.items() if k not in ["labels", "offset_mapping"]}

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
        torch.cat(layer_chunks, dim=0).numpy()
        for layer_chunks in all_layer_hidden
    ]

    attention_masks = torch.cat(all_attention_masks, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    return x_layers_full, attention_masks, y


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


def build_quote_mask_from_offsets(texts, encodings, tokenizer):
    """
    Build mask for tokens inside Japanese quotation marks 「...」.

    texts: list[str]
    encodings: tokenizer output with return_offsets_mapping=True
    tokenizer: Hugging Face tokenizer

    return:
        quote_mask: np.ndarray, shape (N, T)
    """
    offset_mapping = encodings["offset_mapping"]
    attention_mask = encodings["attention_mask"]

    if hasattr(offset_mapping, "cpu"):
        offset_mapping = offset_mapping.cpu().numpy()
    if hasattr(attention_mask, "cpu"):
        attention_mask = attention_mask.cpu().numpy()

    quote_masks = []

    for text, offsets, attn in zip(texts, offset_mapping, attention_mask):
        start = text.find("「")
        end = text.find("」")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not find valid quote span in text: {text}")

        # exclude 「 and 」
        quote_start = start + 1
        quote_end = end

        mask = np.zeros_like(attn)

        for i, ((s, e), is_real_token) in enumerate(zip(offsets, attn)):
            if is_real_token == 0:
                continue

            # Special tokens often have offset (0, 0)
            if s == e:
                continue

            # Token overlaps with inside-quote character span
            if s < quote_end and e > quote_start:
                mask[i] = 1

        quote_masks.append(mask)

    return np.stack(quote_masks, axis=0)


def quote_pool_hs(hidden, quote_mask):
    """
    hidden: (N, T, H)
    quote_mask: (N, T)

    return:
        pooled: (N, H)
    """
    mask = quote_mask[:, :, None]
    summed = (hidden * mask).sum(axis=1)
    lengths = mask.sum(axis=1).clip(min=1)

    return summed / lengths

def extract_quote_text(text):
    start = text.find("「")
    end = text.find("」")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find quote span in text: {text}")

    return text[start + 1:end]


def build_quote_mask_from_token_ids(texts, encodings, tokenizer):
    """
    Build quote mask without offset_mapping.

    It tokenizes the quoted part separately, then finds that token-id
    subsequence inside the full tokenized sentence.
    """
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if hasattr(input_ids, "cpu"):
        input_ids = input_ids.cpu().numpy()
    if hasattr(attention_mask, "cpu"):
        attention_mask = attention_mask.cpu().numpy()

    quote_masks = []

    for text, ids, attn in zip(texts, input_ids, attention_mask):
        quote_text = extract_quote_text(text)

        quote_ids = tokenizer(
            quote_text,
            add_special_tokens=False,
        )["input_ids"]

        real_len = int(attn.sum())
        full_ids = ids[:real_len].tolist()

        found_start = None
        q_len = len(quote_ids)

        for i in range(0, len(full_ids) - q_len + 1):
            if full_ids[i:i + q_len] == quote_ids:
                found_start = i
                break

        if found_start is None:
            raise ValueError(
                "Could not find quote token subsequence.\n"
                f"text: {text}\n"
                f"quote_text: {quote_text}\n"
                f"quote_ids: {quote_ids}\n"
                f"full_ids: {full_ids}\n"
                f"tokens: {tokenizer.convert_ids_to_tokens(full_ids)}"
            )

        mask = np.zeros_like(ids)
        mask[found_start:found_start + q_len] = 1

        quote_masks.append(mask)

    return np.stack(quote_masks, axis=0)


def run_extraction(dataloader, model, device, texts, encodings, tokenizer, desc):
    x_layers_full, masks, y = extract_hs_by_layer(
        dataloader,
        model,
        device,
        desc=desc,
    )

    x_layers_pooled = [
        mean_pool_hs(h, masks)
        for h in x_layers_full
    ]

    quote_masks = build_quote_mask_from_token_ids(
        texts=texts,
        encodings=encodings,
        tokenizer=tokenizer,
    )

    x_quote_layers = [
        quote_pool_hs(h, quote_masks)
        for h in x_layers_full
    ]

    return x_layers_pooled, x_layers_full, masks, y, quote_masks, x_quote_layers