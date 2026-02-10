# Probing Internal Representations of Japanese Politeness

This project investigates **how much Japanese BERT-style models internally encode “politeness”** by combining:

1) **Supervised politeness classification** (baseline fine-tuning)  
2) **Layerwise analysis** to identify where politeness becomes *linearly accessible*  
3) **Representation probing** on extracted embeddings (as “features-as-data”)

The core idea is: **if politeness information is systematically decodable from specific hidden layers (under controlled probes), that’s evidence the model’s representations encode politeness-related pragmatic information.**

---

## Project Goals

- Train a supervised baseline on a 4-class politeness label set.
- Evaluate baseline performance to ensure the model actually learned the task.
- Run **layerwise linear probes** to measure where politeness is accessible in the representation stack.
- Extract embeddings from the most informative layers and build a probing dataset.
- Probe embeddings (e.g., with multinomial L2 logistic regression) under controls to evaluate “how encoded” politeness is.

---

## Dataset

We use **KeiCO / KeiCorpus**:

- Repo: https://github.com/Liumx2020/KeiCO-corpus  
- Based on: *“Construction and Validation of a Japanese Honorific Corpus Based on Systemic Functional Linguistics”*

The dataset provides Japanese text paired with **four politeness levels** (as defined by the authors’ SFL-based framework).  
- This project follows the original definition/structure of politeness used in KeiCO.

#### Dataset Statistics (from Liu & Kobayashi, 2022) -> see [See table 3 in the paper](https://aclanthology.org/2022.dclrl-1.3/)

| Honorific level | Sentences | Avg. sentence length | Avg. kanji per sentence | Word tokens | Word types | Yule’s characteristic K |
|---|---:|---:|---:|---:|---:|---:|
| Level 1 | 2,584 | 18.2 | 2.6 | 47,111 | 4,744 | 135.70 |
| Level 2 | 2,046 | 16.4 | 2.1 | 33,476 | 3,897 | 136.23 |
| Level 3 | 2,694 | 15.2 | 1.8 | 40,980 | 4,448 | 130.28 |
| Level 4 | 2,683 | 13.5 | 1.6 | 36,233 | 4,315 | 129.80 |
| **Total** | **10,007** | **15.8** | **2.0** | **157,806** | **6,465** | **125.54** |

---

## Task Definition

### Supervised baseline task
- Input: Japanese text
- Output: politeness label in **4 classes**


The baseline fine-tuning is used to ensure:
- The model can perform the task reliably (not random / underfit).
- Its learned hidden states provide meaningful representations for later probing.

---

## Models

We fine-tune one or more **Japanese BERT-family models** (e.g., BERT, RoBERTa, DeBERTa variants pretrained on Japanese).

> **TODO:** List the exact model checkpoints used (names + links) once finalized.

> **NOTES:** Could be LLMs but those decoders are removed.

---

## Method Overview

### 1) Baseline training (supervised fine-tuning)
- Fine-tune a Japanese BERT-style encoder for 4-way classification.
- Evaluate on a held-out test set (accuracy, macro-F1, confusion matrix etc... TBD).

Output:
- A trained baseline model
- Baseline performance metrics (sanity check that the task is learned)

---

## 2) Layerwise probing

**Goal:** Identify *where* politeness becomes **linearly accessible** in the model’s representation stack.

We extract a sentence vector from each layer (default: **`[CLS]`**) and train a **multinomial L2 logistic regression probe** per layer.

> **Why logistic regression?** Because (multinomial) logistic regression evaluates each layer in terms of how much it's linearly decodable. It means if linear probes succeed, the representaion has made politeness explicit and easy to read out.

---

### 2.1 Representation extraction (features-as-data)

We run the (fine-tuned) model with:

- `output_hidden_states=True`

This returns:

- `hidden_states[0]`: embedding output (pre-transformer)
- `hidden_states[1]`: output after encoder layer 0
- ...
- `hidden_states[L]`: output after encoder layer (L−1)

**Sentence representation (default):**

- Use the `[CLS]` token hidden state:
  - `x = hidden_states[hs_index][:, 0, :]`  (shape: `(B, H)`)

This produces, for each `hs_index`:

- `X_train[hs_index]` with shape `(N_train, H)`
- `X_dev[hs_index]` with shape `(N_dev, H)`
- (optionally) `X_test[hs_index]` with shape `(N_test, H)`

> Alternative (optional): mean pooling over non-padding tokens. In this project, `[CLS]` is the default.

---

### 2.2 Linear probe per layer

For each `hs_index`:

- Train a multinomial logistic regression probe:
  - L2 regularization
  - solver: `lbfgs`
- Evaluate on dev (recommended metric: **macro-F1**)

Selection rule:

- `best_layer = argmax_hs_index macroF1_dev(hs_index)`

**Important:** We use **dev only** to choose `best_layer` to avoid test leakage.

---

### 2.3 Visualization

We plot dev macro-F1 across layers as:

- a line plot (primary)
- (optionally) a heatmap-style plot over layers

The peak region indicates where politeness is most linearly decodable.

---

### 2.4 Output artifacts

This step produces:

- `dev_f1_macro_by_layer` (array of size `num_hidden_states`)
- `best_layer` and `best_f1_macro`
- a plot of dev macro-F1 by layer

---

## 3) Probing on the best layer (final probe evaluation)

After selecting `best_layer` on dev:

- Extract `X_test[best_layer]`
- Train probe on **train** features at `best_layer`
- Evaluate on **test** features at `best_layer`

This yields the final probing performance:

- accuracy
- macro-F1
- (optional) confusion matrix

> Note: If the probe score at `best_layer` is close to the fine-tuned classifier score, it suggests the politeness signal is already quite explicit in the representation (linearly readable).

---

## 4) Causal intervention via CLS patching (activation patching)

**Goal:** Test whether the representation at `best_layer` is not only *decodable*, but also *causally used* by the classifier.

We perform **CLS patching**:

- Receiver examples: typically the most casual class (e.g., Level 4)
- Donor examples: typically the most polite class (e.g., Level 1)

For each receiver batch:

1. Run **baseline** forward pass on the receiver → `base_logits`, `base_pred`
2. Run donor forward pass with `output_hidden_states=True` and extract donor CLS at `hs_index = best_layer`:
   - `donor_cls = hidden_states[best_layer][:, 0, :]`
3. Register a forward hook on the **encoder layer module** corresponding to `best_layer`
4. In the hook, replace only the receiver CLS vector with `donor_cls`:
   - `patched[:, 0, :] = donor_cls`
5. Run receiver again → `patched_logits`, `patched_pred`
6. Aggregate transition statistics across the dataset

### 4.1 What is actually replaced?

- Only the `[CLS]` vector at the patched layer is replaced.
- All other token vectors remain unchanged.

This is intentional:

- It tests whether a **sentence-level** control signal is sufficient to shift politeness predictions,
- without directly overwriting token-specific honorific markers.

---

### 4.2 Metrics reported

We compute:

- `avg_delta_target_logit`  
  Average change in the target class logit due to patching:
  - `patched_logits[:, target] - base_logits[:, target]`

- `flip_to_target_rate`  
  Fraction of instances where prediction flips **into** the target class:
  - baseline is **not** target AND patched prediction **is** target

- `base_pred_counts`  
  Predicted class counts **before** patching (length 4)

- `patched_pred_counts`  
  Predicted class counts **after** patching (length 4)

- `transition_counts` (4×4)  
  Confusion-like transition matrix:
  - rows = baseline prediction
  - cols = patched prediction
  - entry `(i, j)` counts how many moved from `i → j`

We visualize `transition_counts` as a **4×4 heatmap** for paper-ready figures.

---

## 5) Controls (sanity + strength checks)

We include controls to ensure the effect is not an artifact.

### 5.1 Self-patch (implementation sanity check)

**Definition (strict):**

- Use the **same receiver batch** as the donor
- Extract donor CLS from the receiver itself
- Patch receiver CLS with its own CLS

Expected:

- Almost no change (transition matrix ~ diagonal)

Implementation:

- `mode="self"` sets `donor_no_labels = receiver_no_labels`

---

### 5.2 Random donor CLS (break pairing)

**Definition:**

- Extract donor CLS normally (from Level 1 donors),
- then shuffle donor CLS vectors **within the batch** (break sentence ↔ CLS correspondence).

Expected:

- The effect weakens or becomes less consistent,
- because “this donor sentence’s CLS” is no longer aligned.

Implementation:

- `perm = torch.randperm(B, generator=g)`
- `donor_cls = donor_cls[perm]`

---

### 5.3 Wrong-layer patch (layer sweep)

**Definition:**

- Patch CLS at **every encoder layer** (or every hidden-state index),
- compute effect curves across layers.

Expected:

- Stronger effect near the layer(s) where politeness is encoded/used.
- If it aligns with probe peak layers, that strengthens the causal story.

Implementation:

- Loop over encoder layers, hook each `layer_module`
- Use matching hidden-state index `hs_idx = layer_idx + 1` for donor CLS

---

## 6) Practical notes (indexing: hidden states vs encoder layers)

HuggingFace `hidden_states` indexing and encoder layer indexing differ:

- `hidden_states[0]` is **embeddings**, not an encoder block output.
- Encoder layers are `layer_idx = 0..L-1`
- Their outputs correspond to:
  - `hidden_states[layer_idx + 1]`

Therefore we track two indices:

- `layer_idx` = encoder layer module index (for hooks)
- `hs_index` = hidden states index (for extracting donor CLS)

In code:

- `layer_module = model.distilbert.transformer.layer[layer_idx]`
- `donor_cls = donor_out.hidden_states[hs_index][:, 0, :]`
- Usually: `hs_index = layer_idx + 1`

When using `best_layer` selected from probing:

- If probing was done over `hidden_states` indices, use it directly as `hs_index`.
- Convert to encoder layer index via:
  - `layer_idx = best_layer - 1`
