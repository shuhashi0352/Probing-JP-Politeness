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

### 2) Layerwise probing (heatmap stage)
To identify where politeness is encoded, we extract representations from each layer and train a **multinomial logistic regression (L2-regularized)** per layer:

- For each layer l:
  - Extract hidden states \( h^\ell \) (representation choice: `[CLS]` and/or pooled tokens)
  - Train a linear probe on train features
  - Evaluate on test features
- Visualize results as a **heatmap over layers** (e.g., macro-F1)

Interpretation:
- High performance at layer l suggests politeness is **linearly decodable** from that layer’s representations.
- This provides a *justification* for selecting layer(s) for embedding extraction.

> THOUGH... “decodable” does not automatically mean “causally used.”  

---

### 3) Embedding extraction (features-as-data)
After selecting the best layer(s), we extract embeddings and build a probing dataset:

Each example includes:
- Representation vector(s) from selected layer(s)
- Gold politeness label
- Potentially... metadata such as text length, domain, etc.

Important design choice:
- We primarily collect embeddings for examples the baseline model classified **correctly**, because we want to study how the model supports correct behavior.
- We may also store **incorrect-only** and **all** examples for diagnostic comparisons.

> **TODO:** Finalize whether the default probing dataset is correct-only vs all vs split (or experiment with all of them?)

---

### 4) Probing (main evaluation)
We train probes on the extracted embedding dataset to measure how strongly politeness is encoded.

Default probe (Not yet determined):
- **Multinomial logistic regression (L2)**

Controls to avoid overclaiming (if the time allows):
- Shuffled-label control (probe should drop to chance)
- Random-feature control (same dimensionality)

> **Options:** Logistic regression couldn't be poweful enough => probe family beyond LR (e.g., linear SVM, small MLP, MDL probing).

---

## Evaluation

### Baseline classifier evaluation
- Accuracy
- Macro-F1
- Confusion matrix

### Probing evaluation
- Macro-F1 per layer / per representation type
- Optional: Probe-vs-control gap (true labels vs shuffled labels)