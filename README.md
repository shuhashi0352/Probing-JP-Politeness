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

### 2) Layerwise probing
To identify where politeness is encoded, we extract representations from each layer and train a **multinomial logistic regression (L2-regularized)** per layer:

> **Why logistic regression?** Because (multinomial) logistic regression evaluates each layer in terms of how much it's linearly decodable. It means if linear probes succeed, the representaion has made politeness explicit and easy to read out.

- For each layer $l$:
  - Extract hidden states (representation choice: `[CLS]`)
  > The [CLS] token is designed to represent the whole sentence.
  - Train a linear probe on train features
  - Evaluate on dev features(e.g., macro-F1)
- Visualize results as a **heatmap over layers** 

- **Choose the best layer using dev only** (avoid test leakage).
- Use test only once for final reporting after layer selection.

#### Step A: Extract representations from each layer
- Run the (frozen) encoder with `output_hidden_states=True` to obtain hidden states for all layers in one forward pass.
- For each sentence and each layer *l*, convert token-level outputs into a **single sentence vector**.

Representation choice (default):
- **`[CLS]` vector**
  - Take the hidden state of the `[CLS]` token at layer *l* as the sentence embedding.
  - Rationale: `[CLS]` is designed to represent the whole sentence for classification.

What this produces:
- Let the number of sentences be **N** and hidden size be **768**.
- For each layer *l*, we build a feature matrix:
  - `X_train^l` with shape **(N_train × 768)**
  - `X_dev^l` with shape **(N_dev × 768)**
  - (optionally `X_test^l` for final reporting only)

> Note: Mean pooling over non-padding tokens is a common alternative; we use `[CLS]` as the main setting and may report mean pooling as an ablation.

#### Step B: Train a linear probe per layer
- For each layer *l*:
  - Train a **multinomial logistic regression (L2-regularized)** on `X_train^l` → `y_train`
  - Tune **C** (regularization strength) using dev (e.g., `LogisticRegressionCV` or a small grid search)
  - If class imbalance is strong, consider `class_weight="balanced"`

#### Step C: Evaluate and select the “best layer”
- For each layer *l*:
  - Evaluate on `X_dev^l` (recommended metric: **macro-F1**)
- Select:
  - `best_layer = argmax_l macroF1_dev(l)`

Tie-breaking (recommended):
- If multiple layers are within a small margin:
  - Prefer the layer that is **more stable** across random seeds / folds, OR
  - Use a **1-SE rule** (choose the simplest/earliest layer within one standard error of the best)

#### Step D: Visualize as a heatmap over layers
- Plot dev scores (macro-F1, optionally accuracy) across layers.
- The peak region indicates where politeness is most linearly decodable.

Output:
- Per-layer dev scores (macro-F1, accuracy)
- A heatmap over layers
- Selected `best_layer` for downstream analysis (e.g., embedding extraction / interventions)

---

### Extracting the Politeness Direction
> **Direction $v$**: We are looking for a vector that points to the feature of politeness in the embedding space.

Here's the big picture of how the direction can be extracted (illustrative):

Let's take two sentences "ありがとう" (casual) and "ありがとうございます" (polite), both of which mean "Thank you." LineDistilBERT has 768 dimentions of hidden states, which means there's a 768-dimentional space for each sentence. 

Each sentence produces a vector.

$x_{\text{casual}}, \;x_{\text{polite}} \in \mathbb{R}^{768}$

A direction $v$ represents the arrow between them:

$v = x_{\text{polite}} - x_{\text{casual}}$
- That subtraction gives you an arrow pointing from “casual-ish region” toward “polite-ish region.”

The direction $v$ is normalized:

$\hat v = \frac{v}{\|v\|}$

#### Direction estimation
However, the class means $\mu_{\text{level}}$ are employed since we are interested in the consistent change across many examples when politeness changes. 

Because a single pair is noisy, we estimate v from many labeled examples using class mean embeddings. For each politeness level k, we compute the mean embedding:

$\mu_k = \frac{1}{N_k}\sum_{i: y_i = k} x_i.$

We then define a direction using the extremes (most polite = level 1, most casual = level 4) and normalize it:

$v = \mu_{\text{1}} - \mu_{\text{4}}, \quad \hat v = \frac{v}{\lVert v\rVert}.$

 This captures the typical shift in representation associated with politeness.

---

### 3) Causality Check / Causal Intervention

In this test, we basically change (remove) an internal variable (associated with politeness) on purpose and observe the effect. 

#### Option A: Direction Removal (Ablation)

> **Question:** With the same datapoint, what level would the model predict if the feature of politeness was removed? Would removing that feature cause the model to predict the level involved in "casual"?

#### How we execute direction removal

Given that we have:

- $h$: the hidden-state vector for a sentence (e.g., CLS at layer $l$*), shape $d$

- $\hat v$: a unit vector (length 1) that represents “politeness direction”

and we know that $\hat v$ has length 1, we use the dot product:

$h \cdot \hat v$

 This represents the scalar projection of $h$ onto that direction — basically “how far $h$ goes in the $\hat v$ direction”.

> **Intuition:** $h \cdot \hat v$ is “how much politeness” is in h along that axis

After the dot product, we take that scalar amount and turn it back into a vector by multiplying by $\hat v$:

$\underbrace{(h \cdot \hat v)}_{\text{amount}} \times \underbrace{\hat v}_{\text{direction}} = \text{component of } h \text{ along } \hat v$

This is literally **“the politeness component inside h”**

Finally, we subtract it to remove that component:

$h' = h - (h \cdot \hat v)\hat v$

> **Geometrically:** $h'$ is the projection of $h$ onto the hyperplane perpendicular to $\hat v$.

Below is the blueprint of **intervention** to check causality:

1. start with the full representation $h$
2. subtract the part that lies along politeness axis
3. the result $h'$ is $h$ with “politeness-axis information” removed

#### Option B: Activation Patching (Swap)

In this test, we replace receiver’s hidden state with donor’s hidden state. Concretely, we paste polite internal state into casual sentence and see if output becomes polite.

> **Question:** If we take the internal representation from a polite sentence at layer $l$ and paste it into a casual sentence’s forward pass, does the model’s prediction follow the pasted representation?

#### How we execute activation patching

Run the model normally on each input:

- $y_d = f(x_d)$ → the model’s output for the donor sentence
- $y_r = f(x_r)$ → the model’s output for the receiver sentence

The outputs represent **probabilities** (the value before $\arg\max$ is done).

> We expected them to be different. Otherwise, they can't be flipped.

At layer $l$, the model has hidden states [CLS]:

$h_{\text{CLS}}^{(l)} \in \mathbb{R}^{d}$

We do a forward pass on the receiver $x_r$, but at layer $l$ you overwrite its [CLS] with the donor’s.

$H_{r,\text{patched}}^{(l)}[0] = H_d^{(l)}[0]$

> index 0 is CLS token

This is when **patching** happens.

Then let the model continue forward from layer $l + 1$ to the end and compute output (probability):

$y_{r \leftarrow d} = f_{\text{patched}}(x_r)$

If layer $l$ encodes politeness in a way the model uses, then the probability of donor class increases, meaning that the receiver’s prediction should shift toward the donor’s politeness level.

> **CAUTION:** Both tests A and B are not necessarily capable of holistically supporting causality because the whole instances in the dataset do not only vary across the aspect of politeness, but also other variety of factors can influence the model's prediction (e.g., tense). 

> **Future Implementation:** To eradicate/reduce the *noise*, we need to create a dataset where only minimum pairs exist in terms of politeness.
