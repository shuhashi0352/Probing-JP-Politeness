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

### 2) Layerwise probing (heatmap stage)
To identify where politeness is encoded, we extract representations from each layer and train a **multinomial logistic regression (L2-regularized)** per layer:

> **Why logistic regression?** Because (multinomial) logistic regression evaluates each layer in terms of how much it's linearly decodable. It means if linear probes succeed, the representaion has made politeness explicit and easy to read out.

- For each layer $l$:
  - Extract hidden states (representation choice: `[CLS]` and/or pooled tokens)
  > The [CLS] token is designed to represent the whole sentence.
  - Train a linear probe on train features
  - Evaluate on test features(e.g., macro-F1)
- Visualize results as a **heatmap over layers** 

> **Why Heatmap?** Choosing the best layer based on the logistic regression might not be enough. (Apperance of politeness doesn't necessarily justify the fact that the model **uses the feature of politeness** through its learning process. Thus, we use the heatmap to pick where to look, and check **causality** around the layer.

There are two choices to test causality:
- **Intervention** -> remove the politeness direction at the layer to see if politeness behavior drops (e.g., The model could predict level 4 for the instance that it predicted level 1 before the intervention).

> As for this test, we need to extract the politeness direction $v$ for the model to be trained with. See [Extracting the Politeness Direction](#extracting-the-politeness-direction)

- **Activation patching** -> swap the layer's activation between polite (level 1-2) vs casual (level 4) sentences to see if output also swaps.

Interpretation:
- High performance at layer "l" suggests politeness is **linearly decodable** from that layer’s representations.
- Successful results in the causality check implies that the model uses politeness to learn.
- This provides a **justification** for selecting layer(s) for embedding extraction.

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

### 3) Causality Check

#### Option A: Intervention
Now we have:

- $h$: the hidden-state vector for a sentence (e.g., CLS at layer $l$*), shape $d$

- $\hat v$: a unit vector (length 1) that represents “politeness direction”

Given that $\hat v$ has length 1, we use the dot product:

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