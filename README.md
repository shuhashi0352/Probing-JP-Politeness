# Probing-JP-Politeness

A research project for analyzing how Japanese politeness is represented inside transformer models using probing and causal intervention methods.

---

# Motivation

Japanese politeness is not expressed through a single surface marker. It emerges from interactions between:

- lexical choices
- verb morphology
- sentence endings
- speaker–listener relationships
- discourse context
- pragmatic appropriateness

This makes politeness an ideal test case for representation analysis in language models.

Many NLP systems can classify politeness correctly, but prediction accuracy alone does not reveal **where** politeness is encoded or **how** the model uses that information internally.

This project addresses that gap through two complementary perspectives:

## 1. Geometric Decodability

Can politeness be linearly decoded from hidden states at each layer?

If yes, then the representation contains accessible information about politeness.

## 2. Causal Usefulness

Does changing an internal subspace alter the model’s politeness judgment?

If yes, then the representation is not only decodable but functionally involved in prediction.

To study this, the repository combines:

- layerwise probing
- hidden-state extraction
- activation patching
- Distributed Alignment Search (DAS)
- transition analysis across labels

The broader goal is to move beyond black-box classification and better understand how socially grounded linguistic knowledge is structured inside pretrained models.

---

# Dataset / Model

# Dataset

The project uses a Japanese politeness dataset built from controlled dialogue situations.

Each example includes:

- speaker role
- listener role
- social setting
- utterance
- expected politeness level
- realized politeness level
- naturalness label

Examples are converted into sentence prompts such as:

    {
      "text": "学生は教授に「了解」と言った。",
      "label": "unnatural"
    }

This allows the model to jointly process:

- social relation information
- utterance form
- pragmatic appropriateness

## Label Space

- natural
- unnatural

## Split Strategy

To reduce memorization leakage, the dataset is split by shared utterance identity rather than random row split.

This means the same utterance template does not appear across train / dev / test splits.

---

# Model

The main encoder is:

- LineDistilBERT (Japanese DistilBERT classifier backbone)

Used in frozen form for interpretability experiments.

The encoder outputs hidden states from all transformer layers, enabling layerwise analysis.

---

# Methods

# 1. Prompt Conversion

Structured metadata is converted into natural Japanese sentences:

speaker + listener + quoted utterance + speech event

Example:

    学生は教授に「了解」と言った。

This lets the model reason over both language form and social context in one sequence.

---

# 2. Hidden State Extraction

For every split:

- train
- dev
- test

The system extracts hidden states from all layers.

Representations include:

- pooled full-sequence vectors
- context-only vectors
- quote-only vectors
- context+quote combined vectors

These become inputs for probing and intervention experiments.

---

# 3. Layerwise Linear Probing

A linear classifier is trained separately on each layer representation.

Purpose:

- measure where politeness information becomes decodable
- compare representational quality across layers
- identify the best intervention layer

Output:

- dev accuracy per layer
- best layer index
- best probing score

---

# 4. PyTorch Probe Reproduction

After selecting the best layer, a standardized linear probe is retrained in PyTorch.

Purpose:

- align probe implementation with downstream DAS pipeline
- ensure compatible optimization and tensor handling
- provide a target classifier for interventions

---

# 5. Distributed Alignment Search (DAS)

DAS learns a low-dimensional subspace inside hidden representations that controls classifier behavior.

Instead of replacing the full hidden state, the method:

1. selects donor and receiver examples
2. learns a transformation subspace
3. patches only the learned dimensions
4. observes prediction changes

This tests whether a compact internal direction causally influences politeness judgments.

---

# 6. Donor / Receiver Construction

Examples are aligned by utterance identity.

For each shared utterance:

- natural example ↔ unnatural example

Two directions are created:

- unnatural receiver + natural donor
- natural receiver + unnatural donor

This controls lexical content while changing pragmatic label.

---

# 7. Evaluation Metrics

DAS outputs are evaluated with:

- patched target accuracy
- flip rate
- transition matrix
- layer index
- subspace dimension k

## Interpretation

- High flip rate: intervention strongly changes decisions
- High target accuracy: flips move toward intended labels
- Structured transitions: evidence of meaningful control rather than noise

---

# Research Question

This study investigates:

- Where is Japanese politeness represented inside transformer layers?
- Is politeness merely decodable, or causally actionable?
- Can compact subspaces control pragmatic judgments?
- How much social reasoning is encoded in pretrained language models?