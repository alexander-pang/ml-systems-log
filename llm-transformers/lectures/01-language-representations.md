---
title: "Lecture 01: Evolution of Language Representations"
status: in-progress
date: 2026-05-22
tags: [lecture, language-representations, bag-of-words, word2vec, transformers]
---

# Lecture 01: Evolution of Language Representations

## Overview

Three generations of representing language numerically:

| Method | Context window | Representation |
|--------|---------------|----------------|
| Bag of Words | None (global counts) | Large, sparse vector |
| Word2Vec | A few neighboring words | Dense vector, fixed per word |
| Transformers | Full sentence / paragraph | Dense vector, context-dependent |

---

## The Problem

Text is unstructured — raw characters or bits lose meaning. Language AI needs **structured numerical representations** to do anything useful (classify, generate, embed).

---

## Model Families

```
Non-transformer models
├── Bag of Words
└── Word2Vec

Transformer models
├── Encoder-only   → best at representation (e.g. BERT)
├── Decoder-only   → best at generation (e.g. GPT)
└── Encoder-decoder → both (e.g. T5, BART)
```

Non-transformer models lack **contextualized representations** but are still useful baselines.

---

## Bag of Words (BoW)

### How it works

1. **Tokenize** input text into words (split on whitespace — more on this in the next lesson).
2. Build a **vocabulary** — all unique tokens across all documents.
3. Represent each document as a **count vector** over the vocabulary.

### Example

Documents:
- `"That is a cute dog"`
- `"My cat is cute."`

Vocabulary (union, deduplicated): `[That, is, a, cute, dog, My, cat]`

Representation of `"My cat is cute."`:

```
[0, 1, 0, 1, 0, 1, 1]
 That is  a cute dog My cat
```

The **order of words in the vocabulary** is fixed so vectors can be compared across documents.

### Properties

- Values = raw counts (interpretable).
- Vector is **sparse** (most entries 0 for large vocabularies).
- **No word order** preserved — "dog bites man" == "man bites dog".
- **No meaning** — synonyms are unrelated entries.

---

## Key Terms

- **Token**: a piece of text (often a word, sometimes sub-word — covered next lesson).
- **Vocabulary**: set of all unique tokens in the corpus.
- **Vocabulary size**: number of tokens in the vocabulary.
- **Vector representation**: a list of numbers that encodes an input.
- **Sparse vector**: mostly zeros; length = vocabulary size.

---

## Questions / Gaps

- [ ] How does BoW handle punctuation / casing in practice?
- [ ] What are TF-IDF weightings and when do they beat raw counts?
- [ ] Next video: how are Word2Vec embeddings actually trained?

---

## Notes

<!-- add your own observations here -->
