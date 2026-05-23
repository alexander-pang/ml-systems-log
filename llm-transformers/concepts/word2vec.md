---
title: Word2Vec
status: todo
date: 2026-05-22
tags: [concept, word2vec, embeddings, dense]
---

# Word2Vec

Dense word embeddings trained to predict a word from its neighbors (or vice versa). Each word gets one fixed vector regardless of context.

See also: [[lectures/01-language-representations]]

---

## Key Idea

Words that appear in similar contexts have similar embeddings → meaning is captured geometrically.

```
king − man + woman ≈ queen
```

## Training Objectives

| Variant | Predicts |
|---------|---------|
| CBOW | center word from surrounding context |
| Skip-gram | surrounding context from center word |

## Properties

- **Dense**: every dimension is used (typically 100–300d).
- **Fixed**: one vector per word, independent of sentence context.
- **No OOV**: words not seen at training time have no embedding.

## Limitations

- "Bank" (river) and "bank" (financial) get the same vector.
- Context window is small (5–10 words).
- Superseded by contextual embeddings (BERT, GPT).

---

## Notes

<!-- to fill in after the next lecture -->
