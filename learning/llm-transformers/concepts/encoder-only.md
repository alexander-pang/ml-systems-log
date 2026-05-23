---
title: Encoder-only Models
status: todo
date: 2026-05-22
tags: [concept, encoder, bert, representation]
---

# Encoder-only Models

Transform input text into rich contextual embeddings. Best for tasks that require *understanding* rather than *generation*.

---

## What They Do

Take a sequence of tokens → output a dense vector per token (or a pooled sentence vector). The full input is visible at every position (bidirectional attention).

## Canonical Example

**BERT** (Bidirectional Encoder Representations from Transformers)

## Good For

- Text classification
- Named entity recognition
- Semantic similarity / search
- Question answering (extractive)

## Not Good For

- Open-ended text generation (no decoder)

---

## Notes

<!-- fill in as the course progresses -->
