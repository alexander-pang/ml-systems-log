---
title: Encoder-Decoder Models
status: todo
date: 2026-05-22
tags: [concept, encoder-decoder, seq2seq, t5]
---

# Encoder-Decoder Models

Combine a bidirectional encoder (understands input) with an autoregressive decoder (generates output). Best for sequence-to-sequence tasks.

---

## What They Do

Encoder reads and compresses the full input → decoder attends to the encoder output while generating tokens one at a time.

## Canonical Examples

T5, BART, mT5, MarianMT

## Good For

- Machine translation
- Summarization
- Question answering (abstractive)
- Any task with a structured input → structured output shape

---

## Notes

<!-- fill in as the course progresses -->
