---
title: Dataset Pipeline
status: todo
date: 2026-05-22
tags: [transformer, dataset, tokenization, data]
---

# Dataset Pipeline

Data loading, tokenization, and batching for the transformer project.

---

## Dataset Choices

| Option | Tokens | Notes |
|--------|--------|-------|
| TinyShakespeare | ~1M | good for sanity checks |
| WikiText-103 | ~103M | medium scale |
| OpenWebText | ~9B | GPT-2 scale |
| The Pile (subset) | varies | research-grade |

## Tokenizer

- [ ] BPE via `tiktoken` (GPT-2/GPT-4 vocab) — simplest to start
- [ ] SentencePiece for custom vocab
- Target vocab size: ~50k

## Batching Strategy

```python
# Packed sequences (no padding waste)
# 1. Tokenize all documents
# 2. Concatenate with <EOS> separator
# 3. Chunk into context_length windows
# 4. Shuffle at the chunk level
```

## JAX DataLoader Notes

JAX has no built-in DataLoader. Options:
- `torch.utils.data.DataLoader` with numpy output (detach from torch)
- `tensorflow.data` pipeline
- Custom generator + `jax.device_put`

## TODOs

- [ ] Download and inspect a dataset
- [ ] Implement tokenization + chunking
- [ ] Write a batch iterator that yields `(input_ids, target_ids)` numpy arrays
- [ ] Profile data throughput vs. training throughput

---

## Notes

<!-- bottlenecks, format decisions, file paths -->
