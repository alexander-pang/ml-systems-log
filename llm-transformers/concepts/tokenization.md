---
title: Tokenization
status: in-progress
date: 2026-05-22
tags: [concept, tokenization, vocabulary, preprocessing]
---

# Tokenization

The process of splitting raw text into discrete units (tokens) that a model can process.

Tokenizers have a **fixed vocabulary** — they cannot represent every possible word, so they split unknown words into sub-word pieces.

Example: `vocalization → ["vocal", "ization"]`

*More detail coming in the dedicated tokenization lecture.*

---

## Levels of Tokenization

| Level | Example | Notes |
|-------|---------|-------|
| Word | `["hello", "world"]` | simple but OOV-prone |
| Sub-word (BPE) | `["hel", "lo", "world"]` | balances vocab size and OOV |
| Character | `['h','e','l','l','o']` | no OOV, long sequences |
| Byte | raw UTF-8 bytes | truly universal |

## Key Terms

- **Vocabulary**: the fixed set of tokens the model knows.
- **Vocabulary size**: e.g. GPT-2 uses 50,257 BPE tokens.
- **OOV (out-of-vocabulary)**: a token not seen at training time.
- **Special tokens**: `[CLS]`, `[SEP]`, `<EOS>`, `<PAD>` etc.

---

## Questions

- [ ] How does BPE decide where to split?
- [ ] Why does vocabulary size affect model size?

---

## Notes

<!-- fill in after next lecture -->
