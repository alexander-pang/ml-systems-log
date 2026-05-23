---
title: "Lecture 02: Word2Vec and Vector Embeddings"
status: in-progress
date: 2026-05-22
tags: [lecture, word2vec, embeddings, representation, tokenization]
---

# Lecture 02: Word2Vec and Vector Embeddings

## The Problem with Bag of Words

BoW ignores **semantic meaning** — "cat" and "feline" are completely unrelated entries even though they mean the same thing.

---

## Word2Vec (2013)

Learns dense embeddings by training on large corpora (e.g. all of Wikipedia) using a neural network.

### Training Idea

1. Initialize every vocabulary word with a **random vector** (e.g. 5 values).
2. Sample **pairs of words** from training data.
3. Model predicts: are these two words likely to be **neighbors** in a sentence?
4. Weights (parameters) are updated each step.
5. Words that share the same neighbors end up with **similar embeddings**.

> Words with the same neighbors → embeddings move closer together.

### What the Dimensions Represent

Each dimension loosely encodes some **property** of the word — but you don't get to choose or interpret them directly; they emerge from training.

| Word | animal | plural | newborn | human | fruit |
|------|--------|--------|---------|-------|-------|
| cats | high | high | low | low | low |
| puppy | high | low | high | low | low |
| apple | low | low | low | low | high |

- **Dimensionality**: typically 100–1000+ values per embedding.
- **Range**: values between -1 and 1.
- Similar words cluster together in embedding space; dissimilar words are far apart.

---

## Sub-word Tokenization

Word2Vec introduced the idea that tokenizers have a **fixed vocabulary** and can't represent every word — so rare/unseen words are split into sub-word pieces.

Example: `"Her vocalization was melodic."`

```
vocalization → ["vocal", "ization"]
```

To recover a word embedding from sub-word tokens: **average** the token embeddings.

---

## Types of Embeddings

| Unit | How to get it |
|------|--------------|
| Token embedding | direct output of model |
| Word embedding | average sub-word token embeddings |
| Sentence embedding | average (or pool) all token embeddings |
| Document embedding | same, over longer text |

Models that convert text → embeddings are called **representation models**.

---

## Limitations of Word2Vec

- One **fixed** vector per word regardless of context.
- "Bank" (river) and "bank" (financial) get the same embedding.
- Context window is small (a few neighboring words).
- → Motivates **contextualized** embeddings (next lecture: encoders/decoders).

---

## Related Notes

- [[concepts/word2vec]]
- [[concepts/bag-of-words]]
- [[concepts/tokenization]]

---

## Questions / Gaps

- [ ] What's the difference between Skip-gram and CBOW training objectives?
- [ ] How does averaging sub-word embeddings compare to a learned pooling?
- [ ] Next video: how do encoders/decoders produce contextualized embeddings?

---

## Notes

<!-- your observations -->
