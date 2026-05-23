---
title: Bag of Words
status: in-progress
date: 2026-05-22
tags: [concept, bag-of-words, representation, sparse]
---

# Bag of Words

A sparse count vector over a fixed vocabulary. One of the earliest and still-useful text representations.

See also: [[lectures/01-language-representations]]

---

## Algorithm

```
Input text
  → tokenize (split on whitespace / punctuation)
  → look up each token in the vocabulary
  → increment its index in the count vector
  → output: vector of length |vocab|
```

## Strengths

- Simple and interpretable.
- Fast to compute.
- Good baseline for classification tasks.

## Weaknesses

- No word order (loses syntax).
- No semantics (cat and feline are unrelated).
- High-dimensional and sparse for large vocabularies.
- Out-of-vocabulary (OOV) words are silently dropped.

## Variants

| Variant | Change |
|---------|--------|
| Binary BoW | 1 if present, 0 if absent (no counts) |
| TF-IDF | downweights common words across documents |
| N-gram BoW | counts bigrams/trigrams, not just unigrams |

---

## Notes

<!-- examples, experiments, comparisons -->
