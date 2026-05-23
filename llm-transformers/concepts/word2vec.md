---
title: Word2Vec
status: in-progress
date: 2026-05-22
tags: [concept, word2vec, embeddings, dense]
---

# Word2Vec

Released 2013. Dense word embeddings trained by predicting neighboring words. Each word gets one fixed vector regardless of context.

See also: [[lectures/01-language-representations]], [[lectures/02-word2vec-embeddings]]

---

## Key Idea

Words that appear in similar contexts have similar embeddings → meaning is captured geometrically.

```
king − man + woman ≈ queen
```

## Training

1. Initialize every vocabulary word with a random vector.
2. Sample word pairs from training data.
3. Neural network predicts: are these two words likely neighbors?
4. Words sharing the same neighbors move closer in embedding space.

| Variant | Predicts |
|---------|---------|
| CBOW | center word from surrounding context |
| Skip-gram | surrounding context from center word |

## What Dimensions Represent

Each dimension encodes some emergent property (animal-ness, plurality, etc.) — not hand-crafted, learned from data. Values are between -1 and 1; typical size is 100–1000d.

## Properties

- **Dense**: every dimension carries signal (vs. sparse BoW zeros).
- **Fixed**: one vector per word regardless of sentence context.
- **Sub-word fallback**: unknown words split into sub-word tokens; word embedding = average of token embeddings.

## Limitations

- "Bank" (river) and "bank" (financial) get the same vector.
- Context window is small (5–10 words).
- Superseded by contextual embeddings (BERT, GPT).

---

## Notes

<!-- observations, experiments -->
