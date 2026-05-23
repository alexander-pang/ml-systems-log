---
title: Chinchilla Scaling Experiments
status: todo
date: 2026-05-22
tags: [transformer, chinchilla, scaling-laws, experiments]
---

# Chinchilla Scaling Experiments

Applying Chinchilla-optimal compute allocation to decide model size vs. training tokens.

---

## The Chinchilla Rule of Thumb

> For a compute-optimal run: **N ≈ D** (model params ≈ training tokens)
> Both should scale equally with compute budget C ≈ 6ND.

| Compute (FLOPs) | Params (N) | Tokens (D) |
|-----------------|-----------|------------|
| 1e18 | ~400M | ~8B |
| 1e19 | ~1.4B | ~27B |
| 1e20 | ~4.5B | ~86B |

*From Hoffmann et al. 2022 (Table 3)*

---

## Planned Experiment Grid

| Run | N | D | C (est.) | Goal |
|-----|---|---|----------|------|
| tiny-baseline | 1M | 1M | — | overfit sanity check |
| small-chinchilla | 10M | 200M | — | first scaling point |
| medium-chinchilla | 50M | 1B | — | compare to baseline |

## Metrics to Track

- Final eval loss / perplexity
- Training loss curve shape
- Tokens/sec throughput
- GPU utilization

## Questions to Answer

- [ ] At what N/D ratio does loss diverge from Chinchilla predictions on my setup?
- [ ] How sensitive is final loss to LR schedule vs. compute budget?
- [ ] Does SwiGLU vs. GELU matter at small scale?

## References

- [[papers/chinchilla]] — Hoffmann et al. 2022
- [[papers/gpt3]] — Brown et al. 2020 (for comparison)
- [[transformer-project/architecture]]
- [[transformer-project/training-loop]]

---

## Notes

<!-- results, surprises, plots -->
