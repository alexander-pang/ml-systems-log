---
title: Transformer Architecture
status: todo
date: 2026-05-22
tags: [transformer, architecture, attention, design]
---

# Transformer Architecture

Decoder-only GPT-style transformer built from scratch in JAX/Flax.

---

## Design Decisions

| Choice | Decision | Rationale |
|--------|----------|-----------|
| Architecture | Decoder-only | language modeling focus |
| Norm placement | Pre-LN | more stable training |
| Positional encoding | Learned / RoPE | TBD |
| Activation | SwiGLU / GELU | TBD |
| Attention | Multi-head causal | standard |

## Module Breakdown

```
TransformerLM
├── Embedding (token + positional)
├── [N × TransformerBlock]
│   ├── LayerNorm
│   ├── CausalSelfAttention
│   │   ├── QKV projection
│   │   ├── Scaled dot-product attention (causal mask)
│   │   └── Output projection
│   ├── LayerNorm
│   └── FFN (2-layer MLP or SwiGLU)
├── LayerNorm
└── LM head (tied weights)
```

## Hyperparameters to Tune

- `n_layers`, `d_model`, `n_heads`, `d_ff`
- `dropout`, `weight_decay`
- Context length

## TODOs

- [ ] Implement `CausalSelfAttention` with JAX `lax.dot_general`
- [ ] Add RoPE positional embeddings
- [ ] Verify causal mask is correct with a toy sequence
- [ ] Weight tying between embedding and LM head

---

## Notes

<!-- decisions, diagrams, open questions -->
