---
title: vmap — Vectorization in JAX
status: todo
date: 2026-05-22
tags: [jax, vmap, vectorization, batching]
---

# vmap

`jax.vmap` automatically vectorizes a function written for a single example so it runs over a batch — no manual broadcasting required.

---

## Key Ideas

- Write your function for a **single** input, then `vmap` it over a batch axis.
- Composes cleanly with `jit`: `jax.jit(jax.vmap(f))`.
- `in_axes` / `out_axes` control which axis of each argument is the batch axis.

## Basic Usage

```python
import jax
import jax.numpy as jnp

def dot(a, b):
    return jnp.dot(a, b)

batched_dot = jax.vmap(dot)

A = jnp.ones((4, 3))
B = jnp.ones((4, 3))
print(batched_dot(A, B).shape)  # (4,)
```

## `in_axes` Examples

```python
# broadcast b across the batch of a
jax.vmap(dot, in_axes=(0, None))(A, jnp.ones(3))
```

## Composing vmap

```python
# double vmap for 2D batches
jax.vmap(jax.vmap(f))(batch_2d)
```

## Exercises / TODOs

- [ ] Implement batched per-sample gradients with `vmap(grad(loss))`
- [ ] Compare vmap vs. manual batching on a small MLP forward pass
- [ ] Explore `jax.lax.map` for sequential (memory-constrained) alternatives

---

## Notes

<!-- add observations, surprises, and gotchas here -->
