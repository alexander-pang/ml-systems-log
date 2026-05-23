---
title: pmap — Parallel Execution in JAX
status: todo
date: 2026-05-22
tags: [jax, pmap, parallelism, multi-device]
---

# pmap

`jax.pmap` maps a function over leading axes, executing each slice on a separate device (GPU/TPU). Think `vmap` but for hardware parallelism.

---

## Key Ideas

- Each device gets one slice of the batch along the mapped axis.
- Collective ops (`lax.pmean`, `lax.psum`, `lax.pmax`) communicate across devices.
- `axis_name` is required for collectives so JAX knows which parallel axis to reduce over.

## Basic Usage

```python
import jax
import jax.numpy as jnp

n_devices = jax.device_count()

@jax.pmap
def f(x):
    return x ** 2

xs = jnp.arange(n_devices * 4).reshape(n_devices, 4)
print(f(xs))   # each device processes one row
```

## Gradient Averaging Across Devices

```python
@jax.pmap
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    grads = jax.lax.pmean(grads, axis_name='devices')
    return loss, grads

# replicate params across devices first:
params = jax.device_put_replicated(params, jax.devices())
```

## Replicating vs. Sharding

- **Replicate** (same value on every device): `jax.device_put_replicated`
- **Shard** (different slice per device): split the batch along axis 0

## Exercises / TODOs

- [ ] Run a toy pmap on CPU with `CUDA_VISIBLE_DEVICES` or `XLA_FLAGS`
- [ ] Implement data-parallel training with `pmean` gradient sync
- [ ] Compare pmap vs. `jax.sharding` / GSPMD for larger models

---

## Notes

<!-- add observations, surprises, and gotchas here -->
