---
title: Training Loop
status: todo
date: 2026-05-22
tags: [transformer, training, optax, flax]
---

# Training Loop

JAX/Flax/Optax training loop for the from-scratch transformer.

---

## Stack

- **Model**: Flax `nn.Module`
- **Optimizer**: Optax (AdamW + cosine LR schedule)
- **Training step**: `jax.jit(train_step)` + optional `pmap` for multi-GPU

## Training Step Skeleton

```python
import optax
import jax
import jax.numpy as jnp
from flax.training import train_state

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, ctx_len), jnp.int32))['params']
    tx = optax.adamw(learning_rate, weight_decay=0.1)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input_ids'])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], batch['input_ids'][:, 1:]
        ).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

## LR Schedule

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=2000,
    decay_steps=100_000,
    end_value=3e-5,
)
```

## Checkpointing

- [ ] Use `orbax-checkpoint` (Flax recommended)
- [ ] Save every N steps and keep last K checkpoints

## Logging

- [ ] Log train loss, grad norm, LR per step
- [ ] Eval perplexity every N steps
- [ ] Weights & Biases or TensorBoard

## TODOs

- [ ] Implement `eval_step` (no grad, split key)
- [ ] Add gradient clipping (`optax.clip_by_global_norm`)
- [ ] Smoke test: overfit on a single batch

---

## Notes

<!-- observations on stability, speed, debugging -->
