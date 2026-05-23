---
title: JIT Compilation in JAX
status: todo
date: 2026-05-22
tags: [jax, jit, compilation, xla]
---

# JIT Compilation

`jax.jit` traces a Python function and compiles it to XLA, running it on accelerators with minimal Python overhead on subsequent calls.

---

## Key Ideas

- **Tracing**: JAX replaces concrete values with abstract tracers to build a computation graph.
- **Static vs. dynamic**: shapes must be static at trace time; values are dynamic.
- **Caching**: recompilation happens when static arguments or shapes change.

## Basic Usage

```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

x = jnp.array([1.0, 2.0, 3.0])
print(f(x))  # compiled on first call
print(f(x))  # cached
```

## Gotchas

- Side effects inside jitted functions run only at trace time.
- Python control flow on dynamic values will be traced as a single branch — use `jax.lax.cond` / `jax.lax.while_loop` instead.
- `static_argnums` or `static_argnames` to mark args that trigger recompile when changed.

## Exercises / TODOs

- [ ] Benchmark jitted vs. un-jitted matrix multiply
- [ ] Explore `jax.make_jaxpr` to inspect the computation graph
- [ ] Try `static_argnums` with a function that branches on an integer flag

---

## Notes

<!-- add observations, surprises, and gotchas here -->
