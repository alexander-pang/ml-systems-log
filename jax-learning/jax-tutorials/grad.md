---
title: grad — Automatic Differentiation in JAX
status: todo
date: 2026-05-22
tags: [jax, grad, autodiff, differentiation]
---

# grad

`jax.grad` returns a function that computes the gradient of a scalar-valued function with respect to its first argument (or any specified argument).

---

## Key Ideas

- JAX uses **reverse-mode AD** (`grad`) for scalar outputs and **forward-mode AD** (`jvp`) for Jacobian-vector products.
- `value_and_grad` returns both the value and the gradient in one pass.
- Higher-order derivatives by composing `grad`.

## Basic Usage

```python
import jax

def loss(params, x):
    return jnp.sum((params @ x) ** 2)

grad_loss = jax.grad(loss)          # gradient w.r.t. params (argnums=0 default)
g = grad_loss(params, x)
```

## `value_and_grad`

```python
loss_val, grads = jax.value_and_grad(loss)(params, x)
```

## Higher-Order Derivatives

```python
d2f = jax.grad(jax.grad(f))         # second derivative
```

## `argnums`

```python
# gradient w.r.t. second argument
jax.grad(loss, argnums=1)(params, x)
```

## Jacobians

```python
jax.jacobian(f)(x)                  # full Jacobian (reverse-mode)
jax.jacfwd(f)(x)                    # forward-mode
```

## Exercises / TODOs

- [ ] Implement gradient descent on a quadratic with `grad`
- [ ] Compute per-sample gradients: `vmap(grad(loss))(params, batch)`
- [ ] Explore `jax.hessian` for second-order optimization

---

## Notes

<!-- add observations, surprises, and gotchas here -->
