from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from adj_thermo.utils import is_inexact_leaf, tree_clip_by_global_norm

Params = Any


class AdamState(NamedTuple):
    step: jax.Array
    m: Params
    v: Params


def init_adam(params: Params) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(step=jnp.array(0, dtype=jnp.int32), m=zeros, v=zeros)


def clip_grads(grads: Params, max_norm: float) -> Params:
    return tree_clip_by_global_norm(grads, max_norm)


def adam_update(
    params: Params,
    grads: Params,
    state: AdamState,
    lr: float | jax.Array,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
    weight_decay: float = 0.0,
) -> tuple[Params, AdamState]:
    step = state.step + jnp.array(1, dtype=jnp.int32)

    def grad_or_zero(p: Any, g: Any) -> Any:
        if not is_inexact_leaf(p) or not is_inexact_leaf(g):
            return jnp.zeros_like(p)
        return g

    def update_m(p: Any, m_leaf: Any, g: Any) -> Any:
        if not is_inexact_leaf(p):
            return m_leaf
        clean_g = grad_or_zero(p, g)
        return beta1 * m_leaf + (1.0 - beta1) * clean_g

    def update_v(p: Any, v_leaf: Any, g: Any) -> Any:
        if not is_inexact_leaf(p):
            return v_leaf
        clean_g = grad_or_zero(p, g)
        return beta2 * v_leaf + (1.0 - beta2) * jnp.square(clean_g)

    m = jax.tree_util.tree_map(update_m, params, state.m, grads)
    v = jax.tree_util.tree_map(update_v, params, state.v, grads)
    b1 = 1.0 - jnp.asarray(beta1, dtype=jnp.float32) ** step
    b2 = 1.0 - jnp.asarray(beta2, dtype=jnp.float32) ** step

    def update(p: Any, m_leaf: Any, v_leaf: Any) -> Any:
        if not is_inexact_leaf(p):
            return p
        m_hat = m_leaf / b1
        v_hat = v_leaf / b2
        update_leaf = m_hat / (jnp.sqrt(v_hat) + eps)
        if float(weight_decay) > 0.0:
            update_leaf = update_leaf + float(weight_decay) * p
        return p - lr * update_leaf

    new_params = jax.tree_util.tree_map(update, params, m, v)
    return new_params, AdamState(step=step, m=m, v=v)
