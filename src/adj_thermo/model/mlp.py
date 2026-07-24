from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

Params = Any


def _as_column(value: jax.Array | float, batch_size: int, dtype: jnp.dtype) -> jax.Array:
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return jnp.full((batch_size, 1), arr, dtype=dtype)
    if arr.ndim == 1:
        if arr.shape[0] == batch_size:
            return arr[:, None]
        if arr.shape[0] == 1:
            return jnp.full((batch_size, 1), arr[0], dtype=dtype)
    return jnp.broadcast_to(arr.reshape((-1, 1)), (batch_size, 1))


def _sin_cos_embed(x: jax.Array, freqs: jax.Array, use_2pi: bool) -> jax.Array:
    phase = x * freqs[None, :]
    if use_2pi:
        phase = 2.0 * jnp.pi * phase
    emb = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=-1)
    return emb.reshape((x.shape[0], -1))


def sinusoidal_embed_t(t: jax.Array | float, embed_dim: int = 16) -> jax.Array:
    t_arr = jnp.asarray(t, dtype=jnp.float32)
    if t_arr.ndim == 0:
        t_arr = t_arr.reshape((1, 1))
    elif t_arr.ndim == 1:
        t_arr = t_arr[:, None]
    half = max(1, int(math.ceil(embed_dim / 2)))
    freqs = 2.0 ** jnp.arange(half, dtype=jnp.float32)
    emb = _sin_cos_embed(t_arr, freqs, use_2pi=True)
    return emb[:, : int(embed_dim)]


def sinusoidal_embed_beta(beta: jax.Array | float, embed_dim: int = 8, f_max: float = 10.0) -> jax.Array:
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    if beta_arr.ndim == 0:
        beta_arr = beta_arr.reshape((1, 1))
    elif beta_arr.ndim == 1:
        beta_arr = beta_arr[:, None]
    half = max(1, int(math.ceil(embed_dim / 2)))
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(float(f_max)), half, dtype=jnp.float32))
    emb = _sin_cos_embed(beta_arr, freqs, use_2pi=False)
    return emb[:, : int(embed_dim)]


def silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def init_mlp(key: jax.Array, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int) -> tuple[dict[str, jax.Array], ...]:
    dims = [int(input_dim)] + [int(hidden_dim)] * int(n_layers) + [int(output_dim)]
    keys = jax.random.split(key, len(dims) - 1)
    layers = []
    for k, fan_in, fan_out in zip(keys, dims[:-1], dims[1:]):
        limit = math.sqrt(6.0 / float(fan_in + fan_out))
        w = jax.random.uniform(k, (fan_in, fan_out), minval=-limit, maxval=limit, dtype=jnp.float32)
        b = jnp.zeros((fan_out,), dtype=jnp.float32)
        layers.append({"w": w, "b": b})
    return tuple(layers)


def apply_mlp(
    params: Params,
    x: jax.Array,
    t: jax.Array | float,
    beta: jax.Array | float,
    problem_spec: Any,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim == 1:
        x = x.reshape((-1, int(problem_spec.dim)))
    batch_size = int(x.shape[0])
    t_col = _as_column(t, batch_size, x.dtype)
    beta_col = _as_column(beta, batch_size, x.dtype)
    t_emb = sinusoidal_embed_t(t_col, t_embed_dim)
    beta_emb = sinusoidal_embed_beta(beta_col, beta_embed_dim)
    h = jnp.concatenate([x, t_emb, beta_emb], axis=1)
    for layer in params[:-1]:
        h = silu(h @ layer["w"] + layer["b"])
    out = h @ params[-1]["w"] + params[-1]["b"]
    return problem_spec.project_fn(out)
