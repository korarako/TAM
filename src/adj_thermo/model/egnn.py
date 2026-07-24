from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from adj_thermo.model.mlp import sinusoidal_embed_beta, sinusoidal_embed_t, silu
from adj_thermo.problem.base import ProblemSpec

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


def _init_linear(key: jax.Array, fan_in: int, fan_out: int, scale: float = 1.0) -> dict[str, jax.Array]:
    limit = float(scale) * math.sqrt(6.0 / float(fan_in + fan_out))
    return {
        "w": jax.random.uniform(key, (int(fan_in), int(fan_out)), minval=-limit, maxval=limit, dtype=jnp.float32),
        "b": jnp.zeros((int(fan_out),), dtype=jnp.float32),
    }


def _init_mlp(key: jax.Array, dims: tuple[int, ...], final_scale: float = 1.0) -> tuple[dict[str, jax.Array], ...]:
    keys = jax.random.split(key, len(dims) - 1)
    layers = []
    for i, (k, fan_in, fan_out) in enumerate(zip(keys, dims[:-1], dims[1:])):
        scale = final_scale if i == len(dims) - 2 else 1.0
        layers.append(_init_linear(k, fan_in, fan_out, scale=scale))
    return tuple(layers)


def _apply_mlp(params: tuple[dict[str, jax.Array], ...], x: jax.Array, activate_final: bool = False) -> jax.Array:
    h = x
    for i, layer in enumerate(params):
        h = h @ layer["w"] + layer["b"]
        if i < len(params) - 1 or activate_final:
            h = silu(h)
    return h


def egnn_velocity_config(
    problem: ProblemSpec,
    hidden_dim: int = 128,
    n_layers: int = 5,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    radial_dim: int = 16,
    radial_max: float = 12.0,
    coord_scale: float = 0.1,
) -> dict[str, Any]:
    if problem.n_particles is None or problem.spatial_dim is None:
        raise ValueError("EGNN velocity requires n_particles and spatial_dim.")
    if int(problem.dim) != int(problem.n_particles) * int(problem.spatial_dim):
        raise ValueError("EGNN problem dim must equal n_particles * spatial_dim.")
    return {
        "model_type": "egnn",
        "dim": int(problem.dim),
        "n_particles": int(problem.n_particles),
        "spatial_dim": int(problem.spatial_dim),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "t_embed_dim": int(t_embed_dim),
        "beta_embed_dim": int(beta_embed_dim),
        "radial_dim": int(radial_dim),
        "radial_max": float(radial_max),
        "coord_scale": float(coord_scale),
    }


def _rbf(dist: jax.Array, config: dict[str, Any]) -> jax.Array:
    radial_dim = int(config["radial_dim"])
    radial_max = float(config["radial_max"])
    centers = jnp.linspace(0.0, radial_max, radial_dim, dtype=dist.dtype)
    width = radial_max / max(radial_dim - 1, 1)
    return jnp.exp(-0.5 * ((dist - centers) / (width + 1.0e-6)) ** 2)


def init_egnn_velocity(
    key: jax.Array,
    problem: ProblemSpec,
    hidden_dim: int = 128,
    n_layers: int = 5,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    radial_dim: int = 16,
    radial_max: float = 12.0,
) -> dict[str, Any]:
    config = egnn_velocity_config(
        problem,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        t_embed_dim=t_embed_dim,
        beta_embed_dim=beta_embed_dim,
        radial_dim=radial_dim,
        radial_max=radial_max,
    )
    hidden_dim = int(config["hidden_dim"])
    radial_dim = int(config["radial_dim"])
    cond_dim = int(config["t_embed_dim"]) + int(config["beta_embed_dim"])
    keys = iter(jax.random.split(key, 3 + 3 * int(config["n_layers"])))
    params: dict[str, Any] = {
        "cond_mlp": _init_mlp(next(keys), (cond_dim, hidden_dim, hidden_dim)),
        "layers": [],
    }
    edge_in_dim = 2 * hidden_dim + radial_dim + 1
    for _ in range(int(config["n_layers"])):
        params["layers"].append(
            {
                "edge_mlp": _init_mlp(next(keys), (edge_in_dim, hidden_dim, hidden_dim)),
                "coord_mlp": _init_mlp(next(keys), (hidden_dim, hidden_dim, 1), final_scale=1.0e-2),
                "node_mlp": _init_mlp(next(keys), (2 * hidden_dim, hidden_dim, hidden_dim)),
            }
        )
    params["layers"] = tuple(params["layers"])
    params["out_edge_mlp"] = _init_mlp(next(keys), (edge_in_dim, hidden_dim, hidden_dim))
    params["out_coord_mlp"] = _init_mlp(next(keys), (hidden_dim, hidden_dim, 1), final_scale=1.0e-2)
    return {"model_type": "egnn", "config": config, "params": params}


def is_egnn_velocity_params(params: Any) -> bool:
    return isinstance(params, dict) and params.get("model_type") == "egnn" and "params" in params and "config" in params


def _edge_inputs(coords: jax.Array, h: jax.Array, config: dict[str, Any]) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_particles = int(config["n_particles"])
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist2 = jnp.sum(diff * diff, axis=-1, keepdims=True)
    dist = jnp.sqrt(dist2 + 1.0e-8)
    rbf = _rbf(dist, config)
    h_i = jnp.broadcast_to(h[:, :, None, :], (h.shape[0], n_particles, n_particles, h.shape[-1]))
    h_j = jnp.broadcast_to(h[:, None, :, :], (h.shape[0], n_particles, n_particles, h.shape[-1]))
    edge_in = jnp.concatenate([h_i, h_j, rbf, dist2], axis=-1)
    mask = (1.0 - jnp.eye(n_particles, dtype=coords.dtype))[None, :, :, None]
    return edge_in, diff, mask


def apply_egnn_velocity_params(
    params: Params,
    x: jax.Array,
    t: jax.Array | float,
    beta: jax.Array | float,
    problem: ProblemSpec,
    config: dict[str, Any],
) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim == 1:
        x = x.reshape((-1, int(config["dim"])))
    batch_size = int(x.shape[0])
    n_particles = int(config["n_particles"])
    spatial_dim = int(config["spatial_dim"])
    hidden_dim = int(config["hidden_dim"])
    coord_scale = float(config["coord_scale"])
    denom = max(n_particles - 1, 1)

    x = problem.project_fn(x)
    coords = x.reshape((batch_size, n_particles, spatial_dim))
    t_col = _as_column(t, batch_size, x.dtype)
    beta_col = _as_column(beta, batch_size, x.dtype)
    cond = jnp.concatenate(
        [sinusoidal_embed_t(t_col, int(config["t_embed_dim"])), sinusoidal_embed_beta(beta_col, int(config["beta_embed_dim"]))],
        axis=-1,
    )
    h0 = _apply_mlp(params["cond_mlp"], cond)
    h = jnp.broadcast_to(h0[:, None, :], (batch_size, n_particles, hidden_dim))

    for layer in params["layers"]:
        edge_in, diff, mask = _edge_inputs(coords, h, config)
        m = _apply_mlp(layer["edge_mlp"], edge_in) * mask
        coord_w = _apply_mlp(layer["coord_mlp"], m) * mask
        delta = jnp.sum(diff * coord_w, axis=2) / float(denom)
        coords = coords + coord_scale * delta
        coords = coords - jnp.mean(coords, axis=1, keepdims=True)
        m_sum = jnp.sum(m, axis=2)
        h = h + _apply_mlp(layer["node_mlp"], jnp.concatenate([h, m_sum], axis=-1))

    edge_in, diff, mask = _edge_inputs(coords, h, config)
    m = _apply_mlp(params["out_edge_mlp"], edge_in) * mask
    coord_w = _apply_mlp(params["out_coord_mlp"], m) * mask
    velocity = jnp.sum(diff * coord_w, axis=2) / float(denom)
    return problem.project_fn(velocity.reshape((batch_size, int(config["dim"]))))


def apply_egnn_velocity_package(
    package: dict[str, Any],
    x: jax.Array,
    t: jax.Array | float,
    beta: jax.Array | float,
    problem: ProblemSpec,
) -> jax.Array:
    if not is_egnn_velocity_params(package):
        raise ValueError("Expected EGNN velocity package with model_type/config/params.")
    return apply_egnn_velocity_params(package["params"], x, t, beta, problem, package["config"])
