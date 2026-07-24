from __future__ import annotations

from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from painn_jax import NodeFeatures, PaiNN, cosine_cutoff, gaussian_rbf

from adj_thermo.model.mlp import sinusoidal_embed_beta, sinusoidal_embed_t
from adj_thermo.problem.base import ProblemSpec


def _complete_directed_edges(n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    senders = []
    receivers = []
    for i in range(int(n_particles)):
        for j in range(int(n_particles)):
            if i != j:
                senders.append(i)
                receivers.append(j)
    return np.asarray(senders, dtype=np.int32), np.asarray(receivers, dtype=np.int32)


def painn_velocity_config(
    problem: ProblemSpec,
    hidden_dim: int = 64,
    n_layers: int = 3,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    radius: float | None = None,
    n_rbf: int | None = None,
    atom_identity: str = "auto",
    atom_embed_dim: int = 16,
) -> dict[str, Any]:
    if problem.n_particles is None or problem.spatial_dim != 3:
        raise ValueError("PaiNN velocity requires a molecular problem with n_particles and spatial_dim=3.")
    if problem.name != "lj13":
        raise ValueError("The released PaiNN implementation is configured for LJ13.")
    if radius is None:
        radius = 6.0
    if n_rbf is None:
        n_rbf = 32
    atom_species = tuple(int(v) for v in (problem.atom_species or (1,) * int(problem.n_particles)))
    if len(atom_species) != int(problem.n_particles):
        raise ValueError("problem.atom_species length must match n_particles for PaiNN.")
    identity_mode = str(atom_identity).lower()
    if identity_mode == "auto":
        identity_mode = "none"
    if identity_mode not in {"none", "learned"}:
        raise ValueError(f"Unsupported PaiNN atom identity mode {atom_identity!r}; use auto, learned, or none.")
    atom_identity_dim = int(atom_embed_dim) if identity_mode == "learned" else 0
    return {
        "model_type": "painn",
        "problem_name": str(problem.name),
        "n_particles": int(problem.n_particles),
        "spatial_dim": int(problem.spatial_dim),
        "dim": int(problem.dim),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "t_embed_dim": int(t_embed_dim),
        "beta_embed_dim": int(beta_embed_dim),
        "radius": float(radius),
        "n_rbf": int(n_rbf),
        "atom_species": atom_species,
        "atom_identity": identity_mode,
        "atom_embed_dim": atom_identity_dim,
        "node_feature_dim": 2 + atom_identity_dim + int(t_embed_dim) + int(beta_embed_dim),
        "radial_basis": "painn_jax.gaussian_rbf",
        "cutoff": "painn_jax.cosine_cutoff",
    }


def _as_beta_vector(beta: float | jax.Array, batch_size: int, dtype: jnp.dtype) -> jax.Array:
    arr = jnp.asarray(beta, dtype=dtype)
    if arr.ndim == 0:
        return jnp.full((batch_size,), arr, dtype=dtype)
    return jnp.reshape(arr, (batch_size, -1))[:, 0].astype(dtype)


def _as_time_vector(t: float | jax.Array, batch_size: int, dtype: jnp.dtype) -> jax.Array:
    arr = jnp.asarray(t, dtype=dtype)
    if arr.ndim == 0:
        return jnp.full((batch_size,), arr, dtype=dtype)
    return jnp.reshape(arr, (batch_size, -1))[:, 0].astype(dtype)


def _painn_graph(
    x: jax.Array,
    t: float | jax.Array,
    beta: float | jax.Array,
    config: dict[str, Any],
    atom_embedding: jax.Array | None = None,
) -> jraph.GraphsTuple:
    n_particles = int(config["n_particles"])
    spatial_dim = int(config["spatial_dim"])
    batch_size = int(x.shape[0])
    dtype = x.dtype
    positions = jnp.reshape(x, (batch_size * n_particles, spatial_dim))

    t_vec = _as_time_vector(t, batch_size, dtype)
    beta_vec = _as_beta_vector(beta, batch_size, dtype)
    t_emb = sinusoidal_embed_t(t_vec[:, None], embed_dim=int(config["t_embed_dim"])).astype(dtype)
    beta_emb = sinusoidal_embed_beta(beta_vec[:, None], embed_dim=int(config["beta_embed_dim"])).astype(dtype)
    global_features = jnp.concatenate([t_emb, beta_emb], axis=-1)
    repeated_global = jnp.repeat(global_features, n_particles, axis=0)

    species = jnp.asarray(config.get("atom_species", (1,) * n_particles), dtype=dtype)
    species = jnp.tile(species[None, :], (batch_size, 1)).reshape((-1, 1)) / 10.0
    ones = jnp.ones((batch_size * n_particles, 1), dtype=dtype)
    node_parts = [ones, species]
    if str(config.get("atom_identity", "none")).lower() == "learned":
        if atom_embedding is None:
            raise ValueError("Learned PaiNN atom identity requires atom_embedding parameters.")
        atom_emb = jnp.asarray(atom_embedding, dtype=dtype)
        atom_emb = jnp.tile(atom_emb[None, :, :], (batch_size, 1, 1)).reshape((batch_size * n_particles, -1))
        node_parts.append(atom_emb)
    node_parts.append(repeated_global)
    node_s = jnp.concatenate(node_parts, axis=-1)

    base_senders_np, base_receivers_np = _complete_directed_edges(n_particles)
    base_senders = jnp.asarray(base_senders_np, dtype=jnp.int32)
    base_receivers = jnp.asarray(base_receivers_np, dtype=jnp.int32)
    n_edges = int(base_senders_np.shape[0])
    offsets = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32) * n_particles, n_edges)
    senders = jnp.tile(base_senders, batch_size) + offsets
    receivers = jnp.tile(base_receivers, batch_size) + offsets
    edges = positions[receivers] - positions[senders]

    return jraph.GraphsTuple(
        nodes=NodeFeatures(s=node_s, v=None),
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=None,
        n_node=jnp.full((batch_size,), n_particles, dtype=jnp.int32),
        n_edge=jnp.full((batch_size,), n_edges, dtype=jnp.int32),
    )


def build_painn_velocity_model(config: dict[str, Any]):
    def forward(graph: jraph.GraphsTuple) -> jax.Array:
        model = PaiNN(
            hidden_size=int(config["hidden_dim"]),
            n_layers=int(config["n_layers"]),
            radial_basis_fn=gaussian_rbf,
            cutoff_fn=cosine_cutoff,
            radius=float(config["radius"]),
            n_rbf=int(config["n_rbf"]),
            node_type="continuous",
            task="node",
            pool="sum",
            out_channels=1,
        )
        _, v = model(graph)
        return jnp.asarray(v)

    return hk.transform_with_state(forward)


def init_painn_velocity(
    key: jax.Array,
    problem: ProblemSpec,
    hidden_dim: int = 64,
    n_layers: int = 3,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    atom_identity: str = "auto",
    atom_embed_dim: int = 16,
) -> tuple[dict[str, Any], Any]:
    config = painn_velocity_config(
        problem,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        t_embed_dim=t_embed_dim,
        beta_embed_dim=beta_embed_dim,
        atom_identity=atom_identity,
        atom_embed_dim=atom_embed_dim,
    )
    extra_params: dict[str, Any] = {}
    if str(config.get("atom_identity", "none")).lower() == "learned":
        key, atom_key = jax.random.split(key)
        scale = jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float32))
        extra_params["atom_embedding"] = jax.random.uniform(
            atom_key,
            (int(config["n_particles"]), int(config["atom_embed_dim"])),
            minval=-scale,
            maxval=scale,
            dtype=jnp.float32,
        )
    model = build_painn_velocity_model(config)
    x = problem.project_fn(jax.random.normal(key, (1, int(problem.dim)), dtype=jnp.float32))
    graph = _painn_graph(
        x,
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        config,
        extra_params.get("atom_embedding"),
    )
    params, state = model.init(key, graph)
    package = {"model_type": "painn", "config": config, "params": params, "state": state}
    if extra_params:
        package["extra_params"] = extra_params
    return package, model


def is_painn_velocity_params(params: Any) -> bool:
    return isinstance(params, dict) and params.get("model_type") == "painn" and "params" in params and "state" in params


def painn_trainable_params_from_package(package: dict[str, Any]) -> Any:
    if "extra_params" in package:
        return {"model": package["params"], "extra": package["extra_params"]}
    return package["params"]


def split_painn_trainable_params(params: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(params, dict) and "model" in params and "extra" in params:
        return params["model"], params["extra"]
    return params, {}


def apply_painn_velocity_params(
    model: Any,
    params: Any,
    state: Any,
    x: jax.Array,
    t: float | jax.Array,
    beta: float | jax.Array,
    problem: ProblemSpec,
    config: dict[str, Any],
) -> jax.Array:
    x = problem.project_fn(x)
    model_params, extra_params = split_painn_trainable_params(params)
    graph = _painn_graph(x, t, beta, config, extra_params.get("atom_embedding"))
    v, _ = model.apply(model_params, state, None, graph)
    flat = jnp.reshape(v, (x.shape[0], int(config["dim"])))
    return problem.project_fn(flat)


def apply_painn_velocity_package(
    package: dict[str, Any],
    x: jax.Array,
    t: float | jax.Array,
    beta: float | jax.Array,
    problem: ProblemSpec,
    model: Any | None = None,
) -> jax.Array:
    if not is_painn_velocity_params(package):
        raise ValueError("Expected a PaiNN velocity parameter package.")
    config = package["config"]
    if model is None:
        model = build_painn_velocity_model(config)
    return apply_painn_velocity_params(
        model,
        painn_trainable_params_from_package(package),
        package["state"],
        x,
        t,
        beta,
        problem,
        config,
    )
