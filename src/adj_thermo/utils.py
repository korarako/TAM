from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


def is_inexact_leaf(x: Any) -> bool:
    dtype = getattr(x, "dtype", None)
    if dtype is None or dtype == jax.dtypes.float0:
        return False
    return bool(jnp.issubdtype(dtype, jnp.inexact))


def project_identity(x: jax.Array) -> jax.Array:
    return x


def project_mean_free(x: jax.Array, n_particles: int, spatial_dim: int) -> jax.Array:
    if int(n_particles) <= 0 or int(spatial_dim) <= 0:
        return x
    shape = x.shape
    y = jnp.asarray(x).reshape((-1, int(n_particles), int(spatial_dim)))
    y = y - jnp.mean(y, axis=1, keepdims=True)
    return y.reshape(shape)


def pairwise_distances(x: jax.Array, n_particles: int, spatial_dim: int, eps: float = 1.0e-12) -> jax.Array:
    y = project_mean_free(x, n_particles=n_particles, spatial_dim=spatial_dim)
    y = y.reshape((-1, int(n_particles), int(spatial_dim)))
    diff = y[:, :, None, :] - y[:, None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + float(eps))
    iu = jnp.triu_indices(int(n_particles), k=1)
    return dist[:, iu[0], iu[1]]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def tree_l2_norm(tree: Any) -> jax.Array:
    leaves = [x for x in jax.tree_util.tree_leaves(tree) if is_inexact_leaf(x)]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(jnp.asarray(x))) for x in leaves))


def tree_clip_by_global_norm(tree: Any, max_norm: float) -> Any:
    if max_norm is None or float(max_norm) <= 0.0:
        return tree
    norm = tree_l2_norm(tree)
    scale = jnp.minimum(1.0, float(max_norm) / (norm + 1.0e-12))

    def clip_leaf(x: Any) -> Any:
        if not is_inexact_leaf(x):
            return x
        return x * scale

    return jax.tree_util.tree_map(clip_leaf, tree)
