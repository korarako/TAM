from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

ArrayFn = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    dim: int
    energy_fn: ArrayFn
    grad_energy_fn: ArrayFn
    project_fn: ArrayFn
    default_clip_range: float
    n_particles: int | None = None
    spatial_dim: int | None = None
    atom_species: tuple[int, ...] | None = None


def ensure_batch_dim(x: jax.Array, dim: int) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim == 0:
        return x.reshape((1, 1))
    if x.ndim == 1:
        if int(dim) == 1:
            return x.reshape((-1, 1))
        return x.reshape((-1, int(dim)))
    return x.reshape((-1, int(dim)))
