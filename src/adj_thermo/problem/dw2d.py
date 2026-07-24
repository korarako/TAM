from __future__ import annotations

import jax
import jax.numpy as jnp

from adj_thermo.problem.base import ProblemSpec, ensure_batch_dim
from adj_thermo.utils import project_identity


def make_problem() -> ProblemSpec:
    def energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(x, 2)
        x1, x2 = y[:, 0], y[:, 1]
        return 0.25 * (x1 * x1 - 1.0) ** 2 + 0.5 * x2 * x2

    def grad_energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(x, 2)
        x1, x2 = y[:, 0], y[:, 1]
        return jnp.stack([x1 * (x1 * x1 - 1.0), x2], axis=1)

    return ProblemSpec(
        name="dw2d",
        dim=2,
        energy_fn=energy_fn,
        grad_energy_fn=grad_energy_fn,
        project_fn=project_identity,
        default_clip_range=6.0,
    )
