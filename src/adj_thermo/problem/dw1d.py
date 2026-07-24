from __future__ import annotations

import jax

from adj_thermo.problem.base import ProblemSpec, ensure_batch_dim
from adj_thermo.utils import project_identity


def make_problem(barrier: float = 4.0, tilt: float = 0.5) -> ProblemSpec:
    def energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(x, 1)[:, 0]
        return float(barrier) * (y * y - 1.0) ** 2 + float(tilt) * y

    def grad_energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(x, 1)[:, 0]
        grad = 4.0 * float(barrier) * y * (y * y - 1.0) + float(tilt)
        return grad[:, None]

    return ProblemSpec(
        name="dw1d",
        dim=1,
        energy_fn=energy_fn,
        grad_energy_fn=grad_energy_fn,
        project_fn=project_identity,
        default_clip_range=6.0,
    )
