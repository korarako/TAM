from __future__ import annotations

import jax
import jax.numpy as jnp

from adj_thermo.problem.base import ProblemSpec, ensure_batch_dim
from adj_thermo.utils import project_identity

_A = jnp.asarray([-200.0, -100.0, -170.0, 15.0], dtype=jnp.float32)
_a = jnp.asarray([-1.0, -1.0, -6.5, 0.7], dtype=jnp.float32)
_b = jnp.asarray([0.0, 0.0, 11.0, 0.6], dtype=jnp.float32)
_c = jnp.asarray([-10.0, -10.0, -6.5, 0.7], dtype=jnp.float32)
_x0 = jnp.asarray([1.0, 0.0, -0.5, -1.0], dtype=jnp.float32)
_y0 = jnp.asarray([0.0, 0.5, 1.5, 1.0], dtype=jnp.float32)


def make_problem(scale: float = 0.02, shift: float = 0.0) -> ProblemSpec:
    def energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(x, 2)
        dx = y[:, 0:1] - _x0
        dy = y[:, 1:2] - _y0
        exponent = _a * dx * dx + _b * dx * dy + _c * dy * dy
        return float(scale) * jnp.sum(_A * jnp.exp(exponent), axis=-1) + float(shift)

    def grad_energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(x, 2)
        dx = y[:, 0:1] - _x0
        dy = y[:, 1:2] - _y0
        exponent = _a * dx * dx + _b * dx * dy + _c * dy * dy
        weight = _A * jnp.exp(exponent)
        grad_x = jnp.sum(weight * (2.0 * _a * dx + _b * dy), axis=-1)
        grad_y = jnp.sum(weight * (_b * dx + 2.0 * _c * dy), axis=-1)
        return float(scale) * jnp.stack([grad_x, grad_y], axis=1)

    return ProblemSpec(
        name="mb2d",
        dim=2,
        energy_fn=energy_fn,
        grad_energy_fn=grad_energy_fn,
        project_fn=project_identity,
        default_clip_range=3.0,
    )
