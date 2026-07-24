from __future__ import annotations

import jax
import jax.numpy as jnp

from adj_thermo.problem.base import ProblemSpec, ensure_batch_dim
from adj_thermo.utils import pairwise_distances, project_mean_free


def make_problem(
    eps: float = 1.0,
    rm: float = 1.0,
    oscillator_scale: float = 2.0,
    energy_factor: float = 1.0,
    min_distance: float = 0.1,
) -> ProblemSpec:
    """LJ-13 benchmark potential.

    Defaults follow the common LJ benchmark convention:
    U = sum_ij [(rm / d_ij)^12 - 2 (rm / d_ij)^6]
        + sum_i ||x_i - COM(x)||^2
    with rm=1, eps/tau=1, and c_osc=1.  The implementation writes the harmonic
    term as 0.5 * oscillator_scale, hence oscillator_scale=2.
    """
    n_particles = 13
    spatial_dim = 3
    dim = n_particles * spatial_dim

    def project_fn(x: jax.Array) -> jax.Array:
        return project_mean_free(x, n_particles=n_particles, spatial_dim=spatial_dim)

    def energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(project_fn(x), dim)
        d = pairwise_distances(y, n_particles=n_particles, spatial_dim=spatial_dim)
        r = jnp.maximum(d, float(min_distance))
        inv = float(rm) / r
        inv6 = inv**6
        lj = float(eps) * (inv6 * inv6 - 2.0 * inv6)
        lj_energy = float(energy_factor) * jnp.sum(lj, axis=-1)
        centered = y.reshape((-1, n_particles, spatial_dim))
        osc_energy = 0.5 * float(oscillator_scale) * jnp.sum(centered * centered, axis=(1, 2))
        return lj_energy + osc_energy

    def grad_energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(project_fn(x), dim)

        def single_energy(v: jax.Array) -> jax.Array:
            return energy_fn(v[None, :])[0]

        grad = jax.vmap(jax.grad(single_energy))(y)
        return project_fn(grad)

    return ProblemSpec(
        name="lj13",
        dim=dim,
        energy_fn=energy_fn,
        grad_energy_fn=grad_energy_fn,
        project_fn=project_fn,
        default_clip_range=5.0,
        n_particles=n_particles,
        spatial_dim=spatial_dim,
    )
