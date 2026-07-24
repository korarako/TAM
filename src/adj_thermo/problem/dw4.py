from __future__ import annotations

import jax
import jax.numpy as jnp

from adj_thermo.problem.base import ProblemSpec, ensure_batch_dim
from adj_thermo.utils import pairwise_distances, project_mean_free


def make_problem(
    linear: float = 0.0,
    quadratic: float = -4.0,
    quartic: float = 0.9,
    d0: float = 1.0,
    tau: float = 1.0,
) -> ProblemSpec:
    """DW-4 benchmark potential from the n-body literature.

    Defaults follow the identical-particle DW-4 benchmark:
    U = (1 / tau) * sum_ij [a z_ij + b z_ij^2 + c z_ij^4],
    z_ij = d_ij - d0, with a=0, b=-4, c=0.9, d0=1, tau=1.
    """
    n_particles = 4
    spatial_dim = 2
    dim = n_particles * spatial_dim

    def project_fn(x: jax.Array) -> jax.Array:
        return project_mean_free(x, n_particles=n_particles, spatial_dim=spatial_dim)

    def energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(project_fn(x), dim)
        d = pairwise_distances(y, n_particles=n_particles, spatial_dim=spatial_dim)
        z = d - float(d0)
        pair_energy = float(linear) * z + float(quadratic) * z * z + float(quartic) * z**4
        return jnp.sum(pair_energy, axis=-1) / float(tau)

    def grad_energy_fn(x: jax.Array) -> jax.Array:
        y = ensure_batch_dim(project_fn(x), dim)

        def single_energy(v: jax.Array) -> jax.Array:
            return energy_fn(v[None, :])[0]

        grad = jax.vmap(jax.grad(single_energy))(y)
        return project_fn(grad)

    return ProblemSpec(
        name="dw4",
        dim=dim,
        energy_fn=energy_fn,
        grad_energy_fn=grad_energy_fn,
        project_fn=project_fn,
        default_clip_range=10.0,
        n_particles=n_particles,
        spatial_dim=spatial_dim,
    )
