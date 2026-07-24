from __future__ import annotations

import jax
import jax.numpy as jnp

from adj_thermo.fm import fm_loss_fn
from adj_thermo.model import init_mlp
from adj_thermo.problem import make_problem


def test_fm_loss_and_grad_are_finite():
    problem = make_problem("dw1d")
    beta_grid = [0.25, 0.50]
    datasets = {
        0.25: jnp.linspace(-1, 1, 32, dtype=jnp.float32)[:, None],
        0.50: jnp.linspace(-2, 2, 32, dtype=jnp.float32)[:, None],
    }
    params = init_mlp(jax.random.PRNGKey(1), problem.dim + 16 + 8, 16, problem.dim, 1)
    loss, grads = jax.value_and_grad(lambda p: fm_loss_fn(p, jax.random.PRNGKey(2), datasets, beta_grid, problem, 8, 1.0))(params)
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads))
