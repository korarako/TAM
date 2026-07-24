from __future__ import annotations

import jax
import jax.numpy as jnp

from adj_thermo.am import am_loss_fn, sample_memoryless_sde
from adj_thermo.am_velocity import ANCHOR_RESIDUAL
from adj_thermo.model import init_mlp
from adj_thermo.problem import make_problem


def test_am_loss_and_grad_are_finite():
    problem = make_problem("dw1d")
    params = init_mlp(jax.random.PRNGKey(3), problem.dim + 16 + 8, 16, problem.dim, 1)
    base_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)
    loss, grads = jax.value_and_grad(lambda p: am_loss_fn(p, base_params, jax.random.PRNGKey(4), 1.0, 1.1, problem, 4, 4, 1.0, 2, 1))(params)
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads))

def test_memoryless_sde_prior_scale_zero_removes_path_noise():
    problem = make_problem("dw1d")
    params = init_mlp(jax.random.PRNGKey(5), problem.dim + 16 + 8, 16, problem.dim, 1)
    params = jax.tree_util.tree_map(jnp.zeros_like, params)

    xs, _, xs_after, _, x_final = sample_memoryless_sde(
        params,
        jax.random.PRNGKey(6),
        1.0,
        problem,
        batch_size=3,
        K=5,
        prior_scale=0.0,
    )

    assert jnp.allclose(xs, 0.0)
    assert jnp.allclose(xs_after, 0.0)
    assert jnp.allclose(x_final, 0.0)


def test_anchor_residual_am_loss_and_grad_are_finite():
    problem = make_problem("dw1d")
    params = init_mlp(jax.random.PRNGKey(12), problem.dim + 16 + 8, 16, problem.dim, 1)
    base_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)
    loss, grads = jax.value_and_grad(
        lambda p: am_loss_fn(
            p,
            base_params,
            jax.random.PRNGKey(13),
            1.0,
            1.1,
            problem,
            4,
            4,
            1.0,
            2,
            1,
            am_parameterization=ANCHOR_RESIDUAL,
        )
    )(params)
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads))
