from __future__ import annotations

import jax.numpy as jnp

from adj_thermo.problem import make_problem


def test_problem_shapes():
    for name in ["dw1d", "dw2d", "mb2d", "dw4", "lj13"]:
        problem = make_problem(name)
        x = jnp.ones((5, problem.dim), dtype=jnp.float32)
        e = problem.energy_fn(x)
        g = problem.grad_energy_fn(x)
        assert e.shape == (5,)
        assert g.shape == x.shape
        assert jnp.all(jnp.isfinite(e))
        assert jnp.all(jnp.isfinite(g))


def test_dw4_projection_removes_center_of_mass():
    problem = make_problem("dw4")
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    y = problem.project_fn(x).reshape(2, 4, 2)
    assert jnp.allclose(jnp.mean(y, axis=1), 0.0, atol=1e-6)


def test_lj13_projection_removes_center_of_mass():
    problem = make_problem("lj13")
    x = jnp.arange(78, dtype=jnp.float32).reshape(2, 39)
    y = problem.project_fn(x).reshape(2, 13, 3)
    assert jnp.allclose(jnp.mean(y, axis=1), 0.0, atol=1e-5)
