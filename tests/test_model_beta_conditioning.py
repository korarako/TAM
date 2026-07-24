from __future__ import annotations

import jax
import jax.numpy as jnp

from adj_thermo.model import apply_mlp, init_mlp
from adj_thermo.problem import make_problem


def test_model_accepts_explicit_beta_and_preserves_shape():
    problem = make_problem("dw2d")
    params = init_mlp(jax.random.PRNGKey(0), problem.dim + 16 + 8, 32, problem.dim, 2)
    x = jnp.ones((4, problem.dim), dtype=jnp.float32)
    t = jnp.full((4, 1), 0.5, dtype=jnp.float32)
    va = apply_mlp(params, x, t, 0.5, problem)
    vb = apply_mlp(params, x, t, 1.5, problem)
    assert va.shape == x.shape
    assert vb.shape == x.shape
    assert not jnp.allclose(va, vb)
