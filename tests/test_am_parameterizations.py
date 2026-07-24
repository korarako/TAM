from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adj_thermo.am import train_am
from adj_thermo.am_velocity import (
    ANCHOR_RESIDUAL,
    TARGET_REFINEMENT,
    is_anchor_residual_checkpoint,
    make_anchor_residual_checkpoint,
    total_and_correction_velocity,
)
from adj_thermo.model import init_mlp
from adj_thermo.problem import make_problem
from adj_thermo.sampler import sample_ode
from adj_thermo.utils import load_pickle


def _linear_velocity(params, x, t, beta):
    del t
    return params["scale"] * x + float(beta)


def test_target_refinement_starts_from_target_fm_gap():
    base = {"scale": jnp.asarray(2.0)}
    params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), base)
    x = jnp.asarray([[1.0], [3.0]])
    t = jnp.zeros((2, 1))

    total, correction = total_and_correction_velocity(
        _linear_velocity, params, base, x, t, 0.8, 1.0, TARGET_REFINEMENT
    )

    assert jnp.allclose(total, _linear_velocity(base, x, t, 1.0))
    assert jnp.allclose(correction, 0.2)


def test_anchor_residual_has_zero_initial_correction_and_nonzero_gradient():
    base = {"scale": jnp.asarray(2.0)}
    params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), base)
    x = jnp.asarray([[1.0], [3.0]])
    t = jnp.zeros((2, 1))

    total, correction = total_and_correction_velocity(
        _linear_velocity, params, base, x, t, 0.8, 1.0, ANCHOR_RESIDUAL
    )
    grad = jax.grad(
        lambda p: jnp.sum(
            total_and_correction_velocity(
                _linear_velocity, p, base, x, t, 0.8, 1.0, ANCHOR_RESIDUAL
            )[1]
        )
    )(params)

    assert jnp.allclose(total, _linear_velocity(base, x, t, 0.8))
    assert jnp.allclose(correction, 0.0)
    assert float(jnp.abs(grad["scale"])) > 0.0


def test_anchor_residual_checkpoint_samples_like_anchor_at_initialization():
    problem = make_problem("dw1d")
    base = init_mlp(jax.random.PRNGKey(8), problem.dim + 16 + 8, 12, problem.dim, 1)
    checkpoint = make_anchor_residual_checkpoint(base, base, beta0=0.8, beta1=1.0)
    key = jax.random.PRNGKey(9)

    anchor_samples = sample_ode(base, key, problem, 0.8, 4, 3)
    residual_samples = sample_ode(checkpoint, key, problem, 1.0, 4, 3)

    assert np.allclose(residual_samples, anchor_samples, rtol=1.0e-6, atol=1.0e-6)
    with pytest.raises(ValueError, match="targets beta"):
        sample_ode(checkpoint, key, problem, 0.9, 1, 1)


def test_train_am_saves_self_contained_anchor_residual_checkpoint(tmp_path):
    problem = make_problem("dw1d")
    base = init_mlp(jax.random.PRNGKey(10), problem.dim + 16 + 8, 8, problem.dim, 1)

    _, history = train_am(
        problem,
        base,
        tmp_path,
        jax.random.PRNGKey(11),
        beta0=0.8,
        beta1=1.0,
        steps=1,
        batch_size=2,
        K=2,
        am_num_loss_steps=1,
        am_keep_last_steps=1,
        log_every=0,
        model_type="mlp",
        am_parameterization=ANCHOR_RESIDUAL,
    )
    checkpoint = load_pickle(tmp_path / "am_params.pkl")
    samples = sample_ode(checkpoint, jax.random.PRNGKey(14), problem, 1.0, 2, 2)

    assert history.shape == (1,)
    assert is_anchor_residual_checkpoint(checkpoint)
    assert checkpoint["beta0"] == 0.8
    assert checkpoint["beta1"] == 1.0
    assert np.isfinite(samples).all()
