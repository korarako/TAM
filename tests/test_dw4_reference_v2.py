from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from adj_thermo.problem.dw4 import make_problem
from adj_thermo.reference.dw4_smc import (
    DW4SMCConfig,
    coordinates_to_reduced_np,
    dw4_energy_np,
    helmert_basis_np,
    reduced_to_coordinates_np,
    run_dw4_smc,
)


def test_helmert_basis_and_roundtrip() -> None:
    basis = helmert_basis_np()
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1.0e-14)
    np.testing.assert_allclose(np.ones(4) @ basis, np.zeros(3), atol=1.0e-14)

    rng = np.random.default_rng(123)
    x = rng.normal(size=(32, 4, 2))
    x -= x.mean(axis=1, keepdims=True)
    reconstructed = reduced_to_coordinates_np(coordinates_to_reduced_np(x))
    np.testing.assert_allclose(reconstructed, x, atol=1.0e-13)


def test_numpy_energy_matches_runtime_problem() -> None:
    rng = np.random.default_rng(456)
    x = rng.normal(size=(64, 4, 2))
    x -= x.mean(axis=1, keepdims=True)
    expected = dw4_energy_np(x)
    actual = np.asarray(make_problem().energy_fn(jnp.asarray(x.reshape((64, 8)))))
    # The training runtime intentionally casts coordinates to float32.
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_small_smc_is_finite_centered_and_reproducible() -> None:
    config = DW4SMCConfig(
        target_cess=0.7,
        mala_steps_per_stage=2,
        final_mala_steps=4,
        mala_step_size=0.003,
    )
    first, first_diagnostics = run_dw4_smc(
        beta=1.0,
        n_particles=512,
        seed=789,
        config=config,
    )
    second, second_diagnostics = run_dw4_smc(
        beta=1.0,
        n_particles=512,
        seed=789,
        config=config,
    )
    np.testing.assert_array_equal(first, second)
    assert first_diagnostics["n_stages"] == second_diagnostics["n_stages"]
    assert first.shape == (512, 8)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    coordinates = first.reshape((-1, 4, 2))
    np.testing.assert_allclose(coordinates.mean(axis=1), 0.0, atol=2.0e-6)
    assert first_diagnostics["summary"]["energy_mean"] < -18.0
    assert 0.05 < first_diagnostics["final_mala_acceptance_mean"] <= 1.0
