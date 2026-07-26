from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.problem import make_problem
from adj_thermo.reference.lj13_rehmc import (
    AMBIENT_DIM,
    INTRINSIC_DIM,
    LJ13REHMCConfig,
    coordinates_to_reduced_np,
    energy_and_grad_reduced_jax,
    helmert_basis_np,
    initialize_ensembles_np,
    lj13_energy_np,
    lj13_virial_np,
    make_beta_ladder,
    reduced_to_coordinates_np,
    run_lj13_rehmc,
)


jax.config.update("jax_enable_x64", True)


def _safe_coordinates(seed: int = 0, batch: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(batch, 13, 3))
    values -= np.mean(values, axis=1, keepdims=True)
    # A broad scale avoids the numerical min-distance guard in parity tests.
    return 2.0 * values


def test_lj13_helmert_basis_is_orthonormal_and_com_free():
    basis = helmert_basis_np()
    assert basis.shape == (13, 12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(12), atol=1.0e-14)
    np.testing.assert_allclose(np.ones(13) @ basis, 0.0, atol=1.0e-14)


def test_lj13_reduced_coordinate_round_trip_has_unit_jacobian():
    rng = np.random.default_rng(1)
    reduced = rng.normal(size=(5, INTRINSIC_DIM))
    coordinates = reduced_to_coordinates_np(reduced).reshape((5, AMBIENT_DIM))
    reconstructed = coordinates_to_reduced_np(coordinates)
    np.testing.assert_allclose(reconstructed, reduced, rtol=1.0e-13, atol=1.0e-13)
    gram = np.kron(helmert_basis_np().T @ helmert_basis_np(), np.eye(3))
    sign, logdet = np.linalg.slogdet(gram)
    assert sign == 1.0
    assert abs(logdet) < 1.0e-12


def test_lj13_reference_energy_matches_runtime_and_is_invariant():
    coordinates = _safe_coordinates()
    flat = coordinates.reshape((coordinates.shape[0], AMBIENT_DIM))
    runtime = np.asarray(make_problem("lj13").energy_fn(jnp.asarray(flat)))
    independent = lj13_energy_np(flat)
    reduced = coordinates_to_reduced_np(flat)
    reference_jax, _ = energy_and_grad_reduced_jax(jnp.asarray(reduced))
    np.testing.assert_allclose(
        np.asarray(reference_jax),
        independent,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    # The legacy runtime deliberately casts to float32.  It is still the same
    # physical formula away from its 0.1 training safety clamp.
    np.testing.assert_allclose(runtime, independent, rtol=2.0e-5, atol=2.0e-5)

    rng = np.random.default_rng(2)
    translation = rng.normal(size=(1, 1, 3))
    permutation = rng.permutation(13)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed = (coordinates + translation)[:, permutation] @ q
    np.testing.assert_allclose(
        lj13_energy_np(transformed),
        independent,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_lj13_reduced_gradient_matches_central_difference():
    coordinates = _safe_coordinates(seed=3, batch=1).reshape((1, AMBIENT_DIM))
    reduced = coordinates_to_reduced_np(coordinates)
    energy, gradient = energy_and_grad_reduced_jax(jnp.asarray(reduced))
    assert np.isfinite(np.asarray(energy)).all()
    direction = np.random.default_rng(4).normal(size=reduced.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-5
    plus = reduced_to_coordinates_np(reduced + epsilon * direction)
    minus = reduced_to_coordinates_np(reduced - epsilon * direction)
    finite_difference = (
        lj13_energy_np(plus) - lj13_energy_np(minus)
    ) / (2.0 * epsilon)
    automatic = np.sum(np.asarray(gradient) * direction, axis=-1)
    np.testing.assert_allclose(automatic, finite_difference, rtol=2.0e-5, atol=2.0e-5)
    analytic_virial = lj13_virial_np(coordinates)
    autodiff_virial = np.sum(np.asarray(gradient) * reduced, axis=-1)
    np.testing.assert_allclose(
        autodiff_virial,
        analytic_virial,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_lj13_ladder_contains_requested_targets_exactly():
    config = LJ13REHMCConfig(n_replicas=16, target_betas=(0.8, 1.0, 1.2))
    ladder = make_beta_ladder(config)
    assert ladder.shape == (16,)
    assert np.all(np.diff(ladder) > 0.0)
    for target in config.target_betas:
        assert np.any(ladder == target)
    initial = initialize_ensembles_np(config, ladder)
    coordinates = reduced_to_coordinates_np(initial)
    assert float(np.max(lj13_energy_np(coordinates))) < 500.0
    pair_i, pair_j = np.triu_indices(13, k=1)
    distance = np.linalg.norm(
        coordinates[..., pair_i, :] - coordinates[..., pair_j, :],
        axis=-1,
    )
    assert float(np.min(distance)) > 0.70


def test_lj13_rehmc_tiny_run_is_finite_and_preserves_labels():
    config = LJ13REHMCConfig(
        n_replicas=6,
        target_betas=(0.8, 1.0, 1.2),
        n_ensembles=2,
        warmup_rounds=4,
        settle_rounds=2,
        production_rounds=16,
        storage_stride=2,
        block_rounds=2,
        leapfrog_min=1,
        leapfrog_max=2,
        initial_step_size=1.0e-5,
        seed=7,
    )
    samples, diagnostics, traces = run_lj13_rehmc(config, progress=False)
    for target in config.target_betas:
        assert samples[target].shape == (2, 8, AMBIENT_DIM)
        assert np.isfinite(samples[target]).all()
        centered = samples[target].reshape((2, 8, 13, 3))
        np.testing.assert_allclose(np.mean(centered, axis=2), 0.0, atol=1.0e-12)
    labels = traces["labels"]
    expected = np.arange(config.n_replicas)
    for row in labels.reshape((-1, config.n_replicas)):
        np.testing.assert_array_equal(np.sort(row), expected)
    assert sum(diagnostics["production_divergences"]) == 0
