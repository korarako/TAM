from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from adj_thermo.geometric_alignment import (
    center_configuration,
    joint_alignment_cost_matrix,
    joint_alignment_squared_cost,
)
from adj_thermo.metrics import geometric_w2, geometric_w2_result
from adj_thermo.problem import make_problem


def _generic_cloud(seed: int, *, particles: int = 13, dimension: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = np.linspace(0.7, 1.8, dimension, dtype=np.float64)
    return rng.normal(size=(particles, dimension)) * scales[None, :]


def _orthogonal_transform(
    x: np.ndarray,
    *,
    seed: int,
    reflection: bool,
) -> np.ndarray:
    """Independently permute, rotate/reflect, and translate one configuration."""

    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(x.shape[1], x.shape[1])))
    if (np.linalg.det(q) < 0.0) != reflection:
        q[:, 0] *= -1.0
    translation = rng.normal(size=(1, x.shape[1]))
    return x[rng.permutation(x.shape[0])] @ q + translation


def _icosahedral_lj13() -> np.ndarray:
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    points: list[list[float]] = [[0.0, 0.0, 0.0]]
    points.extend(
        [0.0, first, second * golden_ratio]
        for first in (-1.0, 1.0)
        for second in (-1.0, 1.0)
    )
    points.extend(
        [first, second * golden_ratio, 0.0]
        for first in (-1.0, 1.0)
        for second in (-1.0, 1.0)
    )
    points.extend(
        [second * golden_ratio, 0.0, first]
        for first in (-1.0, 1.0)
        for second in (-1.0, 1.0)
    )
    return np.asarray(points, dtype=np.float64)


@pytest.mark.parametrize("reflection", [False, True])
def test_joint_alignment_zero_for_permutation_o3_and_translation(
    reflection: bool,
) -> None:
    x = _generic_cloud(1)
    transformed = _orthogonal_transform(
        x,
        seed=11 + int(reflection),
        reflection=reflection,
    )

    squared_cost, metadata = joint_alignment_squared_cost(x, transformed)

    assert np.sqrt(squared_cost) < 1.0e-9
    assert metadata["start"] == "exact_distance_isomorphism"


def test_joint_alignment_is_invariant_and_symmetric_for_distinct_clouds() -> None:
    x = _generic_cloud(2)
    y = _generic_cloud(3)
    baseline, _ = joint_alignment_squared_cost(x, y)
    independently_transformed, _ = joint_alignment_squared_cost(
        _orthogonal_transform(x, seed=12, reflection=False),
        _orthogonal_transform(y, seed=13, reflection=True),
    )
    reverse, _ = joint_alignment_squared_cost(y, x)

    baseline_distance = np.sqrt(baseline)
    assert np.sqrt(independently_transformed) == pytest.approx(
        baseline_distance,
        abs=1.0e-9,
    )
    assert np.sqrt(reverse) == pytest.approx(baseline_distance, abs=1.0e-9)


@pytest.mark.parametrize("reflection", [False, True])
def test_exact_icosahedron_handles_repeated_distance_signatures(
    reflection: bool,
) -> None:
    x = _icosahedral_lj13()
    transformed = _orthogonal_transform(
        x,
        seed=19 + int(reflection),
        reflection=reflection,
    )

    squared_cost, metadata = joint_alignment_squared_cost(x, transformed)

    assert np.sqrt(squared_cost) < 1.0e-9
    assert metadata["start"] == "exact_distance_isomorphism"


def test_public_lj13_auto_metric_is_zero_for_same_ensemble_in_new_frames() -> None:
    left = np.stack([_generic_cloud(seed) for seed in (30, 31, 32)])
    sample_order = (2, 0, 1)
    right = np.stack(
        [
            _orthogonal_transform(
                left[index],
                seed=100 + index,
                reflection=bool(index % 2),
            )
            for index in sample_order
        ]
    )

    result = geometric_w2_result(
        left.reshape((len(left), -1)),
        right.reshape((len(right), -1)),
        make_problem("lj13"),
        n_samples=len(left),
        protocol="auto",
        joint_workers=1,
    )

    assert result.value is not None
    assert result.value < 1.0e-9
    assert result.resolved_protocol == "joint"
    assert result.candidate_truncation is None
    assert result.symmetry_consistent is True
    assert result.approximate_particle_alignment is True


def test_public_lj13_metric_uses_frobenius_norm_without_particle_rms() -> None:
    x = center_configuration(_generic_cloud(67))
    scale = 1.001

    value = geometric_w2(
        x.reshape((1, -1)),
        (scale * x).reshape((1, -1)),
        make_problem("lj13"),
        n_samples=1,
        protocol="joint",
        joint_workers=1,
    )

    expected = abs(scale - 1.0) * np.linalg.norm(x)
    assert value is not None
    assert value == pytest.approx(expected, rel=1.0e-8, abs=1.0e-10)


def test_dw4_auto_resolves_to_exact_and_matches_explicit_exact() -> None:
    left = np.stack(
        [
            _generic_cloud(80, particles=4, dimension=2),
            _generic_cloud(81, particles=4, dimension=2),
        ]
    )
    right = np.stack(
        [
            _orthogonal_transform(left[1], seed=182, reflection=True),
            _orthogonal_transform(left[0], seed=181, reflection=False),
        ]
    )
    problem = make_problem("dw4")
    flattened_left = left.reshape((len(left), -1))
    flattened_right = right.reshape((len(right), -1))

    automatic = geometric_w2_result(
        flattened_left,
        flattened_right,
        problem,
        n_samples=len(left),
        protocol="auto",
    )
    explicit = geometric_w2_result(
        flattened_left,
        flattened_right,
        problem,
        n_samples=len(left),
        protocol="exact",
    )

    assert automatic.resolved_protocol == "exact"
    assert explicit.resolved_protocol == "exact"
    assert automatic.approximate_particle_alignment is False
    assert explicit.approximate_particle_alignment is False
    assert automatic.value is not None
    assert explicit.value is not None
    assert automatic.value < 1.0e-7
    assert explicit.value == pytest.approx(automatic.value, abs=1.0e-10)


def test_dw4_legacy_protocol_resolves_to_historical_exact_branch() -> None:
    # A non-finite row is discarded after protocol resolution, so this checks
    # public metadata without running factorial permutation enumeration again.
    invalid = np.full((1, 8), np.nan, dtype=np.float64)

    result = geometric_w2_result(
        invalid,
        invalid,
        make_problem("dw4"),
        protocol="legacy-topk",
    )

    assert result.requested_protocol == "legacy-topk"
    assert result.resolved_protocol == "exact"
    assert result.sample_count == 0
    assert result.candidate_truncation is None
    assert result.approximate_particle_alignment is False


def test_auto_with_legacy_top_k_warns_and_resolves_to_legacy_for_lj13() -> None:
    sample = center_configuration(_generic_cloud(91)).reshape((1, -1))

    with pytest.warns(FutureWarning, match="selects the deprecated legacy-topk"):
        result = geometric_w2_result(
            sample,
            sample,
            make_problem("lj13"),
            n_samples=1,
            protocol="auto",
            dem_refine_top_k=1,
        )

    assert result.requested_protocol == "auto"
    assert result.resolved_protocol == "legacy-topk"
    assert result.sample_count == 1
    assert result.candidate_truncation is None
    assert result.subsampling == "single deterministic RNG draw (generated, then reference)"
    assert result.symmetry_consistent is False
    assert result.approximate_particle_alignment is True


def test_joint_worker_count_does_not_change_cost_matrix() -> None:
    left = np.stack([_generic_cloud(101), _generic_cloud(102)])
    right = np.stack([_generic_cloud(103), _generic_cloud(104)])

    serial = joint_alignment_cost_matrix(left, right, workers=1)
    threaded = joint_alignment_cost_matrix(
        left,
        right,
        workers=2,
        parallel_backend="thread",
    )

    np.testing.assert_allclose(threaded, serial, rtol=0.0, atol=1.0e-12)


def test_particle_feature_width_is_validated() -> None:
    wrong_width = np.zeros((2, 78), dtype=np.float64)

    with pytest.raises(ValueError, match="expected 39"):
        geometric_w2_result(
            wrong_width,
            wrong_width,
            make_problem("lj13"),
            n_samples=2,
        )


def test_heterogeneous_particle_permutations_are_rejected() -> None:
    heterogeneous_problem = SimpleNamespace(
        n_particles=2,
        spatial_dim=1,
        atom_species=(1, 2),
    )
    samples = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="identical particles only"):
        geometric_w2_result(
            samples,
            samples,
            heterogeneous_problem,
            n_samples=1,
        )
