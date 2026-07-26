from __future__ import annotations

import json

import numpy as np

from adj_thermo.problem.mb2d import make_problem
from adj_thermo.reference.mb2d_analytic import (
    _discrete_js,
    AuditTolerances,
    MB2DDomain,
    audit_grid_convergence,
    build_mb2d_grid,
    compare_samples_to_grid,
    generate_mb2d_reference_v2,
    mb2d_energy_numpy,
    normalized_cell_mass,
    sample_piecewise_uniform,
    validate_mb2d_reference_v2,
)


def test_numpy_energy_matches_runtime_problem():
    points = np.asarray(
        [
            [-0.5582236, 1.4417258],
            [-0.0500108, 0.4666941],
            [0.6234994, 0.0280378],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    runtime = np.asarray(make_problem().energy_fn(points), dtype=np.float64)
    analytic = mb2d_energy_numpy(points)
    np.testing.assert_allclose(analytic, runtime, rtol=2.0e-6, atol=2.0e-6)


def test_mass_and_piecewise_uniform_samples_are_reproducible():
    grid = build_mb2d_grid(MB2DDomain(), 64)
    mass, log_z = normalized_cell_mass(grid, beta=1.2)
    assert np.isfinite(log_z)
    assert np.isclose(np.sum(mass), 1.0, rtol=0.0, atol=1.0e-14)
    first = sample_piecewise_uniform(grid, mass, 512, seed=17)
    second = sample_piecewise_uniform(grid, mass, 512, seed=17)
    third = sample_piecewise_uniform(grid, mass, 512, seed=18)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)
    assert np.isfinite(first).all()


def test_reference_bundle_uses_independent_splits_and_verifies_hashes(tmp_path):
    output = tmp_path / "mb2d_reference_v2"
    manifest = generate_mb2d_reference_v2(
        output,
        betas=[1.0],
        n_train=256,
        n_eval=192,
        train_seed=101,
        eval_seed=202,
        resolutions=[32, 64],
        tolerances=AuditTolerances(
            resolution_tv=1.0,
            resolution_log_z=1.0,
            resolution_basin_l1=1.0,
            domain_outside_mass=1.0,
        ),
        strict=True,
    )
    train = np.load(output / "train" / "samples_beta_1.00.npy")
    evaluation = np.load(output / "eval" / "samples_beta_1.00.npy")
    assert train.shape == (256, 2)
    assert evaluation.shape == (192, 2)
    assert manifest["split_policy"]["independent_seeds"] is True
    assert manifest["betas"]["1.00"]["train"]["seed"] != manifest["betas"]["1.00"]["eval"]["seed"]
    report = validate_mb2d_reference_v2(output)
    assert report["pass"], report["errors"]
    assert report["hash_file_count"] > 0
    json.dumps(report, allow_nan=False)
    report_with_missing_comparison = validate_mb2d_reference_v2(
        output,
        comparison_sources={"missing": tmp_path / "does_not_exist"},
    )
    assert report_with_missing_comparison["pass"]
    json.dumps(report_with_missing_comparison, allow_nan=False)
    assert (
        report_with_missing_comparison["comparisons"]["missing"]["1.00"]["status"]
        == "unavailable"
    )
    report_with_missing_template = validate_mb2d_reference_v2(
        output,
        comparison_sources={
            "missing_template": tmp_path / "samples_beta_{beta}.npy"
        },
    )
    assert report_with_missing_template["pass"]
    json.dumps(report_with_missing_template, allow_nan=False)
    assert (
        report_with_missing_template["comparisons"]["missing_template"]["1.00"][
            "status"
        ]
        == "unavailable"
    )
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored["legacy_fixed_step_ula_used"] is False


def test_comparison_penalizes_a_shifted_dataset():
    grid = build_mb2d_grid(MB2DDomain(), 128)
    mass, _ = normalized_cell_mass(grid, beta=1.0)
    reference_like = sample_piecewise_uniform(grid, mass, 10000, seed=7)
    shifted = reference_like + np.asarray([0.8, -0.7], dtype=np.float32)
    good = compare_samples_to_grid(reference_like, grid, mass, beta=1.0)
    bad = compare_samples_to_grid(shifted, grid, mass, beta=1.0)
    assert good["grid_js_nats_including_outside_bin"] < bad["grid_js_nats_including_outside_bin"]
    assert good["basin_probabilities"]["l1"] < bad["basin_probabilities"]["l1"]
    assert good["energy"]["w2_grid_quantiles"] < bad["energy"]["w2_grid_quantiles"]


def test_discrete_js_is_finite_for_subnormal_tail_mass():
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    value = _discrete_js(
        np.asarray([1.0, 0.0], dtype=np.float64),
        np.asarray([1.0, tiny], dtype=np.float64),
    )
    assert np.isfinite(value)
    assert value >= 0.0


def test_formal_default_grid_audit_passes():
    report = audit_grid_convergence(
        betas=[0.25, 0.50, 0.75, 1.00, 1.20, 1.50],
        domain=MB2DDomain(),
        resolutions=[256, 512, 1024],
        domain_padding_fraction=0.15,
        tolerances=AuditTolerances(),
    )
    assert report["pass"], report
