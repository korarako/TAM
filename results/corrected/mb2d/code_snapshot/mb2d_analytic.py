"""Auditable analytic-grid equilibrium references for scaled Mueller--Brown.

The old ADTM MB2D datasets were endpoints of a fixed-length unadjusted
Langevin rollout.  That construction is not an equilibrium sampler and can
retain its initialization bias.  This module instead evaluates

    p_beta(x) = exp(-beta * U(x)) / Z_beta

on a deterministic two-dimensional quadrature grid.  Cell masses are
normalized with a log-sum-exp, samples are drawn independently from the cell
categorical distribution and jittered uniformly inside their selected cells,
and both grid-resolution and domain-truncation convergence are recorded.

The resulting distribution is a piecewise-uniform approximation to the
continuous Boltzmann density.  Its approximation error is made explicit by the
convergence audit; it is not described as an exact iid sampler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REFERENCE_SCHEMA = "adj_thermo.mb2d.reference_v2"
REFERENCE_VERSION = 2

# Keep a float64 NumPy copy of src/adj_thermo/problem/mb2d.py.  The constants
# and the implementation hash are written to the manifest.
_A = np.asarray([-200.0, -100.0, -170.0, 15.0], dtype=np.float64)
_a = np.asarray([-1.0, -1.0, -6.5, 0.7], dtype=np.float64)
_b = np.asarray([0.0, 0.0, 11.0, 0.6], dtype=np.float64)
_c = np.asarray([-10.0, -10.0, -6.5, 0.7], dtype=np.float64)
_x0 = np.asarray([1.0, 0.0, -0.5, -1.0], dtype=np.float64)
_y0 = np.asarray([0.0, 0.5, 1.5, 1.0], dtype=np.float64)
_SCALE = 0.02
_SHIFT = 0.0

# Standard Mueller--Brown minima, used only for an explicit nearest-minimum
# basin partition.  This diagnostic definition does not affect sampling.
_BASIN_NAMES = ("left_upper", "center", "right_lower")
_BASIN_CENTERS = np.asarray(
    [
        [-0.5582236, 1.4417258],
        [-0.0500108, 0.4666941],
        [0.6234994, 0.0280378],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MB2DDomain:
    """Finite quadrature domain, expressed as cell edges."""

    # This is deliberately wider than the legacy plotting box.  At beta=0.25
    # the old [-2.9, 1.4] x [-1.1, 2.9] box omits about 6.5% of the mass.
    x_min: float = -4.0
    x_max: float = 2.5
    y_min: float = -2.5
    y_max: float = 4.5

    def validate(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"MB2D domain must be finite, got {values!r}")
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError(f"Invalid MB2D domain bounds: {values!r}")


@dataclass(frozen=True)
class MB2DGrid:
    domain: MB2DDomain
    nx: int
    ny: int
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    energy: np.ndarray

    @property
    def dx(self) -> float:
        return float(self.x_edges[1] - self.x_edges[0])

    @property
    def dy(self) -> float:
        return float(self.y_edges[1] - self.y_edges[0])

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy


@dataclass(frozen=True)
class AuditTolerances:
    resolution_tv: float = 2.0e-3
    resolution_log_z: float = 2.0e-3
    resolution_basin_l1: float = 2.0e-3
    domain_outside_mass: float = 1.0e-6

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


def mb2d_energy_numpy(x: np.ndarray) -> np.ndarray:
    """Return the repository's scaled Mueller--Brown energy in float64."""

    points = np.asarray(x, dtype=np.float64)
    if points.shape[-1:] != (2,):
        raise ValueError(f"Expected MB2D coordinates with final dimension 2, got {points.shape}")
    flat = points.reshape((-1, 2))
    dx = flat[:, 0:1] - _x0[None, :]
    dy = flat[:, 1:2] - _y0[None, :]
    exponent = _a[None, :] * dx * dx + _b[None, :] * dx * dy + _c[None, :] * dy * dy
    with np.errstate(over="raise", invalid="raise"):
        energy = _SCALE * np.sum(_A[None, :] * np.exp(exponent), axis=-1) + _SHIFT
    return energy.reshape(points.shape[:-1])


def build_mb2d_grid(domain: MB2DDomain, nx: int, ny: int | None = None) -> MB2DGrid:
    """Evaluate the energy on a midpoint quadrature grid."""

    domain.validate()
    nx = int(nx)
    ny = nx if ny is None else int(ny)
    if nx < 2 or ny < 2:
        raise ValueError(f"Grid dimensions must both be >=2, got nx={nx}, ny={ny}")
    x_edges = np.linspace(domain.x_min, domain.x_max, nx + 1, dtype=np.float64)
    y_edges = np.linspace(domain.y_min, domain.y_max, ny + 1, dtype=np.float64)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    energy = mb2d_energy_numpy(np.stack((xx, yy), axis=-1))
    if not np.isfinite(energy).all():
        raise FloatingPointError("Non-finite MB2D energy encountered on the quadrature grid")
    return MB2DGrid(
        domain=domain,
        nx=nx,
        ny=ny,
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        energy=np.asarray(energy, dtype=np.float64),
    )


def normalized_cell_mass(grid: MB2DGrid, beta: float) -> tuple[np.ndarray, float]:
    """Normalize midpoint-quadrature cell masses with log-sum-exp."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be positive and finite, got {beta!r}")
    log_weight = -beta * grid.energy + math.log(grid.cell_area)
    maximum = float(np.max(log_weight))
    shifted = np.exp(log_weight - maximum)
    shifted_sum = float(np.sum(shifted, dtype=np.float64))
    if not math.isfinite(shifted_sum) or shifted_sum <= 0.0:
        raise FloatingPointError(f"Invalid MB2D partition sum at beta={beta}")
    log_z = maximum + math.log(shifted_sum)
    mass = np.asarray(shifted / shifted_sum, dtype=np.float64)
    # Eliminate the last few ulps of normalization drift deterministically.
    # Do not put the correction in an arbitrary tail cell: in a confining
    # potential that cell may have underflowed to exactly zero, so a negative
    # one-ulp correction would create an invalid negative probability.
    mass /= float(np.sum(mass, dtype=np.float64))
    correction_index = int(np.argmax(mass))
    mass.flat[correction_index] += 1.0 - float(np.sum(mass, dtype=np.float64))
    if np.min(mass) < 0.0 or not np.isfinite(mass).all():
        raise FloatingPointError(f"Invalid normalized MB2D mass at beta={beta}")
    return mass, float(log_z)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights, dtype=np.float64)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, ordered_values)


def _basin_indices(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    squared_distance = np.sum(
        (points[:, None, :] - _BASIN_CENTERS[None, :, :]) ** 2,
        axis=-1,
    )
    return np.argmin(squared_distance, axis=1)


def grid_observables(grid: MB2DGrid, mass: np.ndarray, beta: float) -> dict[str, Any]:
    """Compute deterministic basin, marginal, coordinate and energy summaries."""

    mass = np.asarray(mass, dtype=np.float64)
    if mass.shape != grid.energy.shape:
        raise ValueError(f"Mass shape {mass.shape} does not match grid shape {grid.energy.shape}")
    xx, yy = np.meshgrid(grid.x_centers, grid.y_centers, indexing="xy")
    points = np.stack((xx, yy), axis=-1).reshape((-1, 2))
    flat_mass = mass.reshape(-1)
    basin_index = _basin_indices(points)
    basin_probability = np.bincount(
        basin_index,
        weights=flat_mass,
        minlength=len(_BASIN_NAMES),
    )
    x_mass = np.sum(mass, axis=0, dtype=np.float64)
    y_mass = np.sum(mass, axis=1, dtype=np.float64)
    energy = grid.energy.reshape(-1)
    energy_mean = float(np.sum(flat_mass * energy, dtype=np.float64))
    energy_second = float(np.sum(flat_mass * energy * energy, dtype=np.float64))
    probabilities = np.asarray([0.001, 0.01, 0.5, 0.99, 0.999], dtype=np.float64)
    energy_quantiles = _weighted_quantile(energy, flat_mass, probabilities)
    return {
        "beta": float(beta),
        "normalization_mass": float(np.sum(flat_mass, dtype=np.float64)),
        "basin_definition": "nearest Euclidean standard Mueller--Brown minimum",
        "basin_centers": {
            name: _BASIN_CENTERS[i].tolist()
            for i, name in enumerate(_BASIN_NAMES)
        },
        "basin_probabilities": {
            name: float(basin_probability[i])
            for i, name in enumerate(_BASIN_NAMES)
        },
        "coordinate": {
            "x_mean": float(np.sum(grid.x_centers * x_mass, dtype=np.float64)),
            "x_std": float(
                np.sqrt(
                    max(
                        0.0,
                        float(np.sum(grid.x_centers**2 * x_mass, dtype=np.float64))
                        - float(np.sum(grid.x_centers * x_mass, dtype=np.float64)) ** 2,
                    )
                )
            ),
            "y_mean": float(np.sum(grid.y_centers * y_mass, dtype=np.float64)),
            "y_std": float(
                np.sqrt(
                    max(
                        0.0,
                        float(np.sum(grid.y_centers**2 * y_mass, dtype=np.float64))
                        - float(np.sum(grid.y_centers * y_mass, dtype=np.float64)) ** 2,
                    )
                )
            ),
        },
        "energy": {
            "mean": energy_mean,
            "std": float(np.sqrt(max(0.0, energy_second - energy_mean * energy_mean))),
            "quantiles": {
                f"{probability:.3f}": float(value)
                for probability, value in zip(probabilities, energy_quantiles, strict=True)
            },
        },
        "marginals": {
            "x_cell_probability": x_mass.tolist(),
            "y_cell_probability": y_mass.tolist(),
        },
    }


def sample_piecewise_uniform(
    grid: MB2DGrid,
    mass: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Draw iid samples from the grid's piecewise-uniform approximation."""

    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    flat_mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if flat_mass.size != grid.nx * grid.ny:
        raise ValueError("Mass size does not match the grid")
    cdf = np.cumsum(flat_mass, dtype=np.float64)
    cdf[-1] = 1.0
    rng = np.random.default_rng(int(seed))
    cell = np.searchsorted(cdf, rng.random(n_samples), side="right")
    iy, ix = np.divmod(cell, grid.nx)
    jitter = rng.random((n_samples, 2))
    samples = np.empty((n_samples, 2), dtype=np.float64)
    samples[:, 0] = grid.x_edges[ix] + jitter[:, 0] * grid.dx
    samples[:, 1] = grid.y_edges[iy] + jitter[:, 1] * grid.dy
    return samples.astype(np.float32)


def _coarsen_mass(fine_mass: np.ndarray, coarse_shape: tuple[int, int]) -> np.ndarray:
    fine_mass = np.asarray(fine_mass, dtype=np.float64)
    coarse_ny, coarse_nx = map(int, coarse_shape)
    fine_ny, fine_nx = fine_mass.shape
    if fine_nx % coarse_nx or fine_ny % coarse_ny:
        raise ValueError(
            f"Fine shape {fine_mass.shape} is not an integer refinement of {coarse_shape}"
        )
    fy = fine_ny // coarse_ny
    fx = fine_nx // coarse_nx
    return fine_mass.reshape(coarse_ny, fy, coarse_nx, fx).sum(axis=(1, 3))


def _basin_probabilities_on_grid(grid: MB2DGrid, mass: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(grid.x_centers, grid.y_centers, indexing="xy")
    basin = _basin_indices(np.stack((xx, yy), axis=-1))
    return np.bincount(
        basin,
        weights=np.asarray(mass, dtype=np.float64).reshape(-1),
        minlength=len(_BASIN_NAMES),
    )


def audit_grid_convergence(
    betas: Sequence[float],
    domain: MB2DDomain,
    resolutions: Sequence[int],
    domain_padding_fraction: float,
    tolerances: AuditTolerances,
) -> dict[str, Any]:
    """Audit nested resolution convergence and mass outside the base domain."""

    unique_resolutions = tuple(sorted({int(value) for value in resolutions}))
    if len(unique_resolutions) < 2 or unique_resolutions[0] < 2:
        raise ValueError("At least two positive MB2D grid resolutions are required")
    for coarse, fine in zip(unique_resolutions[:-1], unique_resolutions[1:], strict=True):
        if fine % coarse:
            raise ValueError(
                "Successive MB2D resolutions must be integer refinements; "
                f"got {coarse} -> {fine}"
            )
    padding_fraction = float(domain_padding_fraction)
    if not math.isfinite(padding_fraction) or padding_fraction <= 0.0:
        raise ValueError("domain_padding_fraction must be positive and finite")
    tolerances.validate()

    grids = {
        resolution: build_mb2d_grid(domain, resolution, resolution)
        for resolution in unique_resolutions
    }
    per_beta: dict[str, Any] = {}
    all_pass = True
    for beta_value in betas:
        beta = float(beta_value)
        masses: dict[int, np.ndarray] = {}
        log_zs: dict[int, float] = {}
        for resolution, grid in grids.items():
            masses[resolution], log_zs[resolution] = normalized_cell_mass(grid, beta)

        resolution_pairs: list[dict[str, Any]] = []
        for coarse, fine in zip(unique_resolutions[:-1], unique_resolutions[1:], strict=True):
            fine_on_coarse = _coarsen_mass(masses[fine], masses[coarse].shape)
            tv = 0.5 * float(np.sum(np.abs(fine_on_coarse - masses[coarse])))
            coarse_basin = _basin_probabilities_on_grid(grids[coarse], masses[coarse])
            fine_basin = _basin_probabilities_on_grid(grids[fine], masses[fine])
            basin_l1 = float(np.sum(np.abs(coarse_basin - fine_basin)))
            log_z_difference = abs(log_zs[fine] - log_zs[coarse])
            pair_pass = (
                tv <= tolerances.resolution_tv
                and log_z_difference <= tolerances.resolution_log_z
                and basin_l1 <= tolerances.resolution_basin_l1
            )
            resolution_pairs.append(
                {
                    "coarse": coarse,
                    "fine": fine,
                    "total_variation": tv,
                    "abs_log_z_difference": log_z_difference,
                    "basin_probability_l1": basin_l1,
                    "pass": pair_pass,
                }
            )
            all_pass = all_pass and pair_pass

        final_resolution = unique_resolutions[-1]
        base_grid = grids[final_resolution]
        base_mass = masses[final_resolution]
        base_log_z = log_zs[final_resolution]
        pad_x = max(1, int(math.ceil(base_grid.nx * padding_fraction)))
        pad_y = max(1, int(math.ceil(base_grid.ny * padding_fraction)))
        expanded_domain = MB2DDomain(
            x_min=domain.x_min - pad_x * base_grid.dx,
            x_max=domain.x_max + pad_x * base_grid.dx,
            y_min=domain.y_min - pad_y * base_grid.dy,
            y_max=domain.y_max + pad_y * base_grid.dy,
        )
        expanded_grid = build_mb2d_grid(
            expanded_domain,
            base_grid.nx + 2 * pad_x,
            base_grid.ny + 2 * pad_y,
        )
        expanded_mass, expanded_log_z = normalized_cell_mass(expanded_grid, beta)
        interior = expanded_mass[
            pad_y : pad_y + base_grid.ny,
            pad_x : pad_x + base_grid.nx,
        ]
        interior_probability = float(np.sum(interior, dtype=np.float64))
        outside_mass = max(0.0, 1.0 - interior_probability)
        conditional_interior = interior / interior_probability
        conditional_tv = 0.5 * float(np.sum(np.abs(conditional_interior - base_mass)))
        domain_pass = outside_mass <= tolerances.domain_outside_mass
        all_pass = all_pass and domain_pass
        per_beta[f"{beta:.8g}"] = {
            "resolution": {
                "pairs": resolution_pairs,
                "final_log_z": base_log_z,
            },
            "domain": {
                "base": asdict(domain),
                "expanded": asdict(expanded_domain),
                "padding_cells": {"x": pad_x, "y": pad_y},
                "outside_base_mass_under_expanded_grid": outside_mass,
                "conditional_interior_total_variation": conditional_tv,
                "expanded_minus_base_log_z": expanded_log_z - base_log_z,
                "pass": domain_pass,
            },
            "pass": bool(all(pair["pass"] for pair in resolution_pairs) and domain_pass),
        }

    return {
        "schema": f"{REFERENCE_SCHEMA}.convergence_audit",
        "quadrature": "midpoint rectangular cells",
        "resolutions": list(unique_resolutions),
        "domain_padding_fraction": padding_fraction,
        "tolerances": asdict(tolerances),
        "per_beta": per_beta,
        "pass": bool(all_pass),
    }


def _discrete_js(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left = left / np.sum(left)
    right = right / np.sum(right)
    total = left + right

    def kl_to_middle(p: np.ndarray) -> float:
        mask = p > 0.0
        # Evaluate p / ((p + other) / 2) as 2p / (p + other).
        # Forming the half-sum first can underflow to zero for subnormal target
        # cell masses in the far tail even though the exact ratio is finite.
        ratio = (2.0 * p[mask]) / total[mask]
        return float(np.sum(p[mask] * np.log(ratio), dtype=np.float64))

    return 0.5 * (kl_to_middle(left) + kl_to_middle(right))


def compare_samples_to_grid(
    samples: np.ndarray,
    grid: MB2DGrid,
    target_mass: np.ndarray,
    beta: float,
    energy_w2_quantiles: int = 20000,
    histogram_resolution: int = 128,
) -> dict[str, Any]:
    """Compare an empirical dataset with an audited grid reference."""

    raw = np.asarray(samples)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"Expected samples with shape (N, 2), got {raw.shape}")
    finite_mask = np.isfinite(raw).all(axis=1)
    values = np.asarray(raw[finite_mask], dtype=np.float64)
    if values.shape[0] == 0:
        raise ValueError("No finite rows in MB2D comparison dataset")
    histogram_resolution = int(histogram_resolution)
    if (
        histogram_resolution >= 2
        and grid.nx >= histogram_resolution
        and grid.ny >= histogram_resolution
        and grid.nx % histogram_resolution == 0
        and grid.ny % histogram_resolution == 0
    ):
        x_stride = grid.nx // histogram_resolution
        y_stride = grid.ny // histogram_resolution
        histogram_x_edges = grid.x_edges[::x_stride]
        histogram_y_edges = grid.y_edges[::y_stride]
        target_histogram_mass = _coarsen_mass(
            target_mass,
            (histogram_resolution, histogram_resolution),
        )
    else:
        histogram_x_edges = grid.x_edges
        histogram_y_edges = grid.y_edges
        target_histogram_mass = target_mass
    histogram, _, _ = np.histogram2d(
        values[:, 1],
        values[:, 0],
        bins=(histogram_y_edges, histogram_x_edges),
    )
    inside_count = int(np.sum(histogram))
    outside_count = int(values.shape[0] - inside_count)
    empirical_mass = histogram / float(values.shape[0])
    empirical_augmented = np.concatenate(
        (empirical_mass.reshape(-1), np.asarray([outside_count / values.shape[0]]))
    )
    target_augmented = np.concatenate(
        (target_histogram_mass.reshape(-1), np.asarray([0.0]))
    )
    total_variation = 0.5 * float(np.sum(np.abs(empirical_augmented - target_augmented)))
    grid_js = _discrete_js(empirical_augmented, target_augmented)

    empirical_basin_index = _basin_indices(values)
    empirical_basin = np.bincount(
        empirical_basin_index,
        minlength=len(_BASIN_NAMES),
    ).astype(np.float64)
    empirical_basin /= values.shape[0]
    target_basin = _basin_probabilities_on_grid(grid, target_mass)

    x_hist, _ = np.histogram(values[:, 0], bins=histogram_x_edges)
    y_hist, _ = np.histogram(values[:, 1], bins=histogram_y_edges)
    x_outside = values.shape[0] - int(np.sum(x_hist))
    y_outside = values.shape[0] - int(np.sum(y_hist))
    empirical_x = np.concatenate(
        (x_hist.astype(np.float64) / values.shape[0], [x_outside / values.shape[0]])
    )
    empirical_y = np.concatenate(
        (y_hist.astype(np.float64) / values.shape[0], [y_outside / values.shape[0]])
    )
    target_x = np.concatenate(
        (np.sum(target_histogram_mass, axis=0, dtype=np.float64), [0.0])
    )
    target_y = np.concatenate(
        (np.sum(target_histogram_mass, axis=1, dtype=np.float64), [0.0])
    )

    sample_energy = mb2d_energy_numpy(values)
    flat_energy = grid.energy.reshape(-1)
    flat_mass = target_mass.reshape(-1)
    n_quantiles = max(100, min(int(energy_w2_quantiles), values.shape[0]))
    probabilities = (np.arange(n_quantiles, dtype=np.float64) + 0.5) / n_quantiles
    target_energy_quantiles = _weighted_quantile(flat_energy, flat_mass, probabilities)
    sample_energy_quantiles = np.quantile(sample_energy, probabilities, method="linear")
    energy_w2 = float(
        np.sqrt(np.mean((target_energy_quantiles - sample_energy_quantiles) ** 2))
    )

    coordinate_w2: dict[str, float] = {}
    for name, coordinate, centers, marginal_mass in (
        ("x", values[:, 0], grid.x_centers, np.sum(target_mass, axis=0)),
        ("y", values[:, 1], grid.y_centers, np.sum(target_mass, axis=1)),
    ):
        target_quantiles = _weighted_quantile(centers, marginal_mass, probabilities)
        empirical_quantiles = np.quantile(coordinate, probabilities, method="linear")
        coordinate_w2[name] = float(
            np.sqrt(np.mean((target_quantiles - empirical_quantiles) ** 2))
        )

    return {
        "beta": float(beta),
        "rows": int(raw.shape[0]),
        "finite_rows": int(values.shape[0]),
        "finite_fraction": float(values.shape[0] / raw.shape[0]),
        "inside_domain_fraction": float(inside_count / values.shape[0]),
        "histogram_grid_shape_yx": list(target_histogram_mass.shape),
        "grid_total_variation_including_outside_bin": total_variation,
        "grid_js_nats_including_outside_bin": grid_js,
        "basin_probabilities": {
            "target": {
                name: float(target_basin[i])
                for i, name in enumerate(_BASIN_NAMES)
            },
            "empirical": {
                name: float(empirical_basin[i])
                for i, name in enumerate(_BASIN_NAMES)
            },
            "l1": float(np.sum(np.abs(empirical_basin - target_basin))),
        },
        "marginals": {
            "x_js_nats_including_outside_bin": _discrete_js(empirical_x, target_x),
            "y_js_nats_including_outside_bin": _discrete_js(empirical_y, target_y),
            "x_w2": coordinate_w2["x"],
            "y_w2": coordinate_w2["y"],
        },
        "energy": {
            "mean": float(np.mean(sample_energy)),
            "std": float(np.std(sample_energy)),
            "min": float(np.min(sample_energy)),
            "max": float(np.max(sample_energy)),
            "w2_grid_quantiles": energy_w2,
            "w2_quantile_count": n_quantiles,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _beta_tag(beta: float) -> str:
    return f"{float(beta):.2f}"


def _resolve_comparison_path(source: str | Path, beta: float) -> Path:
    text = os.fspath(source)
    if "{beta}" in text:
        return Path(text.replace("{beta}", _beta_tag(beta)))
    path = Path(text)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = (
        path / f"ref_samples_beta_{_beta_tag(beta)}.npy",
        path / f"samples_beta_{_beta_tag(beta)}.npy",
        path / "eval" / f"samples_beta_{_beta_tag(beta)}.npy",
        path / "train" / f"samples_beta_{_beta_tag(beta)}.npy",
    )
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"No MB2D beta={beta:.2f} dataset found under {path}; tried "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    return existing[0]


def parse_comparison_sources(items: Iterable[str] | None) -> dict[str, Path]:
    """Parse repeatable ``LABEL=PATH`` comparison specifications."""

    parsed: dict[str, Path] = {}
    for raw in items or ():
        if "=" not in raw:
            raise ValueError(
                f"Comparison source must be LABEL=PATH (PATH may contain {{beta}}), got {raw!r}"
            )
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid comparison source {raw!r}")
        if label in parsed:
            raise ValueError(f"Duplicate comparison label {label!r}")
        parsed[label] = Path(path)
    return parsed


def _write_sha256sums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in files
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_mb2d_reference_v2(
    output_dir: str | Path,
    betas: Sequence[float],
    n_train: int,
    n_eval: int,
    train_seed: int,
    eval_seed: int,
    *,
    domain: MB2DDomain = MB2DDomain(),
    resolutions: Sequence[int] = (256, 512, 1024),
    domain_padding_fraction: float = 0.15,
    tolerances: AuditTolerances = AuditTolerances(),
    comparison_sources: Mapping[str, str | Path] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Generate a complete MB2D reference-v2 bundle.

    The destination must not already exist.  This deliberate no-overwrite
    policy protects already-audited datasets.
    """

    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing MB2D reference bundle: {root}"
        )
    beta_values = tuple(float(beta) for beta in betas)
    if not beta_values or len(set(beta_values)) != len(beta_values):
        raise ValueError("betas must be a non-empty sequence of unique values")
    if int(train_seed) == int(eval_seed):
        raise ValueError("train_seed and eval_seed must be different")
    if int(n_train) <= 0 or int(n_eval) <= 0:
        raise ValueError("n_train and n_eval must both be positive")
    resolutions = tuple(sorted({int(value) for value in resolutions}))
    domain.validate()
    tolerances.validate()

    convergence = audit_grid_convergence(
        beta_values,
        domain,
        resolutions,
        domain_padding_fraction,
        tolerances,
    )
    if strict and not convergence["pass"]:
        raise ValueError(
            "MB2D grid convergence audit failed under --strict; "
            "increase the domain/resolution or relax explicitly justified tolerances"
        )

    root.mkdir(parents=True)
    train_dir = root / "train"
    eval_dir = root / "eval"
    grid_dir = root / "grid"
    observable_dir = root / "observables"
    comparison_dir = root / "comparisons"
    for path in (train_dir, eval_dir, grid_dir, observable_dir):
        path.mkdir()
    if comparison_sources:
        comparison_dir.mkdir()

    final_resolution = resolutions[-1]
    grid = build_mb2d_grid(domain, final_resolution, final_resolution)
    np.savez_compressed(
        grid_dir / "quadrature_grid.npz",
        x_edges=grid.x_edges,
        y_edges=grid.y_edges,
        x_centers=grid.x_centers,
        y_centers=grid.y_centers,
        energy=grid.energy,
    )
    _json_dump(root / "convergence_audit.json", convergence)

    beta_entries: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for beta_index, beta in enumerate(beta_values):
        tag = _beta_tag(beta)
        mass, log_z = normalized_cell_mass(grid, beta)
        mass_path = grid_dir / f"cell_mass_beta_{tag}.npy"
        np.save(mass_path, mass)
        train_beta_seed = int(train_seed) + beta_index * 1_000_003
        eval_beta_seed = int(eval_seed) + beta_index * 1_000_003
        train_samples = sample_piecewise_uniform(
            grid,
            mass,
            int(n_train),
            train_beta_seed,
        )
        eval_samples = sample_piecewise_uniform(
            grid,
            mass,
            int(n_eval),
            eval_beta_seed,
        )
        train_path = train_dir / f"samples_beta_{tag}.npy"
        eval_path = eval_dir / f"samples_beta_{tag}.npy"
        np.save(train_path, train_samples)
        np.save(eval_path, eval_samples)

        observables = grid_observables(grid, mass, beta)
        observables["log_partition_quadrature"] = log_z
        observables["grid"] = {
            "nx": grid.nx,
            "ny": grid.ny,
            "cell_width": [grid.dx, grid.dy],
            "domain": asdict(domain),
        }
        observable_path = observable_dir / f"beta_{tag}.json"
        _json_dump(observable_path, observables)

        split_checks = {
            "train": compare_samples_to_grid(train_samples, grid, mass, beta),
            "eval": compare_samples_to_grid(eval_samples, grid, mass, beta),
        }
        _json_dump(observable_dir / f"sample_checks_beta_{tag}.json", split_checks)
        beta_entries[tag] = {
            "beta": beta,
            "log_partition_quadrature": log_z,
            "cell_mass": mass_path.relative_to(root).as_posix(),
            "observables": observable_path.relative_to(root).as_posix(),
            "train": {
                "path": train_path.relative_to(root).as_posix(),
                "seed": train_beta_seed,
                "samples": int(n_train),
            },
            "eval": {
                "path": eval_path.relative_to(root).as_posix(),
                "seed": eval_beta_seed,
                "samples": int(n_eval),
            },
        }

        for label, source in (comparison_sources or {}).items():
            try:
                source_path = _resolve_comparison_path(source, beta)
                comparison_values = np.load(source_path, allow_pickle=False)
            except FileNotFoundError as exc:
                comparisons.setdefault(label, {})[tag] = {
                    "status": "unavailable",
                    "reason": str(exc),
                }
                continue
            result = compare_samples_to_grid(
                comparison_values,
                grid,
                mass,
                beta,
            )
            result["source"] = str(source_path)
            result["source_sha256"] = _sha256(source_path)
            comparisons.setdefault(label, {})[tag] = result

    if comparisons:
        _json_dump(comparison_dir / "dataset_comparisons.json", comparisons)

    constants_payload = {
        "A": _A.tolist(),
        "a": _a.tolist(),
        "b": _b.tolist(),
        "c": _c.tolist(),
        "x0": _x0.tolist(),
        "y0": _y0.tolist(),
        "scale": _SCALE,
        "shift": _SHIFT,
    }
    energy_definition_sha = hashlib.sha256(
        json.dumps(constants_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": REFERENCE_SCHEMA,
        "version": REFERENCE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": "p_beta(x) = exp(-beta * U(x)) / Z_beta",
        "sampler": (
            "iid categorical draw from normalized midpoint-quadrature cell masses, "
            "then independent uniform jitter inside each selected cell"
        ),
        "approximation": (
            "piecewise-uniform finite-domain approximation to the continuous "
            "Boltzmann density; convergence_audit.json quantifies grid/domain error"
        ),
        "legacy_fixed_step_ula_used": False,
        "energy": {
            "name": "scaled Mueller--Brown",
            "constants": constants_payload,
            "definition_sha256": energy_definition_sha,
        },
        "quadrature": {
            "method": "midpoint rectangular cells",
            "stable_normalization": "float64 log-sum-exp",
            "domain": asdict(domain),
            "final_resolution": [grid.nx, grid.ny],
            "audited_resolutions": list(resolutions),
            "domain_padding_fraction": float(domain_padding_fraction),
            "audit_pass": bool(convergence["pass"]),
            "strict": bool(strict),
        },
        "split_policy": {
            "train_seed_base": int(train_seed),
            "eval_seed_base": int(eval_seed),
            "independent_seeds": True,
            "beta_seed_stride": 1_000_003,
            "train_path": "train/",
            "eval_path": "eval/",
        },
        "betas": beta_entries,
        "comparison_labels": sorted((comparison_sources or {}).keys()),
        "recommended_cli": {
            "training": f"--data-name {root.name}/train",
            "evaluation": f"--reference-data-name {root.name}/eval",
        },
    }
    _json_dump(root / "manifest.json", manifest)
    _write_sha256sums(root)
    return manifest


def _read_sha256sums(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    path = root / "SHA256SUMS"
    if not path.is_file():
        raise FileNotFoundError(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}: {line!r}") from exc
        result[relative] = digest
    return result


def validate_mb2d_reference_v2(
    root: str | Path,
    *,
    comparison_sources: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Verify hashes/schema/splits and optionally compare external old datasets."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    errors: list[str] = []
    if manifest.get("schema") != REFERENCE_SCHEMA:
        errors.append(f"unexpected schema: {manifest.get('schema')!r}")
    if manifest.get("legacy_fixed_step_ula_used") is not False:
        errors.append("manifest does not explicitly exclude legacy fixed-step ULA")
    split_policy = manifest.get("split_policy", {})
    if split_policy.get("train_seed_base") == split_policy.get("eval_seed_base"):
        errors.append("train/eval base seeds are equal")

    expected_hashes = _read_sha256sums(root)
    hash_results: dict[str, Any] = {}
    actual_relative_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    unexpected_files = sorted(actual_relative_files - set(expected_hashes))
    for relative in unexpected_files:
        errors.append(f"unhashed file present: {relative}")
    for relative, expected in expected_hashes.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing hashed file: {relative}")
            hash_results[relative] = {"expected": expected, "actual": None, "pass": False}
            continue
        actual = _sha256(path)
        passed = actual == expected
        if not passed:
            errors.append(f"SHA256 mismatch: {relative}")
        hash_results[relative] = {"expected": expected, "actual": actual, "pass": passed}

    grid_archive = np.load(root / "grid" / "quadrature_grid.npz", allow_pickle=False)
    domain = MB2DDomain(**manifest["quadrature"]["domain"])
    grid = MB2DGrid(
        domain=domain,
        nx=int(manifest["quadrature"]["final_resolution"][0]),
        ny=int(manifest["quadrature"]["final_resolution"][1]),
        x_edges=np.asarray(grid_archive["x_edges"], dtype=np.float64),
        y_edges=np.asarray(grid_archive["y_edges"], dtype=np.float64),
        x_centers=np.asarray(grid_archive["x_centers"], dtype=np.float64),
        y_centers=np.asarray(grid_archive["y_centers"], dtype=np.float64),
        energy=np.asarray(grid_archive["energy"], dtype=np.float64),
    )
    beta_validation: dict[str, Any] = {}
    dynamic_comparisons: dict[str, Any] = {}
    for tag, entry in manifest.get("betas", {}).items():
        mass = np.load(root / entry["cell_mass"], allow_pickle=False)
        mass_sum = float(np.sum(mass, dtype=np.float64))
        if abs(mass_sum - 1.0) > 1.0e-12:
            errors.append(f"cell mass beta={tag} sums to {mass_sum}")
        split_details: dict[str, Any] = {}
        split_seeds = []
        for split in ("train", "eval"):
            split_entry = entry[split]
            split_seeds.append(int(split_entry["seed"]))
            samples = np.load(root / split_entry["path"], allow_pickle=False)
            passed = bool(
                samples.shape == (int(split_entry["samples"]), 2)
                and np.isfinite(samples).all()
            )
            if not passed:
                errors.append(f"invalid {split} samples for beta={tag}: shape={samples.shape}")
            split_details[split] = {
                "path": split_entry["path"],
                "shape": list(samples.shape),
                "finite": bool(np.isfinite(samples).all()),
                "pass": passed,
            }
        if split_seeds[0] == split_seeds[1]:
            errors.append(f"train/eval per-beta seeds are equal for beta={tag}")
        beta_validation[tag] = {
            "mass_sum": mass_sum,
            "splits": split_details,
            "pass": bool(
                abs(mass_sum - 1.0) <= 1.0e-12
                and all(item["pass"] for item in split_details.values())
                and split_seeds[0] != split_seeds[1]
            ),
        }
        for label, source in (comparison_sources or {}).items():
            try:
                source_path = _resolve_comparison_path(source, float(entry["beta"]))
                comparison_values = np.load(source_path, allow_pickle=False)
            except FileNotFoundError as exc:
                dynamic_comparisons.setdefault(label, {})[tag] = {
                    "status": "unavailable",
                    "reason": str(exc),
                }
                continue
            result = compare_samples_to_grid(
                comparison_values,
                grid,
                mass,
                float(entry["beta"]),
            )
            result["source"] = str(source_path)
            result["source_sha256"] = _sha256(source_path)
            dynamic_comparisons.setdefault(label, {})[tag] = result

    return {
        "schema": f"{REFERENCE_SCHEMA}.validation",
        "root": str(root),
        "manifest_schema": manifest.get("schema"),
        "hash_file_count": len(expected_hashes),
        "unexpected_unhashed_files": unexpected_files,
        "hashes": hash_results,
        "betas": beta_validation,
        "comparisons": dynamic_comparisons,
        "errors": errors,
        "pass": not errors,
    }
