#!/usr/bin/env python3
"""Create hash-bound, standalone LJ13 reference-v2 result figures.

The plotting window is deliberately robust to rare extreme values.  Histogram
heights are nevertheless normalized by the *full* sample count, and every
figure records the number and probability mass outside the displayed window.
Thus plotting never changes the samples used by the numerical score.

Example
-------
python scripts/plot_lj13_reference_v2.py \
  --reference-bundle data/lj13_reference_v2 \
  --primary-run outputs/lj13/lj13_reference_v2_seed2_run1 \
  --primary-am-seed 2 \
  --am-run 1=outputs/lj13/lj13_reference_v2_seed1_run1 \
  --am-run 3=outputs/lj13/lj13_reference_v2_seed3_run1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adj_thermo.reference.lj13_rehmc import (  # noqa: E402
    AMBIENT_DIM,
    N_PARTICLES,
    PAIR_I,
    PAIR_J,
    SPATIAL_DIM,
    lj13_energy_np,
)


SCHEMA = "adtm.lj13_reference_v2.publication_figures.v1"
EXPECTED_REFERENCE_SCHEMA = "adtm.lj13_reference_v2.bundle.v1"
EXPECTED_SCORE_SCHEMA = "adtm.lj13_reference_v2.score2k.v1"
EXPECTED_TARGET_ID = "lj13_bms_eq234_lj1_confinement1_comfree_v1"
EXPECTED_FORMULA = (
    "sum_{i<j}[(1/r_ij)^12 - 2(1/r_ij)^6] + "
    "sum_i ||x_i-x_COM||^2"
)
TARGET_BETA = 1.0
MINIMUM_PAIR_GUARD = 0.1
REFERENCE_COM_TOLERANCE = 1.0e-9
MODEL_COM_TOLERANCE = 1.0e-5
MODEL_COM_RELATIVE_TOLERANCE = 1.0e-7

PALETTE = {
    "md": "#8FD18A",
    "fm": "#F2A270",
    "am_fill": "#86A9E6",
    "am_mean": "#4776CC",
    "am_seeds": ("#9BB9EE", "#6E97DE", "#3267B8", "#174A8B"),
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def parse_sha256s(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line:
            continue
        digest, relative = line.split(maxsplit=1)
        result[relative.lstrip("*").replace("\\", "/")] = digest
    return result


def validate_reference_bundle(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    sums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise FileNotFoundError(
            f"Reference bundle needs manifest.json and SHA256SUMS: {root}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("schema") != EXPECTED_REFERENCE_SCHEMA:
        raise ValueError(f"Unexpected reference schema: {manifest.get('schema')!r}")
    if manifest.get("status") != "validated_equilibrium_reference":
        raise ValueError(f"Reference status is not validated: {manifest.get('status')!r}")
    if manifest.get("cross_split_status") != "pass":
        raise ValueError("Reference train/eval cross-split audit did not pass.")
    if manifest.get("legacy_reference_used") is not False:
        raise ValueError("Refusing a bundle that used the legacy reference.")
    if manifest.get("legacy_unadjusted_langevin_used") is not False:
        raise ValueError("Refusing an unadjusted-Langevin reference.")
    target = manifest.get("target", {})
    if target.get("id") != EXPECTED_TARGET_ID or target.get("formula") != EXPECTED_FORMULA:
        raise ValueError("Reference target identity/formula is not the registered LJ13 target.")
    dimensions = (
        int(target.get("ambient_dimension", -1)),
        int(target.get("intrinsic_com_free_dimension", -1)),
        int(target.get("n_particles", -1)),
        int(target.get("spatial_dim", -1)),
    )
    if dimensions != (39, 36, 13, 3):
        raise ValueError(f"Unexpected LJ13 dimensions: {dimensions}")
    for split in ("train", "eval"):
        if manifest.get("splits", {}).get(split, {}).get("audit_status") != "pass":
            raise ValueError(f"Reference {split} audit did not pass.")
    declared = parse_sha256s(sums_path)
    required = (
        "train/samples_beta_1.00.npy",
        "eval/samples_beta_1.00.npy",
    )
    for relative in required:
        if relative not in declared:
            raise ValueError(f"SHA256SUMS does not bind {relative}.")
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Unsafe reference member: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        actual = sha256_file(candidate)
        if actual != declared[relative]:
            raise ValueError(
                f"Reference checksum mismatch for {relative}: {actual} != "
                f"{declared[relative]}"
            )
    return manifest, declared


def pair_distances(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64).reshape(
        (-1, N_PARTICLES, SPATIAL_DIM)
    )
    return np.linalg.norm(x[:, PAIR_I] - x[:, PAIR_J], axis=-1)


def observables(samples: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64).reshape(
        (-1, N_PARTICLES, SPATIAL_DIM)
    )
    centered = x - np.mean(x, axis=1, keepdims=True)
    distance = pair_distances(x)
    return {
        "energy": lj13_energy_np(x),
        "pair_distance": distance,
        "radius_of_gyration": np.sqrt(
            np.mean(np.sum(centered * centered, axis=-1), axis=1)
        ),
        "minimum_pair_distance": np.min(distance, axis=1),
    }


def validate_samples(
    label: str,
    values: np.ndarray,
    *,
    expected_rows: int,
    com_tolerance: float,
    relative_com_tolerance: float,
) -> dict[str, Any]:
    raw = np.asarray(values)
    if raw.shape != (int(expected_rows), AMBIENT_DIM):
        raise ValueError(
            f"{label}: expected shape ({expected_rows}, {AMBIENT_DIM}), "
            f"got {raw.shape}"
        )
    finite = np.all(np.isfinite(raw), axis=1)
    if not np.all(finite):
        raise ValueError(f"{label}: {int(np.sum(~finite))} non-finite rows")
    x = np.asarray(raw, dtype=np.float64).reshape(
        (-1, N_PARTICLES, SPATIAL_DIM)
    )
    row_abs_com = np.max(np.abs(np.mean(x, axis=1)), axis=1)
    row_max_coordinate = np.max(np.abs(x), axis=(1, 2))
    row_scale = np.maximum(1.0, row_max_coordinate)
    row_limit = (
        float(com_tolerance) + float(relative_com_tolerance) * row_scale
    )
    max_abs_com = float(np.max(row_abs_com))
    max_scale_normalized_com = float(np.max(row_abs_com / row_scale))
    rows_above_absolute = int(np.sum(row_abs_com > float(com_tolerance)))
    rows_failing_scale_aware = int(np.sum(row_abs_com > row_limit))
    if rows_failing_scale_aware:
        raise ValueError(
            f"{label}: {rows_failing_scale_aware} rows fail the scale-aware "
            "COM gate; row_max_abs_com must be <= "
            f"{float(com_tolerance):.6g} + "
            f"{float(relative_com_tolerance):.6g} * "
            "max(1, row_max_abs_coordinate)"
        )
    minimum = np.min(pair_distances(x), axis=1)
    minimum_pair = float(np.min(minimum))
    violating = int(np.sum(minimum <= MINIMUM_PAIR_GUARD))
    if violating:
        raise ValueError(
            f"{label}: {violating} rows have minimum pair distance <= "
            f"{MINIMUM_PAIR_GUARD}; minimum={minimum_pair:.6g}"
        )
    return {
        "shape": [int(value) for value in raw.shape],
        "dtype": str(raw.dtype),
        "finite_rows": int(np.sum(finite)),
        "nonfinite_rows": int(np.sum(~finite)),
        "max_abs_com": max_abs_com,
        "max_abs_coordinate": float(np.max(row_max_coordinate)),
        "max_scale_normalized_com": max_scale_normalized_com,
        "absolute_com_tolerance": float(com_tolerance),
        "relative_com_tolerance": float(relative_com_tolerance),
        "rows_above_absolute_com_tolerance": rows_above_absolute,
        "rows_failing_scale_aware_com_tolerance": rows_failing_scale_aware,
        "com_rule": (
            "row_max_abs_com <= absolute_com_tolerance + "
            "relative_com_tolerance * max(1, row_max_abs_coordinate)"
        ),
        "minimum_pair_distance": minimum_pair,
        "rows_at_or_below_minimum_pair_guard": violating,
        "minimum_pair_guard": MINIMUM_PAIR_GUARD,
    }


def parse_seed_run(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SEED=PATH.")
    raw_seed, raw_path = value.split("=", 1)
    try:
        seed = int(raw_seed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid AM seed: {raw_seed}") from error
    return seed, resolve(raw_path)


def robust_limits(
    series: dict[str, np.ndarray],
    low: float,
    high: float,
) -> tuple[float, float]:
    pooled = np.concatenate(
        [np.asarray(values, dtype=np.float64).reshape(-1) for values in series.values()]
    )
    if not np.isfinite(pooled).all():
        raise ValueError("Plot series contains non-finite values.")
    left, right = np.quantile(pooled, (float(low), float(high)))
    width = max(float(right - left), 1.0e-10)
    return float(left - 0.04 * width), float(right + 0.04 * width)


def full_mass_density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    counts, _ = np.histogram(flat, bins=edges)
    return counts / (flat.size * np.diff(edges))


def save_figure(fig: plt.Figure, base: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for suffix in ("png", "pdf"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(
            path,
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
        )
        files[suffix] = {"path": path.name, "sha256": sha256_file(path)}
    plt.close(fig)
    return files


def distribution_figure(
    *,
    reference: np.ndarray,
    fm: np.ndarray,
    am_by_seed: dict[int, np.ndarray],
    title: str,
    xlabel: str,
    output: Path,
    quantiles: tuple[float, float],
    bins: int,
) -> dict[str, Any]:
    series = {
        "MD reference": np.asarray(reference, dtype=np.float64).reshape(-1),
        "Flow Matching": np.asarray(fm, dtype=np.float64).reshape(-1),
        **{
            f"Adjoint Matching seed{seed}": np.asarray(values, dtype=np.float64).reshape(-1)
            for seed, values in sorted(am_by_seed.items())
        },
    }
    xlim = robust_limits(series, *quantiles)
    edges = np.linspace(xlim[0], xlim[1], int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = {
        label: full_mass_density(values, edges) for label, values in series.items()
    }
    am_density = np.stack(
        [density[f"Adjoint Matching seed{seed}"] for seed in sorted(am_by_seed)],
        axis=0,
    )
    am_mean = np.mean(am_density, axis=0)

    fig, axis = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    axis.fill_between(
        centers,
        density["MD reference"],
        step="mid",
        color=PALETTE["md"],
        alpha=0.76,
        linewidth=0.0,
        label="MD reference",
    )
    axis.fill_between(
        centers,
        density["Flow Matching"],
        step="mid",
        color=PALETTE["fm"],
        alpha=0.64,
        linewidth=0.0,
        label="Flow Matching",
    )
    axis.fill_between(
        centers,
        am_mean,
        step="mid",
        color=PALETTE["am_fill"],
        alpha=0.22,
        linewidth=0.0,
        label="Adjoint Matching",
    )
    for color, seed in zip(PALETTE["am_seeds"], sorted(am_by_seed)):
        axis.step(
            centers,
            density[f"Adjoint Matching seed{seed}"],
            where="mid",
            color=color,
            linewidth=1.15,
            alpha=0.72,
            label=f"AM seed{seed}" if len(am_by_seed) > 1 else None,
        )
    axis.step(
        centers,
        am_mean,
        where="mid",
        color=PALETTE["am_mean"],
        linewidth=2.4,
        label="AM seed mean" if len(am_by_seed) > 1 else "Adjoint Matching outline",
    )
    axis.set_xlim(*xlim)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_title(title)
    axis.grid(alpha=0.12, linewidth=0.7)
    axis.legend(frameon=True, framealpha=0.94, fontsize=9.5)
    files = save_figure(fig, output)

    tails: dict[str, Any] = {}
    for label, values in series.items():
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        below = int(np.sum(flat < xlim[0]))
        above = int(np.sum(flat > xlim[1]))
        tails[label] = {
            "n": int(flat.size),
            "below": below,
            "above": above,
            "below_fraction": float(below / flat.size),
            "above_fraction": float(above / flat.size),
            "inside_fraction": float(
                np.mean((flat >= xlim[0]) & (flat <= xlim[1]))
            ),
            "full_min": float(np.min(flat)),
            "full_max": float(np.max(flat)),
        }
    return {
        "title": title,
        "xlabel": xlabel,
        "bins": int(bins),
        "xlim": [float(value) for value in xlim],
        "display_quantiles": [float(value) for value in quantiles],
        "normalization": (
            "counts / (full validated sample count * bin width); samples outside "
            "the robust display window are not deleted and tails are not renormalized"
        ),
        "tails": tails,
        "files": files,
    }


def equal_rank_w2(a: np.ndarray, b: np.ndarray) -> float:
    left = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    right = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    if left.size != right.size:
        raise ValueError(f"W2 arrays need equal size: {left.size} != {right.size}")
    return float(np.sqrt(np.mean((left - right) ** 2)))


def fixed_row_indices(n_rows: int, count: int, seed: int) -> np.ndarray:
    if int(count) > int(n_rows):
        raise ValueError(f"Cannot choose {count} rows from {n_rows}.")
    return np.random.default_rng(int(seed)).choice(
        int(n_rows), size=int(count), replace=False
    )


def score_observables(
    *,
    reference: dict[str, np.ndarray],
    fm: dict[str, np.ndarray],
    am_by_seed: dict[int, dict[str, np.ndarray]],
    count: int,
    seed: int,
) -> dict[str, Any]:
    n_rows = int(np.asarray(reference["energy"]).shape[0])
    model_index = fixed_row_indices(n_rows, count, int(seed) + 1)
    reference_index = fixed_row_indices(n_rows, count, int(seed) + 2)

    def score_one(observed: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            "energy_w2_2k": equal_rank_w2(
                observed["energy"][model_index],
                reference["energy"][reference_index],
            ),
            "pair_distance_w2_2k": equal_rank_w2(
                observed["pair_distance"][model_index].reshape(-1),
                reference["pair_distance"][reference_index].reshape(-1),
            ),
            "radius_of_gyration_w2_2k": equal_rank_w2(
                observed["radius_of_gyration"][model_index],
                reference["radius_of_gyration"][reference_index],
            ),
            "minimum_pair_distance_w2_2k": equal_rank_w2(
                observed["minimum_pair_distance"][model_index],
                reference["minimum_pair_distance"][reference_index],
            ),
        }

    return {
        "protocol": {
            "rows_each": int(count),
            "model_row_seed": int(seed) + 1,
            "reference_row_seed": int(seed) + 2,
            "same_model_row_indices_across_fm_and_am": True,
        },
        "fm": score_one(fm),
        "am_by_seed": {
            str(seed_value): score_one(value)
            for seed_value, value in sorted(am_by_seed.items())
        },
    }


def reference_floors(
    train: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    *,
    count: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    n_rows = int(train["energy"].shape[0])
    if int(count) * int(repeats) > n_rows:
        raise ValueError(
            "Reference-floor subsets must be disjoint: "
            f"{count} * {repeats} > {n_rows}"
        )
    train_order = np.random.default_rng(int(seed) + 101).permutation(n_rows)
    eval_order = np.random.default_rng(int(seed) + 102).permutation(n_rows)
    metric_keys = (
        "energy",
        "pair_distance",
        "radius_of_gyration",
        "minimum_pair_distance",
    )
    values: dict[str, list[float]] = {key: [] for key in metric_keys}
    runs: list[dict[str, Any]] = []
    for repeat in range(int(repeats)):
        left = repeat * int(count)
        right = left + int(count)
        train_index = train_order[left:right]
        eval_index = eval_order[left:right]
        record: dict[str, Any] = {"repeat": repeat}
        for key in metric_keys:
            a = train[key][train_index]
            b = evaluation[key][eval_index]
            if key == "pair_distance":
                a = a.reshape(-1)
                b = b.reshape(-1)
            value = equal_rank_w2(a, b)
            values[key].append(value)
            record[f"{key}_w2"] = value
        runs.append(record)
    packaged: dict[str, Any] = {}
    for key, raw in values.items():
        array = np.asarray(raw, dtype=np.float64)
        packaged[f"{key}_w2_2k"] = {
            "values": [float(value) for value in array],
            "mean": float(np.mean(array)),
            "sample_std_ddof1": (
                float(np.std(array, ddof=1)) if array.size > 1 else 0.0
            ),
            "n_repeats": int(array.size),
            "rows_each": int(count),
        }
    return {
        "construction": (
            "disjoint subsets of independent validated train/eval RE-HMC pools"
        ),
        "train_permutation_seed": int(seed) + 101,
        "eval_permutation_seed": int(seed) + 102,
        "runs": runs,
        "metrics": packaged,
    }


def load_canonical_score2k(run: Path) -> tuple[dict[str, Any], Path]:
    path = run / "score_lj13_reference_v2.json"
    value = read_json(path)
    if value.get("schema") != EXPECTED_SCORE_SCHEMA:
        raise ValueError(
            f"{path} is not a canonical score2k artifact: "
            f"{value.get('schema')!r}"
        )
    if value.get("status") != "pass":
        raise ValueError(f"{path} is not marked pass.")
    protocol = value.get("protocol", {})
    if protocol.get("energy_pair_rg_minpair_rows_each") != 2_000:
        raise ValueError(f"{path} does not use 2k low-dimensional metrics.")
    if protocol.get("geometric_rows_each") != 2_000:
        raise ValueError(f"{path} does not use 2k geometric metrics.")
    return value, path


def canonical_metric(
    score: dict[str, Any],
    *,
    kind: str,
    name: str,
    label: str,
) -> float:
    value = (
        score.get("per_beta", {})
        .get("1.00", {})
        .get("models", {})
        .get(kind, {})
        .get("metrics", {})
        .get(name)
    )
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} has no finite {kind} {name}.")
    return float(value)


def summary_figure(
    *,
    primary_run: Path,
    am_runs: dict[int, Path],
    output: Path,
) -> dict[str, Any]:
    primary_score, primary_score_path = load_canonical_score2k(primary_run)
    score_by_seed: dict[int, dict[str, Any]] = {}
    score_sources = {"fm": str(primary_score_path)}
    for seed, run in sorted(am_runs.items()):
        score, path = load_canonical_score2k(run)
        score_by_seed[seed] = score
        score_sources[f"am_seed{seed}"] = str(path)

    fm_energy = canonical_metric(
        primary_score,
        kind="fm",
        name="energy_w2_2k",
        label="primary score",
    )
    am_energy = {
        seed: canonical_metric(
            score,
            kind="am",
            name="energy_w2_2k",
            label=f"AM seed{seed} score",
        )
        for seed, score in sorted(score_by_seed.items())
    }
    fm_geo = canonical_metric(
        primary_score,
        kind="fm",
        name="geometric_w2_2k",
        label="primary score",
    )
    am_geo: dict[int, float] = {}
    for seed, score in sorted(score_by_seed.items()):
        am_geo[seed] = canonical_metric(
            score,
            kind="am",
            name="geometric_w2_2k",
            label=f"AM seed{seed} score",
        )
    canonical_floors = (
        primary_score["per_beta"]["1.00"]["reference_floors"]
    )
    for seed, score in sorted(score_by_seed.items()):
        candidate = score["per_beta"]["1.00"]["reference_floors"]
        if candidate.get("metrics") != canonical_floors.get("metrics"):
            raise ValueError(
                f"AM seed{seed} canonical reference floors differ from primary."
            )

    panels = 2
    fig, axes = plt.subplots(
        1,
        panels,
        figsize=(6.2 * panels, 5.2),
        constrained_layout=True,
        squeeze=False,
    )

    def draw(
        axis: plt.Axes,
        *,
        fm_value: float,
        am_values: dict[int, float],
        ylabel: str,
        title: str,
        floor_mean: float | None,
        floor_std: float | None,
    ) -> None:
        if floor_mean is not None and floor_std is not None:
            axis.axhspan(
                max(0.0, floor_mean - floor_std),
                floor_mean + floor_std,
                color=PALETTE["md"],
                alpha=0.34,
                label="Reference floor mean ± SD",
            )
            axis.axhline(floor_mean, color="#4F9D69", linewidth=1.7)
        axis.scatter(
            [0.0],
            [fm_value],
            marker="D",
            s=86,
            color=PALETTE["fm"],
            edgecolor="#8F4E1F",
            linewidth=0.8,
            zorder=5,
            label="Flow Matching",
        )
        seeds = sorted(am_values)
        jitter = np.linspace(-0.13, 0.13, len(seeds)) if len(seeds) > 1 else np.zeros(1)
        for offset, color, seed in zip(jitter, PALETTE["am_seeds"], seeds):
            axis.scatter(
                [1.0 + float(offset)],
                [am_values[seed]],
                s=70,
                color=color,
                edgecolor=PALETTE["am_mean"],
                linewidth=0.7,
                zorder=6,
                label=f"AM seed{seed}",
            )
        array = np.asarray([am_values[seed] for seed in seeds], dtype=np.float64)
        if array.size > 1:
            axis.errorbar(
                [1.0],
                [float(np.mean(array))],
                yerr=[float(np.std(array, ddof=1))],
                fmt="o",
                color=PALETTE["am_mean"],
                ecolor=PALETTE["am_mean"],
                linewidth=2.0,
                capsize=5,
                zorder=7,
                label="AM mean ± sample SD",
            )
        axis.set_xlim(-0.45, 1.45)
        axis.set_xticks((0.0, 1.0), ("FM", "AM seeds"))
        axis.set_ylim(bottom=0.0)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(frameon=True, framealpha=0.94, fontsize=9)

    energy_floor = canonical_floors["metrics"]["energy_w2_2k"]
    draw(
        axes[0, 0],
        fm_value=fm_energy,
        am_values=am_energy,
        ylabel="Energy W2",
        title=r"LJ13 at $\beta=1.00$: energy W2 (2k)",
        floor_mean=float(energy_floor["mean"]),
        floor_std=float(energy_floor["sample_std_ddof1"]),
    )
    geometric_floor = canonical_floors["metrics"]["geometric_w2_2k"]
    draw(
        axes[0, 1],
        fm_value=float(fm_geo),
        am_values=am_geo,
        ylabel="Symmetry-aware geometric W2",
        title=r"LJ13 at $\beta=1.00$: geometric W2 (2k)",
        floor_mean=float(geometric_floor["mean"]),
        floor_std=float(geometric_floor["sample_std_ddof1"]),
    )
    files = save_figure(fig, output)
    return {
        "energy_w2_2k": {
            "fm": fm_energy,
            "am_by_seed": {str(seed): value for seed, value in am_energy.items()},
            "reference_floor": energy_floor,
        },
        "geometric_w2_2k": {
            "fm": float(fm_geo),
            "am_by_seed": {
                str(seed): float(value) for seed, value in am_geo.items()
            },
            "reference_floor": geometric_floor,
        },
        "score_sources": score_sources,
        "reference_floors": canonical_floors,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--primary-am-seed", type=int, default=2)
    parser.add_argument(
        "--am-run",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="Additional AM run. May be supplied more than once.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--metric-rows", type=int, default=2_000)
    parser.add_argument("--reference-floor-repeats", type=int, default=5)
    parser.add_argument("--metric-seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=110)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    reference_root = resolve(args.reference_bundle)
    primary_run = resolve(args.primary_run)
    am_runs = {int(args.primary_am_seed): primary_run}
    for raw in args.am_run:
        seed, run = parse_seed_run(raw)
        if seed in am_runs and am_runs[seed].resolve() != run.resolve():
            raise ValueError(f"AM seed {seed} was bound to two runs.")
        am_runs[seed] = run
    output = (
        resolve(args.output_dir)
        if args.output_dir
        else primary_run / "publication_figures_lj13_reference_v2"
    )
    expected = [
        output / f"lj13_beta1p00_{stem}.{suffix}"
        for stem in (
            "energy_distribution",
            "pair_distance_distribution",
            "radius_of_gyration_distribution",
            "minimum_pair_distance_distribution",
            "primary_metric_seed_summary",
        )
        for suffix in ("png", "pdf")
    ] + [output / "figure_metadata.json"]
    existing = [path for path in expected if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing artifacts; pass --force."
        )
    output.mkdir(parents=True, exist_ok=True)

    manifest, reference_sums = validate_reference_bundle(reference_root)
    reference_paths = {
        "train": reference_root / "train" / "samples_beta_1.00.npy",
        "eval": reference_root / "eval" / "samples_beta_1.00.npy",
    }
    reference_raw = {
        split: np.load(path, allow_pickle=False)
        for split, path in reference_paths.items()
    }
    expected_rows = int(args.expected_rows) or int(reference_raw["eval"].shape[0])
    if int(reference_raw["train"].shape[0]) != expected_rows:
        raise ValueError("Reference train and eval splits do not have equal size.")

    fm_path = primary_run / "fm_beta_sweep" / "fm_samples_beta_1.00.npy"
    am_paths = {
        seed: run / "am_beta_sweep" / "am_samples_beta_1.00.npy"
        for seed, run in am_runs.items()
    }
    for path in [fm_path, *am_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    fm_raw = np.load(fm_path, allow_pickle=False)
    am_raw = {
        seed: np.load(path, allow_pickle=False)
        for seed, path in am_paths.items()
    }
    health = {
        "reference_train": validate_samples(
            "reference train",
            reference_raw["train"],
            expected_rows=expected_rows,
            com_tolerance=REFERENCE_COM_TOLERANCE,
            relative_com_tolerance=0.0,
        ),
        "reference_eval": validate_samples(
            "reference eval",
            reference_raw["eval"],
            expected_rows=expected_rows,
            com_tolerance=REFERENCE_COM_TOLERANCE,
            relative_com_tolerance=0.0,
        ),
        "fm": validate_samples(
            "FM",
            fm_raw,
            expected_rows=expected_rows,
            com_tolerance=MODEL_COM_TOLERANCE,
            relative_com_tolerance=MODEL_COM_RELATIVE_TOLERANCE,
        ),
        "am_by_seed": {
            str(seed): validate_samples(
                f"AM seed{seed}",
                values,
                expected_rows=expected_rows,
                com_tolerance=MODEL_COM_TOLERANCE,
                relative_com_tolerance=MODEL_COM_RELATIVE_TOLERANCE,
            )
            for seed, values in sorted(am_raw.items())
        },
    }
    reference_observed = {
        split: observables(values) for split, values in reference_raw.items()
    }
    fm_observed = observables(fm_raw)
    am_observed = {
        seed: observables(values) for seed, values in sorted(am_raw.items())
    }
    if int(args.metric_rows) != 2_000:
        raise ValueError("--metric-rows must be 2000 for the canonical benchmark.")
    if int(args.reference_floor_repeats) != 5:
        raise ValueError(
            "--reference-floor-repeats must be 5 for the canonical benchmark."
        )

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "legend.fontsize": 10,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.4,
        }
    )
    figures: dict[str, Any] = {}
    figure_specs = {
        "energy_distribution": (
            "energy",
            r"LJ13 at $\beta=1.00$: energy distribution",
            r"$U(x)$",
            (0.0005, 0.9995),
        ),
        "pair_distance_distribution": (
            "pair_distance",
            r"LJ13 at $\beta=1.00$: pair distances",
            "Pair distance",
            (0.001, 0.999),
        ),
        "radius_of_gyration_distribution": (
            "radius_of_gyration",
            r"LJ13 at $\beta=1.00$: radius of gyration",
            r"$R_g$",
            (0.001, 0.999),
        ),
        "minimum_pair_distance_distribution": (
            "minimum_pair_distance",
            r"LJ13 at $\beta=1.00$: minimum pair distance",
            "Minimum pair distance",
            (0.001, 0.999),
        ),
    }
    for name, (key, title, xlabel, quantiles) in figure_specs.items():
        figures[name] = distribution_figure(
            reference=reference_observed["eval"][key],
            fm=fm_observed[key],
            am_by_seed={
                seed: values[key] for seed, values in am_observed.items()
            },
            title=title,
            xlabel=xlabel,
            output=output / f"lj13_beta1p00_{name}",
            quantiles=quantiles,
            bins=int(args.bins),
        )
    figures["primary_metric_seed_summary"] = summary_figure(
        primary_run=primary_run,
        am_runs=am_runs,
        output=output / "lj13_beta1p00_primary_metric_seed_summary",
    )

    metadata = {
        "schema": SCHEMA,
        "target_beta": TARGET_BETA,
        "target": manifest["target"],
        "palette": PALETTE,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "reference_binding": {
            "path": str(reference_root.resolve()),
            "manifest_sha256": sha256_file(reference_root / "manifest.json"),
            "sha256s_sha256": sha256_file(reference_root / "SHA256SUMS"),
            "selected_files": {
                split: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "declared_sha256": reference_sums[
                        f"{split}/samples_beta_1.00.npy"
                    ],
                }
                for split, path in reference_paths.items()
            },
        },
        "run_bindings": {
            "primary_run": str(primary_run.resolve()),
            "primary_am_seed": int(args.primary_am_seed),
            "am_runs": {
                str(seed): str(run.resolve()) for seed, run in sorted(am_runs.items())
            },
        },
        "sample_bindings": {
            "fm": {"path": str(fm_path.resolve()), "sha256": sha256_file(fm_path)},
            "am_by_seed": {
                str(seed): {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for seed, path in sorted(am_paths.items())
            },
        },
        "sample_health": health,
        "observable_definitions": {
            "energy": EXPECTED_FORMULA,
            "pair_distance": "All 78 unordered Euclidean pair distances per frame.",
            "radius_of_gyration": "sqrt(mean_i ||x_i-x_COM||^2).",
            "minimum_pair_distance": "Minimum of the 78 pair distances per frame.",
        },
        "scores": {
            "schema": EXPECTED_SCORE_SCHEMA,
            "source": "canonical per-run score_lj13_reference_v2.json artifacts",
            "primary_metric_summary": figures["primary_metric_seed_summary"],
        },
        "reference_floors": figures[
            "primary_metric_seed_summary"
        ]["reference_floors"],
        "figures": figures,
        "plotting_policy": {
            "benchmark_scores_use_canonical_2k_subsets": True,
            "display_uses_robust_quantile_windows": True,
            "display_tails_are_recorded": True,
            "histograms_are_not_tail_renormalized": True,
        },
        "script_binding": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(output / "figure_metadata.json", metadata)
    print(json.dumps(metadata["scores"], indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
