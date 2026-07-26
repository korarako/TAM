#!/usr/bin/env python3
"""Aggregate corrected-reference DW4 AM seeds against one shared FM model.

This command is intentionally strict.  It accepts the pre-registered DW4
comparison in which FM was trained once with seed 0 and AM was independently
trained with seeds 1, 2 and 3.  It verifies the shared parent FM, corrected
reference, evaluation protocol, arrays and score files before writing an
atomic, hash-bound aggregate directory.

The command does not train or sample a model.  It only aggregates already
completed runs and reconstructs the deterministic Gaussian evaluation input
used by the chunked ODE sampler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adj_thermo.metrics import geometric_w2
from adj_thermo.problem import make_problem
from adj_thermo.reference.dw4_smc import (
    dw4_energy_np,
    equal_rank_w2,
    pair_distances_np,
)


SCHEMA = "adtm.dw4_reference_v2.am_seed_aggregate.v1"
TARGET_BETA = 1.0
EXPECTED_AM_SEEDS = (1, 2, 3)
EXPECTED_ROWS = 100_000
EXPECTED_SHAPE = (EXPECTED_ROWS, 8)
EXPECTED_COM_TOLERANCE = 1.0e-5
EXPECTED_ENERGY_SAMPLES = 20_000
EXPECTED_GEOMETRIC_SAMPLES = 2_000
EXPECTED_EVAL_SEED = 0
EXPECTED_INITIAL_KEY = 7000
EXPECTED_CHUNK_SIZE = 5000
EXPECTED_ODE_STEPS = 150
EXPECTED_ODE_METHOD = "euler"
EXPECTED_PRIOR_SCALE = 1.0
REFERENCE_FLOOR_SEED = 880_001

PALETTE = {
    "reference": "#8FD18A",
    "fm": "#F2A270",
    "am_seeds": ("#8AB4F8", "#5B8DEF", "#2F6BBD"),
    "am_mean": "#174A8B",
    "am_ribbon": "#AFCBF2",
}

DEFAULT_REFERENCE = Path("data/dw4_reference_v2")
DEFAULT_SEED1 = Path(
    "outputs/dw4/"
    "dw4_reference_v2_egnn128x5_anchor_0p8_1p2_target_1p0_seed1_run1"
)
DEFAULT_SEED2 = Path(
    "outputs/dw4/"
    "dw4_reference_v2_egnn128x5_anchor_0p8_1p2_target_1p0_seed2_run1"
)
DEFAULT_SEED3 = Path(
    "outputs/dw4/"
    "dw4_reference_v2_egnn128x5_anchor_0p8_1p2_target_1p0_seed3_run1"
)
DEFAULT_OUTPUT = Path(
    "outputs/dw4/dw4_reference_v2_fixed_fm_seed0_am_seeds1_2_3_aggregate"
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def relative_input(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def require_close(actual: float, expected: float, label: str, atol: float = 1.0e-12) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=float(atol)):
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def load_array(path: Path, label: str, expected_shape: tuple[int, ...] = EXPECTED_SHAPE) -> np.ndarray:
    require_file(path, label)
    value = np.asarray(np.load(path, allow_pickle=False))
    if value.shape != expected_shape:
        raise ValueError(f"{label} shape: expected {expected_shape}, got {value.shape}")
    if value.dtype != np.float32:
        raise ValueError(f"{label} dtype: expected float32, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value


def com_abs_max(samples: np.ndarray) -> float:
    coordinates = np.asarray(samples, dtype=np.float64).reshape((-1, 4, 2))
    return float(np.max(np.abs(np.mean(coordinates, axis=1))))


def validate_com(samples: np.ndarray, label: str) -> float:
    residual = com_abs_max(samples)
    if residual > EXPECTED_COM_TOLERANCE:
        raise ValueError(
            f"{label} COM residual {residual:.6g} exceeds "
            f"{EXPECTED_COM_TOLERANCE:.6g}"
        )
    return residual


def discover_score(run: Path, kinds: tuple[str, ...]) -> Path:
    kind_tag = "_".join(kinds)
    pattern = (
        f"score_samples_metrics_dw4_reference_v2_eval_{kind_tag}_"
        "betas_1p00_geo2000_ew20000.json"
    )
    exact = run / pattern
    if exact.is_file():
        return exact
    candidates = sorted(
        run.glob(
            f"score_samples_metrics_*_{kind_tag}_"
            "betas_1p00_geo2000_ew20000.json"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {kind_tag} score file in {run}, found "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def parse_status(path: Path) -> dict[str, str]:
    require_file(path, "run status")
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            result[key.strip()] = value.strip()
    require_equal(result.get("phase"), "complete", f"{path} phase")
    return result


def validate_score(
    path: Path,
    *,
    required_kinds: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    score = read_json(path)
    require_equal(score.get("problem"), "dw4", f"{path} problem")
    require_equal(
        score.get("data_name"),
        "dw4_reference_v2/train",
        f"{path} training data",
    )
    require_equal(
        score.get("reference_data_name"),
        "dw4_reference_v2/eval",
        f"{path} reference data",
    )
    require_equal(
        [float(value) for value in score.get("eval_betas", [])],
        [TARGET_BETA],
        f"{path} eval beta",
    )
    require_equal(
        int(score.get("energy_w2_samples", -1)),
        EXPECTED_ENERGY_SAMPLES,
        f"{path} energy sample protocol",
    )
    require_equal(
        int(score.get("geometric_w2_samples", -1)),
        EXPECTED_GEOMETRIC_SAMPLES,
        f"{path} geometric sample protocol",
    )
    require_equal(
        tuple(str(value) for value in score.get("score_kinds", [])),
        required_kinds,
        f"{path} score kinds",
    )
    beta = score.get("per_beta", {}).get("1.00")
    if not isinstance(beta, dict):
        raise ValueError(f"{path} has no beta=1.00 metrics")
    require_close(float(beta.get("beta", np.nan)), TARGET_BETA, f"{path} beta")
    require_equal(
        int(beta.get("reference_rows", -1)),
        EXPECTED_ROWS,
        f"{path} reference rows",
    )
    reference_source = str(beta.get("reference_source", "")).replace("\\", "/")
    if not reference_source.endswith(
        "data/dw4_reference_v2/eval/samples_beta_1.00.npy"
    ):
        raise ValueError(
            f"{path} uses an unexpected reference source: {reference_source!r}"
        )
    metrics: dict[str, dict[str, float]] = {}
    for kind in required_kinds:
        item = beta.get(kind)
        if not isinstance(item, dict):
            raise ValueError(f"{path} has no {kind} metrics")
        require_equal(item.get("status"), "ok", f"{path} {kind} status")
        require_equal(
            int(item.get("sample_rows", -1)),
            EXPECTED_ROWS,
            f"{path} {kind} sample rows",
        )
        values = {
            "energy_w2_20k": float(item["ew2_n"]),
            "energy_w2_full": float(item["ew2"]),
            "pairwise_distance_w2_full": float(item["pairwise_distance_w2"]),
            "geometric_w2_2k": float(item["geometric_w2"]),
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"{path} {kind} has non-finite metrics: {values}")
        metrics[kind] = values
    return score, metrics


def validate_replicate_binding(path: Path, seed: int, fm_sha: str, reference_sha: str) -> dict[str, Any]:
    binding = read_json(path)
    require_equal(binding.get("schema"), "adtm.dw4_reference_v2.am_replicate.v1", f"{path} schema")
    require_equal(int(binding.get("shared_fm_seed", -1)), 0, f"{path} shared FM seed")
    require_equal(int(binding.get("am_seed", -1)), int(seed), f"{path} AM seed")
    require_equal(int(binding.get("evaluation_seed", -1)), EXPECTED_EVAL_SEED, f"{path} eval seed")
    require_equal(
        int(binding.get("evaluation_initial_key", -1)),
        EXPECTED_INITIAL_KEY,
        f"{path} initial key",
    )
    require_equal(
        int(binding.get("evaluation_chunk_size", -1)),
        EXPECTED_CHUNK_SIZE,
        f"{path} chunk size",
    )
    require_equal(
        int(binding.get("evaluation_rows", -1)),
        EXPECTED_ROWS,
        f"{path} evaluation rows",
    )
    require_equal(binding.get("parent_fm_sha256"), fm_sha, f"{path} FM SHA")
    require_equal(
        binding.get("reference_manifest_sha256"),
        reference_sha,
        f"{path} reference SHA",
    )
    return binding


def validate_seed2_script(path: Path) -> None:
    text = require_file(path, "seed2 exact run script").read_text(encoding="utf-8")
    required_fragments = (
        "--seed 2",
        "--am-steps 1000",
        "--am-batch-size 512",
        "--am-lr 5.0e-7",
        "--am-lr-min 5.0e-9",
        "--K 40",
        "--am-num-loss-steps 20",
        "--am-keep-last-steps 10",
        "--max-sigma 50.0",
        "--energy-grad-scale 1.0",
        "--grad-clip 1.0",
        "--seed 0",
        "--n-eval 100000",
        "--eval-chunk-size 5000",
        "--ode-steps 150",
        "--ode-method euler",
        "--prior-scale 1.0",
        "--energy-w2-samples 20000",
        "--geometric-w2-samples 2000",
    )
    missing = [value for value in required_fragments if value not in text]
    if missing:
        raise ValueError(f"Seed2 run script is missing protocol fragments: {missing}")


def validate_replicate_scripts(run_by_seed: dict[int, Path]) -> dict[int, str]:
    paths = {
        seed: require_file(
            run_by_seed[seed] / "exact_run_script.sh",
            f"seed{seed} exact replicate script",
        )
        for seed in (1, 3)
    }
    hashes = {seed: sha256_file(path) for seed, path in paths.items()}
    if len(set(hashes.values())) != 1:
        raise ValueError(f"Seeds 1/3 do not use one replicate script: {hashes}")
    text = paths[1].read_text(encoding="utf-8")
    required_fragments = (
        '--seed "${AM_SEED}"',
        "--am-steps 1000",
        "--am-batch-size 512",
        "--am-lr 5.0e-7",
        "--am-lr-min 5.0e-9",
        "--K 40",
        "--am-num-loss-steps 20",
        "--am-keep-last-steps 10",
        "--max-sigma 50.0",
        "--energy-grad-scale 1.0",
        "--grad-clip 1.0",
        "--seed 0",
        "--n-eval 100000",
        "--eval-chunk-size 5000",
        "--ode-steps 150",
        "--ode-method euler",
        "--prior-scale 1.0",
        "--energy-w2-samples 20000",
        "--geometric-w2-samples 2000",
    )
    missing = [value for value in required_fragments if value not in text]
    if missing:
        raise ValueError(
            f"AM replicate script is missing protocol fragments: {missing}"
        )
    return hashes


def verify_input_sha256_manifest(root: Path, path: Path) -> None:
    require_file(path, "reference SHA256SUMS")
    for raw in path.read_text(encoding="ascii").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        candidate = root / relative.lstrip("*")
        require_file(candidate, f"reference manifest member {relative}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(
                f"Reference SHA mismatch for {relative}: {actual} != {expected}"
            )


def deterministic_initial_x(
    *,
    n_rows: int = EXPECTED_ROWS,
    chunk_size: int = EXPECTED_CHUNK_SIZE,
    key_value: int = EXPECTED_INITIAL_KEY,
    prior_scale: float = EXPECTED_PRIOR_SCALE,
) -> np.ndarray:
    """Reconstruct the exact chunked EGNN ODE Gaussian inputs."""

    base_key = jax.random.PRNGKey(int(key_value))
    chunks: list[np.ndarray] = []
    completed = 0
    chunk_id = 0
    while completed < int(n_rows):
        count = min(int(chunk_size), int(n_rows) - completed)
        key = jax.random.fold_in(base_key, chunk_id)
        value = float(prior_scale) * jax.random.normal(
            key,
            (count, 8),
            dtype=jnp.float32,
        )
        value = value.reshape((count, 4, 2))
        value = value - jnp.mean(value, axis=1, keepdims=True)
        chunks.append(np.asarray(jax.device_get(value.reshape((count, 8))), dtype=np.float32))
        completed += count
        chunk_id += 1
    result = np.concatenate(chunks, axis=0)
    if result.shape != EXPECTED_SHAPE or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid deterministic initial_X: {result.shape}")
    validate_com(result, "deterministic initial_X")
    return result


def subset_rows(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    if int(count) > int(values.shape[0]):
        raise ValueError(f"Cannot draw {count} rows from {values.shape[0]}")
    rng = np.random.default_rng(int(seed))
    index = rng.choice(values.shape[0], size=int(count), replace=False)
    return np.asarray(values[index], dtype=np.float32)


def model_pair_w2_20k(samples: np.ndarray, reference: np.ndarray) -> float:
    model = subset_rows(samples, EXPECTED_ENERGY_SAMPLES, 33_001)
    target = subset_rows(reference, EXPECTED_ENERGY_SAMPLES, 33_002)
    return equal_rank_w2(pair_distances_np(model), pair_distances_np(target))


def sample_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size <= 0 or not np.isfinite(array).all():
        raise ValueError(f"Cannot summarize values: {array}")
    return {
        "values": [float(value) for value in array],
        "mean": float(np.mean(array)),
        "sample_std_ddof1": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": float(np.median(array)),
        "n": int(array.size),
    }


def reference_floors(
    train: np.ndarray,
    evaluation: np.ndarray,
    *,
    geometric_chunk_size: int,
) -> dict[str, Any]:
    train_energy = dw4_energy_np(train)
    eval_energy = dw4_energy_np(evaluation)
    train_pair = pair_distances_np(train)
    eval_pair = pair_distances_np(evaluation)
    energy_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for repeat in range(30):
        train_seed = REFERENCE_FLOOR_SEED + 10_000 + repeat * 2
        eval_seed = train_seed + 1
        train_index = np.random.default_rng(train_seed).choice(
            train.shape[0],
            size=EXPECTED_ENERGY_SAMPLES,
            replace=False,
        )
        eval_index = np.random.default_rng(eval_seed).choice(
            evaluation.shape[0],
            size=EXPECTED_ENERGY_SAMPLES,
            replace=False,
        )
        energy_value = equal_rank_w2(
            train_energy[train_index],
            eval_energy[eval_index],
        )
        pair_value = equal_rank_w2(
            train_pair[train_index],
            eval_pair[eval_index],
        )
        energy_rows.append(
            {
                "repeat": repeat,
                "train_row_seed": train_seed,
                "eval_row_seed": eval_seed,
                "n_rows_each": EXPECTED_ENERGY_SAMPLES,
                "value": energy_value,
            }
        )
        pair_rows.append(
            {
                "repeat": repeat,
                "train_row_seed": train_seed,
                "eval_row_seed": eval_seed,
                "n_rows_each": EXPECTED_ENERGY_SAMPLES,
                "value": pair_value,
            }
        )

    problem = make_problem("dw4")
    geometric_rows: list[dict[str, Any]] = []
    for repeat in range(10):
        train_seed = REFERENCE_FLOOR_SEED + 20_000 + repeat * 2
        eval_seed = train_seed + 1
        train_subset = subset_rows(train, EXPECTED_GEOMETRIC_SAMPLES, train_seed)
        eval_subset = subset_rows(evaluation, EXPECTED_GEOMETRIC_SAMPLES, eval_seed)
        value = geometric_w2(
            train_subset,
            eval_subset,
            problem,
            n_samples=EXPECTED_GEOMETRIC_SAMPLES,
            seed=REFERENCE_FLOOR_SEED + 30_000 + repeat,
            cost_chunk_size=int(geometric_chunk_size),
            dem_refine_top_k=32,
        )
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"Non-finite geometric reference floor repeat {repeat}: {value}")
        geometric_rows.append(
            {
                "repeat": repeat,
                "train_row_seed": train_seed,
                "eval_row_seed": eval_seed,
                "metric_seed": REFERENCE_FLOOR_SEED + 30_000 + repeat,
                "n_rows_each": EXPECTED_GEOMETRIC_SAMPLES,
                "value": float(value),
            }
        )

    def package(
        rows: list[dict[str, Any]],
        *,
        metric: str,
        n_rows: int,
    ) -> dict[str, Any]:
        summary = sample_summary(float(row["value"]) for row in rows)
        return {
            "metric": metric,
            "construction": "independent corrected SMC train split versus eval split",
            "n_rows_each": int(n_rows),
            "repeats": len(rows),
            "runs": rows,
            **summary,
        }

    return {
        "schema": "adtm.dw4_reference_v2.reference_floor.v1",
        "base_seed": REFERENCE_FLOOR_SEED,
        "note": (
            "These empirical floors include finite-sample and independent-SMC-split "
            "discrepancy. SMC resampling induces genealogy, so they are not iid "
            "Monte Carlo standard errors."
        ),
        "energy_w2_20k": package(
            energy_rows,
            metric="energy_w2_20k",
            n_rows=EXPECTED_ENERGY_SAMPLES,
        ),
        "pairwise_distance_w2_20k": package(
            pair_rows,
            metric="pairwise_distance_w2_20k",
            n_rows=EXPECTED_ENERGY_SAMPLES,
        ),
        "geometric_w2_2k": package(
            geometric_rows,
            metric="geometric_w2_2k",
            n_rows=EXPECTED_GEOMETRIC_SAMPLES,
        ),
    }


def aggregate_metric(shared_fm: float, am_values: dict[int, float]) -> dict[str, Any]:
    ordered = np.asarray([am_values[seed] for seed in EXPECTED_AM_SEEDS], dtype=np.float64)
    improvements = np.asarray(float(shared_fm) - ordered, dtype=np.float64)
    relative = improvements / float(shared_fm)
    return {
        "shared_fm_value": float(shared_fm),
        "shared_fm_n": 1,
        "am_by_seed": {str(seed): float(am_values[seed]) for seed in EXPECTED_AM_SEEDS},
        "am_summary": sample_summary(ordered),
        "absolute_improvement_fm_minus_am": {
            "by_seed": {
                str(seed): float(improvements[index])
                for index, seed in enumerate(EXPECTED_AM_SEEDS)
            },
            **sample_summary(improvements),
        },
        "relative_improvement": {
            "definition": "(shared_fm - am_seed) / shared_fm",
            "by_seed": {
                str(seed): float(relative[index])
                for index, seed in enumerate(EXPECTED_AM_SEEDS)
            },
            **sample_summary(relative),
        },
        "wins_am_below_shared_fm": int(np.sum(ordered < float(shared_fm))),
        "n_am_training_seeds": len(EXPECTED_AM_SEEDS),
    }


def robust_limits(
    series: dict[str, np.ndarray],
    low: float,
    high: float,
) -> tuple[float, float]:
    pooled = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.values()]
    )
    pooled = pooled[np.isfinite(pooled)]
    left, right = np.quantile(pooled, (float(low), float(high)))
    width = max(float(right - left), 1.0e-8)
    return float(left - 0.04 * width), float(right + 0.04 * width)


def full_mass_density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    count, _ = np.histogram(finite, bins=edges)
    return count / (finite.size * np.diff(edges))


def save_figure(fig: plt.Figure, base: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for suffix in ("png", "pdf"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(
            path,
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
        )
        result[suffix] = {
            "path": path.name,
            "sha256": sha256_file(path),
        }
    plt.close(fig)
    return result


def distribution_figure(
    *,
    reference: np.ndarray,
    fm: np.ndarray,
    am_by_seed: dict[int, np.ndarray],
    title: str,
    xlabel: str,
    output: Path,
    quantiles: tuple[float, float],
    bins: int = 110,
) -> dict[str, Any]:
    series = {
        "Equilibrium reference": np.asarray(reference, dtype=np.float64).reshape(-1),
        "Shared FM seed0": np.asarray(fm, dtype=np.float64).reshape(-1),
        **{
            f"AM seed{seed}": np.asarray(am_by_seed[seed], dtype=np.float64).reshape(-1)
            for seed in EXPECTED_AM_SEEDS
        },
    }
    xlim = robust_limits(series, *quantiles)
    edges = np.linspace(xlim[0], xlim[1], int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = {label: full_mass_density(value, edges) for label, value in series.items()}
    am_density = np.stack(
        [density[f"AM seed{seed}"] for seed in EXPECTED_AM_SEEDS],
        axis=0,
    )
    am_mean = np.mean(am_density, axis=0)
    am_std = np.std(am_density, axis=0, ddof=1)

    fig, axis = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    axis.fill_between(
        centers,
        density["Equilibrium reference"],
        step="mid",
        color=PALETTE["reference"],
        alpha=0.72,
        linewidth=0.0,
        label="Equilibrium reference",
    )
    axis.fill_between(
        centers,
        density["Shared FM seed0"],
        step="mid",
        color=PALETTE["fm"],
        alpha=0.60,
        linewidth=0.0,
        label="Shared FM seed0",
    )
    axis.fill_between(
        centers,
        np.maximum(am_mean - am_std, 0.0),
        am_mean + am_std,
        step="mid",
        color=PALETTE["am_ribbon"],
        alpha=0.38,
        linewidth=0.0,
        label="AM mean ± sample SD",
    )
    for color, seed in zip(PALETTE["am_seeds"], EXPECTED_AM_SEEDS):
        axis.step(
            centers,
            density[f"AM seed{seed}"],
            where="mid",
            color=color,
            alpha=0.82,
            linewidth=1.25,
            label=f"AM seed{seed}",
        )
    axis.step(
        centers,
        am_mean,
        where="mid",
        color=PALETTE["am_mean"],
        linewidth=2.5,
        label="AM seed mean",
    )
    axis.set_xlim(*xlim)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_title(title)
    axis.grid(alpha=0.12, linewidth=0.7)
    axis.legend(frameon=True, framealpha=0.94, fontsize=9.5)
    files = save_figure(fig, output)
    tails = {}
    for label, value in series.items():
        finite = np.asarray(value, dtype=np.float64)
        tails[label] = {
            "n": int(finite.size),
            "below": int(np.sum(finite < xlim[0])),
            "above": int(np.sum(finite > xlim[1])),
            "inside_fraction": float(
                np.mean((finite >= xlim[0]) & (finite <= xlim[1]))
            ),
            "full_min": float(np.min(finite)),
            "full_max": float(np.max(finite)),
        }
    return {
        "kind": "all_seed_distribution",
        "title": title,
        "xlabel": xlabel,
        "bins": int(bins),
        "xlim": [float(value) for value in xlim],
        "quantiles": [float(value) for value in quantiles],
        "normalization": (
            "counts / (full finite sample count * bin width); robust-axis tails "
            "are not renormalized"
        ),
        "am_uncertainty": "binwise sample standard deviation across AM seeds 1/2/3",
        "tails": tails,
        "files": files,
    }


def seed_dot_figure(
    *,
    metric: str,
    shared_fm: float,
    am_values: dict[int, float],
    floor: dict[str, Any],
    ylabel: str,
    output: Path,
) -> dict[str, Any]:
    am = np.asarray([am_values[seed] for seed in EXPECTED_AM_SEEDS], dtype=np.float64)
    floor_mean = float(floor["mean"])
    floor_std = float(floor["sample_std_ddof1"])
    fig, axis = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    axis.axhspan(
        max(0.0, floor_mean - floor_std),
        floor_mean + floor_std,
        color=PALETTE["reference"],
        alpha=0.35,
        label="Reference floor mean ± sample SD",
    )
    axis.axhline(
        floor_mean,
        color="#4F9D69",
        linewidth=1.8,
    )
    axis.scatter(
        [0.0],
        [float(shared_fm)],
        marker="D",
        s=85,
        color=PALETTE["fm"],
        edgecolor="#8F4E1F",
        linewidth=0.8,
        zorder=4,
        label="Shared FM seed0",
    )
    jitter = np.asarray((-0.12, 0.0, 0.12))
    for index, (color, seed) in enumerate(zip(PALETTE["am_seeds"], EXPECTED_AM_SEEDS)):
        axis.scatter(
            [1.0 + jitter[index]],
            [am[index]],
            s=66,
            color=color,
            edgecolor=PALETTE["am_mean"],
            linewidth=0.7,
            zorder=5,
            label=f"AM seed{seed}",
        )
    axis.errorbar(
        [1.0],
        [float(np.mean(am))],
        yerr=[float(np.std(am, ddof=1))],
        fmt="o",
        color=PALETTE["am_mean"],
        ecolor=PALETTE["am_mean"],
        elinewidth=2.0,
        capsize=5,
        markersize=7,
        zorder=6,
        label="AM mean ± sample SD",
    )
    axis.set_xlim(-0.45, 1.45)
    axis.set_xticks((0.0, 1.0), ("Shared FM", "AM seeds"))
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel(ylabel)
    axis.set_title(f"DW4 at β=1.00: {metric}")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=True, framealpha=0.94, fontsize=9)
    files = save_figure(fig, output)
    return {
        "kind": "seed_dot",
        "metric": metric,
        "shared_fm_value": float(shared_fm),
        "am_by_seed": {str(seed): float(am_values[seed]) for seed in EXPECTED_AM_SEEDS},
        "am_mean": float(np.mean(am)),
        "am_sample_std_ddof1": float(np.std(am, ddof=1)),
        "reference_floor_mean": floor_mean,
        "reference_floor_sample_std_ddof1": floor_std,
        "files": files,
    }


def write_per_seed_csv(
    path: Path,
    aggregate: dict[str, Any],
) -> None:
    metrics = aggregate["metrics"]
    fieldnames = [
        "am_seed",
        "metric",
        "shared_fm_value",
        "am_value",
        "absolute_improvement_fm_minus_am",
        "relative_improvement",
        "am_below_shared_fm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, record in metrics.items():
            for seed in EXPECTED_AM_SEEDS:
                seed_key = str(seed)
                writer.writerow(
                    {
                        "am_seed": seed,
                        "metric": metric,
                        "shared_fm_value": record["shared_fm_value"],
                        "am_value": record["am_by_seed"][seed_key],
                        "absolute_improvement_fm_minus_am": record[
                            "absolute_improvement_fm_minus_am"
                        ]["by_seed"][seed_key],
                        "relative_improvement": record["relative_improvement"][
                            "by_seed"
                        ][seed_key],
                        "am_below_shared_fm": int(
                            record["am_by_seed"][seed_key]
                            < record["shared_fm_value"]
                        ),
                    }
                )


def write_aggregate_csv(
    path: Path,
    aggregate: dict[str, Any],
    floors: dict[str, Any],
) -> None:
    fieldnames = [
        "metric",
        "shared_fm_value",
        "am_mean",
        "am_sample_std_ddof1",
        "am_min",
        "am_max",
        "improvement_mean",
        "improvement_sample_std_ddof1",
        "relative_improvement_mean",
        "relative_improvement_sample_std_ddof1",
        "wins",
        "n_am_seeds",
        "reference_floor_mean",
        "reference_floor_sample_std_ddof1",
        "reference_floor_repeats",
        "reference_floor_rows_each",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, record in aggregate["metrics"].items():
            floor = floors[metric]
            writer.writerow(
                {
                    "metric": metric,
                    "shared_fm_value": record["shared_fm_value"],
                    "am_mean": record["am_summary"]["mean"],
                    "am_sample_std_ddof1": record["am_summary"][
                        "sample_std_ddof1"
                    ],
                    "am_min": record["am_summary"]["min"],
                    "am_max": record["am_summary"]["max"],
                    "improvement_mean": record[
                        "absolute_improvement_fm_minus_am"
                    ]["mean"],
                    "improvement_sample_std_ddof1": record[
                        "absolute_improvement_fm_minus_am"
                    ]["sample_std_ddof1"],
                    "relative_improvement_mean": record[
                        "relative_improvement"
                    ]["mean"],
                    "relative_improvement_sample_std_ddof1": record[
                        "relative_improvement"
                    ]["sample_std_ddof1"],
                    "wins": record["wins_am_below_shared_fm"],
                    "n_am_seeds": record["n_am_training_seeds"],
                    "reference_floor_mean": floor["mean"],
                    "reference_floor_sample_std_ddof1": floor[
                        "sample_std_ddof1"
                    ],
                    "reference_floor_repeats": floor["repeats"],
                    "reference_floor_rows_each": floor["n_rows_each"],
                }
            )


def write_comparison_markdown(
    path: Path,
    aggregate: dict[str, Any],
    floors: dict[str, Any],
) -> None:
    labels = {
        "energy_w2_20k": "Energy W2 (20k)",
        "pairwise_distance_w2_20k": "Pair-distance W2 (20k)",
        "geometric_w2_2k": "Geometric W2 (2k)",
    }
    lines = [
        "# Corrected DW4: fixed FM seed0 and AM seeds 1/2/3",
        "",
        "| Metric | Shared FM | AM seed1 | AM seed2 | AM seed3 | "
        "AM mean ± sample SD | Wins | Reference floor mean ± sample SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, label in labels.items():
        record = aggregate["metrics"][metric]
        floor = floors[metric]
        lines.append(
            f"| {label} | {record['shared_fm_value']:.6f} | "
            f"{record['am_by_seed']['1']:.6f} | "
            f"{record['am_by_seed']['2']:.6f} | "
            f"{record['am_by_seed']['3']:.6f} | "
            f"{record['am_summary']['mean']:.6f} ± "
            f"{record['am_summary']['sample_std_ddof1']:.6f} | "
            f"{record['wins_am_below_shared_fm']}/3 | "
            f"{floor['mean']:.6f} ± {floor['sample_std_ddof1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The FM column is one shared checkpoint, not three independent FM runs. "
            "The AM standard deviation describes only AM-training stochasticity "
            "conditional on this FM checkpoint and the fixed corrected dataset.",
            "",
            "Reference floors compare independent corrected-SMC train and eval "
            "splits under the same metric-specific sample protocol. They are "
            "empirical discrepancies, not iid Monte Carlo standard errors.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    path: Path,
    aggregate: dict[str, Any],
    floors: dict[str, Any],
) -> None:
    energy = aggregate["metrics"]["energy_w2_20k"]
    pair = aggregate["metrics"]["pairwise_distance_w2_20k"]
    geom = aggregate["metrics"]["geometric_w2_2k"]
    text = f"""# Corrected DW4 fixed-FM AM-seed aggregate

This directory aggregates one shared corrected-reference FM checkpoint
(`train seed 0`) and three independent AM refinements (`train seeds 1, 2, 3`).
It is not a three-seed end-to-end FM+AM experiment.

## Primary result

- Energy W2 (20k): shared FM `{energy['shared_fm_value']:.6f}`; AM
  `{energy['am_summary']['mean']:.6f} ± {energy['am_summary']['sample_std_ddof1']:.6f}`.
- Pair-distance W2 (20k): shared FM `{pair['shared_fm_value']:.6f}`; AM
  `{pair['am_summary']['mean']:.6f} ± {pair['am_summary']['sample_std_ddof1']:.6f}`.
- Geometric W2 (2k): shared FM `{geom['shared_fm_value']:.6f}`; AM
  `{geom['am_summary']['mean']:.6f} ± {geom['am_summary']['sample_std_ddof1']:.6f}`.

Standard deviations use `ddof=1` across the three AM training seeds.

## Scientific scope

The corrected reference was generated with annealed SMC in the intrinsic
six-dimensional COM-free space, systematic resampling and Metropolis-adjusted
Langevin rejuvenation. It replaces the legacy short fixed-step ULA endpoint
dataset. The SMC particles are finite and genealogically dependent, so the
reference and reference-floor values are empirical approximations rather than
analytic truth or iid Monte Carlo standard errors.

The independent Gaussian-mixture importance calculation and exact virial /
cross-beta identities audit the reference construction, but the importance
calculation has limited effective sample size and is not a second exact ground
truth.

All four controllers were evaluated with the same seed-0 chunked Gaussian
initialization: base key `{EXPECTED_INITIAL_KEY}`, chunk size
`{EXPECTED_CHUNK_SIZE}`, `N={EXPECTED_ROWS}`. The exact array and SHA-256 are
stored as `initial_X.npy` and in `experiment_bindings.json`.

## Reference floors

- Energy W2: {floors['energy_w2_20k']['mean']:.6f} ±
  {floors['energy_w2_20k']['sample_std_ddof1']:.6f}
  ({floors['energy_w2_20k']['repeats']} repeats, 20k rows per split).
- Pair-distance W2: {floors['pairwise_distance_w2_20k']['mean']:.6f} ±
  {floors['pairwise_distance_w2_20k']['sample_std_ddof1']:.6f}
  ({floors['pairwise_distance_w2_20k']['repeats']} repeats, 20k rows per split).
- Geometric W2: {floors['geometric_w2_2k']['mean']:.6f} ±
  {floors['geometric_w2_2k']['sample_std_ddof1']:.6f}
  ({floors['geometric_w2_2k']['repeats']} repeats, 2k rows per split).

The permitted conclusion is that AM refinement is or is not consistently
better than the one shared FM baseline. Full pipeline robustness would require
independent FM training seeds.
"""
    path.write_text(text, encoding="utf-8")


def write_hash_manifest(root: Path) -> Path:
    output = root / "AGGREGATE_SHA256SUMS"
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path != output
        ),
        key=lambda value: value.relative_to(root).as_posix(),
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in paths
    ]
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output


def verify_hash_manifest(root: Path, manifest: Path) -> None:
    for raw in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = raw.split(maxsplit=1)
        path = root / relative.lstrip("*")
        if not path.is_file():
            raise FileNotFoundError(f"Aggregate manifest missing file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Aggregate manifest mismatch for {relative}: "
                f"{actual} != {expected}"
            )


def build(args: argparse.Namespace, output: Path) -> None:
    reference_root = resolve(args.reference)
    run_by_seed = {
        1: resolve(args.seed1_run),
        2: resolve(args.seed2_run),
        3: resolve(args.seed3_run),
    }
    reference_manifest_path = require_file(
        reference_root / "manifest.json",
        "reference manifest",
    )
    reference_manifest = read_json(reference_manifest_path)
    require_equal(
        reference_manifest.get("schema"),
        "adtm.dw4_reference_v2.bundle.v1",
        "reference manifest schema",
    )
    require_equal(
        reference_manifest.get("legacy_fixed_step_ula_used"),
        False,
        "reference legacy ULA flag",
    )
    require_equal(
        reference_manifest.get("audit_status"),
        "pass",
        "reference audit status",
    )
    require_equal(
        [float(value) for value in reference_manifest.get("betas", [])],
        [0.8, 1.0, 1.2],
        "reference beta grid",
    )
    for split in ("train", "eval"):
        require_equal(
            int(reference_manifest.get("splits", {}).get(split, {}).get("n_per_beta", -1)),
            EXPECTED_ROWS,
            f"reference {split} rows",
        )
    reference_manifest_sha = sha256_file(reference_manifest_path)
    verify_input_sha256_manifest(
        reference_root,
        reference_root / "SHA256SUMS",
    )
    convergence_audit = read_json(reference_root / "convergence_audit.json")
    require_equal(
        convergence_audit.get("status"),
        "pass",
        "reference convergence audit",
    )
    require_file(reference_root / "importance_audit.json", "reference importance audit")

    reference_paths: dict[str, dict[float, Path]] = {"train": {}, "eval": {}}
    reference_arrays: dict[str, dict[float, np.ndarray]] = {"train": {}, "eval": {}}
    reference_com: dict[str, dict[float, float]] = {"train": {}, "eval": {}}
    for split in ("train", "eval"):
        for beta in (0.8, 1.0, 1.2):
            path = require_file(
                reference_root / split / f"samples_beta_{beta:.2f}.npy",
                f"reference {split} beta{beta:.2f}",
            )
            array = load_array(path, f"reference {split} beta{beta:.2f}")
            reference_paths[split][beta] = path
            reference_arrays[split][beta] = array
            reference_com[split][beta] = validate_com(
                array,
                f"reference {split} beta{beta:.2f}",
            )
    reference_train_path = reference_paths["train"][1.0]
    reference_eval_path = reference_paths["eval"][1.0]
    reference_train = reference_arrays["train"][1.0]
    reference_eval = reference_arrays["eval"][1.0]

    status_by_seed = {
        seed: parse_status(run / "status.txt")
        for seed, run in run_by_seed.items()
    }
    seed2_score_path = discover_score(run_by_seed[2], ("fm", "am"))
    _, seed2_metrics = validate_score(
        seed2_score_path,
        required_kinds=("fm", "am"),
    )
    score_paths = {2: seed2_score_path}
    score_metrics = {2: seed2_metrics["am"]}
    for seed in (1, 3):
        path = discover_score(run_by_seed[seed], ("am",))
        _, metrics = validate_score(path, required_kinds=("am",))
        score_paths[seed] = path
        score_metrics[seed] = metrics["am"]

    fm_checkpoint_paths = {
        seed: require_file(run / "fm_params.pkl", f"seed{seed} FM checkpoint")
        for seed, run in run_by_seed.items()
    }
    fm_checkpoint_hashes = {
        seed: sha256_file(path)
        for seed, path in fm_checkpoint_paths.items()
    }
    if len(set(fm_checkpoint_hashes.values())) != 1:
        raise ValueError(f"Runs do not share one FM checkpoint: {fm_checkpoint_hashes}")
    shared_fm_sha = fm_checkpoint_hashes[2]

    fm_sample_paths = {
        seed: require_file(
            run / "fm_beta_sweep" / "fm_samples_beta_1.00.npy",
            f"seed{seed} shared FM samples",
        )
        for seed, run in run_by_seed.items()
    }
    fm_sample_hashes = {
        seed: sha256_file(path)
        for seed, path in fm_sample_paths.items()
    }
    if len(set(fm_sample_hashes.values())) != 1:
        raise ValueError(f"Runs do not share one FM sample array: {fm_sample_hashes}")
    shared_fm = load_array(fm_sample_paths[2], "shared FM samples")
    shared_fm_com = validate_com(shared_fm, "shared FM samples")

    am_checkpoint_paths = {
        seed: require_file(run / "am_params.pkl", f"seed{seed} AM checkpoint")
        for seed, run in run_by_seed.items()
    }
    am_checkpoint_hashes = {
        seed: sha256_file(path)
        for seed, path in am_checkpoint_paths.items()
    }
    if len(set(am_checkpoint_hashes.values())) != len(EXPECTED_AM_SEEDS):
        raise ValueError(f"AM checkpoints are not distinct: {am_checkpoint_hashes}")
    am_sample_paths = {
        seed: require_file(
            run / "am_beta_sweep" / "am_samples_beta_1.00.npy",
            f"seed{seed} AM samples",
        )
        for seed, run in run_by_seed.items()
    }
    am_samples = {
        seed: load_array(path, f"seed{seed} AM samples")
        for seed, path in am_sample_paths.items()
    }
    am_com = {
        seed: validate_com(samples, f"seed{seed} AM samples")
        for seed, samples in am_samples.items()
    }

    copied_reference_hashes = {}
    for seed, run in run_by_seed.items():
        path = require_file(
            run / "reference_manifest.json",
            f"seed{seed} copied reference manifest",
        )
        copied_reference_hashes[seed] = sha256_file(path)
    if set(copied_reference_hashes.values()) != {reference_manifest_sha}:
        raise ValueError(
            "Run/reference manifest mismatch: "
            f"bundle={reference_manifest_sha}, runs={copied_reference_hashes}"
        )

    reference_audit_sources = {
        "convergence": reference_root / "convergence_audit.json",
        "importance": reference_root / "importance_audit.json",
        "exact_identity": run_by_seed[2] / "reference_exact_identity_audit.json",
    }
    exact_identity = read_json(reference_audit_sources["exact_identity"])
    require_equal(
        exact_identity.get("status"),
        "pass",
        "reference exact-identity audit",
    )
    reference_audit_hashes = {
        name: sha256_file(path)
        for name, path in reference_audit_sources.items()
    }
    copied_audit_names = {
        "convergence": "reference_convergence_audit.json",
        "importance": "reference_importance_audit.json",
        "exact_identity": "reference_exact_identity_audit.json",
    }
    for seed, run in run_by_seed.items():
        for name, filename in copied_audit_names.items():
            copied = require_file(
                run / filename,
                f"seed{seed} copied {name} audit",
            )
            actual = sha256_file(copied)
            expected = reference_audit_hashes[name]
            if actual != expected:
                raise ValueError(
                    f"seed{seed} {name} audit SHA {actual} != {expected}"
                )

    seed2_script_path = run_by_seed[2] / "exact_run_script.sh"
    validate_seed2_script(seed2_script_path)
    replicate_script_hashes = validate_replicate_scripts(run_by_seed)
    replicate_bindings = {
        seed: validate_replicate_binding(
            run_by_seed[seed] / "replicate_binding.json",
            seed,
            shared_fm_sha,
            reference_manifest_sha,
        )
        for seed in (1, 3)
    }

    pair20k = {
        "fm": model_pair_w2_20k(shared_fm, reference_eval),
        **{
            f"am{seed}": model_pair_w2_20k(am_samples[seed], reference_eval)
            for seed in EXPECTED_AM_SEEDS
        },
    }
    shared_metrics = {
        "energy_w2_20k": float(seed2_metrics["fm"]["energy_w2_20k"]),
        "pairwise_distance_w2_20k": float(pair20k["fm"]),
        "geometric_w2_2k": float(seed2_metrics["fm"]["geometric_w2_2k"]),
    }
    am_metric_values = {
        "energy_w2_20k": {
            seed: float(score_metrics[seed]["energy_w2_20k"])
            for seed in EXPECTED_AM_SEEDS
        },
        "pairwise_distance_w2_20k": {
            seed: float(pair20k[f"am{seed}"])
            for seed in EXPECTED_AM_SEEDS
        },
        "geometric_w2_2k": {
            seed: float(score_metrics[seed]["geometric_w2_2k"])
            for seed in EXPECTED_AM_SEEDS
        },
    }
    floors = reference_floors(
        reference_train,
        reference_eval,
        geometric_chunk_size=int(args.geometric_chunk_size),
    )
    aggregate = {
        "schema": SCHEMA,
        "target_beta": TARGET_BETA,
        "experimental_unit": (
            "three independent AM training seeds conditional on one shared "
            "FM seed0 checkpoint and one corrected reference bundle"
        ),
        "not_an_end_to_end_three_seed_claim": True,
        "metric_protocol": {
            "energy_w2_20k": {
                "n_rows_each": EXPECTED_ENERGY_SAMPLES,
                "score_seed_model": 33_001,
                "score_seed_reference": 33_002,
            },
            "pairwise_distance_w2_20k": {
                "n_rows_each": EXPECTED_ENERGY_SAMPLES,
                "score_seed_model": 33_001,
                "score_seed_reference": 33_002,
                "note": (
                    "Recomputed at 20k for protocol-matched reference floors; "
                    "the source score JSON also retains full-array pair W2."
                ),
            },
            "geometric_w2_2k": {
                "n_rows_each": EXPECTED_GEOMETRIC_SAMPLES,
                "score_seed": 31_001,
                "exact_particle_permutations": True,
                "orthogonal_procrustes": True,
            },
        },
        "metrics": {
            metric: aggregate_metric(shared_metrics[metric], am_metric_values[metric])
            for metric in shared_metrics
        },
        "source_score_full_array_metrics": {
            "shared_fm": seed2_metrics["fm"],
            "am_by_seed": {
                str(seed): score_metrics[seed]
                for seed in EXPECTED_AM_SEEDS
            },
        },
        "reference_floors": floors,
    }

    output.mkdir(parents=True, exist_ok=False)
    initial_x = deterministic_initial_x()
    initial_x_path = output / "initial_X.npy"
    np.save(initial_x_path, initial_x)

    bindings = {
        "schema": "adtm.dw4_reference_v2.am_seed_bindings.v1",
        "reference": {
            "root": relative_input(reference_root),
            "manifest_path": relative_input(reference_manifest_path),
            "manifest_sha256": reference_manifest_sha,
            "sha256sums_path": relative_input(reference_root / "SHA256SUMS"),
            "sha256sums_sha256": sha256_file(reference_root / "SHA256SUMS"),
            "audit_sha256": reference_audit_hashes,
            "arrays": {
                split: {
                    f"{beta:.2f}": {
                        "path": relative_input(reference_paths[split][beta]),
                        "sha256": sha256_file(reference_paths[split][beta]),
                        "shape": list(reference_arrays[split][beta].shape),
                        "dtype": str(reference_arrays[split][beta].dtype),
                        "com_abs_max": reference_com[split][beta],
                    }
                    for beta in (0.8, 1.0, 1.2)
                }
                for split in ("train", "eval")
            },
        },
        "shared_fm": {
            "train_seed": 0,
            "checkpoint_sha256": shared_fm_sha,
            "checkpoint_hashes_by_run": {
                str(seed): value for seed, value in fm_checkpoint_hashes.items()
            },
            "sample_sha256": fm_sample_hashes[2],
            "sample_hashes_by_run": {
                str(seed): value for seed, value in fm_sample_hashes.items()
            },
            "sample_shape": list(shared_fm.shape),
            "sample_com_abs_max": shared_fm_com,
        },
        "am_runs": {
            str(seed): {
                "train_seed": seed,
                "run_dir": relative_input(run_by_seed[seed]),
                "checkpoint_path": relative_input(am_checkpoint_paths[seed]),
                "checkpoint_sha256": am_checkpoint_hashes[seed],
                "sample_path": relative_input(am_sample_paths[seed]),
                "sample_sha256": sha256_file(am_sample_paths[seed]),
                "sample_shape": list(am_samples[seed].shape),
                "sample_com_abs_max": am_com[seed],
                "score_path": relative_input(score_paths[seed]),
                "score_sha256": sha256_file(score_paths[seed]),
                "status": status_by_seed[seed],
                "replicate_binding": replicate_bindings.get(seed),
                "exact_run_script_sha256": (
                    sha256_file(seed2_script_path)
                    if seed == 2
                    else replicate_script_hashes[seed]
                ),
            }
            for seed in EXPECTED_AM_SEEDS
        },
        "evaluation_protocol": {
            "evaluation_seed": EXPECTED_EVAL_SEED,
            "initial_key": EXPECTED_INITIAL_KEY,
            "chunk_size": EXPECTED_CHUNK_SIZE,
            "n_rows": EXPECTED_ROWS,
            "prior_scale": EXPECTED_PRIOR_SCALE,
            "ode_steps": EXPECTED_ODE_STEPS,
            "ode_method": EXPECTED_ODE_METHOD,
            "initial_X_relative_path": "initial_X.npy",
            "initial_X_sha256": sha256_file(initial_x_path),
            "initial_X_shape": list(initial_x.shape),
            "initial_X_dtype": str(initial_x.dtype),
            "initial_X_com_abs_max": com_abs_max(initial_x),
        },
        "source_script": {
            "path": relative_input(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    write_json(output / "aggregate_metrics.json", aggregate)
    write_json(
        output / "per_seed_metrics.json",
        {
            "schema": "adtm.dw4_reference_v2.am_per_seed_metrics.v1",
            "shared_fm_seed": 0,
            "am_training_seeds": list(EXPECTED_AM_SEEDS),
            "note": (
                "FM is one shared observation. Rows indexed by AM seed quantify "
                "AM-training variability conditional on that FM checkpoint."
            ),
            "metrics": {
                metric: {
                    "shared_fm_value": record["shared_fm_value"],
                    "am_by_seed": record["am_by_seed"],
                    "absolute_improvement_fm_minus_am": record[
                        "absolute_improvement_fm_minus_am"
                    ]["by_seed"],
                    "relative_improvement": record["relative_improvement"][
                        "by_seed"
                    ],
                }
                for metric, record in aggregate["metrics"].items()
            },
        },
    )
    write_json(output / "reference_floor.json", floors)
    write_json(output / "experiment_bindings.json", bindings)
    write_per_seed_csv(output / "per_seed_metrics.csv", aggregate)
    write_aggregate_csv(output / "aggregate_metrics.csv", aggregate, floors)
    write_comparison_markdown(
        output / "comparison_table.md",
        aggregate,
        floors,
    )
    write_readme(output / "README.md", aggregate, floors)

    figure_root = output / "figures"
    figure_root.mkdir()
    reference_energy = dw4_energy_np(reference_eval)
    fm_energy = dw4_energy_np(shared_fm)
    am_energy = {
        seed: dw4_energy_np(am_samples[seed])
        for seed in EXPECTED_AM_SEEDS
    }
    reference_pair = pair_distances_np(reference_eval).reshape(-1)
    fm_pair = pair_distances_np(shared_fm).reshape(-1)
    am_pair = {
        seed: pair_distances_np(am_samples[seed]).reshape(-1)
        for seed in EXPECTED_AM_SEEDS
    }
    figure_metadata: dict[str, Any] = {
        "schema": "adtm.dw4_reference_v2.am_seed_figures.v1",
        "target_beta": TARGET_BETA,
        "palette": PALETTE,
        "sample_bindings": {
            "reference_sha256": bindings["reference"]["arrays"]["eval"]["1.00"][
                "sha256"
            ],
            "shared_fm_sha256": bindings["shared_fm"]["sample_sha256"],
            "am_by_seed_sha256": {
                str(seed): bindings["am_runs"][str(seed)]["sample_sha256"]
                for seed in EXPECTED_AM_SEEDS
            },
        },
        "figures": {},
    }
    figure_metadata["figures"]["energy_distribution"] = distribution_figure(
        reference=reference_energy,
        fm=fm_energy,
        am_by_seed=am_energy,
        title=r"DW4 at $\beta=1.00$: energy distribution",
        xlabel=r"$U(x)$",
        output=figure_root / "dw4_beta1p00_energy_all_am_seeds",
        quantiles=(0.0005, 0.9995),
    )
    figure_metadata["figures"]["pairwise_distance_distribution"] = distribution_figure(
        reference=reference_pair,
        fm=fm_pair,
        am_by_seed=am_pair,
        title=r"DW4 at $\beta=1.00$: pairwise distances",
        xlabel="Pairwise distance",
        output=figure_root / "dw4_beta1p00_pairwise_distance_all_am_seeds",
        quantiles=(0.001, 0.999),
    )
    for coordinate_index in range(4):
        figure_metadata["figures"][f"x{coordinate_index + 1}_distribution"] = (
            distribution_figure(
                reference=reference_eval[:, coordinate_index],
                fm=shared_fm[:, coordinate_index],
                am_by_seed={
                    seed: am_samples[seed][:, coordinate_index]
                    for seed in EXPECTED_AM_SEEDS
                },
                title=(
                    rf"DW4 at $\beta=1.00$: "
                    rf"$x_{{{coordinate_index + 1}}}$ marginal"
                ),
                xlabel=rf"$x_{{{coordinate_index + 1}}}$",
                output=(
                    figure_root
                    / f"dw4_beta1p00_x{coordinate_index + 1}_all_am_seeds"
                ),
                quantiles=(0.001, 0.999),
            )
        )
    dot_labels = {
        "energy_w2_20k": "Energy W2 (20k)",
        "pairwise_distance_w2_20k": "Pair-distance W2 (20k)",
        "geometric_w2_2k": "Geometric W2 (2k)",
    }
    for metric, label in dot_labels.items():
        record = aggregate["metrics"][metric]
        figure_metadata["figures"][f"{metric}_seed_dots"] = seed_dot_figure(
            metric=label,
            shared_fm=float(record["shared_fm_value"]),
            am_values={
                seed: float(record["am_by_seed"][str(seed)])
                for seed in EXPECTED_AM_SEEDS
            },
            floor=floors[metric],
            ylabel=label,
            output=figure_root / f"dw4_beta1p00_{metric}_seed_dots",
        )
    write_json(figure_root / "figure_metadata.json", figure_metadata)
    manifest = write_hash_manifest(output)
    verify_hash_manifest(output, manifest)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    value.add_argument("--seed1-run", type=Path, default=DEFAULT_SEED1)
    value.add_argument("--seed2-run", type=Path, default=DEFAULT_SEED2)
    value.add_argument("--seed3-run", type=Path, default=DEFAULT_SEED3)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--geometric-chunk-size", type=int, default=16)
    value.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing aggregate only after the new temporary build succeeds.",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    final_output = resolve(args.output)
    if final_output.exists() and not bool(args.force):
        raise FileExistsError(
            f"Refusing to overwrite existing aggregate: {final_output}. "
            "Pass --force to replace it after a successful temporary build."
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.tmp-",
            dir=final_output.parent,
        )
    )
    build_root = temporary / final_output.name
    try:
        build(args, build_root)
        if final_output.exists():
            shutil.rmtree(final_output)
        build_root.replace(final_output)
        temporary.rmdir()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(final_output)


if __name__ == "__main__":
    main()
