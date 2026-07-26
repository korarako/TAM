#!/usr/bin/env python3
"""Build the extended, sample-only ADTM Ala2 evaluation freeze.

The script never trains a model or generates molecular configurations.  It
uses fixed index positions from already-saved MD/FM/AM arrays, evaluates the
same 10,000 configurations per series, and writes only derived metrics,
figures, and a compact energy cache.

Two historical cases are registered by default:

* ``best_mixed``: grid35 seed2, 500 K -> 400 K, 100k saved samples.
* ``failure``: grid3579, 900 K -> 800 K, 10k saved samples.

The four-temperature 200k grid3579 FM evaluation is used only for independent
MD-vs-FM Ramachandran free-energy sanity figures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "adtm.ala2.extended_freeze.v1"
N_COMPARE = 10_000
RAMA_BINS = 80
MARGINAL_BINS = 160
ENERGY_BINS = 120
MIN_PAIR_BINS = 120
CHIRALITY_BINS = 120
ENERGY_QUANTILES = (0.01, 0.99)
MIN_PAIR_UPPER_QUANTILE = 0.9995
FES_FLOOR = 1.0e-12
FES_MAX_KBT = 8.0
PAIR_CHUNK_SIZE = 2_048
TORSION_CHUNK_SIZE = 50_000

COLORS = {
    "md": "#8FD18A",
    "fm": "#F2A270",
    "am": "#4776CC",
}
LABELS = {
    "md": "MD Reference",
    "fm": "FM",
    "am": "AM",
}

REMOTE_ADTM_ROOT = Path("${ADTM_ROOT}")
REMOTE_ALA2_ROOT = REMOTE_ADTM_ROOT / "outputs/ala2"
REMOTE_PASTTRY_ROOT = REMOTE_ALA2_ROOT / "pasttry"
REMOTE_DATA_ROOT = Path("${AMBOLTZ_ROOT}/data/ala2")

DEFAULT_BEST_RUN = REMOTE_PASTTRY_ROOT / (
    "ala2_chiro_painn_jax_grid35_am_500K_to_400K_1k_lr1e-6_const_bs16_"
    "K40_loss20_keep10_eg0p35_seed2_adam095_wd0p01_gc0_fixedsigma"
)
DEFAULT_BEST_FM_SOURCE = REMOTE_PASTTRY_ROOT / "ala2_chiro_painn_jax_grid35_30k_constlr1e-4"
DEFAULT_FAILURE_RUN = REMOTE_PASTTRY_ROOT / (
    "ala2_chiro_painn_jax_grid3579_am_900K_to_800K_1k_lr5e-7_bs16_"
    "K40_loss20_keep10_eg1_adam095_wd0p01_gc0_fixedsigma"
)
DEFAULT_FAILURE_FM_SOURCE = REMOTE_PASTTRY_ROOT / "ala2_chiro_painn_jax_grid3579_30k_constlr1e-4"
DEFAULT_BASELINE_ROOT = REMOTE_PASTTRY_ROOT / "ala2_chiro_painn_jax_grid3579_eval34579_200k_gpu2"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    interpretation: str
    source_temperature_k: int
    target_temperature_k: int
    target_beta: float
    beta_key: str
    expected_rows: int
    md_samples: Path
    fm_samples: Path
    am_samples: Path
    legacy_fm_metrics: Path
    legacy_am_metrics: Path
    fm_checkpoint: Path
    am_checkpoint: Path
    fm_config: Path
    run_root: Path


def parse_args() -> argparse.Namespace:
    local_adtm_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adtm-root", type=Path, default=local_adtm_root)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--best-run-root", type=Path, default=DEFAULT_BEST_RUN)
    parser.add_argument("--best-fm-source-root", type=Path, default=DEFAULT_BEST_FM_SOURCE)
    parser.add_argument("--failure-run-root", type=Path, default=DEFAULT_FAILURE_RUN)
    parser.add_argument("--failure-fm-source-root", type=Path, default=DEFAULT_FAILURE_FM_SOURCE)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--pdb-path", type=Path, default=REMOTE_DATA_ROOT / "structure_vac.pdb")
    parser.add_argument(
        "--cases",
        choices=("best_mixed", "failure"),
        nargs="+",
        default=("best_mixed", "failure"),
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--openmm-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--openmm-precision", choices=("single", "mixed", "double"), default="mixed")
    parser.add_argument("--energy-chunk-size", type=int, default=256)
    parser.add_argument("--force-recompute-energy", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values)
    if canonical.dtype == np.int64:
        canonical = canonical.astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def json_dump_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_record(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    resolved = path.resolve()
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(resolved)
        return None
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def validate_sample_array(path: Path, expected_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    array = np.load(resolved, mmap_mode="r", allow_pickle=False)
    expected_shape = (int(expected_rows), 66)
    if array.shape != expected_shape:
        raise ValueError(f"{resolved}: expected shape {expected_shape}, got {array.shape}")
    if array.dtype != np.float32:
        raise ValueError(f"{resolved}: expected float32, got {array.dtype}")
    for start in range(0, array.shape[0], TORSION_CHUNK_SIZE):
        chunk = np.asarray(array[start : start + TORSION_CHUNK_SIZE])
        if not bool(np.isfinite(chunk).all()):
            raise ValueError(f"{resolved}: contains NaN or Inf; rows are never filtered")
    record = {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "all_source_rows_finite": True,
    }
    return array, record


def fixed_index_positions(n_rows: int) -> np.ndarray:
    if int(n_rows) < N_COMPARE:
        raise ValueError(f"Need at least {N_COMPARE} rows, got {n_rows}")
    if int(n_rows) == N_COMPARE:
        indices = np.arange(N_COMPARE, dtype=np.int64)
    else:
        indices = np.linspace(0, int(n_rows) - 1, num=N_COMPARE, dtype=np.int64)
    if indices.shape != (N_COMPARE,) or np.unique(indices).size != N_COMPARE:
        raise AssertionError("Fixed index construction did not produce 10,000 unique positions")
    return indices


def select_rows(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(array[indices], dtype=np.float32)
    if selected.shape != (N_COMPARE, 66):
        raise ValueError(f"Unexpected selected shape: {selected.shape}")
    if not bool(np.isfinite(selected).all()):
        raise ValueError("Selected rows contain NaN or Inf; no row filtering is permitted")
    return selected


def phi_psi(samples: np.ndarray, chunk_size: int = TORSION_CHUNK_SIZE) -> np.ndarray:
    """Match ``adj_thermo.metrics_ala2.ALA2_TORSION_INDICES`` exactly."""

    outputs: list[np.ndarray] = []
    for start in range(0, len(samples), int(chunk_size)):
        coords = np.asarray(samples[start : start + int(chunk_size)], dtype=np.float64).reshape((-1, 22, 3))
        angles: list[np.ndarray] = []
        for i, j, k, ell in ((4, 6, 7, 8), (6, 7, 8, 16)):
            p0, p1, p2, p3 = coords[:, i], coords[:, j], coords[:, k], coords[:, ell]
            b0 = p0 - p1
            b1 = p2 - p1
            b2 = p3 - p2
            b1 = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1.0e-12)
            v = b0 - b1 * np.sum(b0 * b1, axis=-1, keepdims=True)
            w = b2 - b1 * np.sum(b2 * b1, axis=-1, keepdims=True)
            y = np.sum(np.cross(b1, v) * w, axis=-1)
            x = np.sum(v * w, axis=-1)
            angles.append(np.arctan2(y, x))
        outputs.append(np.stack(angles, axis=-1))
    return np.concatenate(outputs, axis=0)


def pair_distance_matrix(samples: np.ndarray, chunk_size: int = PAIR_CHUNK_SIZE) -> np.ndarray:
    coords = np.asarray(samples, dtype=np.float32).reshape((-1, 22, 3))
    tri = np.triu_indices(22, k=1)
    output = np.empty((coords.shape[0], tri[0].size), dtype=np.float64)
    for start in range(0, coords.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), coords.shape[0])
        chunk = coords[start:stop]
        delta = chunk[:, tri[0], :] - chunk[:, tri[1], :]
        output[start:stop] = np.linalg.norm(delta, axis=-1)
    return output


def normalized_chirality(samples: np.ndarray) -> np.ndarray:
    """Return the normalized CA handedness triple product in data atom order.

    Data-order atom indices are CA=7, N=6, C=8, and CB=12.  Positive values
    are the handedness observed throughout the registered MD references.
    """

    coords = np.asarray(samples, dtype=np.float64).reshape((-1, 22, 3))
    center = coords[:, 7]
    unit_vectors: list[np.ndarray] = []
    for atom_index in (6, 8, 12):
        vector = coords[:, atom_index] - center
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        if np.any(norm <= 0.0):
            raise ValueError("Zero-length CA bond encountered in chirality diagnostic")
        unit_vectors.append(vector / norm)
    return np.sum(unit_vectors[0] * np.cross(unit_vectors[1], unit_vectors[2]), axis=-1)


def wasserstein2_equal(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Fair W2 requires equal shapes, got {left.shape} and {right.shape}")
    if not bool(np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("Fair W2 inputs must be finite; values are never filtered")
    return float(np.sqrt(np.mean((np.sort(left) - np.sort(right)) ** 2)))


def histogram2_counts(values: np.ndarray, bins: int = RAMA_BINS) -> np.ndarray:
    histogram, _, _ = np.histogram2d(
        values[:, 0],
        values[:, 1],
        bins=int(bins),
        range=((-math.pi, math.pi), (-math.pi, math.pi)),
        density=False,
    )
    return histogram.T.astype(np.float64, copy=False)


def js_divergence_counts(p_counts: np.ndarray, q_counts: np.ndarray, eps: float = 1.0e-12) -> float:
    p = np.asarray(p_counts, dtype=np.float64).reshape(-1)
    q = np.asarray(q_counts, dtype=np.float64).reshape(-1)
    p = p / (float(np.sum(p)) + eps)
    q = q / (float(np.sum(q)) + eps)
    midpoint = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0.0, p * np.log((p + eps) / (midpoint + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0.0, q * np.log((q + eps) / (midpoint + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def fes_from_counts(counts: np.ndarray) -> np.ndarray:
    probability = np.asarray(counts, dtype=np.float64)
    probability = probability / max(float(np.sum(probability)), FES_FLOOR)
    free_energy = -np.log(np.maximum(probability, FES_FLOOR))
    free_energy -= float(np.nanmin(free_energy))
    return np.clip(free_energy, 0.0, FES_MAX_KBT)


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not bool(np.isfinite(array).all()):
        raise ValueError("Summary inputs must be finite; values are never filtered")
    quantiles = np.quantile(array, (0.01, 0.05, 0.50, 0.90, 0.99, 0.999))
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "q99_9": float(quantiles[5]),
        "max": float(np.max(array)),
        "finite_fraction": 1.0,
    }


def circular_summary(values: np.ndarray) -> dict[str, float]:
    z = np.exp(1j * np.asarray(values, dtype=np.float64))
    mean_z = np.mean(z)
    return {
        "circular_mean_rad": float(np.angle(mean_z)),
        "resultant_length": float(np.abs(mean_z)),
    }


def com_norm_mean(samples: np.ndarray) -> float:
    coords = np.asarray(samples, dtype=np.float64).reshape((-1, 22, 3))
    return float(np.mean(np.linalg.norm(np.mean(coords, axis=1), axis=-1)))


def openmm_runtime_version() -> str:
    try:
        return str(importlib.metadata.version("OpenMM"))
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def openmm_installation_record() -> dict[str, Any]:
    record: dict[str, Any] = {"version": openmm_runtime_version()}
    for label, module_name in (
        ("python_package", "openmm"),
        ("native_extension", "openmm._openmm"),
    ):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError):
            spec = None
        origin = None if spec is None else spec.origin
        if origin is not None and Path(origin).is_file():
            record[label] = file_record(Path(origin))
        else:
            record[label] = None
    return record


def energy_binding(
    spec: CaseSpec,
    records: dict[str, dict[str, Any]],
    indices: np.ndarray,
    args: argparse.Namespace,
    script_sha256: str,
) -> dict[str, Any]:
    adtm_root = args.adtm_root.resolve()
    bridge_path = adtm_root / "src/adj_thermo/openmm_ala2.py"
    problem_path = adtm_root / "src/adj_thermo/problem/ala2.py"
    pdb_path = args.pdb_path.resolve()
    binding_sources = {}
    for key in ("md", "fm", "am"):
        binding_sources[key] = {
            "path": records[key]["path"],
            "sha256": records[key]["sha256"],
            "shape": records[key]["shape"],
            "dtype": records[key]["dtype"],
            "indices_sha256": sha256_array(indices),
            "selected_shape": [N_COMPARE, 66],
        }
    return {
        "schema": "adtm.ala2.openmm_energy_cache.v1",
        "case_id": spec.case_id,
        "selection": {
            "policy": "same deterministic evenly-spaced index positions; all positions when N=10000",
            "count": N_COMPARE,
            "indices_sha256": sha256_array(indices),
            "first": int(indices[0]),
            "last": int(indices[-1]),
            "no_content_filtering": True,
        },
        "sources": binding_sources,
        "potential": {
            "system_name": "amber99sbildn_obc1_xml",
            "forcefield": "amber99sbildn.xml",
            "forcefield_water": "implicit/obc1.xml",
            "coordinates_unit": "nm",
            "energy_unit": "kJ/mol",
            "constraints": None,
            "nonbonded_method": "NoCutoff",
            "remove_cmmotion": False,
            "center_before_evaluation": True,
            "gradient_clip": None,
            "openmm_device": str(args.openmm_device),
            "openmm_precision": str(args.openmm_precision),
            "openmm_installation": openmm_installation_record(),
            "pdb": file_record(pdb_path),
            "openmm_bridge": file_record(bridge_path),
            "ala2_problem_source": file_record(problem_path),
        },
        "plot_script_sha256": script_sha256,
    }


def compute_openmm_energies(
    selected: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    source_dir = (args.adtm_root.resolve() / "src")
    sys.path.insert(0, str(source_dir))
    from adj_thermo import openmm_ala2 as bridge_module

    expected_module = (source_dir / "adj_thermo/openmm_ala2.py").resolve()
    actual_module = Path(bridge_module.__file__).resolve()
    if actual_module != expected_module:
        raise RuntimeError(f"Imported unexpected OpenMM bridge: {actual_module} != {expected_module}")

    output: dict[str, np.ndarray] = {}
    chunk_size = max(1, int(args.energy_chunk_size))
    for key in ("md", "fm", "am"):
        pieces: list[np.ndarray] = []
        samples = selected[key]
        print(
            f"[energy] case series={key} device={args.openmm_device} "
            f"precision={args.openmm_precision} rows={samples.shape[0]}",
            flush=True,
        )
        for start in range(0, samples.shape[0], chunk_size):
            stop = min(start + chunk_size, samples.shape[0])
            energy, _ = bridge_module.openmm_energy_and_grad_np(
                samples[start:stop],
                device=str(args.openmm_device),
                precision=str(args.openmm_precision),
                system_name="amber99sbildn_obc1_xml",
                forcefield="amber99sbildn.xml",
                forcefield_water="implicit/obc1.xml",
                pdb_path=args.pdb_path.resolve(),
                grad_clip=None,
                center=True,
            )
            pieces.append(np.asarray(energy, dtype=np.float64))
            if stop == samples.shape[0] or stop % max(10 * chunk_size, 1) == 0:
                print(f"[energy] {key}: {stop}/{samples.shape[0]}", flush=True)
        values = np.concatenate(pieces)
        if values.shape != (N_COMPARE,) or not bool(np.isfinite(values).all()):
            raise ValueError(f"OpenMM returned invalid {key} energies: {values.shape}")
        output[key] = values
    return output


def load_or_compute_energy_cache(
    cache_path: Path,
    binding: dict[str, Any],
    indices: np.ndarray,
    selected: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], bool, dict[str, Any]]:
    expected_binding_json = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if cache_path.is_file() and not bool(args.force_recompute_energy):
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_binding_json = str(np.asarray(archive["binding_json"]).item())
            if cached_binding_json != expected_binding_json:
                raise ValueError(
                    f"Energy cache binding mismatch: {cache_path}. "
                    "Refuse reuse; inspect inputs or pass --force-recompute-energy."
                )
            energies = {
                key: np.asarray(archive[f"energy_{key}"], dtype=np.float64)
                for key in ("md", "fm", "am")
            }
            cached_indices = np.asarray(archive["indices"], dtype=np.int64)
            runtime = json.loads(str(np.asarray(archive["runtime_json"]).item()))
        expected_indices_sha = binding["selection"]["indices_sha256"]
        if sha256_array(cached_indices) != expected_indices_sha:
            raise ValueError(f"Energy cache index payload mismatch: {cache_path}")
        for key, values in energies.items():
            if values.shape != (N_COMPARE,) or not bool(np.isfinite(values).all()):
                raise ValueError(f"Energy cache contains invalid {key} values: {cache_path}")
        return energies, True, runtime

    energies = compute_openmm_energies(selected, args)
    runtime = {
        "openmm_runtime_version": openmm_runtime_version(),
        "device": str(args.openmm_device),
        "precision": str(args.openmm_precision),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        binding_json=np.asarray(expected_binding_json),
        runtime_json=np.asarray(json.dumps(runtime, sort_keys=True, separators=(",", ":"))),
        indices=np.asarray(indices, dtype=np.int64),
        energy_md=energies["md"],
        energy_fm=energies["fm"],
        energy_am=energies["am"],
    )
    os.replace(temporary, cache_path)
    return energies, False, runtime


def style_axis(axis: Any) -> None:
    axis.grid(True, color="#C9C9C9", alpha=0.35, linewidth=0.7)
    axis.tick_params(direction="out", width=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def save_figure(
    figure: Any,
    output_root: Path,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        files.append(path.resolve().relative_to(output_root.resolve()).as_posix())
    plt.close(figure)
    return files


def angle_ticks(axis: Any) -> None:
    ticks = (-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi)
    labels = (r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$")
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_yticks(ticks)
    axis.set_yticklabels(labels)


def plot_fes_single(
    counts: np.ndarray,
    title: str,
    output_root: Path,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> list[str]:
    free_energy = fes_from_counts(counts)
    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    image = axis.imshow(
        free_energy,
        origin="lower",
        extent=(-math.pi, math.pi, -math.pi, math.pi),
        aspect="equal",
        cmap="viridis_r",
        vmin=0.0,
        vmax=FES_MAX_KBT,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel(r"$\phi$ (rad)")
    axis.set_ylabel(r"$\psi$ (rad)")
    axis.set_xlim(-math.pi, math.pi)
    axis.set_ylim(-math.pi, math.pi)
    angle_ticks(axis)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$F(\phi,\psi) / k_B T$")
    style_axis(axis)
    figure.tight_layout()
    return save_figure(figure, output_root, output_dir, stem, dpi)


def histogram_density(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density, _ = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def plot_three_way_filled(
    series: dict[str, np.ndarray],
    edges: np.ndarray,
    title: str,
    xlabel: str,
    output_root: Path,
    output_dir: Path,
    stem: str,
    dpi: int,
    label_suffixes: dict[str, str] | None = None,
    zero_line: bool = False,
) -> list[str]:
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    for key in ("md", "fm", "am"):
        centers, density = histogram_density(series[key], edges)
        suffix = "" if label_suffixes is None else label_suffixes.get(key, "")
        axis.plot(centers, density, color=COLORS[key], linewidth=2.2, label=LABELS[key] + suffix)
        axis.fill_between(centers, 0.0, density, color=COLORS[key], alpha=0.20)
    if zero_line:
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0, label="handedness boundary")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Probability density")
    axis.set_xlim(float(edges[0]), float(edges[-1]))
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    return save_figure(figure, output_root, output_dir, stem, dpi)


def robust_energy_range(
    energies: dict[str, np.ndarray],
) -> tuple[float, float, dict[str, dict[str, float]]]:
    pooled = np.concatenate([np.asarray(energies[key], dtype=np.float64) for key in ("md", "fm", "am")])
    lower, upper = (float(value) for value in np.quantile(pooled, ENERGY_QUANTILES))
    if not upper > lower:
        upper = lower + 1.0
    tails: dict[str, dict[str, float]] = {}
    for key in ("md", "fm", "am"):
        values = np.asarray(energies[key], dtype=np.float64)
        below = float(np.mean(values < lower))
        above = float(np.mean(values > upper))
        tails[key] = {
            "below_axis_fraction": below,
            "above_axis_fraction": above,
            "outside_axis_fraction": below + above,
            "shown_fraction": float(np.mean((values >= lower) & (values <= upper))),
        }
    return lower, upper, tails


def load_legacy_metric(path: Path, beta_key: str) -> dict[str, Any]:
    record = file_record(path)
    assert record is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload["per_beta"][str(beta_key)]
    fixed_count = int(item.get("energy_w2_samples", payload.get("energy_w2_samples", 0)))
    normalized = {
        "energy_w2_native_full": float(item["ew2"]),
        "energy_w2_fixed_1k": float(item["ew2_2k"]),
        "energy_w2_fixed_sample_count": fixed_count,
        "pair_distance_w2": float(item["w2"]),
        "min_pair_distance_w2": float(item["min_pair_distance_w2"]),
        "phi_w2_linear_cut": float(item["phi_w2"]),
        "psi_w2_linear_cut": float(item["psi_w2"]),
        "rama_js_80bin": float(item["rama_js"]),
    }
    if fixed_count != 1_000:
        normalized["energy_w2_fixed_1k_label_warning"] = (
            f"Expected historical fixed count 1000, found {fixed_count}"
        )
    return {
        "source": record,
        "native_n_eval": int(payload["n_eval"]),
        "ode_steps": int(payload["ode_steps"]),
        "ode_method": str(payload["ode_method"]),
        "raw_item": item,
        "normalized": normalized,
        "corrections": {
            "ew2_2k": "Historical key is misnamed: energy_w2_samples is 1000, so this is fixed1k.",
            "w2": "This is pooled pair-distance W2, not symmetry-aware geometric W2.",
        },
    }


def build_case_specs(args: argparse.Namespace) -> dict[str, CaseSpec]:
    best_run = args.best_run_root.resolve()
    best_fm_source = args.best_fm_source_root.resolve()
    failure_run = args.failure_run_root.resolve()
    failure_fm_source = args.failure_fm_source_root.resolve()
    return {
        "best_mixed": CaseSpec(
            case_id="best_mixed",
            interpretation=(
                "Best available mixed Ala2 case: legacy fixed1k energy, Rama, phi, and psi improve, "
                "while the full energy tail remains a required caveat."
            ),
            source_temperature_k=500,
            target_temperature_k=400,
            target_beta=0.3006808876068151,
            beta_key="0.30",
            expected_rows=100_000,
            md_samples=best_run / "am_beta_sweep/md_samples_beta_0.30.npy",
            fm_samples=best_run / "fm_beta_sweep/fm_samples_beta_0.30.npy",
            am_samples=best_run / "am_beta_sweep/am_samples_beta_0.30.npy",
            legacy_fm_metrics=best_run / "fm_beta_sweep/fm_beta_sweep_metrics.json",
            legacy_am_metrics=best_run / "am_beta_sweep/am_beta_sweep_metrics.json",
            fm_checkpoint=best_run / "fm_params.pkl",
            am_checkpoint=best_run / "am_params.pkl",
            fm_config=best_fm_source / "config.yaml",
            run_root=best_run,
        ),
        "failure": CaseSpec(
            case_id="failure",
            interpretation=(
                "Representative 900 K -> 800 K failure/mixed case: only limited observables improve "
                "and the result must not be generalized beyond this historical seed."
            ),
            source_temperature_k=900,
            target_temperature_k=800,
            target_beta=0.15034044380340755,
            beta_key="0.15",
            expected_rows=10_000,
            md_samples=failure_run / "am_beta_sweep/md_samples_beta_0.15.npy",
            fm_samples=failure_fm_source / "fm_beta_sweep/fm_samples_beta_0.15.npy",
            am_samples=failure_run / "am_beta_sweep/am_samples_beta_0.15.npy",
            legacy_fm_metrics=failure_fm_source / "fm_beta_sweep/fm_beta_sweep_metrics.json",
            legacy_am_metrics=failure_run / "am_beta_sweep/am_beta_sweep_metrics.json",
            fm_checkpoint=failure_run / "fm_params.pkl",
            am_checkpoint=failure_run / "am_params.pkl",
            fm_config=failure_fm_source / "config.yaml",
            run_root=failure_run,
        ),
    }


def process_case(
    spec: CaseSpec,
    args: argparse.Namespace,
    output_root: Path,
    script_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print(f"[case] {spec.case_id}: loading registered arrays", flush=True)
    sources: dict[str, np.ndarray] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for key, path in (
        ("md", spec.md_samples),
        ("fm", spec.fm_samples),
        ("am", spec.am_samples),
    ):
        sources[key], source_records[key] = validate_sample_array(path, spec.expected_rows)

    indices = fixed_index_positions(spec.expected_rows)
    selected = {key: select_rows(sources[key], indices) for key in ("md", "fm", "am")}
    selection_record = {
        "policy": "same deterministic evenly-spaced index positions; all positions when N=10000",
        "source_rows": int(spec.expected_rows),
        "selected_rows": N_COMPARE,
        "first": int(indices[0]),
        "last": int(indices[-1]),
        "indices_sha256": sha256_array(indices),
        "no_content_filtering": True,
        "same_positions_for_md_fm_am": True,
    }

    binding = energy_binding(spec, source_records, indices, args, script_sha256)
    cache_path = output_root / "derived" / f"{spec.case_id}_openmm_energy_10k.npz"
    energies, cache_reused, openmm_runtime = load_or_compute_energy_cache(
        cache_path,
        binding,
        indices,
        selected,
        args,
    )

    torsions = {key: phi_psi(selected[key]) for key in ("md", "fm", "am")}
    pair_matrices = {key: pair_distance_matrix(selected[key]) for key in ("md", "fm", "am")}
    minimum_pairs = {key: np.min(pair_matrices[key], axis=1) for key in ("md", "fm", "am")}
    chiralities = {key: normalized_chirality(selected[key]) for key in ("md", "fm", "am")}
    rama_counts = {key: histogram2_counts(torsions[key]) for key in ("md", "fm", "am")}

    reference_sign = 1.0 if float(np.median(chiralities["md"])) >= 0.0 else -1.0
    series_summary: dict[str, Any] = {}
    for key in ("md", "fm", "am"):
        series_summary[key] = {
            "energy_kj_mol": finite_summary(energies[key]),
            "pair_distance_nm": finite_summary(pair_matrices[key]),
            "minimum_pair_distance_nm": finite_summary(minimum_pairs[key]),
            "phi": {
                **finite_summary(torsions[key][:, 0]),
                **circular_summary(torsions[key][:, 0]),
            },
            "psi": {
                **finite_summary(torsions[key][:, 1]),
                **circular_summary(torsions[key][:, 1]),
            },
            "chirality": {
                **finite_summary(chiralities[key]),
                "reference_handedness_sign": int(reference_sign),
                "wrong_handed_fraction": float(np.mean(reference_sign * chiralities[key] <= 0.0)),
            },
            "com_norm_mean_nm": com_norm_mean(selected[key]),
        }

    comparisons: dict[str, Any] = {}
    for key in ("fm", "am"):
        comparisons[key] = {
            "energy_w2_10k_kj_mol": wasserstein2_equal(energies[key], energies["md"]),
            "pair_distance_w2_10k_nm": wasserstein2_equal(
                pair_matrices[key],
                pair_matrices["md"],
            ),
            "min_pair_distance_w2_10k_nm": wasserstein2_equal(
                minimum_pairs[key],
                minimum_pairs["md"],
            ),
            "phi_w2_10k_linear_cut_rad": wasserstein2_equal(
                torsions[key][:, 0],
                torsions["md"][:, 0],
            ),
            "psi_w2_10k_linear_cut_rad": wasserstein2_equal(
                torsions[key][:, 1],
                torsions["md"][:, 1],
            ),
            "rama_js_10k_80bin": js_divergence_counts(
                rama_counts[key],
                rama_counts["md"],
            ),
        }
    delta = {
        metric: float(comparisons["fm"][metric] - comparisons["am"][metric])
        for metric in comparisons["fm"]
    }

    energy_lower, energy_upper, energy_tails = robust_energy_range(energies)
    figures_dir = output_root / "figures" / spec.case_id
    figure_records: list[dict[str, Any]] = []
    for key in ("md", "fm", "am"):
        stem = f"{spec.case_id}_{spec.target_temperature_k}k_rama_fes_{key}"
        files = plot_fes_single(
            rama_counts[key],
            f"ADTM Ala2 {spec.target_temperature_k} K - {LABELS[key]}",
            output_root,
            figures_dir,
            stem,
            args.dpi,
        )
        figure_records.append(
            {
                "stem": stem,
                "files": files,
                "kind": "ramachandran_fes",
                "series": [key],
                "shared_scale_kbt": [0.0, FES_MAX_KBT],
            }
        )

    angle_edges = np.linspace(-math.pi, math.pi, MARGINAL_BINS + 1)
    for torsion_index, name, symbol in ((0, "phi", r"$\phi$ (rad)"), (1, "psi", r"$\psi$ (rad)")):
        stem = f"{spec.case_id}_{spec.target_temperature_k}k_{name}_marginal"
        files = plot_three_way_filled(
            {key: torsions[key][:, torsion_index] for key in ("md", "fm", "am")},
            angle_edges,
            f"ADTM Ala2 {spec.target_temperature_k} K - {name} marginal",
            symbol,
            output_root,
            figures_dir,
            stem,
            args.dpi,
        )
        figure_records.append(
            {
                "stem": stem,
                "files": files,
                "kind": f"{name}_marginal",
                "series": ["md", "fm", "am"],
            }
        )

    energy_edges = np.linspace(energy_lower, energy_upper, ENERGY_BINS + 1)
    energy_suffixes = {
        key: f" (tail mass {100.0 * energy_tails[key]['outside_axis_fraction']:.2f}%)"
        for key in ("md", "fm", "am")
    }
    energy_stem = f"{spec.case_id}_{spec.target_temperature_k}k_energy_robust"
    energy_files = plot_three_way_filled(
        energies,
        energy_edges,
        f"ADTM Ala2 {spec.target_temperature_k} K - potential energy",
        "Potential energy (kJ/mol)",
        output_root,
        figures_dir,
        energy_stem,
        args.dpi,
        label_suffixes=energy_suffixes,
    )
    figure_records.append(
        {
            "stem": energy_stem,
            "files": energy_files,
            "kind": "robust_energy_distribution",
            "series": ["md", "fm", "am"],
            "axis_kj_mol": [energy_lower, energy_upper],
            "tail_mass": energy_tails,
        }
    )

    min_pair_upper = float(
        np.quantile(
            np.concatenate([minimum_pairs[key] for key in ("md", "fm", "am")]),
            MIN_PAIR_UPPER_QUANTILE,
        )
    )
    min_pair_edges = np.linspace(0.0, min_pair_upper, MIN_PAIR_BINS + 1)
    min_pair_stem = f"{spec.case_id}_{spec.target_temperature_k}k_min_pair_distance"
    min_pair_files = plot_three_way_filled(
        minimum_pairs,
        min_pair_edges,
        f"ADTM Ala2 {spec.target_temperature_k} K - minimum pair distance",
        "Minimum pair distance (nm)",
        output_root,
        figures_dir,
        min_pair_stem,
        args.dpi,
    )
    figure_records.append(
        {
            "stem": min_pair_stem,
            "files": min_pair_files,
            "kind": "minimum_pair_distance_distribution",
            "series": ["md", "fm", "am"],
            "axis_nm": [0.0, min_pair_upper],
        }
    )

    chirality_edges = np.linspace(-1.0, 1.0, CHIRALITY_BINS + 1)
    chirality_suffixes = {
        key: (
            f" (wrong-handed {100.0 * series_summary[key]['chirality']['wrong_handed_fraction']:.3f}%)"
        )
        for key in ("md", "fm", "am")
    }
    chirality_stem = f"{spec.case_id}_{spec.target_temperature_k}k_chirality"
    chirality_files = plot_three_way_filled(
        chiralities,
        chirality_edges,
        f"ADTM Ala2 {spec.target_temperature_k} K - CA chirality",
        "Normalized signed triple product",
        output_root,
        figures_dir,
        chirality_stem,
        args.dpi,
        label_suffixes=chirality_suffixes,
        zero_line=True,
    )
    figure_records.append(
        {
            "stem": chirality_stem,
            "files": chirality_files,
            "kind": "chirality_distribution",
            "series": ["md", "fm", "am"],
            "atom_indices_data_order": {"CA": 7, "N": 6, "C": 8, "CB": 12},
            "axis": [-1.0, 1.0],
        }
    )

    provenance = {
        "run_root": str(spec.run_root.resolve()),
        "sample_sources": source_records,
        "fm_checkpoint": file_record(spec.fm_checkpoint),
        "am_checkpoint": file_record(spec.am_checkpoint),
        "fm_config": file_record(spec.fm_config),
        "am_config_note": (
            "Historical AM directory has no untouched config.yaml; AM settings must be explicitly "
            "marked as reconstructed from the run name and implementation snapshot."
        ),
    }
    legacy_fm_md = file_record(
        spec.legacy_fm_metrics.parent / f"md_samples_beta_{spec.beta_key}.npy"
    )
    legacy_am_md = file_record(
        spec.legacy_am_metrics.parent / f"md_samples_beta_{spec.beta_key}.npy"
    )
    assert legacy_fm_md is not None and legacy_am_md is not None
    legacy = {
        "fm": load_legacy_metric(spec.legacy_fm_metrics, spec.beta_key),
        "am": load_legacy_metric(spec.legacy_am_metrics, spec.beta_key),
        "reference_arrays": {
            "fm_evaluation_md": legacy_fm_md,
            "am_evaluation_md": legacy_am_md,
            "same_sha256": legacy_fm_md["sha256"] == legacy_am_md["sha256"],
            "canonical_fair_10k_md": source_records["md"],
        },
        "scope_note": (
            "Legacy metrics are retained verbatim and are not substituted for the fair fixed-index "
            "10k recomputation."
        ),
    }
    metrics = {
        "case_id": spec.case_id,
        "interpretation": spec.interpretation,
        "source_temperature_k": spec.source_temperature_k,
        "target_temperature_k": spec.target_temperature_k,
        "target_beta_kj_mol_inverse": spec.target_beta,
        "selection": selection_record,
        "metric_protocol": {
            "reference": "one canonical MD array and the exact same 10k index positions for MD/FM/AM",
            "no_content_filtering": True,
            "energy_w2": "1D empirical W2 on all selected 10k OpenMM energies",
            "pair_distance_w2": "1D empirical W2 over all 231 pair distances per selected structure",
            "min_pair_distance_w2": "1D empirical W2 over one minimum pair distance per structure",
            "phi_psi_w2": (
                "legacy-compatible linear sorted W2 on the fixed [-pi,pi] cut; periodic histogram "
                "Rama JS and circular moments are reported alongside it"
            ),
            "rama_js": f"{RAMA_BINS}x{RAMA_BINS} fixed histogram over [-pi,pi]^2",
        },
        "series_summary": series_summary,
        "comparisons_to_same_md": comparisons,
        "fm_minus_am": {
            "meaning": "positive means lower AM discrepancy (AM improvement)",
            "values": delta,
        },
        "legacy_metrics": legacy,
        "provenance": provenance,
    }
    figure_metadata = {
        "case_id": spec.case_id,
        "selection": selection_record,
        "energy": {
            "cache": str(cache_path.resolve()),
            "cache_sha256": sha256_file(cache_path.resolve()),
            "cache_reused": cache_reused,
            "binding": binding,
            "runtime": openmm_runtime,
            "robust_quantiles": list(ENERGY_QUANTILES),
            "axis_kj_mol": [energy_lower, energy_upper],
            "tail_mass": energy_tails,
        },
        "figures": figure_records,
    }
    return metrics, figure_metadata


def process_baseline(
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = args.baseline_root.resolve()
    beta_rows = (
        ("0.40", 300, 0.40090785014242014),
        ("0.24", 500, 0.24054471008545208),
        ("0.17", 700, 0.1718176500610372),
        ("0.13", 900, 0.13363595004747336),
    )
    figures_dir = output_root / "figures" / "fm_baseline_200k"
    metrics: dict[str, Any] = {
        "role": "FM source-temperature sanity baseline only",
        "root": str(root),
        "checkpoint": file_record(root / "fm_params.pkl"),
        "per_temperature": {},
    }
    figure_records: list[dict[str, Any]] = []
    for beta_key, temperature_k, beta in beta_rows:
        fm_path = root / f"fm_beta_sweep/fm_samples_beta_{beta_key}.npy"
        md_path = root / f"fm_beta_sweep/md_samples_beta_{beta_key}.npy"
        md, md_record = validate_sample_array(md_path, 200_000)
        fm, fm_record = validate_sample_array(fm_path, 200_000)
        md_torsions = phi_psi(md)
        fm_torsions = phi_psi(fm)
        counts = {
            "md": histogram2_counts(md_torsions),
            "fm": histogram2_counts(fm_torsions),
        }
        temperature_figures: list[dict[str, Any]] = []
        for key in ("md", "fm"):
            stem = f"fm_baseline_200k_{temperature_k}k_rama_fes_{key}"
            files = plot_fes_single(
                counts[key],
                f"ADTM Ala2 {temperature_k} K - {LABELS[key]} (200k)",
                output_root,
                figures_dir,
                stem,
                args.dpi,
            )
            record = {
                "stem": stem,
                "files": files,
                "kind": "ramachandran_fes",
                "series": [key],
                "shared_scale_kbt": [0.0, FES_MAX_KBT],
                "sample_count": 200_000,
            }
            temperature_figures.append(record)
            figure_records.append(record)
        metrics["per_temperature"][str(temperature_k)] = {
            "beta_kj_mol_inverse": float(beta),
            "selection": "all 200000 rows; no filtering or subsampling",
            "sources": {"md": md_record, "fm": fm_record},
            "rama_js_200k_80bin": js_divergence_counts(counts["fm"], counts["md"]),
            "figures": temperature_figures,
        }
    return metrics, {
        "protocol": {
            "selection": "all 200000 rows for every MD/FM temperature pair",
            "rama_bins": RAMA_BINS,
            "fes_floor": FES_FLOOR,
            "fes_display_kbt": [0.0, FES_MAX_KBT],
        },
        "figures": figure_records,
    }


def main() -> None:
    args = parse_args()
    args.adtm_root = args.adtm_root.resolve()
    if args.output_root is None:
        args.output_root = (
            args.adtm_root / "frozen_results/2026-07-26/ala2_adtm_extended"
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    script_sha256 = sha256_file(script_path)
    specs = build_case_specs(args)
    requested_cases = list(dict.fromkeys(str(value) for value in args.cases))

    comparison_metrics: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "script": {
            "path": str(script_path),
            "sha256": script_sha256,
        },
        "global_protocol": {
            "comparison_rows_per_series": N_COMPARE,
            "same_index_positions_for_md_fm_am": True,
            "no_content_filtering": True,
            "colors": COLORS,
            "rama_bins": RAMA_BINS,
            "marginal_bins": MARGINAL_BINS,
            "energy_bins": ENERGY_BINS,
            "energy_robust_quantiles": list(ENERGY_QUANTILES),
            "fes_floor": FES_FLOOR,
            "fes_display_kbt": [0.0, FES_MAX_KBT],
            "legacy_corrections": {
                "ew2_2k": "fixed1k because historical energy_w2_samples=1000",
                "w2": "pair_distance_w2, not geometric_w2",
            },
        },
        "cases": {},
    }
    figure_metadata: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha256,
        "output_root": str(output_root),
        "colors": COLORS,
        "cases": {},
    }

    for case_id in requested_cases:
        metrics, figures = process_case(specs[case_id], args, output_root, script_sha256)
        comparison_metrics["cases"][case_id] = metrics
        figure_metadata["cases"][case_id] = figures

    if not args.skip_baseline:
        baseline_metrics, baseline_figures = process_baseline(args, output_root)
        comparison_metrics["fm_baseline_200k"] = baseline_metrics
        figure_metadata["fm_baseline_200k"] = baseline_figures

    comparison_path = output_root / "comparison_metrics.json"
    figure_path = output_root / "figure_metadata.json"
    json_dump_atomic(comparison_path, comparison_metrics)
    json_dump_atomic(figure_path, figure_metadata)
    print(comparison_path, flush=True)
    print(figure_path, flush=True)


if __name__ == "__main__":
    main()
