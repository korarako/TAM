#!/usr/bin/env python3
"""Create standalone publication panels for the corrected DW4 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adj_thermo.reference.dw4_smc import (
    compare_dw4_samples,
    dw4_energy_np,
    pair_distances_np,
    sha256_file,
)


PALETTE = {
    "Equilibrium reference": "#8FD18A",
    "Flow Matching": "#F2A270",
    "Adjoint Matching": "#4776CC",
}

EXPECTED_ENERGY_FORMULA = "sum_{i<j}[-4*(d_ij-1)^2 + 0.9*(d_ij-1)^4]"
EXPECTED_REFERENCE_SCHEMA = "adtm.dw4_reference_v2.bundle.v1"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def validate_reference_bundle(
    reference_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    reference_path = reference_path.resolve()
    if reference_path.name != "samples_beta_1.00.npy" or reference_path.parent.name != "eval":
        raise ValueError(
            "The equilibrium reference must be the held-out eval/samples_beta_1.00.npy split."
        )
    root = reference_path.parents[1]
    manifest_path = root / "manifest.json"
    checksum_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("Reference manifest.json or SHA256SUMS is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_REFERENCE_SCHEMA:
        raise ValueError(f"Unexpected reference schema: {manifest.get('schema')!r}")
    if manifest.get("audit_status") != "pass":
        raise ValueError("Reference bundle has not passed its audit.")
    if manifest.get("legacy_fixed_step_ula_used") is not False:
        raise ValueError("Refusing a legacy fixed-step ULA reference.")
    energy = manifest.get("energy", {})
    if energy.get("formula") != EXPECTED_ENERGY_FORMULA:
        raise ValueError("Reference energy formula does not match corrected DW4.")
    if (
        energy.get("ambient_dimension") != 8
        or energy.get("intrinsic_com_free_dimension") != 6
        or energy.get("n_particles") != 4
        or energy.get("spatial_dim") != 2
    ):
        raise ValueError("Reference dimension metadata does not match DW4.")
    if 1.0 not in [float(value) for value in manifest.get("betas", [])]:
        raise ValueError("Reference manifest does not contain beta=1.00.")

    declared: dict[str, str] = {}
    for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe checksum path: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        actual = sha256_file(candidate)
        if actual != digest:
            raise ValueError(f"Reference checksum mismatch: {relative}")
        declared[relative] = digest
    expected_relative = str(reference_path.relative_to(root)).replace("\\", "/")
    if expected_relative not in declared:
        raise ValueError("Selected reference is not bound by SHA256SUMS.")
    return root, manifest, declared


def validate_samples(
    label: str,
    samples: np.ndarray,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    raw = np.asarray(samples)
    if raw.ndim != 2 or raw.shape != (int(expected_rows), 8):
        raise ValueError(
            f"{label} must have exact shape ({expected_rows}, 8), got {raw.shape}."
        )
    finite_rows = np.all(np.isfinite(raw), axis=1)
    if not np.all(finite_rows):
        raise ValueError(f"{label} contains {int(np.sum(~finite_rows))} non-finite rows.")
    coordinates = np.asarray(raw, dtype=np.float64).reshape((-1, 4, 2))
    com = np.mean(coordinates, axis=1)
    max_abs_com = float(np.max(np.abs(com)))
    if max_abs_com > 1.0e-5:
        raise ValueError(f"{label} is not COM-free: max_abs_com={max_abs_com:.6g}.")
    return {
        "shape": [int(value) for value in raw.shape],
        "dtype": str(raw.dtype),
        "finite_rows": int(np.sum(finite_rows)),
        "nonfinite_rows": int(np.sum(~finite_rows)),
        "max_abs_com": max_abs_com,
    }


def robust_limits(series: dict[str, np.ndarray], low: float, high: float) -> tuple[float, float]:
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


def plot_distribution(
    series: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    output: Path,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
    bins: int = 110,
) -> dict[str, Any]:
    xlim = robust_limits(series, quantile_low, quantile_high)
    edges = np.linspace(xlim[0], xlim[1], int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = {label: full_mass_density(value, edges) for label, value in series.items()}

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.4,
        }
    )
    fig, axis = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    axis.fill_between(
        centers,
        density["Equilibrium reference"],
        step="mid",
        color=PALETTE["Equilibrium reference"],
        alpha=0.76,
        linewidth=0.0,
        label="Equilibrium reference",
    )
    axis.fill_between(
        centers,
        density["Flow Matching"],
        step="mid",
        color=PALETTE["Flow Matching"],
        alpha=0.66,
        linewidth=0.0,
        label="Flow Matching",
    )
    axis.fill_between(
        centers,
        density["Adjoint Matching"],
        step="mid",
        color=PALETTE["Adjoint Matching"],
        alpha=0.16,
        linewidth=0.0,
    )
    axis.step(
        centers,
        density["Adjoint Matching"],
        where="mid",
        color=PALETTE["Adjoint Matching"],
        linewidth=2.3,
        label="Adjoint Matching",
    )
    axis.set_xlim(*xlim)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_title(title)
    axis.legend(frameon=True, framealpha=0.93)
    axis.grid(alpha=0.12, linewidth=0.7)
    for suffix in ("png", "pdf"):
        path = output.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=320 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    tails: dict[str, Any] = {}
    for label, value in series.items():
        finite = np.asarray(value, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        tails[label] = {
            "n": int(finite.size),
            "below": int(np.sum(finite < xlim[0])),
            "above": int(np.sum(finite > xlim[1])),
            "inside_fraction": float(np.mean((finite >= xlim[0]) & (finite <= xlim[1]))),
            "full_min": float(np.min(finite)),
            "full_max": float(np.max(finite)),
        }
    return {
        "title": title,
        "xlabel": xlabel,
        "bins": int(bins),
        "xlim": [float(v) for v in xlim],
        "quantiles": [float(quantile_low), float(quantile_high)],
        "normalization": "counts / (full finite sample count * bin width); clipped tails are not renormalized",
        "tails": tails,
        "png": {
            "path": output.with_suffix(".png").name,
            "sha256": sha256_file(output.with_suffix(".png")),
        },
        "pdf": {
            "path": output.with_suffix(".pdf").name,
            "sha256": sha256_file(output.with_suffix(".pdf")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/dw4_reference_v2/eval/samples_beta_1.00.npy"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing publication_figures directory.",
    )
    args = parser.parse_args()
    run = resolve(args.run_dir)
    reference_path = resolve(args.reference)
    output = resolve(args.output_dir) if args.output_dir else run / "publication_figures"
    expected_outputs = [
        output / f"dw4_beta1p00_{stem}.{suffix}"
        for stem in (
            "energy",
            "pairwise_distance",
            "x1_marginal",
            "x2_marginal",
            "x3_marginal",
            "x4_marginal",
        )
        for suffix in ("png", "pdf")
    ] + [output / "figure_metadata.json"]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing figure artifacts; pass --force."
        )
    output.mkdir(parents=True, exist_ok=True)

    reference_root, reference_manifest, reference_checksums = validate_reference_bundle(
        reference_path
    )
    expected_rows = int(reference_manifest["splits"]["eval"]["n_per_beta"])
    paths = {
        "Equilibrium reference": reference_path,
        "Flow Matching": run / "fm_beta_sweep" / "fm_samples_beta_1.00.npy",
        "Adjoint Matching": run / "am_beta_sweep" / "am_samples_beta_1.00.npy",
    }
    samples = {
        label: np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        for label, path in paths.items()
    }
    validations = {
        label: validate_samples(label, value, expected_rows=expected_rows)
        for label, value in samples.items()
    }
    reference = samples["Equilibrium reference"]
    fm = samples["Flow Matching"]
    am = samples["Adjoint Matching"]
    energies = {label: dw4_energy_np(value) for label, value in samples.items()}
    distances = {label: pair_distances_np(value) for label, value in samples.items()}

    metadata: dict[str, Any] = {
        "schema": "adtm.dw4_reference_v2.publication_figures.v1",
        "target_beta": 1.0,
        "palette": PALETTE,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "reference_bundle": {
            "root": str(reference_root),
            "manifest_sha256": sha256_file(reference_root / "manifest.json"),
            "checksum_manifest_sha256": sha256_file(reference_root / "SHA256SUMS"),
            "manifest": reference_manifest,
            "verified_files": reference_checksums,
        },
        "provenance_bindings": {
            "plot_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "dw4_smc": {
                "path": str(SRC / "adj_thermo" / "reference" / "dw4_smc.py"),
                "sha256": sha256_file(SRC / "adj_thermo" / "reference" / "dw4_smc.py"),
            },
            "run_config": {
                "path": str(run / "config.yaml"),
                "sha256": sha256_file(run / "config.yaml"),
            },
            "exact_run_script": {
                "path": str(run / "exact_run_script.sh"),
                "sha256": sha256_file(run / "exact_run_script.sh"),
            },
            "score_metrics": {
                "path": str(
                    run
                    / "score_samples_metrics_dw4_reference_v2_eval_fm_am_betas_1p00_geo2000_ew20000.json"
                ),
                "sha256": sha256_file(
                    run
                    / "score_samples_metrics_dw4_reference_v2_eval_fm_am_betas_1p00_geo2000_ew20000.json"
                ),
            },
        },
        "sample_bindings": {
            label: {
                "path": str(path),
                "sha256": sha256_file(path),
                "validation": validations[label],
            }
            for label, path in paths.items()
        },
        "observable_definitions": {
            "pairwise_distance": "Six unordered Euclidean inter-particle distances per frame, pooled.",
            "coordinate_marginals": (
                "Flattened ambient columns 0..3: particle-1 x/y followed by particle-2 x/y. "
                "These rotation- and label-dependent panels are auxiliary."
            ),
            "energy": EXPECTED_ENERGY_FORMULA,
        },
        "comparisons": {
            "fm_vs_reference": compare_dw4_samples(fm, reference),
            "am_vs_reference": compare_dw4_samples(am, reference),
        },
        "figures": {},
    }
    metadata["figures"]["energy"] = plot_distribution(
        energies,
        title=r"DW4 at $\beta=1.00$: energy distribution",
        xlabel=r"$U(x)$",
        output=output / "dw4_beta1p00_energy",
        quantile_low=0.0005,
        quantile_high=0.9995,
    )
    metadata["figures"]["pairwise_distance"] = plot_distribution(
        {label: value.reshape(-1) for label, value in distances.items()},
        title=r"DW4 at $\beta=1.00$: pairwise distances",
        xlabel="Pairwise distance",
        output=output / "dw4_beta1p00_pairwise_distance",
    )
    flattened = {label: value.reshape((-1, 8)) for label, value in samples.items()}
    for axis_index in range(4):
        key = f"x{axis_index + 1}_marginal"
        metadata["figures"][key] = plot_distribution(
            {label: value[:, axis_index] for label, value in flattened.items()},
            title=rf"DW4 at $\beta=1.00$: $x_{{{axis_index + 1}}}$ marginal",
            xlabel=rf"$x_{{{axis_index + 1}}}$",
            output=output / f"dw4_beta1p00_x{axis_index + 1}_marginal",
        )
    metadata_path = output / "figure_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata["comparisons"], indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
