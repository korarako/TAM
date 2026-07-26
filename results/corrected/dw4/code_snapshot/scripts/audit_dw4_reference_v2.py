#!/usr/bin/env python3
"""Supplementary exact-identity audit for a DW4 reference-v2 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adj_thermo.reference.dw4_smc import (
    compare_dw4_samples,
    dw4_energy_np,
    pair_distances_np,
    summarize_dw4,
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def virial_values(samples: np.ndarray) -> np.ndarray:
    """Return x dot grad U = sum_ij d_ij * dU/dd_ij."""

    distance = pair_distances_np(samples)
    z = distance - 1.0
    derivative = -8.0 * z + 3.6 * z**3
    return np.sum(distance * derivative, axis=-1)


def virial_summary(samples: np.ndarray, beta: float) -> dict[str, float]:
    values = virial_values(samples)
    scaled = float(beta) * values
    mean = float(np.mean(scaled))
    std = float(np.std(scaled))
    standard_error = std / np.sqrt(values.size)
    error = abs(mean - 6.0)
    return {
        "identity": "beta * E[x dot grad U] = intrinsic_dimension = 6",
        "beta_virial_mean": mean,
        "beta_virial_std": std,
        "naive_standard_error": standard_error,
        "absolute_error_from_6": error,
        "naive_z_from_6": error / standard_error,
    }


def energy_ratio_slope(
    lower: np.ndarray,
    upper: np.ndarray,
    beta_lower: float,
    beta_upper: float,
    *,
    bins: int = 120,
    min_count: int = 80,
) -> dict[str, float | int]:
    """Fit log p_beta2(E)-log p_beta1(E) against E."""

    energy_lower = dw4_energy_np(lower)
    energy_upper = dw4_energy_np(upper)
    pooled = np.concatenate((energy_lower, energy_upper))
    low, high = np.quantile(pooled, (0.002, 0.998))
    edges = np.linspace(float(low), float(high), int(bins) + 1)
    count_lower, _ = np.histogram(energy_lower, bins=edges)
    count_upper, _ = np.histogram(energy_upper, bins=edges)
    mask = (count_lower >= int(min_count)) & (count_upper >= int(min_count))
    centers = 0.5 * (edges[:-1] + edges[1:])
    response = np.log(count_upper[mask] / energy_upper.size) - np.log(
        count_lower[mask] / energy_lower.size
    )
    design = np.column_stack((centers[mask], np.ones(np.sum(mask))))
    slope, intercept = np.linalg.lstsq(design, response, rcond=None)[0]
    fitted = design @ np.asarray((slope, intercept))
    residual = response - fitted
    expected = -(float(beta_upper) - float(beta_lower))
    return {
        "beta_lower": float(beta_lower),
        "beta_upper": float(beta_upper),
        "expected_slope": expected,
        "fitted_slope": float(slope),
        "absolute_slope_error": abs(float(slope) - expected),
        "intercept": float(intercept),
        "n_bins_used": int(np.sum(mask)),
        "residual_rmse": float(np.sqrt(np.mean(residual * residual))),
    }


def load(root: Path, split: str, beta: float) -> np.ndarray:
    return np.asarray(
        np.load(root / split / f"samples_beta_{beta:.2f}.npy", allow_pickle=False),
        dtype=np.float64,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve(args.bundle)
    legacy = resolve(args.legacy_dir) if args.legacy_dir else None
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    betas = tuple(float(value) for value in manifest["betas"])
    report: dict[str, Any] = {
        "schema": "adtm.dw4_reference_v2.exact_identity_audit.v1",
        "bundle": str(root),
        "bundle_manifest_sha256": sha256(root / "manifest.json"),
        "thresholds": {
            "combined_virial_absolute_error_max": float(args.max_virial_error),
            "split_virial_naive_z_max": float(args.max_virial_z),
            "energy_ratio_slope_absolute_error_max": float(args.max_slope_error),
        },
        "betas": {},
        "energy_ratio_slopes": {},
        "failures": [],
    }
    eval_samples: dict[float, np.ndarray] = {}
    for beta in betas:
        train = load(root, "train", beta)
        evaluation = load(root, "eval", beta)
        eval_samples[beta] = evaluation
        record: dict[str, Any] = {
            "train": {
                "sha256": sha256(root / "train" / f"samples_beta_{beta:.2f}.npy"),
                "summary": summarize_dw4(train),
                "virial": virial_summary(train, beta),
            },
            "eval": {
                "sha256": sha256(root / "eval" / f"samples_beta_{beta:.2f}.npy"),
                "summary": summarize_dw4(evaluation),
                "virial": virial_summary(evaluation, beta),
            },
            "train_vs_eval": compare_dw4_samples(train, evaluation),
            "combined_virial": virial_summary(
                np.concatenate((train, evaluation), axis=0),
                beta,
            ),
        }
        if legacy is not None:
            path = legacy / f"samples_beta_{beta:.2f}.npy"
            if path.is_file():
                old = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
                record["legacy"] = {
                    "sha256": sha256(path),
                    "summary": summarize_dw4(old),
                    "virial": virial_summary(old, beta),
                    "legacy_vs_eval": compare_dw4_samples(old, evaluation),
                }
        report["betas"][f"{beta:.2f}"] = record
        for split in ("train", "eval"):
            z_value = float(record[split]["virial"]["naive_z_from_6"])
            if z_value > float(args.max_virial_z):
                report["failures"].append(
                    f"beta={beta:.2f} {split} virial z {z_value:.6g}"
                )
        combined_error = float(record["combined_virial"]["absolute_error_from_6"])
        if combined_error > float(args.max_virial_error):
            report["failures"].append(
                f"beta={beta:.2f} combined virial error {combined_error:.6g}"
            )

    for lower, upper in zip(betas[:-1], betas[1:]):
        value = energy_ratio_slope(
            eval_samples[lower],
            eval_samples[upper],
            lower,
            upper,
        )
        report["energy_ratio_slopes"][f"{lower:.2f}_to_{upper:.2f}"] = value
        if float(value["absolute_slope_error"]) > float(args.max_slope_error):
            report["failures"].append(
                f"beta={lower:.2f}->{upper:.2f} slope error "
                f"{float(value['absolute_slope_error']):.6g}"
            )
    report["status"] = "pass" if not report["failures"] else "fail"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/dw4_reference_v2"))
    parser.add_argument("--legacy-dir", type=Path, default=Path("data/dw4_paper"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-virial-error", type=float, default=0.12)
    parser.add_argument("--max-virial-z", type=float, default=3.0)
    parser.add_argument("--max-slope-error", type=float, default=0.03)
    args = parser.parse_args()
    report = run(args)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(output)
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
