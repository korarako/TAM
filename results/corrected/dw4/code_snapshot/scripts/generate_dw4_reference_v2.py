#!/usr/bin/env python3
"""Generate and validate an equilibrium DW4 reference bundle.

Examples
--------
Pilot:

    JAX_ENABLE_X64=true python scripts/generate_dw4_reference_v2.py generate \
      --output data/dw4_reference_v2_pilot --n-train 32768 --n-eval 32768

Formal:

    JAX_ENABLE_X64=true python scripts/generate_dw4_reference_v2.py generate \
      --output data/dw4_reference_v2 --n-train 100000 --n-eval 100000
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from adj_thermo.reference.dw4_smc import (
    DW4SMCConfig,
    compare_dw4_samples,
    dw4_energy_np,
    pair_distances_np,
    run_dw4_smc,
    sha256_file,
    summarize_dw4,
    verify_sha256_manifest,
    write_json,
    write_sha256_manifest,
)


DEFAULT_BETAS = (0.8, 1.0, 1.2)
IMPORTANCE_SCALES = np.asarray((0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0), dtype=np.float64)


def beta_tag(beta: float) -> str:
    return f"{float(beta):.2f}"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _mixture_logq(radius_squared: np.ndarray) -> np.ndarray:
    terms = np.stack(
        [
            -3.0 * np.log(2.0 * np.pi * scale * scale)
            - radius_squared / (2.0 * scale * scale)
            - np.log(float(IMPORTANCE_SCALES.size))
            for scale in IMPORTANCE_SCALES
        ],
        axis=1,
    )
    maximum = np.max(terms, axis=1)
    return maximum + np.log(np.sum(np.exp(terms - maximum[:, None]), axis=1))


def independent_importance_audit(
    betas: tuple[float, ...],
    *,
    n_samples: int,
    seed: int,
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    """Independent Gaussian-mixture IS audit in the intrinsic 6D space."""

    rng = np.random.default_rng(int(seed))
    energies: list[np.ndarray] = []
    mean_distances: list[np.ndarray] = []
    logq_values: list[np.ndarray] = []
    completed = 0
    while completed < int(n_samples):
        n = min(int(chunk_size), int(n_samples) - completed)
        component = rng.integers(0, IMPORTANCE_SCALES.size, size=n)
        scale = IMPORTANCE_SCALES[component]
        x = rng.normal(size=(n, 4, 2)) * scale[:, None, None]
        x = x - np.mean(x, axis=1, keepdims=True)
        energy = dw4_energy_np(x)
        distance = pair_distances_np(x)
        radius_squared = np.sum(x * x, axis=(1, 2))
        energies.append(energy)
        mean_distances.append(np.mean(distance, axis=1))
        logq_values.append(_mixture_logq(radius_squared))
        completed += n
        print(f"[dw4-is] samples={completed}/{int(n_samples)}", flush=True)

    energy = np.concatenate(energies)
    mean_distance = np.concatenate(mean_distances)
    logq = np.concatenate(logq_values)
    result: dict[str, Any] = {
        "schema": "adtm.dw4_reference_v2.independent_importance_audit.v1",
        "method": "self_normalized_importance_sampling",
        "proposal": {
            "kind": "equal_weight_isotropic_gaussian_scale_mixture_in_com_free_6d",
            "scales": [float(v) for v in IMPORTANCE_SCALES],
        },
        "n_samples": int(n_samples),
        "seed": int(seed),
        "betas": {},
    }
    for beta in betas:
        log_weight = -float(beta) * energy - logq
        maximum = float(np.max(log_weight))
        weight = np.exp(log_weight - maximum)
        weight /= np.sum(weight)
        ess = float(1.0 / np.sum(weight * weight))
        energy_mean = float(np.sum(weight * energy))
        distance_mean = float(np.sum(weight * mean_distance))
        energy_variance = float(np.sum(weight * (energy - energy_mean) ** 2))
        distance_variance = float(np.sum(weight * (mean_distance - distance_mean) ** 2))
        result["betas"][beta_tag(beta)] = {
            "effective_sample_size": ess,
            "effective_sample_size_fraction": ess / int(n_samples),
            "maximum_normalized_weight": float(np.max(weight)),
            "energy_mean": energy_mean,
            "energy_mean_se_rough": float(np.sqrt(energy_variance / ess)),
            "pair_distance_mean": distance_mean,
            "pair_distance_mean_se_rough": float(np.sqrt(distance_variance / ess)),
        }
    return result


def _load_split(root: Path, split: str, beta: float) -> np.ndarray:
    return np.asarray(
        np.load(root / split / f"samples_beta_{beta_tag(beta)}.npy", allow_pickle=False),
        dtype=np.float32,
    )


def build_audit(
    root: Path,
    betas: tuple[float, ...],
    *,
    importance: dict[str, Any],
    legacy_dir: Path | None,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema": "adtm.dw4_reference_v2.audit.v1",
        "status": "pending",
        "betas": {},
        "thresholds": {
            "energy_split_w2_max": 0.15,
            "pair_distance_split_w2_max": 0.04,
            "energy_importance_mean_abs_delta_max": 0.25,
            "pair_distance_importance_mean_abs_delta_max": 0.05,
            "com_abs_max": 1.0e-5,
        },
        "independent_importance_audit": importance,
    }
    failures: list[str] = []
    for beta in betas:
        tag = beta_tag(beta)
        train = _load_split(root, "train", beta)
        evaluation = _load_split(root, "eval", beta)
        train_summary = summarize_dw4(train)
        eval_summary = summarize_dw4(evaluation)
        split_comparison = compare_dw4_samples(train, evaluation)
        is_beta = importance["betas"][tag]
        record: dict[str, Any] = {
            "train": train_summary,
            "eval": eval_summary,
            "train_vs_eval": split_comparison,
            "importance_crosscheck": {
                "train_energy_mean_abs_delta": abs(
                    float(train_summary["energy_mean"]) - float(is_beta["energy_mean"])
                ),
                "eval_energy_mean_abs_delta": abs(
                    float(eval_summary["energy_mean"]) - float(is_beta["energy_mean"])
                ),
                "train_pair_distance_mean_abs_delta": abs(
                    float(train_summary["pair_distance_mean"]) - float(is_beta["pair_distance_mean"])
                ),
                "eval_pair_distance_mean_abs_delta": abs(
                    float(eval_summary["pair_distance_mean"]) - float(is_beta["pair_distance_mean"])
                ),
            },
        }
        if legacy_dir is not None:
            legacy_path = legacy_dir / f"samples_beta_{tag}.npy"
            if legacy_path.is_file():
                legacy = np.asarray(np.load(legacy_path, allow_pickle=False), dtype=np.float32)
                record["legacy"] = summarize_dw4(legacy)
                record["legacy_vs_new_eval"] = compare_dw4_samples(legacy, evaluation)
            else:
                record["legacy"] = {"status": "unavailable", "path": str(legacy_path)}
        audit["betas"][tag] = record

        thresholds = audit["thresholds"]
        if float(split_comparison["energy_w2"]) > float(thresholds["energy_split_w2_max"]):
            failures.append(f"beta={tag}: train/eval energy W2 too large")
        if float(split_comparison["pair_distance_w2"]) > float(
            thresholds["pair_distance_split_w2_max"]
        ):
            failures.append(f"beta={tag}: train/eval pair-distance W2 too large")
        for split_name, summary in (("train", train_summary), ("eval", eval_summary)):
            if float(summary["com_abs_max"]) > float(thresholds["com_abs_max"]):
                failures.append(f"beta={tag}: {split_name} COM residual too large")
        for key, value in record["importance_crosscheck"].items():
            limit = (
                thresholds["energy_importance_mean_abs_delta_max"]
                if "energy" in key
                else thresholds["pair_distance_importance_mean_abs_delta_max"]
            )
            if float(value) > float(limit):
                failures.append(f"beta={tag}: {key}={float(value):.6g} exceeds {float(limit):.6g}")

    audit["failures"] = failures
    audit["status"] = "pass" if not failures else "fail"
    return audit


def generate(args: argparse.Namespace) -> None:
    root = resolve(args.output)
    if root.exists():
        if not bool(args.force):
            raise FileExistsError(f"Refusing to overwrite existing bundle: {root}")
        shutil.rmtree(root)
    (root / "train").mkdir(parents=True)
    (root / "eval").mkdir(parents=True)
    (root / "diagnostics").mkdir(parents=True)

    betas = tuple(float(v) for v in args.betas)
    config = DW4SMCConfig(
        source_sigma=float(args.source_sigma),
        target_cess=float(args.target_cess),
        mala_steps_per_stage=int(args.mala_steps_per_stage),
        final_mala_steps=int(args.final_mala_steps),
        mala_step_size=float(args.mala_step_size),
        mala_step_decay=float(args.mala_step_decay),
        max_stages=int(args.max_stages),
    )
    generated: list[Path] = []
    for split_index, (split, count, seed_base) in enumerate(
        (
            ("train", int(args.n_train), int(args.train_seed_base)),
            ("eval", int(args.n_eval), int(args.eval_seed_base)),
        )
    ):
        for beta_index, beta in enumerate(betas):
            seed = seed_base + 1009 * beta_index
            print(
                f"[dw4-reference-v2] split={split} beta={beta:.3f} n={count} seed={seed}",
                flush=True,
            )
            samples, diagnostics = run_dw4_smc(
                beta=beta,
                n_particles=count,
                seed=seed,
                config=config,
                progress=True,
            )
            sample_path = root / split / f"samples_beta_{beta_tag(beta)}.npy"
            diagnostic_path = root / "diagnostics" / f"{split}_beta_{beta_tag(beta)}.json"
            np.save(sample_path, samples)
            write_json(diagnostic_path, diagnostics)
            generated.extend((sample_path, diagnostic_path))

    importance = independent_importance_audit(
        betas,
        n_samples=int(args.importance_samples),
        seed=int(args.importance_seed),
    )
    importance_path = root / "importance_audit.json"
    write_json(importance_path, importance)
    generated.append(importance_path)

    legacy_dir = resolve(args.legacy_dir) if args.legacy_dir else None
    audit = build_audit(root, betas, importance=importance, legacy_dir=legacy_dir)
    audit_path = root / "convergence_audit.json"
    write_json(audit_path, audit)
    generated.append(audit_path)

    module_path = PROJECT_ROOT / "src" / "adj_thermo" / "reference" / "dw4_smc.py"
    script_path = Path(__file__).resolve()
    manifest = {
        "schema": "adtm.dw4_reference_v2.bundle.v1",
        "method": "annealed SMC from exact Gaussian source with systematic resampling and MALA",
        "legacy_fixed_step_ula_used": False,
        "energy": {
            "formula": "sum_{i<j}[-4*(d_ij-1)^2 + 0.9*(d_ij-1)^4]",
            "n_particles": 4,
            "spatial_dim": 2,
            "ambient_dimension": 8,
            "intrinsic_com_free_dimension": 6,
        },
        "betas": [float(v) for v in betas],
        "splits": {
            "train": {"n_per_beta": int(args.n_train), "seed_base": int(args.train_seed_base)},
            "eval": {"n_per_beta": int(args.n_eval), "seed_base": int(args.eval_seed_base)},
        },
        "smc_config": asdict(config),
        "importance_audit": {
            "n_samples": int(args.importance_samples),
            "seed": int(args.importance_seed),
        },
        "audit_status": audit["status"],
        "source_sha256": {
            "dw4_smc.py": sha256_file(module_path),
            "generate_dw4_reference_v2.py": sha256_file(script_path),
        },
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    generated.append(manifest_path)
    write_sha256_manifest(root, generated)

    validation = validate_bundle(root)
    print(json.dumps(validation, indent=2, sort_keys=True), flush=True)
    if validation["status"] != "pass":
        raise RuntimeError("DW4 reference-v2 validation failed.")
    print(root, flush=True)


def validate_bundle(root: str | Path) -> dict[str, Any]:
    bundle = resolve(root)
    errors: list[str] = []
    required = ("manifest.json", "convergence_audit.json", "importance_audit.json", "SHA256SUMS")
    for name in required:
        if not (bundle / name).is_file():
            errors.append(f"missing: {name}")
    if errors:
        return {"status": "fail", "root": str(bundle), "errors": errors}

    errors.extend(verify_sha256_manifest(bundle, bundle / "SHA256SUMS"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((bundle / "convergence_audit.json").read_text(encoding="utf-8"))
    if manifest.get("legacy_fixed_step_ula_used") is not False:
        errors.append("manifest does not explicitly reject legacy fixed-step ULA")
    if manifest.get("audit_status") != "pass":
        errors.append(f"manifest audit_status={manifest.get('audit_status')!r}")
    if audit.get("status") != "pass":
        errors.extend(str(value) for value in audit.get("failures", []))

    for split in ("train", "eval"):
        expected = int(manifest["splits"][split]["n_per_beta"])
        for beta in manifest["betas"]:
            path = bundle / split / f"samples_beta_{beta_tag(float(beta))}.npy"
            if not path.is_file():
                errors.append(f"missing: {path.relative_to(bundle).as_posix()}")
                continue
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (expected, 8):
                errors.append(f"shape mismatch: {path.name} {values.shape} != {(expected, 8)}")
            if values.dtype != np.float32:
                errors.append(f"dtype mismatch: {path.name} {values.dtype} != float32")
            if not np.isfinite(np.asarray(values)).all():
                errors.append(f"non-finite rows: {path.name}")

    return {
        "status": "pass" if not errors else "fail",
        "root": str(bundle),
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "sha256sums_sha256": sha256_file(bundle / "SHA256SUMS"),
        "errors": errors,
    }


def validate(args: argparse.Namespace) -> None:
    result = validate_bundle(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output", type=Path, default=Path("data/dw4_reference_v2"))
    generate_parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    generate_parser.add_argument("--n-train", type=int, default=100_000)
    generate_parser.add_argument("--n-eval", type=int, default=100_000)
    generate_parser.add_argument("--train-seed-base", type=int, default=37001)
    generate_parser.add_argument("--eval-seed-base", type=int, default=47001)
    generate_parser.add_argument("--source-sigma", type=float, default=1.0)
    generate_parser.add_argument("--target-cess", type=float, default=0.80)
    generate_parser.add_argument("--mala-steps-per-stage", type=int, default=8)
    generate_parser.add_argument("--final-mala-steps", type=int, default=64)
    generate_parser.add_argument("--mala-step-size", type=float, default=0.020)
    generate_parser.add_argument("--mala-step-decay", type=float, default=0.75)
    generate_parser.add_argument("--max-stages", type=int, default=256)
    generate_parser.add_argument("--importance-samples", type=int, default=2_000_000)
    generate_parser.add_argument("--importance-seed", type=int, default=57001)
    generate_parser.add_argument("--legacy-dir", type=Path, default=Path("data/dw4_paper"))
    generate_parser.add_argument("--force", action="store_true")
    generate_parser.set_defaults(func=generate)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, default=Path("data/dw4_reference_v2"))
    validate_parser.set_defaults(func=validate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
