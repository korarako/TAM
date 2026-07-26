#!/usr/bin/env python3
"""Generate and validate an auditable LJ13 reference-v2 bundle.

The first run should be a diagnostic T0 pilot, not a publication dataset:

    JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python scripts/generate_lj13_reference_v2.py generate \
      --preset t0 --output data/lj13_reference_v2_t0

After a T0 audit has acceptable HMC/swap/round-trip behavior, generate two
independent formal pools:

    JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python scripts/generate_lj13_reference_v2.py generate \
      --preset formal --output data/lj13_reference_v2
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
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

from adj_thermo.reference.lj13_rehmc import (
    AMBIENT_DIM,
    INTRINSIC_DIM,
    LJ13REHMCConfig,
    N_PARTICLES,
    SPATIAL_DIM,
    audit_pass_fail,
    helmert_basis_np,
    lj13_energy_np,
    run_lj13_rehmc,
)


TARGET_BETAS = (0.8, 1.0, 1.2)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def beta_tag(beta: float) -> str:
    return f"{float(beta):.2f}"


def _w2_1d(a: np.ndarray, b: np.ndarray, n_quantiles: int = 20_000) -> float:
    n = min(int(n_quantiles), int(a.size), int(b.size))
    probability = (np.arange(n, dtype=np.float64) + 0.5) / n
    qa = np.quantile(np.asarray(a, dtype=np.float64), probability)
    qb = np.quantile(np.asarray(b, dtype=np.float64), probability)
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def _observables(samples: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64).reshape((-1, N_PARTICLES, SPATIAL_DIM))
    x = x - np.mean(x, axis=1, keepdims=True)
    pair_i, pair_j = np.triu_indices(N_PARTICLES, k=1)
    distance = np.linalg.norm(x[:, pair_i] - x[:, pair_j], axis=-1)
    return {
        "energy": lj13_energy_np(x),
        "radius_of_gyration": np.sqrt(np.mean(np.sum(x * x, axis=-1), axis=1)),
        "mean_pair_distance": np.mean(distance, axis=1),
        "minimum_pair_distance": np.min(distance, axis=1),
    }


def _cross_split_audit(
    root: Path,
    split_a: str,
    split_b: str,
    betas: tuple[float, ...],
    diagnostics_by_split: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "adtm.lj13_reference_v2.cross_split.v1",
        "split_a": split_a,
        "split_b": split_b,
        "betas": {},
        "status": "pass",
        "failures": [],
    }
    for beta in betas:
        a = np.load(
            root / split_a / f"samples_beta_{beta_tag(beta)}.npy",
            allow_pickle=False,
        )
        b = np.load(
            root / split_b / f"samples_beta_{beta_tag(beta)}.npy",
            allow_pickle=False,
        )
        obs_a = _observables(a)
        obs_b = _observables(b)
        record: dict[str, Any] = {}
        for name in obs_a:
            mean_a = float(np.mean(obs_a[name]))
            mean_b = float(np.mean(obs_b[name]))
            diag_a = diagnostics_by_split[split_a]["targets"][beta_tag(beta)][
                "observables"
            ][name]
            diag_b = diagnostics_by_split[split_b]["targets"][beta_tag(beta)][
                "observables"
            ][name]
            se_a = float(diag_a["std"]) / np.sqrt(
                max(float(diag_a["bulk_ess"]), 1.0)
            )
            se_b = float(diag_b["std"]) / np.sqrt(
                max(float(diag_b["bulk_ess"]), 1.0)
            )
            combined_se = float(np.sqrt(se_a * se_a + se_b * se_b))
            z = abs(mean_a - mean_b) / max(combined_se, 1.0e-15)
            record[name] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "difference": mean_a - mean_b,
                "combined_mcse_approx": combined_se,
                "absolute_z": z,
                "w2_1d": _w2_1d(obs_a[name], obs_b[name]),
            }
            if z > 3.0:
                result["failures"].append(
                    f"beta={beta_tag(beta)} {name}: independent pools differ by {z:.3g} MCSE"
                )
        result["betas"][beta_tag(beta)] = record
    result["status"] = "pass" if not result["failures"] else "fail"
    return result


def _preset_config(args: argparse.Namespace, split_index: int) -> LJ13REHMCConfig:
    if args.preset == "t0":
        base = LJ13REHMCConfig(
            n_replicas=16,
            n_ensembles=4,
            warmup_rounds=2_000,
            production_rounds=5_000,
            storage_stride=1,
            block_rounds=50,
            seed=int(args.seed_base) + 100_003 * split_index,
        )
    elif args.preset == "t1":
        base = LJ13REHMCConfig(
            n_replicas=20,
            n_ensembles=4,
            warmup_rounds=5_000,
            production_rounds=20_000,
            storage_stride=2,
            block_rounds=50,
            seed=int(args.seed_base) + 100_003 * split_index,
        )
    elif args.preset == "formal":
        base = LJ13REHMCConfig(
            n_replicas=24,
            n_ensembles=4,
            warmup_rounds=20_000,
            settle_rounds=2_000,
            production_rounds=100_000,
            storage_stride=4,
            block_rounds=50,
            seed=int(args.seed_base) + 100_003 * split_index,
        )
    else:
        raise ValueError(args.preset)
    overrides: dict[str, Any] = {}
    for name in (
        "n_replicas",
        "n_ensembles",
        "warmup_rounds",
        "settle_rounds",
        "production_rounds",
        "storage_stride",
        "block_rounds",
        "leapfrog_min",
        "leapfrog_max",
        "initial_step_size",
        "maximum_step_size",
        "target_accept",
        "adaptation_rate",
        "beta_min",
        "beta_max",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    overrides["target_betas"] = tuple(float(value) for value in args.target_betas)
    return replace(base, **overrides)


def _write_sha256s(root: Path, paths: list[Path]) -> Path:
    manifest = root / "SHA256SUMS"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(set(paths))
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def generate(args: argparse.Namespace) -> None:
    root = resolve(args.output)
    if root.exists():
        if not args.force:
            raise FileExistsError(f"refusing to overwrite existing bundle: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "diagnostics").mkdir()
    (root / "traces").mkdir()

    splits = tuple(str(value) for value in args.splits)
    if len(set(splits)) != len(splits):
        raise ValueError("split names must be unique")
    if args.preset == "formal" and set(splits) != {"train", "eval"}:
        raise ValueError("formal preset requires exactly --splits train eval")

    generated: list[Path] = []
    diagnostics_by_split: dict[str, dict[str, Any]] = {}
    audits_by_split: dict[str, dict[str, Any]] = {}
    configs_by_split: dict[str, LJ13REHMCConfig] = {}
    for split_index, split in enumerate(splits):
        config = _preset_config(args, split_index)
        configs_by_split[split] = config
        split_dir = root / split
        split_dir.mkdir()
        print(
            f"[lj13-reference-v2] split={split} preset={args.preset} "
            f"seed={config.seed} replicas={config.n_replicas} "
            f"warmup={config.warmup_rounds} production={config.production_rounds}",
            flush=True,
        )
        samples, diagnostics, traces = run_lj13_rehmc(config, progress=True)
        diagnostics_by_split[split] = diagnostics
        pilot = args.preset != "formal"
        audit = audit_pass_fail(diagnostics, pilot=pilot)
        audits_by_split[split] = audit
        for beta, value in samples.items():
            path = split_dir / f"samples_beta_{beta_tag(beta)}.npy"
            chain_path = split_dir / f"chains_beta_{beta_tag(beta)}.npy"
            # Coordinates are stored as centered Cartesian float64.  The
            # intrinsic density and sampler nevertheless live in 36D.
            np.save(path, value.reshape((-1, AMBIENT_DIM)).astype(np.float64))
            np.save(chain_path, value.astype(np.float64))
            generated.extend((path, chain_path))
        diagnostic_path = root / "diagnostics" / f"{split}.json"
        audit_path = root / "diagnostics" / f"{split}_audit.json"
        trace_path = root / "traces" / f"{split}.npz"
        write_json(diagnostic_path, diagnostics)
        write_json(audit_path, audit)
        np.savez_compressed(trace_path, **traces)
        generated.extend((diagnostic_path, audit_path, trace_path))
        print(
            f"[lj13-reference-v2] split={split} audit={audit['status']} "
            f"publishable={audit['publishable']}",
            flush=True,
        )

    cross_split: dict[str, Any] | None = None
    if set(splits) == {"train", "eval"}:
        cross_split = _cross_split_audit(
            root,
            "train",
            "eval",
            tuple(configs_by_split["train"].target_betas),
            diagnostics_by_split,
        )
        cross_path = root / "diagnostics" / "train_vs_eval.json"
        write_json(cross_path, cross_split)
        generated.append(cross_path)

    source_module = (
        PROJECT_ROOT / "src" / "adj_thermo" / "reference" / "lj13_rehmc.py"
    )
    problem_module = PROJECT_ROOT / "src" / "adj_thermo" / "problem" / "lj13.py"
    script_path = Path(__file__).resolve()
    basis = helmert_basis_np()
    basis_bytes = np.asarray(basis, dtype="<f8").tobytes(order="C")
    all_split_pass = all(audit["status"] == "pass" for audit in audits_by_split.values())
    formal = args.preset == "formal"
    cross_pass = cross_split is None or cross_split["status"] == "pass"
    publishable = bool(formal and all_split_pass and cross_pass)
    manifest = {
        "schema": "adtm.lj13_reference_v2.bundle.v1",
        "status": (
            "validated_equilibrium_reference"
            if publishable
            else "diagnostic_only"
        ),
        "preset": args.preset,
        "method": "replica_exchange_hmc_with_metropolis_corrected_trajectories_and_swaps",
        "legacy_reference_used": False,
        "legacy_unadjusted_langevin_used": False,
        "target": {
            "id": "lj13_bms_eq234_lj1_confinement1_comfree_v1",
            "formula": (
                "sum_{i<j}[(1/r_ij)^12 - 2(1/r_ij)^6] "
                "+ sum_i ||x_i-x_COM||^2"
            ),
            "BMS_parameters": {
                "r_m": 1.0,
                "tau": 1.0,
                "epsilon": 1.0,
                "c_osc": 1.0,
            },
            "n_particles": N_PARTICLES,
            "spatial_dim": SPATIAL_DIM,
            "ambient_dimension": AMBIENT_DIM,
            "intrinsic_com_free_dimension": INTRINSIC_DIM,
            "minimum_distance_numerical_guard": 1.0e-6,
            "runtime_training_guard_note": (
                "problem/lj13.py currently uses a 0.1 safety clamp; the formal "
                "reference uses only a 1e-6 numerical guard and audits that no "
                "stored sample approaches either guard"
            ),
        },
        "support": {
            "kind": "com_free",
            "coordinate_transform": "fixed_orthonormal_Helmert",
            "helmert_basis_sha256": hashlib.sha256(basis_bytes).hexdigest(),
            "absolute_jacobian_determinant_in_intrinsic_coordinates": 1.0,
        },
        "splits": {
            split: {
                "config": asdict(configs_by_split[split]),
                "audit_status": audits_by_split[split]["status"],
                "publishable": audits_by_split[split]["publishable"],
            }
            for split in splits
        },
        "cross_split_status": None if cross_split is None else cross_split["status"],
        "source_sha256": {
            "problem/lj13.py": sha256_file(problem_module),
            "reference/lj13_rehmc.py": sha256_file(source_module),
            "generate_lj13_reference_v2.py": sha256_file(script_path),
        },
        "runtime": {
            "jax_version": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    generated.append(manifest_path)
    _write_sha256s(root, generated)
    validation = validate_bundle(root)
    print(json.dumps(validation, indent=2, sort_keys=True), flush=True)
    if validation["status"] != "pass":
        raise RuntimeError("LJ13 reference-v2 bundle validation failed")
    print(root, flush=True)


def _verify_sha256s(root: Path) -> list[str]:
    path = root / "SHA256SUMS"
    if not path.is_file():
        return ["missing SHA256SUMS"]
    errors: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", maxsplit=1)
        target = root / relative
        if not target.is_file():
            errors.append(f"missing checksummed file: {relative}")
        elif sha256_file(target) != expected:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def validate_bundle(path: str | Path) -> dict[str, Any]:
    root = resolve(path)
    errors: list[str] = []
    for required in ("manifest.json", "SHA256SUMS"):
        if not (root / required).is_file():
            errors.append(f"missing {required}")
    if errors:
        return {"status": "fail", "root": str(root), "errors": errors}
    errors.extend(_verify_sha256s(root))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("legacy_reference_used") is not False:
        errors.append("manifest does not reject the legacy reference")
    if manifest.get("legacy_unadjusted_langevin_used") is not False:
        errors.append("manifest does not reject unadjusted Langevin")
    if manifest.get("target", {}).get("intrinsic_com_free_dimension") != INTRINSIC_DIM:
        errors.append("intrinsic dimension is not 36")
    for split, record in manifest.get("splits", {}).items():
        config = record["config"]
        draws = (
            int(config["n_ensembles"])
            * int(config["production_rounds"])
            // int(config["storage_stride"])
        )
        for beta in config["target_betas"]:
            sample_path = root / split / f"samples_beta_{beta_tag(beta)}.npy"
            chain_path = root / split / f"chains_beta_{beta_tag(beta)}.npy"
            if not sample_path.is_file():
                errors.append(f"missing {sample_path.relative_to(root).as_posix()}")
                continue
            if not chain_path.is_file():
                errors.append(f"missing {chain_path.relative_to(root).as_posix()}")
                continue
            values = np.load(sample_path, mmap_mode="r", allow_pickle=False)
            chains = np.load(chain_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (draws, AMBIENT_DIM):
                errors.append(
                    f"shape mismatch {sample_path.name}: {values.shape} != {(draws, AMBIENT_DIM)}"
                )
            if values.dtype != np.float64:
                errors.append(f"dtype mismatch {sample_path.name}: {values.dtype}")
            expected_chain_shape = (
                int(config["n_ensembles"]),
                int(config["production_rounds"]) // int(config["storage_stride"]),
                AMBIENT_DIM,
            )
            if chains.shape != expected_chain_shape:
                errors.append(
                    f"shape mismatch {chain_path.name}: {chains.shape} != "
                    f"{expected_chain_shape}"
                )
            if chains.dtype != np.float64:
                errors.append(f"dtype mismatch {chain_path.name}: {chains.dtype}")
            elif not np.array_equal(
                np.asarray(chains).reshape((-1, AMBIENT_DIM)),
                np.asarray(values),
            ):
                errors.append(f"chain provenance mismatch: {chain_path.name}")
            com_max = 0.0
            minimum_pair = float("inf")
            finite = True
            energy_finite = True
            pair_i, pair_j = np.triu_indices(N_PARTICLES, k=1)
            for start in range(0, values.shape[0], 4096):
                x = np.asarray(values[start : start + 4096], dtype=np.float64).reshape(
                    (-1, N_PARTICLES, SPATIAL_DIM)
                )
                finite = finite and bool(np.isfinite(x).all())
                com_max = max(
                    com_max,
                    float(np.max(np.abs(np.mean(x, axis=1)))),
                )
                distance = np.linalg.norm(x[:, pair_i] - x[:, pair_j], axis=-1)
                minimum_pair = min(minimum_pair, float(np.min(distance)))
                energy_finite = energy_finite and bool(np.isfinite(lj13_energy_np(x)).all())
            if not finite:
                errors.append(f"non-finite values in {sample_path.name}")
            if not energy_finite:
                errors.append(f"non-finite target energy in {sample_path.name}")
            if com_max > 1.0e-9:
                errors.append(f"raw COM residual in {sample_path.name}: {com_max:.6g}")
            if minimum_pair < 0.20:
                errors.append(
                    f"minimum pair distance in {sample_path.name}: {minimum_pair:.6g}"
                )
    return {
        "status": "pass" if not errors else "fail",
        "root": str(root),
        "manifest_status": manifest.get("status"),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
        "errors": errors,
    }


def validate(args: argparse.Namespace) -> None:
    result = validate_bundle(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument(
        "--preset",
        choices=("t0", "t1", "formal"),
        default="t0",
    )
    generate_parser.add_argument("--splits", nargs="+", default=("pilot",))
    generate_parser.add_argument("--seed-base", type=int, default=61001)
    generate_parser.add_argument(
        "--target-betas",
        type=float,
        nargs="+",
        default=TARGET_BETAS,
    )
    generate_parser.add_argument("--n-replicas", type=int)
    generate_parser.add_argument("--n-ensembles", type=int)
    generate_parser.add_argument("--warmup-rounds", type=int)
    generate_parser.add_argument("--settle-rounds", type=int)
    generate_parser.add_argument("--production-rounds", type=int)
    generate_parser.add_argument("--storage-stride", type=int)
    generate_parser.add_argument("--block-rounds", type=int)
    generate_parser.add_argument("--leapfrog-min", type=int)
    generate_parser.add_argument("--leapfrog-max", type=int)
    generate_parser.add_argument("--initial-step-size", type=float)
    generate_parser.add_argument("--maximum-step-size", type=float)
    generate_parser.add_argument("--target-accept", type=float)
    generate_parser.add_argument("--adaptation-rate", type=float)
    generate_parser.add_argument("--beta-min", type=float)
    generate_parser.add_argument("--beta-max", type=float)
    generate_parser.add_argument("--force", action="store_true")
    generate_parser.set_defaults(func=generate)

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.set_defaults(func=validate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
