#!/usr/bin/env python3
"""Freeze LJ13 reference-v2 result artifacts without checkpoint payloads.

The frozen bundle contains configs, exact run scripts, numerical metrics,
standalone figures, reference bindings, checkpoint/sample hashes, relevant
code, and a complete SHA-256 manifest.  It deliberately excludes model
checkpoints and raw ``.npy/.npz`` payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "adtm.lj13_reference_v2.result_freeze.v1"
EXPECTED_REFERENCE_SCHEMA = "adtm.lj13_reference_v2.bundle.v1"
EXPECTED_SCORE_SCHEMA = "adtm.lj13_reference_v2.score2k.v1"
EXPECTED_SCORE_KEYS = (
    "energy_w2_2k",
    "pair_distance_w2_2k",
    "radius_of_gyration_w2_2k",
    "minimum_pair_distance_w2_2k",
    "geometric_w2_2k",
)
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".npy",
    ".npz",
}
CHECKPOINT_NAMES = (
    "fm_params.pkl",
    "am_params.pkl",
)
SAMPLE_PATTERNS = (
    "fm_beta_sweep/fm_samples_beta_*.npy",
    "am_beta_sweep/am_samples_beta_*.npy",
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_seed_run(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SEED=PATH.")
    seed, path = value.split("=", 1)
    return int(seed), resolve(path)


def copy_bound_file(
    *,
    source: Path,
    destination: Path,
    records: list[dict[str, Any]],
    category: str,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Refusing payload in frozen bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_sha = sha256_file(source)
    frozen_sha = sha256_file(destination)
    if source_sha != frozen_sha:
        raise RuntimeError(f"Copy checksum mismatch: {source}")
    records.append(
        {
            "category": category,
            "source_path": str(source.resolve()),
            "frozen_path": destination.as_posix(),
            "sha256": source_sha,
            "bytes": int(destination.stat().st_size),
        }
    )


def safe_metric_files(run: Path) -> list[Path]:
    exact_names = {
        "status.txt",
        "config.yaml",
        "parent_fm_config.yaml",
        "exact_run_script.sh",
        "exact_score_script.py",
        "exact_score_recovery_script.sh",
        "score_recovery_binding.json",
        "status_failed_pre_score_recovery.txt",
        "score_lj13_reference_v2.json",
        "replicate_binding.json",
        "experiment_binding.json",
        "result_SHA256SUMS",
        "fm_params.sha256",
        "am_params.sha256",
    }
    output: set[Path] = set()
    for name in exact_names:
        candidate = run / name
        if candidate.is_file():
            output.add(candidate)
    patterns = (
        "**/*metrics*.json",
        "**/*audit*.json",
        "**/*binding*.json",
        "**/*comparison*.json",
        "**/*summary*.json",
        "**/*metrics*.csv",
        "**/*comparison*.csv",
        "**/*.md",
    )
    for pattern in patterns:
        for candidate in run.glob(pattern):
            if candidate.is_file() and candidate.suffix.lower() not in FORBIDDEN_SUFFIXES:
                output.add(candidate)
    return sorted(output)


def validate_reference(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != EXPECTED_REFERENCE_SCHEMA:
        raise ValueError(f"Unexpected reference schema: {manifest.get('schema')!r}")
    if manifest.get("status") != "validated_equilibrium_reference":
        raise ValueError("Reference is not marked validated_equilibrium_reference.")
    if manifest.get("cross_split_status") != "pass":
        raise ValueError("Reference cross-split audit did not pass.")
    if manifest.get("legacy_reference_used") is not False:
        raise ValueError("Reference used legacy data.")
    if manifest.get("legacy_unadjusted_langevin_used") is not False:
        raise ValueError("Reference used unadjusted Langevin.")
    if not (root / "SHA256SUMS").is_file():
        raise FileNotFoundError(root / "SHA256SUMS")
    return manifest


def validate_score2k(run: Path) -> dict[str, Any]:
    path = run / "score_lj13_reference_v2.json"
    score = read_json(path)
    if score.get("schema") != EXPECTED_SCORE_SCHEMA:
        raise ValueError(
            f"{path} is not a canonical score2k artifact: "
            f"{score.get('schema')!r}"
        )
    if score.get("status") != "pass":
        raise ValueError(f"{path} is not marked pass.")
    protocol = score.get("protocol", {})
    if protocol.get("energy_pair_rg_minpair_rows_each") != 2_000:
        raise ValueError(f"{path} does not use 2k low-dimensional metrics.")
    if protocol.get("geometric_rows_each") != 2_000:
        raise ValueError(f"{path} does not use 2k geometric metrics.")
    beta = score.get("per_beta", {}).get("1.00", {})
    models = beta.get("models", {})
    for kind in ("fm", "am"):
        metrics = models.get(kind, {}).get("metrics", {})
        missing = [key for key in EXPECTED_SCORE_KEYS if key not in metrics]
        if missing:
            raise ValueError(f"{path} {kind} is missing score2k keys: {missing}")
    return score


def run_payload_bindings(run: Path) -> dict[str, Any]:
    checkpoints: dict[str, Any] = {}
    for name in CHECKPOINT_NAMES:
        path = run / name
        if path.is_file():
            checkpoints[name] = {
                "source_path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
                "payload_copied": False,
            }
    samples: dict[str, Any] = {}
    for pattern in SAMPLE_PATTERNS:
        for path in sorted(run.glob(pattern)):
            key = path.relative_to(run).as_posix()
            samples[key] = {
                "source_path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
                "payload_copied": False,
            }
    return {"checkpoints": checkpoints, "samples": samples}


def write_sha256s(root: Path) -> Path:
    output = root / "SHA256SUMS"
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path != output
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in paths
    ]
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output


def verify_no_payloads(root: Path) -> None:
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise ValueError(
            "Frozen bundle contains forbidden payloads: "
            + ", ".join(str(path.relative_to(root)) for path in forbidden)
        )


def verify_sha256s(root: Path, path: Path) -> None:
    for raw in path.read_text(encoding="ascii").splitlines():
        expected, relative = raw.split(maxsplit=1)
        candidate = root / relative.lstrip("*")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(f"SHA mismatch for {relative}: {actual} != {expected}")


def unique_existing(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and resolved.is_file():
            result.append(resolved)
            seen.add(resolved)
    return result


def build(args: argparse.Namespace, output: Path) -> None:
    reference = resolve(args.reference_bundle)
    primary = resolve(args.primary_run)
    plots = (
        resolve(args.plots_dir)
        if args.plots_dir
        else primary / "publication_figures_lj13_reference_v2"
    )
    if not primary.is_dir():
        raise FileNotFoundError(primary)
    if not plots.is_dir():
        raise FileNotFoundError(plots)
    reference_manifest = validate_reference(reference)
    run_by_seed = {int(args.primary_am_seed): primary}
    for raw in args.am_run:
        seed, run = parse_seed_run(raw)
        if seed in run_by_seed and run.resolve() != run_by_seed[seed].resolve():
            raise ValueError(f"AM seed {seed} is bound to two runs.")
        run_by_seed[seed] = run
    score_by_seed: dict[int, dict[str, Any]] = {}
    for seed, run in run_by_seed.items():
        if not run.is_dir():
            raise FileNotFoundError(run)
        score_by_seed[seed] = validate_score2k(run)

    records: list[dict[str, Any]] = []
    reference_files = [
        reference / "manifest.json",
        reference / "SHA256SUMS",
        *sorted((reference / "diagnostics").glob("*.json")),
    ]
    for source in reference_files:
        copy_bound_file(
            source=source,
            destination=output / "reference_binding" / source.name,
            records=records,
            category="reference_binding",
        )

    run_bindings: dict[str, Any] = {}
    for seed, run in sorted(run_by_seed.items()):
        run_name = f"am_seed{seed}"
        for source in safe_metric_files(run):
            relative = source.relative_to(run)
            copy_bound_file(
                source=source,
                destination=output / "runs" / run_name / relative,
                records=records,
                category="run_config_script_or_metric",
            )
        run_bindings[str(seed)] = {
            "source_path": str(run.resolve()),
            **run_payload_bindings(run),
        }

    for source in sorted(plots.rglob("*")):
        if source.is_file():
            if source.suffix.lower() not in {".png", ".pdf", ".json", ".csv", ".md"}:
                continue
            copy_bound_file(
                source=source,
                destination=output / "plots" / source.relative_to(plots),
                records=records,
                category="plot_or_plot_metadata",
            )

    default_code = [
        PROJECT_ROOT / "scripts" / "plot_lj13_reference_v2.py",
        PROJECT_ROOT / "scripts" / "freeze_lj13_reference_v2_results.py",
        PROJECT_ROOT / "scripts" / "score_lj13_reference_v2.py",
        PROJECT_ROOT / "scripts" / "generate_lj13_reference_v2.py",
        PROJECT_ROOT / "scripts" / "run_lj13_reference_v2_seed2.sh",
        PROJECT_ROOT / "scripts" / "run_lj13_reference_v2_am_replicate.sh",
        PROJECT_ROOT / "scripts" / "recover_lj13_reference_v2_primary_score.sh",
        PROJECT_ROOT / "scripts" / "run_lj13_reference_v2_full_closure.sh",
        PROJECT_ROOT / "scripts" / "queue_lj13_score_recovery_and_closure.sh",
        PROJECT_ROOT / "scripts" / "aggregate_lj13_reference_v2_am_seeds.py",
        PROJECT_ROOT / "src" / "adj_thermo" / "reference" / "lj13_rehmc.py",
        PROJECT_ROOT / "src" / "adj_thermo" / "problem" / "lj13.py",
        PROJECT_ROOT / "requirements.txt",
    ]
    code = unique_existing(
        [*default_code, *(resolve(path) for path in args.include_code)]
    )
    for source in code:
        try:
            relative = source.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = Path("external") / source.name
        copy_bound_file(
            source=source,
            destination=output / "code_snapshot" / relative,
            records=records,
            category="code_snapshot",
        )

    reference_samples: dict[str, Any] = {}
    for split in ("train", "eval"):
        for path in sorted((reference / split).glob("samples_beta_*.npy")):
            key = path.relative_to(reference).as_posix()
            reference_samples[key] = {
                "source_path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
                "payload_copied": False,
            }
    binding = {
        "schema": SCHEMA,
        "reference": {
            "source_path": str(reference.resolve()),
            "manifest_sha256": sha256_file(reference / "manifest.json"),
            "sha256s_sha256": sha256_file(reference / "SHA256SUMS"),
            "status": reference_manifest["status"],
            "target": reference_manifest["target"],
            "sample_payloads": reference_samples,
        },
        "primary_am_seed": int(args.primary_am_seed),
        "benchmark_protocol": {
            "score_schema": EXPECTED_SCORE_SCHEMA,
            "rows_per_metric": 2_000,
            "metric_keys": list(EXPECTED_SCORE_KEYS),
            "historical_comparability": (
                "All frozen LJ13 benchmark metrics use the historical 2k protocol."
            ),
        },
        "runs": run_bindings,
        "plots_source": str(plots.resolve()),
        "checkpoint_and_sample_policy": (
            "SHA-256, source path, and byte count are frozen; checkpoint and raw "
            "sample payloads are deliberately not copied."
        ),
    }
    write_json(output / "artifact_bindings.json", binding)

    readme = """# LJ13 reference-v2 frozen ADTM results

This is a result/provenance bundle, not a checkpoint archive.  It contains
validated-reference metadata, exact configs and run scripts, numerical
metrics, standalone figures, relevant code, and SHA-256 bindings.

Model checkpoints and raw sample/reference arrays are intentionally omitted.
Their source paths, byte counts, and SHA-256 digests are recorded in
`artifact_bindings.json`.

The plots may use robust display windows, but the numerical metrics do not
discard tail samples.  Exact display limits and omitted tail mass are stored
in `plots/figure_metadata.json`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    write_json(
        output / "FREEZE_MANIFEST.json",
        {
            "schema": SCHEMA,
            "files_before_sha_manifest": records,
            "n_bound_source_files": len(records),
            "checkpoint_payloads_copied": 0,
            "raw_array_payloads_copied": 0,
        },
    )
    verify_no_payloads(output)
    sums = write_sha256s(output)
    verify_sha256s(output, sums)


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
    )
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-code",
        action="append",
        default=[],
        type=Path,
        help="Additional source file to preserve (repeatable).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = resolve(args.output_dir)
    if destination.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --force.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        build(args, staging)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(staging, destination)
    verify_no_payloads(destination)
    verify_sha256s(destination, destination / "SHA256SUMS")
    print(destination)


if __name__ == "__main__":
    main()
