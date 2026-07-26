#!/usr/bin/env python3
"""Freeze the extended ADTM Ala2 result without checkpoints or raw arrays.

This command is intentionally strict about the scientific claim and metric
names.  It freezes:

* one single-seed, mixed/partial 500 K -> 400 K FM-to-AM correction;
* one single-seed 900 K -> 800 K negative/failure diagnostic;
* the 200k FM source-temperature baseline as supporting evidence.

The historical Ala2 metric key ``w2`` is the Wasserstein-2 distance between
pooled pair distances.  It is therefore exported as ``pair_distance_w2`` and
must never be labelled geometric W2.  Likewise the historical key
``ew2_2k`` used 1,000 energy samples in these runs and is exported as
``energy_w2_fixed_1k``.

Checkpoint, sample, loss-array, and dataset payloads are never copied.  Their
SHA-256 digests, source paths, sizes, and array metadata are recorded in
``artifact_bindings.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "adtm.ala2.extended_result_freeze.v1"
FORBIDDEN_SUFFIXES = {
    ".pkl",
    ".npy",
    ".npz",
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
}
CHECKPOINT_SUFFIXES = {".pkl", ".ckpt", ".pt", ".pth", ".safetensors"}
ARRAY_SUFFIXES = {".npy", ".npz"}
EXPECTED_DIM = 66
EXPECTED_PARTICLES = 22
EXPECTED_SPATIAL_DIM = 3


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
    if not path.is_file():
        raise FileNotFoundError(path)
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


def parse_label_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH.")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("LABEL must not be empty.")
    return label, resolve(path)


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def safe_copy(
    *,
    source: Path,
    destination: Path,
    root: Path,
    category: str,
    records: list[dict[str, Any]],
) -> None:
    source = require_file(source, category)
    if source.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Refusing to copy forbidden payload: {source}")
    destination = destination.resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe frozen destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_sha = sha256_file(source)
    frozen_sha = sha256_file(destination)
    if source_sha != frozen_sha:
        raise RuntimeError(f"Copy SHA mismatch: {source}")
    records.append(
        {
            "category": category,
            "source_path": str(source.resolve()),
            "frozen_path": destination.relative_to(root).as_posix(),
            "sha256": source_sha,
            "bytes": int(source.stat().st_size),
        }
    )


def npy_metadata(path: Path) -> dict[str, Any]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"Expected a NumPy array: {path}")
    result = {
        "kind": "npy",
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
    }
    if value.ndim == 2 and value.shape[1] == EXPECTED_DIM:
        result["representation"] = {
            "ambient_dimension": EXPECTED_DIM,
            "n_particles": EXPECTED_PARTICLES,
            "spatial_dim": EXPECTED_SPATIAL_DIM,
        }
    return result


def npz_metadata(path: Path) -> dict[str, Any]:
    members: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            value = archive[name]
            members[name] = {
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
    return {"kind": "npz", "members": members}


def payload_binding(path: Path, role: str) -> dict[str, Any]:
    path = require_file(path, role)
    binding: dict[str, Any] = {
        "role": role,
        "source_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "payload_copied": False,
    }
    suffix = path.suffix.lower()
    if suffix == ".npy":
        binding["array"] = npy_metadata(path)
    elif suffix == ".npz":
        binding["array"] = npz_metadata(path)
    return binding


def discover_payloads(root: Path, role: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower()
            in (CHECKPOINT_SUFFIXES | ARRAY_SUFFIXES)
        ),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        result.append(
            payload_binding(
                path,
                f"{role}:{path.relative_to(root).as_posix()}",
            )
        )
    return result


def discover_metric(path: Path, kind: str) -> Path:
    exact = path / f"{kind}_beta_sweep" / f"{kind}_beta_sweep_metrics.json"
    if exact.is_file():
        return exact
    candidates = sorted(path.rglob(f"{kind}_beta_sweep_metrics.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {kind} metric JSON below {path}, found {candidates}"
        )
    return candidates[0]


def beta_record(path: Path, beta_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document = read_json(path)
    if document.get("problem") != "ala2":
        raise ValueError(f"{path} is not an Ala2 metric artifact.")
    item = document.get("per_beta", {}).get(str(beta_key))
    if not isinstance(item, dict):
        raise ValueError(f"{path} has no beta key {beta_key!r}.")
    if int(document.get("energy_w2_samples", -1)) != 1_000:
        raise ValueError(
            f"{path}: legacy ew2_2k is only registered here when "
            "energy_w2_samples=1000."
        )
    if int(item.get("energy_w2_samples", -1)) != 1_000:
        raise ValueError(f"{path}: per-beta energy_w2_samples is not 1000.")
    return document, item


def finite_float(item: dict[str, Any], key: str, label: str) -> float:
    value = item.get(key)
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError(f"{label}: missing/non-finite metric {key!r}.")
    return float(value)


def canonical_metrics(item: dict[str, Any], label: str) -> dict[str, float]:
    return {
        # Historical ew2_2k: the JSON explicitly says energy_w2_samples=1000.
        "energy_w2_fixed_1k": finite_float(item, "ew2_2k", label),
        "energy_w2_full": finite_float(item, "ew2", label),
        # Historical w2 is defined in metrics_ala2.py as pooled pair-distance W2.
        "pair_distance_w2": finite_float(item, "w2", label),
        "ramachandran_js": finite_float(item, "rama_js", label),
        "phi_w2": finite_float(item, "phi_w2", label),
        "psi_w2": finite_float(item, "psi_w2", label),
        "minimum_pair_distance_w2": finite_float(
            item, "min_pair_distance_w2", label
        ),
        "energy_finite_fraction": finite_float(
            item, "energy_finite_fraction", label
        ),
        "energy_mean": finite_float(item, "energy_mean", label),
        "energy_std": finite_float(item, "energy_std", label),
        "energy_q50": finite_float(item, "energy_finite_q50", label),
        "energy_q90": finite_float(item, "energy_finite_q90", label),
        "energy_q99": finite_float(item, "energy_finite_q99", label),
        "minimum_pair_distance_q01": finite_float(
            item, "min_pair_distance_q01", label
        ),
        "minimum_pair_distance_q05": finite_float(
            item, "min_pair_distance_q05", label
        ),
        "minimum_pair_distance_q50": finite_float(
            item, "min_pair_distance_q50", label
        ),
        "com_norm_mean": finite_float(item, "com_norm_mean", label),
    }


def reference_metrics(item: dict[str, Any], label: str) -> dict[str, float]:
    return {
        "energy_finite_fraction": finite_float(
            item, "md_energy_finite_fraction", label
        ),
        "energy_mean": finite_float(item, "md_energy_mean", label),
        "energy_std": finite_float(item, "md_energy_std", label),
        "energy_q50": finite_float(item, "md_energy_finite_q50", label),
        "energy_q90": finite_float(item, "md_energy_finite_q90", label),
        "energy_q99": finite_float(item, "md_energy_finite_q99", label),
        "minimum_pair_distance_q01": finite_float(
            item, "md_min_pair_distance_q01", label
        ),
        "minimum_pair_distance_q05": finite_float(
            item, "md_min_pair_distance_q05", label
        ),
        "minimum_pair_distance_q50": finite_float(
            item, "md_min_pair_distance_q50", label
        ),
    }


def comparison_case(
    *,
    case_id: str,
    status: str,
    claim: str,
    beta_key: str,
    fm_path: Path,
    am_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    fm_document, fm_item = beta_record(fm_path, beta_key)
    am_document, am_item = beta_record(am_path, beta_key)
    if int(fm_document.get("n_eval", -1)) != int(expected_rows):
        raise ValueError(
            f"{fm_path}: expected n_eval={expected_rows}, "
            f"got {fm_document.get('n_eval')}"
        )
    if int(am_document.get("n_eval", -1)) != int(expected_rows):
        raise ValueError(
            f"{am_path}: expected n_eval={expected_rows}, "
            f"got {am_document.get('n_eval')}"
        )
    fm = canonical_metrics(fm_item, f"{case_id} FM")
    am = canonical_metrics(am_item, f"{case_id} AM")
    comparable = (
        "energy_w2_fixed_1k",
        "energy_w2_full",
        "pair_distance_w2",
        "ramachandran_js",
        "phi_w2",
        "psi_w2",
        "minimum_pair_distance_w2",
    )
    comparison = {
        metric: {
            "fm": fm[metric],
            "am": am[metric],
            "delta_am_minus_fm": am[metric] - fm[metric],
            "am_improves": bool(am[metric] < fm[metric]),
        }
        for metric in comparable
    }
    return {
        "case_id": case_id,
        "status": status,
        "claim": claim,
        "beta_key": str(beta_key),
        "beta": finite_float(fm_item, "beta", f"{case_id} FM"),
        "n_eval_per_method": int(expected_rows),
        "ode": {
            "fm_method": fm_document.get("ode_method"),
            "fm_steps": int(fm_document.get("ode_steps", -1)),
            "am_method": am_document.get("ode_method"),
            "am_steps": int(am_document.get("ode_steps", -1)),
        },
        "metric_sources": {
            "fm": {
                "path": str(fm_path.resolve()),
                "sha256": sha256_file(fm_path),
            },
            "am": {
                "path": str(am_path.resolve()),
                "sha256": sha256_file(am_path),
            },
        },
        "reference_source_fm": fm_item.get("reference_source"),
        "reference_source_am": am_item.get("reference_source"),
        "fm": fm,
        "am": am,
        "reference_from_fm_evaluation": reference_metrics(
            fm_item, f"{case_id} FM reference"
        ),
        "reference_from_am_evaluation": reference_metrics(
            am_item, f"{case_id} AM reference"
        ),
        "comparison": comparison,
        "n_improved_of_7_reported_metrics": int(
            sum(record["am_improves"] for record in comparison.values())
        ),
    }


def copy_run_metadata(
    *,
    root: Path,
    role: str,
    run: Path,
    records: list[dict[str, Any]],
) -> None:
    candidates: set[Path] = set()
    for name in (
        "config.yaml",
        "parent_fm_config.yaml",
        "exact_run_script.sh",
        "status.txt",
        "replicate_binding.json",
        "experiment_binding.json",
    ):
        path = run / name
        if path.is_file():
            candidates.add(path)
    for pattern in (
        "**/*metrics*.json",
        "**/*binding*.json",
        "**/*audit*.json",
        "**/*summary*.json",
    ):
        candidates.update(
            path
            for path in run.glob(pattern)
            if path.is_file() and path.suffix.lower() not in FORBIDDEN_SUFFIXES
        )
    for source in sorted(
        candidates, key=lambda path: path.relative_to(run).as_posix()
    ):
        safe_copy(
            source=source,
            destination=root
            / "raw_metrics_and_configs"
            / role
            / source.relative_to(run),
            root=root,
            category=f"{role}_raw_metric_or_config",
            records=records,
        )


def write_legacy_comparison_csv(path: Path, comparison: dict[str, Any]) -> None:
    fields = (
        "case_id",
        "case_status",
        "metric",
        "fm",
        "am",
        "delta_am_minus_fm",
        "am_improves",
        "n_eval_per_method",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in comparison["cases"]:
            for metric, record in case["comparison"].items():
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "case_status": case["status"],
                        "metric": metric,
                        "fm": record["fm"],
                        "am": record["am"],
                        "delta_am_minus_fm": record["delta_am_minus_fm"],
                        "am_improves": int(record["am_improves"]),
                        "n_eval_per_method": case["n_eval_per_method"],
                    }
                )


def load_fair_fixed10k_metrics(plots: Path) -> dict[str, Any]:
    path = plots / "comparison_metrics.json"
    document = read_json(path)
    cases = document.get("cases")
    if not isinstance(cases, dict) or set(cases) != {"best_mixed", "failure"}:
        raise ValueError(f"{path}: expected best_mixed and failure cases.")
    for case_id, case in cases.items():
        if not isinstance(case, dict):
            raise ValueError(f"{path}: malformed case {case_id!r}.")
        selection = case.get("selection")
        if not isinstance(selection, dict):
            raise ValueError(f"{path}: missing selection for {case_id!r}.")
        if int(selection.get("selected_rows", -1)) != 10_000:
            raise ValueError(f"{path}: {case_id!r} is not a fixed-10k result.")
        if selection.get("same_positions_for_md_fm_am") is not True:
            raise ValueError(f"{path}: {case_id!r} does not use common indices.")
        if selection.get("no_content_filtering") is not True:
            raise ValueError(f"{path}: {case_id!r} filtered its evaluation rows.")
        comparisons = case.get("comparisons_to_same_md")
        if not isinstance(comparisons, dict) or set(comparisons) != {"fm", "am"}:
            raise ValueError(f"{path}: malformed comparisons for {case_id!r}.")
    return document


def write_fair_comparison_csv(path: Path, comparison: dict[str, Any]) -> None:
    fields = (
        "case_id",
        "case_status",
        "metric",
        "fm",
        "am",
        "fm_minus_am",
        "am_improves",
        "n_eval_per_method",
        "same_md_and_indices",
        "content_filtering",
    )
    statuses = {
        "best_mixed": "single_seed_mixed_partial_correction",
        "failure": "single_seed_negative_failure_diagnostic",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id in ("best_mixed", "failure"):
            case = comparison["cases"][case_id]
            fm = case["comparisons_to_same_md"]["fm"]
            am = case["comparisons_to_same_md"]["am"]
            if set(fm) != set(am):
                raise ValueError(f"Fair metric keys differ for {case_id}.")
            for metric in sorted(fm):
                delta = float(fm[metric]) - float(am[metric])
                writer.writerow(
                    {
                        "case_id": case_id,
                        "case_status": statuses[case_id],
                        "metric": metric,
                        "fm": fm[metric],
                        "am": am[metric],
                        "fm_minus_am": delta,
                        "am_improves": int(delta > 0.0),
                        "n_eval_per_method": 10_000,
                        "same_md_and_indices": 1,
                        "content_filtering": 0,
                    }
                )


def write_readme(path: Path) -> None:
    text = """# Extended ADTM Ala2 frozen results

This checkpoint-free bundle preserves two single-seed Ala2 FM-to-AM cases:

1. a best available **mixed/partial correction** at 500 K -> 400 K.  Its
   registered source arrays contain 100,000 configurations per series, while
   the canonical comparison uses the same deterministic 10,000 index
   positions for MD, FM, and AM;
2. a 900 K -> 800 K **negative/failure diagnostic**, evaluated with 10,000
   configurations per method using all rows.

It does not establish full Ala2 success and does not establish three-seed
robustness.  In particular, the best case retains a serious high-energy tail
and worsens the psi marginal.

## Authoritative evaluation

`comparison_metrics.json/csv` is the authoritative **fair fixed-10k**
evaluation.  Both cases use one canonical MD array, exactly the same index
positions for MD/FM/AM, no row filtering, and double-precision OpenMM energy
evaluation.  The energy W2 includes every selected value and is therefore
tail-sensitive.  The robust plotting window affects display only; its omitted
tail mass is recorded explicitly.

Historical native-run metrics are retained separately as
`legacy_native_metrics.json/csv`.  They use the original run-specific
evaluation sizes (100k or 10k, plus a fixed-1k energy diagnostic) and are not
substituted for the fair fixed-10k result.

## Metric corrections

- Historical Ala2 `w2` is pooled **pair-distance W2**.  It is not geometric
  W2, and this bundle never labels it geometric W2.
- Historical `ew2_2k` was computed with `energy_w2_samples=1000`; this bundle
  calls it `energy_w2_fixed_1k`.

## Payload policy

No `.pkl`, `.npy`, `.npz`, `.ckpt`, `.pt`, `.pth`, or `.safetensors` payload
is copied.  `artifact_bindings.json` records their source paths, sizes,
SHA-256 digests, and (for NumPy arrays) shapes and dtypes.

See `RESULTS.md` for the scientific summary, `comparison_metrics.json/csv`
for canonical values, `legacy_native_metrics.json/csv` for historical
diagnostics, `plots/` for standalone figures, and `SHA256SUMS` for integrity.
"""
    path.write_text(text, encoding="utf-8")


def write_results(
    path: Path,
    fair: dict[str, Any],
    legacy: dict[str, Any],
) -> None:
    best = fair["cases"]["best_mixed"]
    failure = fair["cases"]["failure"]

    def fair_lines(case: dict[str, Any]) -> list[str]:
        fm = case["comparisons_to_same_md"]["fm"]
        am = case["comparisons_to_same_md"]["am"]
        result: list[str] = []
        for metric in sorted(fm):
            improves = float(am[metric]) < float(fm[metric])
            result.append(
                f"- {metric}: `{float(fm[metric]):.6g}` -> "
                f"`{float(am[metric]):.6g}` "
                f"({'improves' if improves else 'worsens'})."
            )
        return result

    legacy_by_id = {case["case_id"]: case for case in legacy["cases"]}
    legacy_best = legacy_by_id["best_500K_to_400K_seed2"]
    legacy_failure = legacy_by_id["failure_900K_to_800K_seed0"]

    lines = [
        "# Ala2 result summary",
        "",
        "## Canonical protocol",
        "",
        "All primary values below use the same canonical MD array and exactly",
        "the same fixed 10,000 index positions for MD, FM, and AM.  No values",
        "are filtered.  OpenMM energies are evaluated in double precision.",
        "",
        "## Best available mixed/partial correction: 500 K -> 400 K",
        "",
        "Single seed; 100k registered rows and a fixed fair 10k comparison.",
        "",
        *fair_lines(best),
        "",
        "This case is not a complete success.  Energy, minimum-pair, phi, and",
        "Rama JS improve, while pooled pair-distance and psi worsen.  The",
        "unclipped energy W2 remains dominated by rare high-energy values;",
        "the robust energy figure reports, but does not discard, its tail.",
        "",
        "## Negative/failure diagnostic: 900 K -> 800 K",
        "",
        "Single seed; all 10,000 rows per method.",
        "",
        *fair_lines(failure),
        "",
        "Energy and minimum-pair improve, but pooled pair-distance, phi, and",
        "psi worsen; Rama JS changes only marginally.  This is retained as a",
        "negative/mixed diagnostic, not averaged with the best case.",
        "",
        "## Historical native-run diagnostics",
        "",
        f"- 500 K -> 400 K native rows: `{legacy_best['n_eval_per_method']}`.",
        f"- 900 K -> 800 K native rows: `{legacy_failure['n_eval_per_method']}`.",
        "- Historical `ew2_2k` used 1,000 energy samples and is renamed",
        "  `energy_w2_fixed_1k`.",
        "- Historical Ala2 `w2` is pooled `pair_distance_w2`, not Geo W2.",
        "- Exact historical values are in `legacy_native_metrics.json/csv`.",
        "",
        "## Scope",
        "",
        "- No three-seed claim.",
        "- No claim of universally successful Ala2 temperature transfer.",
        "- No symmetry-aware geometric W2 is reported for Ala2.",
        "- Unclipped energy-tail values and robust-display tail masses must be",
        "  shown together; the plotting window is not metric clipping.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sha256s(root: Path) -> Path:
    output = root / "SHA256SUMS"
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path != output
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    output.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in files
        )
        + "\n",
        encoding="ascii",
    )
    return output


def verify_sha256s(root: Path, manifest: Path) -> None:
    for raw in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = raw.split(maxsplit=1)
        path = (root / relative.lstrip("*")).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Invalid SHA manifest member: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA mismatch for {relative}: {actual} != {expected}")


def verify_no_forbidden_payloads(root: Path) -> None:
    bad = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if bad:
        raise ValueError(f"Frozen bundle contains forbidden payloads: {bad}")


def unique_files(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            require_file(resolved, "source file")
            seen.add(resolved)
            result.append(resolved)
    return result


def build(args: argparse.Namespace, root: Path) -> None:
    plots = require_directory(resolve(args.plots_dir), "extended plot directory")
    best = require_directory(resolve(args.best_run), "best 100k run")
    failure = require_directory(resolve(args.failure_run), "failure 10k run")
    best_fm_source = require_directory(
        resolve(args.best_fm_source), "best-case FM source"
    )
    failure_fm_source = require_directory(
        resolve(args.failure_fm_source), "failure-case FM source"
    )
    baseline = require_directory(
        resolve(args.baseline_200k_run), "200k FM baseline"
    )

    best_fm_metrics = (
        resolve(args.best_fm_metrics)
        if args.best_fm_metrics
        else discover_metric(best, "fm")
    )
    best_am_metrics = discover_metric(best, "am")
    failure_fm_metrics = (
        resolve(args.failure_fm_metrics)
        if args.failure_fm_metrics
        else discover_metric(failure_fm_source, "fm")
    )
    failure_am_metrics = discover_metric(failure, "am")
    legacy_comparison = {
        "schema": "adtm.ala2.extended_comparison_metrics.v1",
        "metric_definitions": {
            "energy_w2_fixed_1k": (
                "Historical ew2_2k evaluated with exactly 1000 energy samples."
            ),
            "energy_w2_full": "W2 over all evaluated per-frame energies.",
            "pair_distance_w2": (
                "Historical Ala2 w2: 1D W2 over all pooled inter-atomic distances."
            ),
            "ramachandran_js": "JS divergence of the 80x80 phi/psi histogram.",
            "phi_w2": "1D W2 of the phi marginal.",
            "psi_w2": "1D W2 of the psi marginal.",
            "minimum_pair_distance_w2": (
                "1D W2 of the per-frame minimum inter-atomic distance."
            ),
        },
        "forbidden_metric_label": "geometric_w2",
        "cases": [
            comparison_case(
                case_id="best_500K_to_400K_seed2",
                status="single_seed_mixed_partial_correction",
                claim=(
                    "AM corrects several FM observables, but psi worsens and "
                    "the high-energy tail remains unresolved."
                ),
                beta_key=str(args.best_beta_key),
                fm_path=best_fm_metrics,
                am_path=best_am_metrics,
                expected_rows=100_000,
            ),
            comparison_case(
                case_id="failure_900K_to_800K_seed0",
                status="single_seed_negative_failure_diagnostic",
                claim=(
                    "Overall unsuccessful refinement; retained as a failure "
                    "case rather than positive evidence."
                ),
                beta_key=str(args.failure_beta_key),
                fm_path=failure_fm_metrics,
                am_path=failure_am_metrics,
                expected_rows=10_000,
            ),
        ],
        "global_claim_limit": (
            "Single-seed evidence only. Do not claim complete Ala2 success or "
            "three-seed robustness."
        ),
    }
    fair_comparison = load_fair_fixed10k_metrics(plots)
    write_json(root / "comparison_metrics.json", fair_comparison)
    write_fair_comparison_csv(root / "comparison_metrics.csv", fair_comparison)
    write_json(root / "legacy_native_metrics.json", legacy_comparison)
    write_legacy_comparison_csv(
        root / "legacy_native_metrics.csv",
        legacy_comparison,
    )
    write_readme(root / "README.md")
    write_results(root / "RESULTS.md", fair_comparison, legacy_comparison)

    copied: list[dict[str, Any]] = []
    roles = {
        "best_100k_run": best,
        "best_fm_source": best_fm_source,
        "failure_10k_run": failure,
        "failure_fm_source": failure_fm_source,
        "baseline_200k": baseline,
    }
    for role, run in roles.items():
        copy_run_metadata(root=root, role=role, run=run, records=copied)

    plot_payload_bindings: list[dict[str, Any]] = []
    for source in sorted(
        (path for path in plots.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(plots).as_posix(),
    ):
        if source.suffix.lower() in FORBIDDEN_SUFFIXES:
            plot_payload_bindings.append(
                payload_binding(
                    source,
                    f"plot_output:{source.relative_to(plots).as_posix()}",
                )
            )
            continue
        if source.suffix.lower() not in {".png", ".pdf", ".json", ".csv", ".md"}:
            continue
        safe_copy(
            source=source,
            destination=root / "plots" / source.relative_to(plots),
            root=root,
            category="extended_plot_or_metadata",
            records=copied,
        )

    datasets = dict(parse_label_path(raw) for raw in args.dataset)
    dataset_bindings = {
        label: payload_binding(path, f"dataset:{label}")
        for label, path in sorted(datasets.items())
    }
    pdb = require_file(resolve(args.pdb), "Ala2 PDB")
    safe_copy(
        source=pdb,
        destination=root / "reference_metadata" / pdb.name,
        root=root,
        category="pdb",
        records=copied,
    )

    plot_script = (
        resolve(args.plot_script)
        if args.plot_script
        else PROJECT_ROOT / "scripts" / "plot_ala2_adtm_extended.py"
    )
    code_files = unique_files(
        [
            Path(__file__).resolve(),
            plot_script,
            *(resolve(path) for path in args.source_file),
        ]
    )
    code_bindings: list[dict[str, Any]] = []
    for source in code_files:
        try:
            relative = source.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = Path("external") / source.name
        safe_copy(
            source=source,
            destination=root / "code_snapshot" / relative,
            root=root,
            category="code_snapshot",
            records=copied,
        )
        code_bindings.append(
            {
                "source_path": str(source),
                "sha256": sha256_file(source),
            }
        )

    role_payloads = {
        role: discover_payloads(run, role) for role, run in roles.items()
    }
    best_fm_checkpoint = best_fm_source / "fm_params.pkl"
    best_run_fm_checkpoint = best / "fm_params.pkl"
    if best_fm_checkpoint.is_file() and best_run_fm_checkpoint.is_file():
        source_sha = sha256_file(best_fm_checkpoint)
        run_sha = sha256_file(best_run_fm_checkpoint)
        if source_sha != run_sha:
            raise ValueError(
                "Best run FM checkpoint does not match its declared FM source: "
                f"{run_sha} != {source_sha}"
            )
    failure_fm_checkpoint = failure_fm_source / "fm_params.pkl"
    failure_run_fm_checkpoint = failure / "fm_params.pkl"
    if failure_fm_checkpoint.is_file() and failure_run_fm_checkpoint.is_file():
        source_sha = sha256_file(failure_fm_checkpoint)
        run_sha = sha256_file(failure_run_fm_checkpoint)
        if source_sha != run_sha:
            raise ValueError(
                "Failure run FM checkpoint does not match its FM source: "
                f"{run_sha} != {source_sha}"
            )

    binding = {
        "schema": SCHEMA,
        "claim_scope": {
            "best_case": "single_seed_mixed_partial_correction",
            "failure_case": "single_seed_negative_failure_diagnostic",
            "complete_ala2_success": False,
            "three_seed_robustness": False,
        },
        "metric_name_corrections": {
            "legacy_w2": "pair_distance_w2",
            "legacy_w2_is_geometric_w2": False,
            "legacy_ew2_2k": "energy_w2_fixed_1k",
            "legacy_energy_w2_samples": 1_000,
        },
        "source_roles": {
            role: str(run.resolve()) for role, run in roles.items()
        },
        "payload_policy": (
            "Checkpoint, sample, loss-array, and dataset payloads are not "
            "copied; SHA/size/shape provenance is retained."
        ),
        "payloads_by_role": role_payloads,
        "plot_payloads_not_copied": plot_payload_bindings,
        "datasets": dataset_bindings,
        "pdb": {
            "source_path": str(pdb.resolve()),
            "sha256": sha256_file(pdb),
            "bytes": int(pdb.stat().st_size),
            "payload_copied": True,
        },
        "code": code_bindings,
        "comparison_metrics": {
            "path": "comparison_metrics.json",
            "sha256": sha256_file(root / "comparison_metrics.json"),
        },
    }
    write_json(root / "artifact_bindings.json", binding)

    write_json(
        root / "FREEZE_MANIFEST.json",
        {
            "schema": SCHEMA,
            "n_copied_source_artifacts": len(copied),
            "copied_source_artifacts": copied,
            "forbidden_payload_suffixes": sorted(FORBIDDEN_SUFFIXES),
            "checkpoint_payloads_copied": 0,
            "array_or_dataset_payloads_copied": 0,
            "required_top_level": [
                "README.md",
                "RESULTS.md",
                "comparison_metrics.json",
                "comparison_metrics.csv",
                "artifact_bindings.json",
                "raw_metrics_and_configs/",
                "code_snapshot/",
                "plots/",
                "SHA256SUMS",
            ],
        },
    )
    verify_no_forbidden_payloads(root)
    sums = write_sha256s(root)
    verify_sha256s(root, sums)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--best-run", type=Path, required=True)
    parser.add_argument("--failure-run", type=Path, required=True)
    parser.add_argument("--best-fm-source", type=Path, required=True)
    parser.add_argument("--failure-fm-source", type=Path, required=True)
    parser.add_argument("--baseline-200k-run", type=Path, required=True)
    parser.add_argument("--best-fm-metrics", type=Path, default=None)
    parser.add_argument("--failure-fm-metrics", type=Path, default=None)
    parser.add_argument("--best-beta-key", default="0.30")
    parser.add_argument("--failure-beta-key", default="0.15")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Reference dataset binding; repeat for every relevant temperature.",
    )
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument(
        "--source-file",
        action="append",
        default=[],
        type=Path,
        help="Relevant source file to preserve; repeatable.",
    )
    parser.add_argument("--plot-script", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = resolve(args.output_dir)
    if destination.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {destination}; pass --force."
        )
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
        shutil.move(str(staging), str(destination))
    verify_no_forbidden_payloads(destination)
    verify_sha256s(destination, destination / "SHA256SUMS")
    print(destination)


if __name__ == "__main__":
    main()
