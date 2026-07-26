#!/usr/bin/env python3
"""Strict, reproducible scoring against the validated LJ13 reference-v2 bundle.

This entry point deliberately does not use the generic ``score-samples`` path:
that path drops non-finite rows and evaluates LJ13 with the float32 training
guard.  Here every input row must be healthy, and energies are evaluated with
the independent float64 reference implementation of the exact BMS target.

Example
-------

    python scripts/score_lj13_reference_v2.py \
      --bundle data/lj13_reference_v2 \
      --run-dir outputs/lj13/formal_seed0 \
      --beta 1.0 \
      --kinds fm am \
      --energy-samples 2000 \
      --geometric-samples 2000 \
      --reference-floor-repeats 5 \
      --seed 0 \
      --output outputs/lj13/formal_seed0/score_lj13_reference_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from adj_thermo.metrics import geometric_w2
from adj_thermo.problem import make_problem
from adj_thermo.reference.lj13_rehmc import (
    AMBIENT_DIM,
    N_PARTICLES,
    PAIR_I,
    PAIR_J,
    SPATIAL_DIM,
    lj13_energy_np,
)


SCHEMA = "adtm.lj13_reference_v2.score2k.v1"
EXPECTED_BUNDLE_SCHEMA = "adtm.lj13_reference_v2.bundle.v1"
EXPECTED_BUNDLE_STATUS = "validated_equilibrium_reference"
EXPECTED_TARGET_ID = "lj13_bms_eq234_lj1_confinement1_comfree_v1"
ENERGY_SAMPLE_COUNT = 2_000
GEOMETRIC_SAMPLE_COUNT = 2_000
REFERENCE_FLOOR_REPEATS = 5
MINIMUM_PAIR_DISTANCE = 0.1
REFERENCE_COM_TOLERANCE = 1.0e-9
MODEL_COM_TOLERANCE = 1.0e-5
MODEL_COM_RELATIVE_TOLERANCE = 1.0e-7
GEOMETRIC_CHUNK_SIZE = 16
GEOMETRIC_TOP_K = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ScoreValidationError(ValueError):
    """An input violates the pre-registered LJ13 scoring protocol."""


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_sha256(index: np.ndarray) -> str:
    canonical = np.asarray(index, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreValidationError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScoreValidationError(f"Expected a JSON object in {path}")
    return value


def write_canonical_json(path: str | Path, value: dict[str, Any]) -> Path:
    """Write deterministic, sorted JSON and reject NaN/Infinity."""

    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target


def _normalise_bundle_path(path: str | Path) -> tuple[Path, Path]:
    """Return ``(bundle_root, eval_dir)``.

    ``--reference`` may explicitly name the bundle's ``eval`` directory;
    ``--bundle`` normally names its root.  No repository-default reference is
    ever inferred.
    """

    supplied = _resolve(path)
    if supplied.name == "eval" and (supplied.parent / "manifest.json").is_file():
        root = supplied.parent
        eval_dir = supplied
    elif (supplied / "manifest.json").is_file() and (supplied / "eval").is_dir():
        root = supplied
        eval_dir = supplied / "eval"
    else:
        raise ScoreValidationError(
            f"{supplied} is neither an LJ13 reference-v2 bundle nor its eval directory"
        )
    return root, eval_dir


def _checksummed_paths(root: Path) -> dict[str, str]:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise ScoreValidationError(f"Missing bundle checksum file: {sums_path}")

    records: dict[str, str] = {}
    root_resolved = root.resolve()
    for line_number, line in enumerate(
        sums_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        fields = line.split("  ", maxsplit=1)
        if len(fields) != 2:
            raise ScoreValidationError(
                f"Malformed SHA256SUMS line {line_number}: expected two-space separator"
            )
        expected, relative_text = fields
        if not _SHA256_RE.fullmatch(expected):
            raise ScoreValidationError(
                f"Malformed SHA-256 on SHA256SUMS line {line_number}"
            )
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise ScoreValidationError(
                f"Unsafe path on SHA256SUMS line {line_number}: {relative_text!r}"
            )
        key = relative.as_posix()
        if key in records:
            raise ScoreValidationError(f"Duplicate SHA256SUMS entry: {key}")
        target = root.joinpath(*relative.parts).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError as exc:
            raise ScoreValidationError(
                f"Checksummed path escapes bundle root: {relative_text!r}"
            ) from exc
        if not target.is_file():
            raise ScoreValidationError(f"Missing checksummed bundle file: {key}")
        actual = sha256_file(target)
        if actual != expected:
            raise ScoreValidationError(
                f"SHA-256 mismatch for {key}: expected {expected}, got {actual}"
            )
        records[key] = expected
    if not records:
        raise ScoreValidationError(f"No checksum records in {sums_path}")
    return records


def _require_checksummed(records: dict[str, str], relative: str) -> None:
    if relative not in records:
        raise ScoreValidationError(
            f"Critical bundle file is not covered by SHA256SUMS: {relative}"
        )


def _validate_audit(
    root: Path,
    records: dict[str, str],
    relative: str,
    *,
    require_publishable: bool,
) -> dict[str, Any]:
    _require_checksummed(records, relative)
    audit = _read_json(root / relative)
    if audit.get("status") != "pass":
        raise ScoreValidationError(
            f"Reference audit {relative} has status {audit.get('status')!r}, not 'pass'"
        )
    failures = audit.get("failures")
    if not isinstance(failures, list) or failures:
        raise ScoreValidationError(
            f"Reference audit {relative} has non-empty or malformed failures"
        )
    if require_publishable and audit.get("publishable") is not True:
        raise ScoreValidationError(f"Reference audit {relative} is not publishable")
    return audit


def validate_reference_bundle(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    """Verify checksum coverage, manifest gates, and all formal audits."""

    root, eval_dir = _normalise_bundle_path(path)
    records = _checksummed_paths(root)
    _require_checksummed(records, "manifest.json")
    manifest = _read_json(root / "manifest.json")

    if manifest.get("schema") != EXPECTED_BUNDLE_SCHEMA:
        raise ScoreValidationError(
            f"Unexpected bundle schema: {manifest.get('schema')!r}"
        )
    if manifest.get("status") != EXPECTED_BUNDLE_STATUS:
        raise ScoreValidationError(
            f"Bundle status is {manifest.get('status')!r}, expected "
            f"{EXPECTED_BUNDLE_STATUS!r}"
        )
    if manifest.get("legacy_reference_used") is not False:
        raise ScoreValidationError("Manifest does not explicitly reject legacy reference data")
    if manifest.get("legacy_unadjusted_langevin_used") is not False:
        raise ScoreValidationError("Manifest does not explicitly reject legacy ULA data")
    if manifest.get("cross_split_status") != "pass":
        raise ScoreValidationError("Manifest cross-split audit status is not 'pass'")

    target = manifest.get("target")
    if not isinstance(target, dict):
        raise ScoreValidationError("Manifest target record is missing")
    expected_target_fields = {
        "id": EXPECTED_TARGET_ID,
        "ambient_dimension": AMBIENT_DIM,
        "n_particles": N_PARTICLES,
        "spatial_dim": SPATIAL_DIM,
    }
    for key, expected in expected_target_fields.items():
        if target.get(key) != expected:
            raise ScoreValidationError(
                f"Manifest target.{key} is {target.get(key)!r}, expected {expected!r}"
            )

    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ScoreValidationError("Manifest splits record is missing")
    for split_name in ("train", "eval"):
        split = splits.get(split_name)
        if not isinstance(split, dict):
            raise ScoreValidationError(f"Manifest split {split_name!r} is missing")
        if split.get("audit_status") != "pass":
            raise ScoreValidationError(
                f"Manifest split {split_name!r} audit status is not 'pass'"
            )
        if split.get("publishable") is not True:
            raise ScoreValidationError(
                f"Manifest split {split_name!r} is not publishable"
            )

    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ScoreValidationError("Manifest source_sha256 record is missing")
    for source, digest in source_hashes.items():
        if not isinstance(source, str) or not isinstance(digest, str):
            raise ScoreValidationError("Manifest source_sha256 record is malformed")
        if not _SHA256_RE.fullmatch(digest):
            raise ScoreValidationError(f"Invalid source SHA-256 for {source!r}")

    audit_paths = (
        ("diagnostics/train_audit.json", True),
        ("diagnostics/eval_audit.json", True),
        ("diagnostics/train_vs_eval.json", False),
    )
    audit_reports = {
        relative: _validate_audit(
            root,
            records,
            relative,
            require_publishable=require_publishable,
        )
        for relative, require_publishable in audit_paths
    }

    report = {
        "status": "pass",
        "bundle_root": str(root),
        "scoring_split": "eval",
        "eval_dir": str(eval_dir),
        "manifest_schema": manifest["schema"],
        "manifest_status": manifest["status"],
        "manifest_sha256": records["manifest.json"],
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
        "checksummed_files": len(records),
        "audits": {
            relative: {
                "status": audit["status"],
                **(
                    {"publishable": audit["publishable"]}
                    if "publishable" in audit
                    else {}
                ),
            }
            for relative, audit in audit_reports.items()
        },
        "_manifest": manifest,
        "_checksums": records,
    }
    return root, eval_dir, report


def validate_lj13_sample_health(
    values: np.ndarray,
    label: str,
    *,
    com_tolerance: float = MODEL_COM_TOLERANCE,
    relative_com_tolerance: float = 0.0,
    minimum_pair_distance: float = MINIMUM_PAIR_DISTANCE,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    """Hard-fail on malformed, non-finite, off-support, or colliding rows."""

    array = np.asanyarray(values)
    if array.ndim != 2 or array.shape[1:] != (AMBIENT_DIM,):
        raise ScoreValidationError(
            f"{label} must have shape (N, {AMBIENT_DIM}); got {array.shape}"
        )
    if array.shape[0] <= 0:
        raise ScoreValidationError(f"{label} has no rows")
    if not np.issubdtype(array.dtype, np.number):
        raise ScoreValidationError(f"{label} must be numeric; got dtype {array.dtype}")
    if not np.isfinite(float(com_tolerance)) or float(com_tolerance) < 0.0:
        raise ValueError("com_tolerance must be finite and non-negative")
    if (
        not np.isfinite(float(relative_com_tolerance))
        or float(relative_com_tolerance) < 0.0
    ):
        raise ValueError(
            "relative_com_tolerance must be finite and non-negative"
        )
    if not np.isfinite(float(minimum_pair_distance)) or float(minimum_pair_distance) < 0.0:
        raise ValueError("minimum_pair_distance must be finite and non-negative")

    maximum_com = 0.0
    maximum_coordinate = 0.0
    maximum_relative_com = 0.0
    rows_above_absolute_com_tolerance = 0
    rows_failing_scale_aware_com_tolerance = 0
    global_minimum_pair = float("inf")
    step = max(1, int(chunk_size))
    for start in range(0, int(array.shape[0]), step):
        stop = min(start + step, int(array.shape[0]))
        block = np.asarray(array[start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            bad = np.argwhere(~np.isfinite(block))[0]
            row = start + int(bad[0])
            column = int(bad[1])
            raise ScoreValidationError(
                f"{label} contains a non-finite value at row {row}, column {column}; "
                "rows are never filtered"
            )
        coordinates = block.reshape((-1, N_PARTICLES, SPATIAL_DIM))
        block_com = np.mean(coordinates, axis=1)
        row_com = np.max(np.abs(block_com), axis=1)
        row_coordinate = np.max(np.abs(coordinates), axis=(1, 2))
        row_scale = np.maximum(1.0, row_coordinate)
        row_limit = (
            float(com_tolerance)
            + float(relative_com_tolerance) * row_scale
        )
        maximum_com = max(maximum_com, float(np.max(row_com)))
        maximum_coordinate = max(
            maximum_coordinate, float(np.max(row_coordinate))
        )
        maximum_relative_com = max(
            maximum_relative_com,
            float(np.max(row_com / row_scale)),
        )
        rows_above_absolute_com_tolerance += int(
            np.sum(row_com > float(com_tolerance))
        )
        rows_failing_scale_aware_com_tolerance += int(
            np.sum(row_com > row_limit)
        )
        difference = coordinates[:, PAIR_I] - coordinates[:, PAIR_J]
        distances = np.linalg.norm(difference, axis=-1)
        global_minimum_pair = min(
            global_minimum_pair,
            float(np.min(distances)),
        )

    if rows_failing_scale_aware_com_tolerance > 0:
        raise ScoreValidationError(
            f"{label} COM is unhealthy in "
            f"{rows_failing_scale_aware_com_tolerance} rows: "
            f"max |COM coordinate|={maximum_com:.12g}, "
            f"max scale-normalized COM={maximum_relative_com:.12g}; "
            "row-wise tolerance is absolute + relative * "
            f"max(1, max|coordinate|) = {float(com_tolerance):.12g} + "
            f"{float(relative_com_tolerance):.12g} * scale"
        )
    if global_minimum_pair <= float(minimum_pair_distance):
        raise ScoreValidationError(
            f"{label} has minimum pair distance {global_minimum_pair:.12g}; "
            f"the protocol requires strictly > {float(minimum_pair_distance):.12g}"
        )
    return {
        "status": "pass",
        "rows": int(array.shape[0]),
        "columns": int(array.shape[1]),
        "input_dtype": str(array.dtype),
        "all_finite": True,
        "max_abs_com_coordinate": maximum_com,
        "max_abs_coordinate": maximum_coordinate,
        "max_scale_normalized_com": maximum_relative_com,
        "absolute_com_tolerance": float(com_tolerance),
        "relative_com_tolerance": float(relative_com_tolerance),
        "com_tolerance_rule": (
            "row_max_abs_com <= absolute_com_tolerance + "
            "relative_com_tolerance * max(1, row_max_abs_coordinate)"
        ),
        "rows_above_absolute_com_tolerance": (
            rows_above_absolute_com_tolerance
        ),
        "rows_failing_scale_aware_com_tolerance": (
            rows_failing_scale_aware_com_tolerance
        ),
        "minimum_pair_distance": global_minimum_pair,
        "minimum_pair_distance_exclusive_threshold": float(minimum_pair_distance),
    }


def exact_lj13_energy(values: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
    """Evaluate exact float64 ``L + 2Q`` energy via ``lj13_energy_np``.

    The scoring health gate guarantees every pair distance is above 0.1, so
    the reference implementation's 1e-6 numerical guard is inactive.
    """

    array = np.asanyarray(values)
    if array.ndim != 2 or array.shape[1:] != (AMBIENT_DIM,):
        raise ScoreValidationError(
            f"Exact LJ13 energy expects shape (N, {AMBIENT_DIM}); got {array.shape}"
        )
    energies: list[np.ndarray] = []
    step = max(1, int(chunk_size))
    for start in range(0, int(array.shape[0]), step):
        block = np.asarray(array[start : start + step], dtype=np.float64)
        result = np.asarray(
            lj13_energy_np(block, min_distance=1.0e-6),
            dtype=np.float64,
        ).reshape(-1)
        if not np.isfinite(result).all():
            raise ScoreValidationError(
                f"Exact LJ13 energy is non-finite for rows {start}:{start + result.size}"
            )
        energies.append(result)
    return (
        np.concatenate(energies).astype(np.float64, copy=False)
        if energies
        else np.empty((0,), dtype=np.float64)
    )


def _strict_w2_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Repository-compatible rank W2 without silent finite-value filtering."""

    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size <= 0 or b.size <= 0:
        raise ScoreValidationError("Cannot compute W2 from an empty observable")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ScoreValidationError("Cannot compute W2 from non-finite observables")
    a = np.sort(a)
    b = np.sort(b)
    n = min(a.size, b.size)
    if a.size != n:
        a = a[np.rint(np.linspace(0, a.size - 1, n)).astype(np.int64)]
    if b.size != n:
        b = b[np.rint(np.linspace(0, b.size - 1, n)).astype(np.int64)]
    result = float(np.sqrt(np.mean((a - b) ** 2)))
    if not np.isfinite(result):
        raise ScoreValidationError("W2 calculation produced a non-finite result")
    return result


def _observables(values: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    coordinates = array.reshape((-1, N_PARTICLES, SPATIAL_DIM))
    centered = coordinates - np.mean(coordinates, axis=1, keepdims=True)
    difference = centered[:, PAIR_I] - centered[:, PAIR_J]
    pair_distance = np.linalg.norm(difference, axis=-1)
    return {
        "energy": exact_lj13_energy(array),
        "pair_distance": pair_distance,
        "radius_of_gyration": np.sqrt(
            np.mean(np.sum(centered * centered, axis=-1), axis=1)
        ),
        "minimum_pair_distance": np.min(pair_distance, axis=1),
    }


def _derived_seed(
    base_seed: int,
    beta: float,
    purpose: int,
    repeat: int = 0,
    side: int = 0,
) -> int:
    beta_code = int(round(float(beta) * 1000.0))
    sequence = np.random.SeedSequence(
        [int(base_seed), beta_code, int(purpose), int(repeat), int(side)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _sample_indices(rows: int, count: int, seed: int) -> np.ndarray:
    if int(rows) < int(count):
        raise ScoreValidationError(
            f"Need {int(count)} rows without replacement, but only {int(rows)} exist"
        )
    rng = np.random.default_rng(int(seed))
    return np.asarray(
        rng.choice(int(rows), size=int(count), replace=False),
        dtype=np.int64,
    )


def _disjoint_indices(
    rows: int,
    count: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    total = int(count) * int(repeats)
    if int(rows) < total:
        raise ScoreValidationError(
            f"Need {total} rows for {repeats} globally disjoint subsets of "
            f"{count}, but only {rows} exist"
        )
    selected = _sample_indices(rows, total, seed)
    return selected.reshape((int(repeats), int(count)))


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size <= 0 or not np.isfinite(array).all():
        raise ScoreValidationError(f"Cannot summarize values: {array}")
    return {
        "values": [float(value) for value in array],
        "mean": float(np.mean(array)),
        "sample_std_ddof1": (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        ),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": float(np.median(array)),
        "n": int(array.size),
    }


def _geometric_score(
    left: np.ndarray,
    right: np.ndarray,
    problem: Any,
    *,
    seed: int,
) -> float:
    if left.shape[0] != GEOMETRIC_SAMPLE_COUNT:
        raise ScoreValidationError(
            f"Geometric left subset has {left.shape[0]} rows, expected "
            f"{GEOMETRIC_SAMPLE_COUNT}"
        )
    if right.shape[0] != GEOMETRIC_SAMPLE_COUNT:
        raise ScoreValidationError(
            f"Geometric right subset has {right.shape[0]} rows, expected "
            f"{GEOMETRIC_SAMPLE_COUNT}"
        )
    value = geometric_w2(
        left,
        right,
        problem,
        n_samples=GEOMETRIC_SAMPLE_COUNT,
        seed=int(seed),
        cost_chunk_size=GEOMETRIC_CHUNK_SIZE,
        dem_refine_top_k=GEOMETRIC_TOP_K,
    )
    if value is None or not np.isfinite(float(value)):
        raise ScoreValidationError(f"geometric_w2 returned invalid value {value!r}")
    return float(value)


def _metric_values(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> dict[str, float]:
    return {
        "energy_w2_2k": _strict_w2_1d(left["energy"], right["energy"]),
        "pair_distance_w2_2k": _strict_w2_1d(
            left["pair_distance"],
            right["pair_distance"],
        ),
        "radius_of_gyration_w2_2k": _strict_w2_1d(
            left["radius_of_gyration"],
            right["radius_of_gyration"],
        ),
        "minimum_pair_distance_w2_2k": _strict_w2_1d(
            left["minimum_pair_distance"],
            right["minimum_pair_distance"],
        ),
    }


def _reference_floors(
    train: np.ndarray,
    evaluation: np.ndarray,
    problem: Any,
    *,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    train_2k_seed = _derived_seed(seed, beta, purpose=300, side=0)
    eval_2k_seed = _derived_seed(seed, beta, purpose=300, side=1)
    train_2k_indices = _disjoint_indices(
        train.shape[0],
        ENERGY_SAMPLE_COUNT,
        REFERENCE_FLOOR_REPEATS,
        train_2k_seed,
    )
    eval_2k_indices = _disjoint_indices(
        evaluation.shape[0],
        ENERGY_SAMPLE_COUNT,
        REFERENCE_FLOOR_REPEATS,
        eval_2k_seed,
    )

    rows: list[dict[str, Any]] = []
    for repeat in range(REFERENCE_FLOOR_REPEATS):
        left_index = train_2k_indices[repeat]
        right_index = eval_2k_indices[repeat]
        left = np.asarray(train[left_index], dtype=np.float64)
        right = np.asarray(evaluation[right_index], dtype=np.float64)
        low_dimensional = _metric_values(
            _observables(left),
            _observables(right),
        )
        metric_seed = _derived_seed(seed, beta, purpose=301, repeat=repeat)
        geometric = _geometric_score(
            left,
            right,
            problem,
            seed=metric_seed,
        )
        row = {
            "repeat": repeat,
            **low_dimensional,
            "geometric_w2_2k": geometric,
            "energy_pair_rg_minpair_rows_each": ENERGY_SAMPLE_COUNT,
            "geometric_rows_each": GEOMETRIC_SAMPLE_COUNT,
            "train_2k_indices_sha256": _index_sha256(left_index),
            "eval_2k_indices_sha256": _index_sha256(right_index),
            "same_2k_subset_for_all_metrics": True,
            "geometric_metric_seed": metric_seed,
        }
        rows.append(row)
        print(
            f"[lj13-score] beta={beta:.2f} floor={repeat + 1}/"
            f"{REFERENCE_FLOOR_REPEATS} "
            f"energy={row['energy_w2_2k']:.6g} "
            f"geometry={row['geometric_w2_2k']:.6g}",
            flush=True,
        )

    metric_names = (
        "energy_w2_2k",
        "pair_distance_w2_2k",
        "radius_of_gyration_w2_2k",
        "minimum_pair_distance_w2_2k",
        "geometric_w2_2k",
    )
    return {
        "construction": (
            "independent corrected RE-HMC train split versus eval split; "
            "subsets are globally disjoint across all five repeats within "
            "each split; every metric in a repeat uses the same 2k subset"
        ),
        "repeats": REFERENCE_FLOOR_REPEATS,
        "energy_pair_rg_minpair_rows_each": ENERGY_SAMPLE_COUNT,
        "geometric_rows_each": GEOMETRIC_SAMPLE_COUNT,
        "selection_seeds": {
            "train_2k": train_2k_seed,
            "eval_2k": eval_2k_seed,
        },
        "runs": rows,
        "metrics": {
            name: _summary(float(row[name]) for row in rows)
            for name in metric_names
        },
    }


def _sample_candidates(run_dir: Path, kind: str, beta: float) -> list[Path]:
    tag = f"{float(beta):.2f}"
    return [
        run_dir / f"{kind}_beta_sweep" / f"{kind}_samples_beta_{tag}.npy",
        run_dir / f"{kind}_samples_beta_{tag}.npy",
    ]


def _resolve_sample_path(run_dir: Path, kind: str, beta: float) -> Path:
    existing = [path.resolve() for path in _sample_candidates(run_dir, kind, beta) if path.is_file()]
    if not existing:
        candidates = ", ".join(str(path) for path in _sample_candidates(run_dir, kind, beta))
        raise FileNotFoundError(
            f"Missing {kind.upper()} samples for beta={beta:.2f}; checked {candidates}"
        )
    if len(existing) > 1:
        hashes = {sha256_file(path) for path in existing}
        if len(hashes) != 1:
            raise ScoreValidationError(
                f"Ambiguous non-identical {kind.upper()} samples for beta={beta:.2f}: "
                f"{existing}"
            )
    return existing[0]


def _load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ScoreValidationError(f"Cannot load numeric NPY file {path}: {exc}") from exc


def _available_betas(manifest: dict[str, Any], split: str) -> tuple[float, ...]:
    try:
        values = manifest["splits"][split]["config"]["target_betas"]
    except (KeyError, TypeError) as exc:
        raise ScoreValidationError(
            f"Manifest does not declare target betas for split {split!r}"
        ) from exc
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ScoreValidationError(
            f"Manifest target betas for split {split!r} are malformed"
        ) from exc
    if not result or not np.isfinite(result).all():
        raise ScoreValidationError(
            f"Manifest target betas for split {split!r} are malformed"
        )
    return result


def _score_beta(
    *,
    root: Path,
    eval_dir: Path,
    checksums: dict[str, str],
    manifest: dict[str, Any],
    run_dir: Path,
    beta: float,
    kinds: tuple[str, ...],
    seed: int,
    problem: Any,
) -> dict[str, Any]:
    for split in ("train", "eval"):
        declared = _available_betas(manifest, split)
        if not any(np.isclose(beta, candidate, rtol=0.0, atol=1.0e-12) for candidate in declared):
            raise ScoreValidationError(
                f"beta={beta:.12g} is not declared by manifest split {split!r}: "
                f"{declared}"
            )

    tag = f"{float(beta):.2f}"
    train_relative = f"train/samples_beta_{tag}.npy"
    eval_relative = f"eval/samples_beta_{tag}.npy"
    _require_checksummed(checksums, train_relative)
    _require_checksummed(checksums, eval_relative)
    train_path = root / train_relative
    eval_path = eval_dir / f"samples_beta_{tag}.npy"
    train = _load_npy(train_path)
    evaluation = _load_npy(eval_path)
    train_health = validate_lj13_sample_health(
        train,
        f"reference train beta={beta:.2f}",
        com_tolerance=REFERENCE_COM_TOLERANCE,
    )
    eval_health = validate_lj13_sample_health(
        evaluation,
        f"reference eval beta={beta:.2f}",
        com_tolerance=REFERENCE_COM_TOLERANCE,
    )

    reference_2k_seed = _derived_seed(seed, beta, purpose=100)
    model_2k_seed = _derived_seed(seed, beta, purpose=110)
    geometric_metric_seed = _derived_seed(seed, beta, purpose=120)
    reference_2k_index = _sample_indices(
        evaluation.shape[0],
        ENERGY_SAMPLE_COUNT,
        reference_2k_seed,
    )
    reference_2k = np.asarray(evaluation[reference_2k_index], dtype=np.float64)
    reference_observables = _observables(reference_2k)
    reference_full_energy = exact_lj13_energy(evaluation)

    models: dict[str, Any] = {}
    model_row_count: int | None = None
    shared_model_2k_index: np.ndarray | None = None
    for kind in kinds:
        sample_path = _resolve_sample_path(run_dir, kind, beta)
        samples = _load_npy(sample_path)
        health = validate_lj13_sample_health(
            samples,
            f"{kind.upper()} samples beta={beta:.2f}",
            com_tolerance=MODEL_COM_TOLERANCE,
            relative_com_tolerance=MODEL_COM_RELATIVE_TOLERANCE,
        )
        if model_row_count is None:
            model_row_count = int(samples.shape[0])
            shared_model_2k_index = _sample_indices(
                model_row_count,
                ENERGY_SAMPLE_COUNT,
                model_2k_seed,
            )
        elif int(samples.shape[0]) != model_row_count:
            raise ScoreValidationError(
                "FM and AM must have the same number of rows so the canonical "
                "2k benchmark can use identical model index positions; got "
                f"{model_row_count} and {int(samples.shape[0])}"
            )
        assert shared_model_2k_index is not None
        sample_2k = np.asarray(
            samples[shared_model_2k_index],
            dtype=np.float64,
        )
        sample_observables = _observables(sample_2k)
        metrics = {
            "energy_w2_full_tail_diagnostic": _strict_w2_1d(
                exact_lj13_energy(samples),
                reference_full_energy,
            ),
            **_metric_values(sample_observables, reference_observables),
            "geometric_w2_2k": _geometric_score(
                sample_2k,
                reference_2k,
                problem,
                seed=geometric_metric_seed,
            ),
        }
        models[kind] = {
            "status": "pass",
            "sample_path": str(sample_path),
            "sample_sha256": sha256_file(sample_path),
            "health": health,
            "shared_model_2k_selection_seed": model_2k_seed,
            "model_2k_indices_sha256": _index_sha256(
                shared_model_2k_index
            ),
            "same_2k_subset_for_all_metrics": True,
            "metrics": metrics,
        }
        print(
            f"[lj13-score] beta={beta:.2f} kind={kind} "
            f"energy2k={metrics['energy_w2_2k']:.6g} "
            f"geometry2k={metrics['geometric_w2_2k']:.6g}",
            flush=True,
        )

    floors = _reference_floors(
        train,
        evaluation,
        problem,
        beta=beta,
        seed=seed,
    )
    return {
        "beta": float(beta),
        "reference": {
            "scoring_split": "eval",
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "train_sha256": checksums[train_relative],
            "eval_sha256": checksums[eval_relative],
            "train_health": train_health,
            "eval_health": eval_health,
            "shared_by_all_model_kinds": True,
            "shared_2k_selection_seed": reference_2k_seed,
            "shared_2k_indices_sha256": _index_sha256(reference_2k_index),
            "same_2k_subset_for_all_metrics": True,
            "shared_model_2k_selection_seed": model_2k_seed,
            "shared_geometric_metric_seed": geometric_metric_seed,
        },
        "models": models,
        "reference_floors": floors,
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.energy_samples) != ENERGY_SAMPLE_COUNT:
        raise ScoreValidationError(
            f"Canonical protocol requires --energy-samples {ENERGY_SAMPLE_COUNT}"
        )
    if int(args.geometric_samples) != GEOMETRIC_SAMPLE_COUNT:
        raise ScoreValidationError(
            f"Canonical protocol requires --geometric-samples {GEOMETRIC_SAMPLE_COUNT}"
        )
    if int(args.reference_floor_repeats) != REFERENCE_FLOOR_REPEATS:
        raise ScoreValidationError(
            "Canonical protocol requires --reference-floor-repeats "
            f"{REFERENCE_FLOOR_REPEATS}"
        )
    if int(args.seed) < 0:
        raise ScoreValidationError("--seed must be non-negative")

    betas = tuple(float(beta) for beta in args.beta)
    if not betas or not np.isfinite(betas).all() or any(beta <= 0.0 for beta in betas):
        raise ScoreValidationError("--beta values must be finite and positive")
    if len(set(betas)) != len(betas):
        raise ScoreValidationError("--beta values must be unique")
    kinds = tuple(str(kind).lower() for kind in args.kinds)
    if not kinds or len(set(kinds)) != len(kinds):
        raise ScoreValidationError("--kinds must contain unique fm/am values")

    run_dir = _resolve(args.run_dir)
    if not run_dir.is_dir():
        raise ScoreValidationError(f"--run-dir is not a directory: {run_dir}")
    root, eval_dir, bundle_report = validate_reference_bundle(args.bundle)
    manifest = bundle_report.pop("_manifest")
    checksums = bundle_report.pop("_checksums")
    print(
        f"[lj13-score] validated bundle={root} files={len(checksums)}; "
        "scoring eval split only",
        flush=True,
    )

    problem = make_problem("lj13")
    per_beta = {
        f"{beta:.2f}": _score_beta(
            root=root,
            eval_dir=eval_dir,
            checksums=checksums,
            manifest=manifest,
            run_dir=run_dir,
            beta=beta,
            kinds=kinds,
            seed=int(args.seed),
            problem=problem,
        )
        for beta in betas
    }
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "problem": "lj13",
        "protocol": {
            "energy_definition": (
                "float64 exact L+2Q via "
                "adj_thermo.reference.lj13_rehmc.lj13_energy_np"
            ),
            "energy_w2_full_tail_diagnostic": {
                "enabled": True,
                "role": (
                    "secondary tail diagnostic only; canonical cross-run "
                    "benchmark uses the shared 2k subset"
                ),
            },
            "energy_pair_rg_minpair_rows_each": ENERGY_SAMPLE_COUNT,
            "geometric_rows_each": GEOMETRIC_SAMPLE_COUNT,
            "same_2k_subset_for_all_canonical_metrics": True,
            "same_model_2k_index_positions_across_kinds": True,
            "geometric_metric": (
                "adj_thermo.metrics.geometric_w2 symmetry-aware "
                "DEM-style approximation"
            ),
            "geometric_dem_refine_top_k": GEOMETRIC_TOP_K,
            "geometric_cost_chunk_size": GEOMETRIC_CHUNK_SIZE,
            "reference_floor_repeats": REFERENCE_FLOOR_REPEATS,
            "reference_floor_subsets_disjoint_across_repeats": True,
            "minimum_pair_distance_exclusive_threshold": MINIMUM_PAIR_DISTANCE,
            "reference_com_tolerance": REFERENCE_COM_TOLERANCE,
            "model_com_tolerance": MODEL_COM_TOLERANCE,
            "model_com_relative_tolerance": MODEL_COM_RELATIVE_TOLERANCE,
            "model_com_tolerance_rule": (
                "absolute + relative * max(1, row max absolute coordinate); "
                "this is a float32 scale-aware COM-free support check and "
                "does not filter, recenter, or alter any sample"
            ),
            "nonfinite_policy": "hard_fail_without_filtering",
        },
        "seed": int(args.seed),
        "betas": [float(beta) for beta in betas],
        "kinds": list(kinds),
        "inputs": {
            "explicit_reference_argument": str(_resolve(args.bundle)),
            "bundle_validation": bundle_report,
            "run_dir": str(run_dir),
        },
        "per_beta": per_beta,
    }
    output = write_canonical_json(args.output, result)
    print(f"[lj13-score] wrote canonical JSON: {output}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        "--bundle",
        dest="bundle",
        type=Path,
        required=True,
        help="Explicit reference-v2 bundle root or its eval directory.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, nargs="+", required=True)
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=("fm", "am"),
        default=("fm", "am"),
    )
    parser.add_argument(
        "--energy-samples",
        type=int,
        choices=(ENERGY_SAMPLE_COUNT,),
        default=ENERGY_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--geometric-samples",
        type=int,
        choices=(GEOMETRIC_SAMPLE_COUNT,),
        default=GEOMETRIC_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--reference-floor-repeats",
        type=int,
        choices=(REFERENCE_FLOOR_REPEATS,),
        default=REFERENCE_FLOOR_REPEATS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    score(build_parser().parse_args())


if __name__ == "__main__":
    main()
