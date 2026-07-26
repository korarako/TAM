#!/usr/bin/env python3
"""Aggregate the LJ13 corrected-reference AM-seed robustness experiment.

This script deliberately treats the three AM runs as *conditional* replicates:
all three must use one bit-identical FM seed-0 proposal and the same validated
reference-v2 bundle.  It therefore does not make an end-to-end three-seed FM
robustness claim.

Example
-------
python scripts/aggregate_lj13_reference_v2_am_seeds.py \
  --metric 1=/path/seed1/score_lj13_reference_v2.json \
  --metric 2=/path/seed2/score_lj13_reference_v2.json \
  --metric 3=/path/seed3/score_lj13_reference_v2.json \
  --output-dir /path/to/aggregate
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "adtm.lj13_reference_v2.am_seed_aggregate.v1"
SCORE_SCHEMA = "adtm.lj13_reference_v2.score2k.v1"
EXPECTED_SEEDS = (1, 2, 3)
EXPECTED_BETA_KEY = "1.00"
EXPECTED_REFERENCE_MANIFEST_SHA = (
    "5487261ba70d2e4c3c10189432dfa31a67c6f32ba3575fe3ec1c388a7039fdb1"
)
METRICS = (
    "energy_w2_2k",
    "pair_distance_w2_2k",
    "radius_of_gyration_w2_2k",
    "minimum_pair_distance_w2_2k",
    "geometric_w2_2k",
)
PRIMARY_METRICS = ("energy_w2_2k", "geometric_w2_2k")


class AggregateValidationError(RuntimeError):
    """Raised when a supposedly controlled replicate is not comparable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregateValidationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregateValidationError(f"Expected a JSON object in {path}")
    return value


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AggregateValidationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AggregateValidationError(f"{label} is not finite: {result!r}")
    if result < 0.0:
        raise AggregateValidationError(f"{label} must be non-negative: {result!r}")
    return result


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AggregateValidationError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise AggregateValidationError(
            f"{label} mismatch: expected {expected:.17g}, got {actual:.17g}"
        )


def _parse_metric_argument(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("--metric must be formatted SEED=/path/file.json")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed in --metric {value!r}") from exc
    path = Path(path_text).expanduser().resolve()
    return seed, path


def _summary(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "n": len(rows),
        "values": rows,
        "mean": statistics.fmean(rows),
        "sample_std_ddof1": statistics.stdev(rows) if len(rows) > 1 else 0.0,
        "median": statistics.median(rows),
        "min": min(rows),
        "max": max(rows),
    }


def _validated_floor(
    payload: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    try:
        floor = payload["per_beta"][EXPECTED_BETA_KEY]["reference_floors"]
        metrics = floor["metrics"]
    except (KeyError, TypeError) as exc:
        raise AggregateValidationError(
            f"seed{seed}: malformed reference_floors structure"
        ) from exc
    _require_equal(floor.get("repeats"), 5, f"seed{seed} floor repeats")
    _require_equal(
        floor.get("energy_pair_rg_minpair_rows_each"),
        2_000,
        f"seed{seed} floor low-dimensional rows",
    )
    _require_equal(
        floor.get("geometric_rows_each"),
        2_000,
        f"seed{seed} floor geometric rows",
    )
    normalized: dict[str, dict[str, float]] = {}
    for name in METRICS:
        try:
            record = metrics[name]
        except (KeyError, TypeError) as exc:
            raise AggregateValidationError(
                f"seed{seed}: missing reference floor metric {name}"
            ) from exc
        if not isinstance(record, dict):
            raise AggregateValidationError(
                f"seed{seed}: floor metric {name} must be an object"
            )
        values = [
            _finite_float(value, f"seed{seed} floor {name} value")
            for value in record.get("values", [])
        ]
        _require_equal(len(values), 5, f"seed{seed} floor {name} value count")
        recomputed = _summary(values)
        normalized[name] = {
            key: float(recomputed[key])
            for key in ("mean", "sample_std_ddof1", "median", "min", "max")
        }
        normalized[name]["n"] = float(recomputed["n"])
        for key in ("mean", "sample_std_ddof1", "median", "min", "max"):
            _require_close(
                _finite_float(record.get(key), f"seed{seed} floor {name} {key}"),
                float(recomputed[key]),
                f"seed{seed} floor {name} {key}",
            )
        _require_equal(record.get("n"), 5, f"seed{seed} floor {name} n")
        normalized[name]["values"] = values  # type: ignore[assignment]
    return floor, normalized


def _validate_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    path: Path,
) -> dict[str, Any]:
    _require_equal(payload.get("schema"), SCORE_SCHEMA, f"seed{seed} score schema")
    _require_equal(payload.get("status"), "pass", f"seed{seed} score status")
    _require_equal(payload.get("problem"), "lj13", f"seed{seed} problem")
    _require_equal(payload.get("seed"), 0, f"seed{seed} scoring seed")
    _require_equal(payload.get("betas"), [1.0], f"seed{seed} scored betas")
    kinds = payload.get("kinds")
    if set(kinds or []) != {"fm", "am"}:
        raise AggregateValidationError(
            f"seed{seed}: strict score must contain both FM and AM; got {kinds!r}"
        )

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise AggregateValidationError(f"seed{seed}: protocol is missing")
    _require_equal(
        protocol.get("energy_pair_rg_minpair_rows_each"),
        2_000,
        f"seed{seed} low-dimensional metric rows",
    )
    _require_equal(
        protocol.get("geometric_rows_each"),
        2_000,
        f"seed{seed} geometric metric rows",
    )
    _require_equal(
        protocol.get("reference_floor_repeats"),
        5,
        f"seed{seed} floor repeats",
    )
    _require_equal(
        protocol.get("nonfinite_policy"),
        "hard_fail_without_filtering",
        f"seed{seed} nonfinite policy",
    )

    try:
        beta = payload["per_beta"][EXPECTED_BETA_KEY]
        reference = beta["reference"]
        models = beta["models"]
        fm = models["fm"]
        am = models["am"]
        manifest_sha = payload["inputs"]["bundle_validation"]["manifest_sha256"]
    except (KeyError, TypeError) as exc:
        raise AggregateValidationError(
            f"seed{seed}: canonical score structure is incomplete"
        ) from exc
    _require_equal(manifest_sha, EXPECTED_REFERENCE_MANIFEST_SHA, "reference manifest")

    for kind, model in (("fm", fm), ("am", am)):
        _require_equal(model.get("status"), "pass", f"seed{seed} {kind} status")
        health = model.get("health")
        if not isinstance(health, dict):
            raise AggregateValidationError(f"seed{seed}: {kind} health is missing")
        _require_equal(health.get("status"), "pass", f"seed{seed} {kind} health")
        _require_equal(health.get("rows"), 100_000, f"seed{seed} {kind} rows")
        _require_equal(health.get("columns"), 39, f"seed{seed} {kind} columns")
        _require_equal(
            health.get("all_finite"), True, f"seed{seed} {kind} all_finite"
        )
        _require_equal(
            model.get("same_2k_subset_for_all_metrics"),
            True,
            f"seed{seed} {kind} same 2k subset for all metrics",
        )

    _require_equal(
        reference.get("same_2k_subset_for_all_metrics"),
        True,
        f"seed{seed} reference same 2k subset for all metrics",
    )
    reference_model_seed = reference.get("shared_model_2k_selection_seed")
    fm_model_seed = fm.get("shared_model_2k_selection_seed")
    am_model_seed = am.get("shared_model_2k_selection_seed")
    _require_equal(
        fm_model_seed,
        reference_model_seed,
        f"seed{seed} FM shared model 2k selection seed",
    )
    _require_equal(
        am_model_seed,
        reference_model_seed,
        f"seed{seed} AM shared model 2k selection seed",
    )
    _require_equal(
        am.get("model_2k_indices_sha256"),
        fm.get("model_2k_indices_sha256"),
        f"seed{seed} FM/AM model 2k index positions",
    )

    metric_values: dict[str, dict[str, float]] = {"fm": {}, "am": {}}
    for kind, model in (("fm", fm), ("am", am)):
        raw_metrics = model.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise AggregateValidationError(
                f"seed{seed}: {kind} metrics object is missing"
            )
        for name in METRICS:
            metric_values[kind][name] = _finite_float(
                raw_metrics.get(name), f"seed{seed} {kind} {name}"
            )

    floor, normalized_floor = _validated_floor(payload, seed=seed)
    return {
        "seed": seed,
        "path": str(path),
        "file_sha256": _sha256(path),
        "manifest_sha256": manifest_sha,
        "fm_sample_sha256": fm.get("sample_sha256"),
        "am_sample_sha256": am.get("sample_sha256"),
        "reference_train_sha256": reference.get("train_sha256"),
        "reference_eval_sha256": reference.get("eval_sha256"),
        "reference_2k_indices_sha256": reference.get(
            "shared_2k_indices_sha256"
        ),
        "reference_2k_selection_seed": reference.get(
            "shared_2k_selection_seed"
        ),
        "model_2k_selection_seed": reference_model_seed,
        "geometric_metric_seed": reference.get(
            "shared_geometric_metric_seed"
        ),
        "fm_2k_indices_sha256": fm.get("model_2k_indices_sha256"),
        "metrics": metric_values,
        "reference_floor_raw": floor,
        "reference_floor": normalized_floor,
    }


def _validate_shared_inputs(rows: dict[int, dict[str, Any]]) -> None:
    baseline = rows[2]
    hash_fields = (
        "manifest_sha256",
        "fm_sample_sha256",
        "reference_train_sha256",
        "reference_eval_sha256",
        "reference_2k_indices_sha256",
        "fm_2k_indices_sha256",
    )
    seed_fields = (
        "reference_2k_selection_seed",
        "model_2k_selection_seed",
        "geometric_metric_seed",
    )
    for seed, row in sorted(rows.items()):
        for field in hash_fields:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise AggregateValidationError(f"seed{seed}: missing {field}")
            _require_equal(value, baseline[field], f"seed{seed} shared {field}")
        for field in seed_fields:
            value = row.get(field)
            if not isinstance(value, int):
                raise AggregateValidationError(f"seed{seed}: missing {field}")
            _require_equal(value, baseline[field], f"seed{seed} shared {field}")
        for metric in METRICS:
            _require_close(
                row["metrics"]["fm"][metric],
                baseline["metrics"]["fm"][metric],
                f"seed{seed} shared FM {metric}",
            )
            floor = row["reference_floor"][metric]
            baseline_floor = baseline["reference_floor"][metric]
            for key in ("mean", "sample_std_ddof1", "median", "min", "max"):
                _require_close(
                    float(floor[key]),
                    float(baseline_floor[key]),
                    f"seed{seed} shared floor {metric} {key}",
                )
            _require_equal(
                floor["values"],
                baseline_floor["values"],
                f"seed{seed} shared floor {metric} values",
            )


def _metric_verdict(wins: int) -> str:
    if wins == len(EXPECTED_SEEDS):
        return "consistent_positive"
    if wins == 0:
        return "non_improving"
    return "mixed"


def _overall_conclusion(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = [metrics[name]["verdict"] for name in PRIMARY_METRICS]
    if all(value == "consistent_positive" for value in primary):
        verdict = "positive"
        text = (
            "AM improves both pre-registered primary metrics for all three AM "
            "training seeds, conditional on the one shared FM seed-0 checkpoint."
        )
    elif all(value == "non_improving" for value in primary):
        verdict = "negative"
        text = (
            "AM improves neither pre-registered primary metric across the three "
            "AM seeds, conditional on the one shared FM seed-0 checkpoint."
        )
    else:
        verdict = "mixed"
        text = (
            "The pre-registered primary metrics or AM seeds disagree; the result "
            "is mixed conditional on the one shared FM seed-0 checkpoint."
        )
    return {
        "verdict": verdict,
        "text": text,
        "claim_scope": (
            "AM-seed robustness conditional on one shared FM seed-0 checkpoint "
            "and one validated LJ13 reference-v2 bundle"
        ),
        "not_an_end_to_end_three_seed_fm_claim": True,
        "not_a_three_seed_fm_robustness_claim": True,
    }


def _build_aggregate(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    shared = rows[2]
    metrics: dict[str, dict[str, Any]] = {}
    for name in METRICS:
        fm = float(shared["metrics"]["fm"][name])
        am_by_seed = {
            str(seed): float(rows[seed]["metrics"]["am"][name])
            for seed in EXPECTED_SEEDS
        }
        am_values = [am_by_seed[str(seed)] for seed in EXPECTED_SEEDS]
        absolute = [fm - value for value in am_values]
        relative = [(fm - value) / fm if fm > 0.0 else math.nan for value in am_values]
        wins = sum(value < fm for value in am_values)
        floor = shared["reference_floor"][name]
        am_summary = _summary(am_values)
        metrics[name] = {
            "shared_fm_seed": 0,
            "shared_fm_value": fm,
            "am_by_seed": am_by_seed,
            "am_summary": am_summary,
            "absolute_improvement_fm_minus_am": {
                "by_seed": {
                    str(seed): absolute[index]
                    for index, seed in enumerate(EXPECTED_SEEDS)
                },
                **_summary(absolute),
            },
            "relative_improvement": {
                "definition": "(shared_fm - am_seed) / shared_fm",
                "by_seed": {
                    str(seed): relative[index]
                    for index, seed in enumerate(EXPECTED_SEEDS)
                },
                **_summary(relative),
            },
            "wins_am_below_shared_fm": wins,
            "losses_or_ties": len(EXPECTED_SEEDS) - wins,
            "verdict": _metric_verdict(wins),
            "reference_floor": floor,
            "shared_fm_to_floor_mean_ratio": (
                fm / float(floor["mean"]) if float(floor["mean"]) > 0.0 else None
            ),
            "am_mean_to_floor_mean_ratio": (
                float(am_summary["mean"]) / float(floor["mean"])
                if float(floor["mean"]) > 0.0
                else None
            ),
        }

    return {
        "schema": SCHEMA,
        "status": "pass",
        "problem": "lj13",
        "target_beta": 1.0,
        "experimental_unit": (
            "three independent AM training seeds conditional on one shared FM "
            "seed-0 checkpoint and one corrected reference-v2 bundle"
        ),
        "shared_inputs": {
            key: shared[key]
            for key in (
                "manifest_sha256",
                "fm_sample_sha256",
                "reference_train_sha256",
                "reference_eval_sha256",
                "reference_2k_indices_sha256",
                "reference_2k_selection_seed",
                "model_2k_selection_seed",
                "geometric_metric_seed",
                "fm_2k_indices_sha256",
            )
        },
        "score_inputs": {
            str(seed): {
                "path": rows[seed]["path"],
                "sha256": rows[seed]["file_sha256"],
                "am_sample_sha256": rows[seed]["am_sample_sha256"],
            }
            for seed in EXPECTED_SEEDS
        },
        "metric_protocol": {
            "energy_pair_rg_minpair_rows_each": 2_000,
            "geometric_rows_each": 2_000,
            "reference_floor_repeats": 5,
            "same_evaluation_seed": 0,
            "same_2k_subset_for_all_canonical_metrics": True,
            "same_model_2k_index_positions_across_fm_and_am": True,
            "strict_nonfinite_policy": "hard_fail_without_filtering",
        },
        "metrics": metrics,
        "conclusion": _overall_conclusion(metrics),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_per_seed_csv(
    path: Path,
    aggregate: dict[str, Any],
) -> None:
    fields = (
        "metric",
        "am_seed",
        "shared_fm",
        "am",
        "absolute_improvement_fm_minus_am",
        "relative_improvement",
        "am_below_shared_fm",
        "reference_floor_mean",
        "reference_floor_sample_std",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            row = aggregate["metrics"][metric]
            for seed in EXPECTED_SEEDS:
                am = row["am_by_seed"][str(seed)]
                writer.writerow(
                    {
                        "metric": metric,
                        "am_seed": seed,
                        "shared_fm": f"{row['shared_fm_value']:.17g}",
                        "am": f"{am:.17g}",
                        "absolute_improvement_fm_minus_am": (
                            f"{row['absolute_improvement_fm_minus_am']['by_seed'][str(seed)]:.17g}"
                        ),
                        "relative_improvement": (
                            f"{row['relative_improvement']['by_seed'][str(seed)]:.17g}"
                        ),
                        "am_below_shared_fm": str(am < row["shared_fm_value"]).lower(),
                        "reference_floor_mean": (
                            f"{row['reference_floor']['mean']:.17g}"
                        ),
                        "reference_floor_sample_std": (
                            f"{row['reference_floor']['sample_std_ddof1']:.17g}"
                        ),
                    }
                )


def _write_aggregate_csv(path: Path, aggregate: dict[str, Any]) -> None:
    fields = (
        "metric",
        "shared_fm",
        "am_mean",
        "am_sample_std_ddof1",
        "am_median",
        "wins_am_below_shared_fm",
        "am_seed_count",
        "mean_absolute_improvement",
        "mean_relative_improvement",
        "reference_floor_mean",
        "reference_floor_sample_std",
        "shared_fm_to_floor_mean_ratio",
        "am_mean_to_floor_mean_ratio",
        "verdict",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            row = aggregate["metrics"][metric]
            writer.writerow(
                {
                    "metric": metric,
                    "shared_fm": f"{row['shared_fm_value']:.17g}",
                    "am_mean": f"{row['am_summary']['mean']:.17g}",
                    "am_sample_std_ddof1": (
                        f"{row['am_summary']['sample_std_ddof1']:.17g}"
                    ),
                    "am_median": f"{row['am_summary']['median']:.17g}",
                    "wins_am_below_shared_fm": row["wins_am_below_shared_fm"],
                    "am_seed_count": len(EXPECTED_SEEDS),
                    "mean_absolute_improvement": (
                        f"{row['absolute_improvement_fm_minus_am']['mean']:.17g}"
                    ),
                    "mean_relative_improvement": (
                        f"{row['relative_improvement']['mean']:.17g}"
                    ),
                    "reference_floor_mean": (
                        f"{row['reference_floor']['mean']:.17g}"
                    ),
                    "reference_floor_sample_std": (
                        f"{row['reference_floor']['sample_std_ddof1']:.17g}"
                    ),
                    "shared_fm_to_floor_mean_ratio": (
                        ""
                        if row["shared_fm_to_floor_mean_ratio"] is None
                        else f"{row['shared_fm_to_floor_mean_ratio']:.17g}"
                    ),
                    "am_mean_to_floor_mean_ratio": (
                        ""
                        if row["am_mean_to_floor_mean_ratio"] is None
                        else f"{row['am_mean_to_floor_mean_ratio']:.17g}"
                    ),
                    "verdict": row["verdict"],
                }
            )


def _write_hashes(output: Path, paths: Iterable[Path]) -> None:
    rows = [
        f"{_sha256(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(paths)
    ]
    (output / "AGGREGATE_SHA256SUMS").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> Path:
    provided: dict[int, Path] = {}
    for seed, path in args.metric:
        if seed in provided:
            raise AggregateValidationError(f"Duplicate --metric for seed {seed}")
        provided[seed] = path
    _require_equal(tuple(sorted(provided)), EXPECTED_SEEDS, "AM seed set")

    rows: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        path = provided[seed]
        if not path.is_file():
            raise AggregateValidationError(f"Missing seed{seed} metrics: {path}")
        rows[seed] = _validate_payload(_read_json(path), seed=seed, path=path)
    _validate_shared_inputs(rows)

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise AggregateValidationError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    aggregate = _build_aggregate(rows)
    aggregate_json = output / "aggregate_metrics.json"
    per_seed_csv = output / "per_seed_metrics.csv"
    aggregate_csv = output / "aggregate_metrics.csv"
    _write_json(aggregate_json, aggregate)
    _write_per_seed_csv(per_seed_csv, aggregate)
    _write_aggregate_csv(aggregate_csv, aggregate)
    _write_hashes(output, (aggregate_json, per_seed_csv, aggregate_csv))
    print(
        f"Wrote LJ13 AM-seed aggregate ({aggregate['conclusion']['verdict']}): "
        f"{output}"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        action="append",
        type=_parse_metric_argument,
        required=True,
        metavar="SEED=PATH",
        help="Canonical strict score JSON; provide exactly seeds 1, 2, and 3.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
