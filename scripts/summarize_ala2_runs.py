#!/usr/bin/env python3
"""Summarize historical Ala2 beta-sweep metrics without loading checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _scalar(row: dict, *names: str):
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)):
            return value
    return None


def _temperature(beta: float | None) -> float | None:
    # The project nondimensionalizes beta so beta * T = 120.272355 K.
    if not beta:
        return None
    return 120.27235504272604 / beta


def _rows(path: Path, kind: str) -> list[dict]:
    payload = json.loads(path.read_text())
    metrics = payload.get("per_beta", payload.get("metrics", payload))
    sample_count = payload.get("energy_w2_samples")
    result = []
    for beta_key, row in metrics.items():
        if not isinstance(row, dict):
            continue
        try:
            beta = float(beta_key)
        except (TypeError, ValueError):
            beta = _scalar(row, "beta")
        beta = _scalar(row, "beta") or beta
        energy_sample = row.get("sample_energy", {})
        reference_energy = row.get("reference_energy", row.get("md_energy", {}))
        result.append(
            {
                "kind": kind,
                "run": path.parent.parent.name,
                "metrics_path": str(path),
                "beta": beta,
                "temperature_K": _temperature(beta),
                "energy_w2_fixed": _scalar(row, "energy_w2_2k", "ew2_2k"),
                "energy_w2_full": _scalar(row, "energy_w2", "ew2"),
                "energy_w2_samples": row.get("energy_w2_samples", sample_count),
                "rama_js": _scalar(row, "rama_js", "ramachandran_js"),
                "phi_w2": _scalar(row, "phi_w2"),
                "psi_w2": _scalar(row, "psi_w2"),
                "pair_w2": _scalar(row, "pairwise_distance_w2", "pair_distance_w2", "w2"),
                "min_pair_w2": _scalar(row, "min_pair_distance_w2", "min_pair_w2"),
                "sample_energy_q99": _scalar(
                    row, "energy_finite_q99", "energy_q99"
                ) or _scalar(energy_sample, "q99"),
                "reference_energy_q99": _scalar(
                    row, "md_energy_finite_q99", "reference_energy_q99"
                ) or _scalar(reference_energy, "q99"),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("am_beta_sweep_metrics.json")):
        rows.extend(_rows(path, "am"))
    for path in sorted(args.root.rglob("fm_beta_sweep_metrics.json")):
        rows.extend(_rows(path, "fm"))

    text = json.dumps(rows, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
