"""Generate or validate an auditable MB2D equilibrium reference-v2 bundle.

Examples
--------
Generate the formal train/eval splits::

    python scripts/generate_mb2d_reference_v2.py generate \
      --output data/mb2d_reference_v2 \
      --betas 0.25 0.50 0.75 1.00 1.20 1.50 \
      --train-samples 100000 --eval-samples 100000

Validate hashes and compare a legacy run's reference samples::

    python scripts/generate_mb2d_reference_v2.py validate \
      --output data/mb2d_reference_v2 \
      --compare legacy=outputs/mb2d/old_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adj_thermo.reference.mb2d_analytic import (  # noqa: E402
    AuditTolerances,
    MB2DDomain,
    generate_mb2d_reference_v2,
    parse_comparison_sources,
    validate_mb2d_reference_v2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate/validate independent train and eval samples from a "
            "convergence-audited analytic MB2D Boltzmann grid."
        )
    )
    parser.add_argument("action", choices=("generate", "validate"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "mb2d_reference_v2",
        help="Reference bundle root (default: data/mb2d_reference_v2).",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00, 1.20, 1.50],
    )
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--eval-samples", type=int, default=100000)
    parser.add_argument("--train-seed", type=int, default=17001)
    parser.add_argument("--eval-seed", type=int, default=27001)
    parser.add_argument(
        "--domain",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=[-4.0, 2.5, -2.5, 4.5],
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Nested midpoint-grid resolutions used for convergence audit.",
    )
    parser.add_argument("--domain-padding-fraction", type=float, default=0.15)
    parser.add_argument("--resolution-tv-tol", type=float, default=2.0e-3)
    parser.add_argument("--resolution-logz-tol", type=float, default=2.0e-3)
    parser.add_argument("--resolution-basin-l1-tol", type=float, default=2.0e-3)
    parser.add_argument("--domain-mass-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--allow-audit-failure",
        action="store_true",
        help=(
            "Write a bundle whose convergence audit failed. Formal generation "
            "is strict by default."
        ),
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Compare an old dataset with the grid. PATH may be a directory or "
            "contain the literal {beta}; repeat for multiple datasets."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    comparisons = parse_comparison_sources(args.compare)
    if args.action == "generate":
        domain = MB2DDomain(*map(float, args.domain))
        tolerances = AuditTolerances(
            resolution_tv=float(args.resolution_tv_tol),
            resolution_log_z=float(args.resolution_logz_tol),
            resolution_basin_l1=float(args.resolution_basin_l1_tol),
            domain_outside_mass=float(args.domain_mass_tol),
        )
        manifest = generate_mb2d_reference_v2(
            output,
            args.betas,
            int(args.train_samples),
            int(args.eval_samples),
            int(args.train_seed),
            int(args.eval_seed),
            domain=domain,
            resolutions=args.resolutions,
            domain_padding_fraction=float(args.domain_padding_fraction),
            tolerances=tolerances,
            comparison_sources=comparisons,
            strict=not bool(args.allow_audit_failure),
        )
        report = validate_mb2d_reference_v2(output)
        if not report["pass"]:
            raise SystemExit(
                "Generated bundle failed self-validation: "
                + "; ".join(report["errors"])
            )
        result = {
            "action": "generate",
            "output": str(output),
            "audit_pass": bool(manifest["quadrature"]["audit_pass"]),
            "validation_pass": True,
            "betas": sorted(manifest["betas"]),
            "train_samples_per_beta": int(args.train_samples),
            "eval_samples_per_beta": int(args.eval_samples),
        }
    else:
        report = validate_mb2d_reference_v2(
            output,
            comparison_sources=comparisons,
        )
        result = report
        if not report["pass"]:
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            raise SystemExit(1)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
