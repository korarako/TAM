#!/usr/bin/env python3
"""Build the checkpoint/sample-free TAM v0.2 corrected result tree.

The source freeze is the local, hash-verified research archive.  This exporter
uses an explicit whitelist, sanitizes machine-local paths in text artifacts,
rejects model/sample payloads, and emits a new release-level manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".sha256",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
INTERNAL_MANIFEST_NAMES = {
    "FREEZE_MANIFEST.json",
    "FREEZE_SHA256SUMS.txt",
    "SHA256SUMS",
    "SHA256SUMS.txt",
    "AGGREGATE_SHA256SUMS",
    "result_SHA256SUMS",
    "reference_SHA256SUMS",
}
PATH_REPLACEMENTS = (
    ("/ds/project/weilong/ke/adtm", "${ADTM_ROOT}"),
    ("/ds/project/weilong/ke/amboltz", "${AMBOLTZ_ROOT}"),
    ("/ds/project/weilong/ke/miniconda3", "${CONDA_ROOT}"),
    ("D:\\\\masterthesis\\\\demoprojects\\\\ke\\\\adtm", "${LOCAL_ADTM_ARCHIVE}"),
    ("D:\\masterthesis\\demoprojects\\ke\\adtm", "${LOCAL_ADTM_ARCHIVE}"),
    ("D:/masterthesis/demoprojects/ke/adtm", "${LOCAL_ADTM_ARCHIVE}"),
)
UNSANITIZED_MARKERS = (
    "/ds/project/weilong/",
    "D:\\\\masterthesis\\\\",
    "D:\\masterthesis\\",
    "D:/masterthesis/",
)
PUBLIC_README = """# TAM v0.2 corrected result export

This directory is the checkpoint- and sample-free public export of the
2026-07-26 corrected research freeze.

- `mb2d/`: corrected analytic-reference single-seed positive pilot.
- `dw4/`: corrected conditional positive result (one FM seed, three AM seeds).
- `lj13/`: corrected conditional negative result (one FM seed, three AM seeds).
- `ala2/`: fair fixed-10k single-seed mixed diagnostics.
- `publication_tables/`: standalone CSV/TeX/PDF/PNG tables.

`MANIFEST.json` and `SHA256SUMS` verify this exported tree.
`PROVENANCE.json` binds it to the full local research freeze. Model
checkpoints, optimizer states, and sample/reference arrays are intentionally
excluded. Machine-local paths in text records are replaced by symbolic roots.

Use `../summary.json`, `../../RESULTS.md`, and `../../LIMITATIONS.md` for the
release-level scientific interpretation.
"""
SYSTEM_EXPORT_NOTICE = (
    "> **Public export note:** this directory was derived from a larger local "
    "research freeze. Model parameters, optimizer state, and all `.npy`/`.npz` "
    "sample or reference payloads are intentionally omitted here. Their "
    "identities remain available through hashes and manifests.\n\n"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if source.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Refusing forbidden payload: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES or source.name in {
        "status.txt",
        "am_params.sha256",
        "fm_params.sha256",
    }:
        text = source.read_text(encoding="utf-8")
        for old, new in PATH_REPLACEMENTS:
            text = text.replace(old, new)
        destination.write_text(text, encoding="utf-8", newline="\n")
    else:
        shutil.copy2(source, destination)


def copy_relatives(source_root: Path, destination_root: Path, relatives: list[str]) -> None:
    for relative in relatives:
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        copy_file(source, destination_root / relative)


def copy_tree_filtered(source_root: Path, destination_root: Path) -> None:
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        if source.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        if source.name in INTERNAL_MANIFEST_NAMES:
            continue
        if source.suffix.lower() == ".log":
            continue
        copy_file(source, destination_root / source.relative_to(source_root))


def export_mb2d(freeze: Path, destination: Path) -> None:
    source = freeze / "mb2d_reference_v2"
    relatives = [
        "README.md",
        "SUMMARY.json",
        "reference_bundle/convergence_audit.json",
        "reference_bundle/manifest.json",
        "seed0_run/am_params.sha256",
        "seed0_run/analytic_grid_comparison.json",
        "seed0_run/config.yaml",
        "seed0_run/exact_run_script.sh",
        "seed0_run/fm_params.sha256",
        "seed0_run/reference_validation.json",
        (
            "seed0_run/"
            "score_samples_metrics_mb2d_reference_v2_eval_fm_am_"
            "betas_1p00_1p20_geo2000_ew20000.json"
        ),
        "seed0_run/status.txt",
    ]
    copy_relatives(source, destination, relatives)
    for subtree in ("code_snapshot", "reference_bundle/observables", "seed0_run/publication_figures"):
        copy_tree_filtered(source / subtree, destination / subtree)
    readme = destination / "README.md"
    readme.write_text(
        SYSTEM_EXPORT_NOTICE + readme.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )


def export_dw4(freeze: Path, destination: Path) -> None:
    source = freeze / "dw4_reference_v2"
    copy_relatives(
        source,
        destination,
        ["ENVIRONMENT.md", "LEGACY_REFERENCE_NOTICE.md", "README.md", "SUMMARY.json"],
    )
    copy_tree_filtered(source / "aggregate", destination / "aggregate")
    copy_tree_filtered(source / "reference_bundle", destination / "reference_bundle")
    for run in ("formal_seed2_run", "am_seed_replicates/seed1", "am_seed_replicates/seed3"):
        run_source = source / run
        for item in sorted(path for path in run_source.rglob("*") if path.is_file()):
            if item.suffix.lower() in FORBIDDEN_SUFFIXES or item.suffix.lower() == ".log":
                continue
            if item.name in INTERNAL_MANIFEST_NAMES:
                continue
            if (
                item.suffix.lower() in {".json", ".yaml", ".yml", ".sh", ".sha256", ".txt"}
                or item.name == "status.txt"
            ):
                copy_file(item, destination / run / item.relative_to(run_source))
    code_relatives = [
        "scripts/aggregate_dw4_reference_v2_am_seeds.py",
        "scripts/audit_dw4_reference_v2.py",
        "scripts/generate_dw4_reference_v2.py",
        "scripts/plot_dw4_reference_v2.py",
        "scripts/run_dw4_reference_v2_am_replicate.sh",
        "scripts/run_dw4_reference_v2_seed2.sh",
        "src/adj_thermo/reference/dw4_smc.py",
        "tests/test_dw4_reference_v2.py",
    ]
    copy_relatives(source / "code_snapshot", destination / "code_snapshot", code_relatives)
    readme = destination / "README.md"
    readme.write_text(
        SYSTEM_EXPORT_NOTICE + readme.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )


def export_compact_bundle(freeze: Path, name: str, destination: Path) -> None:
    copy_tree_filtered(freeze / name, destination)


def validate_and_manifest(destination: Path, source_freeze: Path) -> None:
    forbidden = [
        path
        for path in destination.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden payloads entered release: {forbidden}")

    unsanitized: list[str] = []
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in UNSANITIZED_MARKERS):
            unsanitized.append(path.relative_to(destination).as_posix())
    if unsanitized:
        raise RuntimeError(f"Unsanitized machine paths: {unsanitized}")

    source_manifest = source_freeze / "SHA256SUMS.txt"
    provenance = {
        "schema": "tam.corrected_release_provenance.v1",
        "release": "v0.2.0",
        "source_freeze_date": "2026-07-26",
        "source_freeze_manifest_sha256": sha256(source_manifest),
        "policy": {
            "whitelist_export": True,
            "machine_paths_sanitized": True,
            "checkpoints_included": False,
            "sample_arrays_included": False,
            "forbidden_suffixes": sorted(FORBIDDEN_SUFFIXES),
        },
    }
    (destination / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    records = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if path.name in {"MANIFEST.json", "SHA256SUMS"}:
            continue
        records.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "tam.corrected_release_manifest.v1",
        "release": "v0.2.0",
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sums = "\n".join(f"{item['sha256']}  {item['path']}" for item in records) + "\n"
    (destination / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--tam", type=Path, required=True)
    args = parser.parse_args()

    freeze = args.freeze.resolve()
    tam = args.tam.resolve()
    destination = tam / "results" / "corrected"
    expected_suffix = Path("results") / "corrected"
    if destination.relative_to(tam) != expected_suffix:
        raise RuntimeError(f"Unsafe destination: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    copy_relatives(
        freeze,
        destination,
        ["TAM_CORRECTED_FREEZE_README.md", "TAM_RESULT_STATUS.json"],
    )
    copy_tree_filtered(freeze / "publication_tables", destination / "publication_tables")
    export_mb2d(freeze, destination / "mb2d")
    export_dw4(freeze, destination / "dw4")
    export_compact_bundle(freeze, "lj13_adtm_reference_v2", destination / "lj13")
    export_compact_bundle(freeze, "ala2_adtm_extended", destination / "ala2")
    (destination / "README.md").write_text(
        PUBLIC_README,
        encoding="utf-8",
        newline="\n",
    )
    validate_and_manifest(destination, freeze)

    manifest = json.loads((destination / "MANIFEST.json").read_text(encoding="utf-8"))
    print(f"destination={destination}")
    print(f"files={manifest['file_count']}")
    print(f"bytes={manifest['total_bytes']}")


if __name__ == "__main__":
    main()
