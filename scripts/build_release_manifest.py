#!/usr/bin/env python3
"""Regenerate and validate the source-release SHA256SUMS file."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "dist", "local_archive"}
FORBIDDEN_SUFFIXES = {".ckpt", ".npy", ".npz", ".pkl", ".pt", ".pth", ".safetensors"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = []
    forbidden = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.as_posix() == "SHA256SUMS":
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(relative.as_posix())
        files.append(path)
    if forbidden:
        raise RuntimeError(f"Forbidden release payloads: {forbidden}")

    lines = [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())
    ]
    (ROOT / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"files={len(lines)}")
    print(f"bytes={sum(path.stat().st_size for path in files)}")
    print("forbidden=0")


if __name__ == "__main__":
    main()
