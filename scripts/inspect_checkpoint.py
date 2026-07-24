"""Inspect a TAM checkpoint without mutating it."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Any


def _summary(value: Any) -> str:
    if isinstance(value, dict):
        return f"dict(keys={sorted(value)})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)}, items={[ _summary(item) for item in value ]})"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return f"{type(value).__name__}(shape={tuple(shape)}, dtype={dtype})"
    return type(value).__name__


def _array_leaves(value: Any, prefix: str = "root") -> list[str]:
    if isinstance(value, dict):
        leaves: list[str] = []
        for key, item in value.items():
            leaves.extend(_array_leaves(item, f"{prefix}.{key}"))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_array_leaves(item, f"{prefix}[{index}]"))
        return leaves
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [f"{prefix}: shape={tuple(shape)}, dtype={getattr(value, 'dtype', None)}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.checkpoints:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".npy":
            import numpy as np

            array = np.load(path, mmap_mode="r", allow_pickle=False)
            print(path)
            print(f"  sha256: {digest}")
            print(f"  array: shape={array.shape}, dtype={array.dtype}")
            continue
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)

        print(path)
        print(f"  sha256: {digest}")
        print(f"  type: {type(checkpoint).__name__}")
        print(f"  summary: {_summary(checkpoint)}")
        for leaf in _array_leaves(checkpoint)[:12]:
            print(f"  leaf: {leaf}")
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                print(f"  {key}: {_summary(value)}")
            if "config" in checkpoint:
                print(f"  config_value: {checkpoint['config']!r}")


if __name__ == "__main__":
    main()
