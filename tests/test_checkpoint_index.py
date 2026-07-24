from __future__ import annotations

import hashlib

import yaml

from adj_thermo.cli import ROOT


def _verify(item: dict[str, str]) -> None:
    path = ROOT / item["path"]
    assert path.exists(), path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == item["sha256"], path


def test_released_checkpoint_index_paths_and_hashes():
    index = yaml.safe_load((ROOT / "checkpoints" / "index.yaml").read_text(encoding="utf-8"))
    for section in ("checkpoints", "experimental_checkpoints"):
        for entry in index.get(section, {}).values():
            _verify(entry["fm"])
            _verify(entry["am"])
            for item in entry.get("additional_formal_am", []):
                _verify(item)
            if "exploratory_scan_am" in entry:
                _verify(entry["exploratory_scan_am"])
