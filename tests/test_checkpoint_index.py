from __future__ import annotations

import re

import yaml

from adj_thermo.cli import ROOT


def test_v0_2_checkpoint_registry_is_hash_only():
    index = yaml.safe_load((ROOT / "checkpoints" / "index.yaml").read_text(encoding="utf-8"))
    assert index["payloads_included"] is False
    for record in index["corrected_checkpoint_hashes"].values():
        for key, value in record.items():
            assert key.endswith("_sha256")
            assert re.fullmatch(r"[0-9a-f]{64}", value)


def test_source_tree_has_no_checkpoint_payloads():
    forbidden = {".pkl", ".ckpt", ".pt", ".pth", ".safetensors"}
    found = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "dist" not in path.parts
        and "local_archive" not in path.parts
        and path.suffix.lower() in forbidden
    ]
    assert found == []
