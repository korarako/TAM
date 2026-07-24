from __future__ import annotations

from argparse import Namespace

import yaml

from adj_thermo.cli import ROOT, _profile_cli_args, build_parser


def test_all_reproduction_profiles_parse_for_their_available_phases():
    profiles = yaml.safe_load((ROOT / "configs" / "reproduction_profiles.yaml").read_text(encoding="utf-8"))["profiles"]
    parser = build_parser()
    for name, profile in profiles.items():
        common = [
            "--problem",
            profile["problem"],
            "--model",
            profile["model"],
            "--run-dir",
            f"outputs/{name}",
        ]
        fm = parser.parse_args(["train-fm", *common, "--data-dir", profile["data_dir"], *_profile_cli_args(profile["fm"])])
        assert isinstance(fm, Namespace)
        assert fm.steps == profile["fm"]["steps"]
        if "am" in profile:
            am = parser.parse_args(
                [
                    "train-am",
                    *common,
                    "--base-checkpoint",
                    f"outputs/{name}/fm_params.pkl",
                    *_profile_cli_args(profile["am"]),
                ]
            )
            assert am.steps == profile["am"]["steps"]
            assert am.am_parameterization == profile["am"]["am_parameterization"]


def test_recorded_molecular_best_settings_are_not_generic_defaults():
    profiles = yaml.safe_load((ROOT / "configs" / "reproduction_profiles.yaml").read_text(encoding="utf-8"))["profiles"]
    assert profiles["dw4_best_formal_seed2"]["am"]["grad_clip"] == 1.0
    assert profiles["lj13_best"]["fm"]["batch_size"] == 64
    assert profiles["lj13_best"]["am"]["grad_clip"] == 0.0
    assert profiles["ala2_fm_baseline"]["status"] == "exact_fm_only_no_successful_am"
