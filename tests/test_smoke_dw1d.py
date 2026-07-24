from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(root: Path, args: list[str], env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, str(root / "main.py"), *args],
        cwd=root,
        check=True,
        timeout=120,
        env=env,
    )


def test_tiny_dw1d_pipeline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "dw1d_tiny"
    data_dir = tmp_path / "data"
    samples = tmp_path / "samples.npy"
    metrics = tmp_path / "metrics.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    _run(
        root,
        [
            "generate-data", "--problem", "dw1d", "--beta-grid", "1.25", "1.30",
            "--data-dir", str(data_dir), "--n-per-beta", "16", "--langevin-steps", "5",
        ],
        env,
    )
    _run(
        root,
        [
            "train-fm", "--problem", "dw1d", "--model", "mlp", "--beta-grid", "1.25", "1.30",
            "--data-dir", str(data_dir), "--run-dir", str(run_dir), "--steps", "2",
            "--batch-size", "4", "--hidden-dim", "8", "--n-layers", "1", "--log-every", "1",
        ],
        env,
    )
    _run(
        root,
        [
            "train-am", "--problem", "dw1d", "--model", "mlp", "--run-dir", str(run_dir),
            "--beta0", "1.25", "--beta1", "1.30", "--steps", "1", "--batch-size", "2",
            "--sde-steps", "2", "--loss-steps", "1", "--keep-last-steps", "1",
            "--hidden-dim", "8", "--n-layers", "1", "--log-every", "1",
        ],
        env,
    )
    _run(
        root,
        [
            "sample", "--problem", "dw1d", "--checkpoint", str(run_dir / "am_params.pkl"),
            "--beta", "1.30", "--num-samples", "8", "--batch-size", "4", "--ode-steps", "2",
            "--output", str(samples),
        ],
        env,
    )
    _run(
        root,
        [
            "evaluate", "--problem", "dw1d", "--samples", str(samples),
            "--reference", str(data_dir / "samples_beta_1.30.npy"), "--metric-samples", "8",
            "--output", str(metrics),
        ],
        env,
    )

    for path in (
        run_dir / "fm_params.pkl",
        run_dir / "am_params.pkl",
        run_dir / "fm_loss_history.npy",
        run_dir / "am_loss_history.npy",
        samples,
        metrics,
    ):
        assert path.exists(), path

    report = json.loads(metrics.read_text(encoding="utf-8"))
    assert report["metric_samples"] == 8
    assert "energy_w2_n" in report["runs"][0]
    assert "geometric_w2_n" in report["runs"][0]
