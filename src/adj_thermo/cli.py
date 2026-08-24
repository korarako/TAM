from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import yaml

from adj_thermo.am import train_am
from adj_thermo.am_velocity import (
    AM_PARAMETERIZATIONS,
    TARGET_REFINEMENT,
    is_anchor_residual_checkpoint,
)
from adj_thermo.fm import load_beta_datasets, train_fm
from adj_thermo.langevin import generate_langevin_dataset
from adj_thermo.lj13_langevin import generate_lj13_langevin_dataset
from adj_thermo.metrics import energy_w2_n, geometric_w2_result, pairwise_distance_w2
from adj_thermo.problem import make_problem
from adj_thermo.sampler import sample_ode
from adj_thermo.utils import ensure_dir, load_pickle

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS = ("dw1d", "dw2d", "mb2d", "dw4", "lj13", "ala2")
MODELS = ("mlp", "egnn", "painn", "ala2_chiro_painn_jax")


def _path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _run_dir(args: argparse.Namespace) -> Path:
    return _path(args.run_dir)


def _data_dir(args: argparse.Namespace) -> Path:
    if args.problem == "ala2" and getattr(args, "ala2_data_root", None):
        return _path(args.ala2_data_root)
    if getattr(args, "data_dir", None):
        return _path(args.data_dir)
    return ROOT / "data" / args.problem


def _make_problem(args: argparse.Namespace):
    if args.problem != "ala2":
        return make_problem(args.problem)
    return make_problem(
        "ala2",
        openmm_device=args.ala2_openmm_device,
        openmm_grad_clip=args.ala2_openmm_grad_clip,
        openmm_system=args.ala2_openmm_system,
        data_root=_data_dir(args),
    )


def _default_langevin_step_size(problem: str) -> float:
    return {
        "dw1d": 5.0e-3,
        "dw2d": 5.0e-3,
        "mb2d": 1.0e-4,
        "dw4": 5.0e-5,
        "lj13": 1.0e-4,
    }[problem]


def _write_config(run_dir: Path, args: argparse.Namespace) -> None:
    payload = {key: value for key, value in vars(args).items() if key != "handler"}
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def generate_data_stage(args: argparse.Namespace) -> None:
    if args.problem == "ala2":
        from adj_thermo.problem.ala2 import data_file_for_beta

        missing = [str(data_file_for_beta(beta, _data_dir(args))) for beta in args.beta_grid if not data_file_for_beta(beta, _data_dir(args)).exists()]
        if missing:
            raise FileNotFoundError(
                "Ala2 uses external OpenMM MD datasets; missing files:\n" + "\n".join(missing)
            )
        print(f"[data] verified external Ala2 datasets in {_data_dir(args)}", flush=True)
        return
    problem = _make_problem(args)
    output = ensure_dir(_data_dir(args))
    step_size = args.langevin_step_size or _default_langevin_step_size(args.problem)
    if args.problem == "lj13":
        generate_lj13_langevin_dataset(
            args.beta_grid,
            args.n_per_beta,
            args.warmup_steps,
            output,
            seed=args.seed,
            step_size=step_size,
            init_file=args.init_file,
            n_chains=args.n_chains,
        )
    else:
        generate_langevin_dataset(
            problem,
            args.beta_grid,
            args.n_per_beta,
            args.langevin_steps,
            step_size,
            float(args.clip_range or problem.default_clip_range),
            jax.random.PRNGKey(args.seed),
            args.init_mode,
            output,
        )
    print(f"[data] wrote datasets to {output}", flush=True)


def _validate_model(problem: str, model: str) -> None:
    if model == "painn" and problem != "lj13":
        raise ValueError("PaiNN is supported only for LJ13 in the TAM release.")
    if model == "egnn" and problem != "dw4":
        raise ValueError("The released EGNN checkpoint and configuration are for DW4.")
    if model == "ala2_chiro_painn_jax" and problem != "ala2":
        raise ValueError("ala2_chiro_painn_jax is supported only for Ala2.")
    if problem == "ala2" and model != "ala2_chiro_painn_jax":
        raise ValueError("The retained Ala2 route uses --model ala2_chiro_painn_jax.")


def _load_training_datasets(args: argparse.Namespace) -> dict[float, np.ndarray]:
    if args.problem != "ala2":
        return load_beta_datasets(
            _data_dir(args),
            args.beta_grid,
            samples_per_beta=args.samples_per_beta,
            seed=args.seed,
        )
    from adj_thermo.problem.ala2 import load_ala2_positions_for_beta

    n = int(args.samples_per_beta) if int(args.samples_per_beta) > 0 else None
    return {
        float(beta): load_ala2_positions_for_beta(
            beta,
            n=n,
            seed=args.seed + 1009 * index,
            data_root=_data_dir(args),
        )
        for index, beta in enumerate(args.beta_grid)
    }


def train_fm_stage(args: argparse.Namespace) -> None:
    _validate_model(args.problem, args.model)
    problem = _make_problem(args)
    run_dir = ensure_dir(_run_dir(args))
    datasets = _load_training_datasets(args)
    _write_config(run_dir, args)
    train_fm(
        problem,
        datasets,
        args.beta_grid,
        run_dir,
        jax.random.PRNGKey(args.seed + 101),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        prior_scale=args.prior_scale,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        t_embed_dim=args.t_embed_dim,
        beta_embed_dim=args.beta_embed_dim,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        config={key: value for key, value in vars(args).items() if key != "handler"},
        model_type=args.model,
        painn_backend="jax",
        painn_atom_embed_dim=args.painn_atom_embed_dim,
        ala2_temp_embed_min=args.ala2_temp_embed_min,
        ala2_temp_embed_max=args.ala2_temp_embed_max,
        ala2_temp_embed_l0=args.ala2_temp_embed_l0,
        save_checkpoints=args.save_checkpoints,
    )
    print(f"[FM] checkpoint written to {run_dir / 'fm_params.pkl'}", flush=True)


def train_am_stage(args: argparse.Namespace) -> None:
    _validate_model(args.problem, args.model)
    problem = _make_problem(args)
    run_dir = ensure_dir(_run_dir(args))
    checkpoint = _path(args.base_checkpoint) if args.base_checkpoint else run_dir / "fm_params.pkl"
    base_params = load_pickle(checkpoint)
    _write_config(run_dir, args)
    train_am(
        problem,
        base_params,
        run_dir,
        jax.random.PRNGKey(args.seed + 202),
        beta0=args.beta0,
        beta1=args.beta1,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        K=args.sde_steps,
        prior_scale=args.prior_scale,
        am_num_loss_steps=args.loss_steps,
        am_keep_last_steps=args.keep_last_steps,
        max_sigma=args.max_sigma,
        energy_grad_scale=args.energy_grad_scale,
        t_embed_dim=args.t_embed_dim,
        beta_embed_dim=args.beta_embed_dim,
        grad_clip=args.grad_clip,
        adam_b1=args.adam_b1,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        model_type=args.model,
        save_checkpoints=args.save_checkpoints,
        am_parameterization=args.am_parameterization,
    )
    print(f"[AM] checkpoint written to {run_dir / 'am_params.pkl'}", flush=True)


def sample_stage(args: argparse.Namespace) -> None:
    problem = _make_problem(args)
    checkpoint = _path(args.checkpoint)
    params = load_pickle(checkpoint)
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(args.num_samples, int(problem.dim)),
    )
    done = 0
    chunk_index = 0
    while done < args.num_samples:
        count = min(args.batch_size, args.num_samples - done)
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed), chunk_index)
        samples[done : done + count] = sample_ode(
            params,
            key,
            problem,
            args.beta,
            count,
            args.ode_steps,
            prior_scale=args.prior_scale,
            t_embed_dim=args.t_embed_dim,
            beta_embed_dim=args.beta_embed_dim,
            method=args.method,
        )
        done += count
        chunk_index += 1
        samples.flush()
        print(f"[sample] wrote {done}/{args.num_samples}", flush=True)
    manifest = {
        "problem": args.problem,
        "checkpoint": str(checkpoint),
        "beta": args.beta,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "ode_steps": args.ode_steps,
        "method": args.method,
        "prior_scale": args.prior_scale,
        "seed": args.seed,
        "output": str(output),
    }
    if is_anchor_residual_checkpoint(params):
        manifest.update(
            {
                "am_parameterization": params["parameterization"],
                "checkpoint_beta0": float(params["beta0"]),
                "checkpoint_beta1": float(params["beta1"]),
            }
        )
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _finite_rows(array: np.ndarray) -> tuple[np.ndarray, float]:
    flat = np.asarray(array).reshape((len(array), -1))
    finite = np.isfinite(flat).all(axis=1)
    return flat[finite].astype(np.float32, copy=False), float(1.0 - np.mean(finite))


def _subsample(array: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    n = min(int(n_samples), len(array))
    if n == len(array):
        return np.asarray(array)
    rng = np.random.default_rng(seed)
    return np.asarray(array)[rng.choice(len(array), size=n, replace=False)]


def _load_sample_array(path: Path) -> np.ndarray:
    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "R" not in loaded:
                raise KeyError(f"NPZ sample archive has no 'R' array: {path}")
            return np.asarray(loaded["R"], dtype=np.float32).reshape((loaded["R"].shape[0], -1))
        finally:
            loaded.close()
    return np.asarray(loaded)


def evaluate_stage(args: argparse.Namespace) -> None:
    if (
        args.geometric_refine_top_k is not None
        and args.geometric_protocol != "legacy-topk"
    ):
        raise ValueError(
            "--geometric-refine-top-k requires --geometric-protocol legacy-topk"
        )
    problem = _make_problem(args)
    generated, generated_invalid = _finite_rows(_load_sample_array(_path(args.samples)))
    reference, reference_invalid = _finite_rows(_load_sample_array(_path(args.reference)))
    if len(generated) < args.metric_samples or len(reference) < args.metric_samples:
        raise ValueError("Not enough finite rows for the requested metric sample count.")

    rows: list[dict[str, Any]] = []
    energy_key = "energy_w2_2k" if args.metric_samples == 2000 else "energy_w2_n"
    geometry_key = "geometric_w2_2k" if args.metric_samples == 2000 else "geometric_w2_n"
    for repeat in range(args.repeats):
        seed = args.seed + repeat * 1009
        sample_subset = _subsample(generated, args.metric_samples, seed)
        reference_subset = _subsample(reference, args.metric_samples, seed + 1)
        if args.problem == "ala2":
            from adj_thermo.metrics_ala2 import ala2_summary_metrics

            ala2_metrics = ala2_summary_metrics(
                sample_subset,
                reference_subset,
                problem,
                energy_w2_samples=args.metric_samples,
                seed=seed,
            )
            row = {
                "repeat": repeat,
                "seed": seed,
                **ala2_metrics,
                energy_key: ala2_metrics["ew2_2k"],
                geometry_key: None,
                "pairwise_distance_w2": ala2_metrics["w2"],
            }
        else:
            geometry = geometric_w2_result(
                generated,
                reference,
                problem,
                n_samples=args.metric_samples,
                seed=seed,
                cost_chunk_size=args.geometric_chunk_size,
                dem_refine_top_k=args.geometric_refine_top_k,
                protocol=args.geometric_protocol,
                joint_max_iterations=args.geometric_max_iterations,
                joint_workers=args.geometric_workers,
                joint_parallel_backend=args.geometric_parallel_backend,
            )
            row = {
                "repeat": repeat,
                "seed": seed,
                energy_key: energy_w2_n(
                    sample_subset,
                    reference_subset,
                    problem,
                    n_samples=args.metric_samples,
                    seed=seed,
                ),
                geometry_key: geometry.value,
                "geometric_w2_metadata": geometry.to_dict(),
                "pairwise_distance_w2": pairwise_distance_w2(sample_subset, reference_subset, problem),
            }
        rows.append(row)
        print(f"[evaluate] repeat={repeat + 1}/{args.repeats} {row}", flush=True)

    aggregate: dict[str, Any] = {}
    aggregate_keys = [energy_key, geometry_key, "pairwise_distance_w2"]
    if args.problem == "ala2":
        aggregate_keys.extend(["phi_w2", "psi_w2", "rama_js", "min_pair_distance_w2"])
    for key in aggregate_keys:
        values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values)) if values.size else None
        aggregate[f"{key}_std"] = float(np.std(values)) if values.size else None
    result = {
        "problem": args.problem,
        "metric_samples": args.metric_samples,
        "repeats": args.repeats,
        "generated_rows": len(generated),
        "reference_rows": len(reference),
        "generated_invalid_fraction": generated_invalid,
        "reference_invalid_fraction": reference_invalid,
        "geometric_protocol_requested": args.geometric_protocol,
        "runs": rows,
        "aggregate": aggregate,
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[evaluate] wrote {output}", flush=True)


def _common_problem(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--problem", choices=PROBLEMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ala2-data-root", default=os.environ.get("TAM_ALA2_DATA_ROOT"))
    parser.add_argument("--ala2-openmm-device", choices=("cpu", "cuda"), default=os.environ.get("TAM_ALA2_OPENMM_DEVICE", "cpu"))
    parser.add_argument(
        "--ala2-openmm-system",
        choices=("alanine_dipeptide_implicit", "amber99sbildn_obc1_xml"),
        default=os.environ.get("TAM_ALA2_OPENMM_SYSTEM", "amber99sbildn_obc1_xml"),
    )
    parser.add_argument("--ala2-openmm-grad-clip", type=float, default=None)


def _common_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--t-embed-dim", type=int, default=16)
    parser.add_argument("--beta-embed-dim", type=int, default=8)
    parser.add_argument("--prior-scale", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tam", description="Thermodynamic Adjoint Matching")
    stages = parser.add_subparsers(dest="stage", required=True)

    data = stages.add_parser("generate-data", help="Generate reference data with Langevin dynamics.")
    _common_problem(data)
    data.add_argument("--beta-grid", nargs="+", type=float, required=True)
    data.add_argument("--data-dir", default=None)
    data.add_argument("--n-per-beta", type=int, default=100000)
    data.add_argument("--langevin-steps", type=int, default=5000)
    data.add_argument("--langevin-step-size", type=float, default=None)
    data.add_argument("--warmup-steps", type=int, default=2000)
    data.add_argument("--n-chains", type=int, default=1024)
    data.add_argument("--init-file", default=None)
    data.add_argument("--init-mode", choices=("normal", "uniform"), default="normal")
    data.add_argument("--clip-range", type=float, default=None)
    data.set_defaults(handler=generate_data_stage)

    fm = stages.add_parser("train-fm", help="Train a beta-conditioned Flow Matching model.")
    _common_problem(fm)
    _common_model(fm)
    fm.add_argument("--run-dir", required=True)
    fm.add_argument("--data-dir", default=None)
    fm.add_argument("--beta-grid", nargs="+", type=float, required=True)
    fm.add_argument("--samples-per-beta", type=int, default=0)
    fm.add_argument("--steps", type=int, default=30000)
    fm.add_argument("--batch-size", type=int, default=512)
    fm.add_argument("--lr", type=float, default=1.0e-3)
    fm.add_argument("--lr-min", type=float, default=1.0e-5)
    fm.add_argument("--grad-clip", type=float, default=1.0)
    fm.add_argument("--log-every", type=int, default=100)
    fm.add_argument("--painn-atom-embed-dim", type=int, default=16)
    fm.add_argument("--ala2-temp-embed-min", type=float, default=300.0)
    fm.add_argument("--ala2-temp-embed-max", type=float, default=1000.0)
    fm.add_argument("--ala2-temp-embed-l0", type=float, default=75.0)
    fm.add_argument("--save-checkpoints", nargs="*", type=int, default=None)
    fm.set_defaults(handler=train_fm_stage)

    am = stages.add_parser("train-am", help="Fine-tune a frozen FM anchor with Adjoint Matching.")
    _common_problem(am)
    _common_model(am)
    am.add_argument("--run-dir", required=True)
    am.add_argument("--base-checkpoint", default=None)
    am.add_argument("--beta0", type=float, required=True)
    am.add_argument("--beta1", type=float, required=True)
    am.add_argument("--steps", type=int, default=1000)
    am.add_argument("--batch-size", type=int, default=512)
    am.add_argument("--lr", type=float, default=5.0e-7)
    am.add_argument("--lr-min", type=float, default=5.0e-9)
    am.add_argument("--sde-steps", type=int, default=40)
    am.add_argument("--loss-steps", type=int, default=20)
    am.add_argument("--keep-last-steps", type=int, default=10)
    am.add_argument("--max-sigma", type=float, default=50.0)
    am.add_argument("--energy-grad-scale", type=float, default=1.0)
    am.add_argument("--grad-clip", type=float, default=1.0)
    am.add_argument("--adam-b1", type=float, default=0.9)
    am.add_argument("--weight-decay", type=float, default=0.0)
    am.add_argument("--log-every", type=int, default=100)
    am.add_argument("--save-checkpoints", nargs="*", type=int, default=None)
    am.add_argument(
        "--am-parameterization",
        choices=AM_PARAMETERIZATIONS,
        default=TARGET_REFINEMENT,
        help="Use the legacy target-FM refinement or a zero-control anchor residual.",
    )
    am.set_defaults(handler=train_am_stage)

    sample = stages.add_parser("sample", help="Generate samples from a released checkpoint.")
    _common_problem(sample)
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--beta", type=float, required=True)
    sample.add_argument("--num-samples", type=int, required=True)
    sample.add_argument("--batch-size", type=int, default=1000)
    sample.add_argument("--ode-steps", type=int, default=150)
    sample.add_argument("--method", choices=("euler", "heun"), default="euler")
    sample.add_argument("--prior-scale", type=float, default=1.0)
    sample.add_argument("--t-embed-dim", type=int, default=16)
    sample.add_argument("--beta-embed-dim", type=int, default=8)
    sample.add_argument("--output", required=True)
    sample.set_defaults(handler=sample_stage)

    evaluate = stages.add_parser("evaluate", help="Evaluate saved samples against a reference dataset.")
    _common_problem(evaluate)
    evaluate.add_argument("--samples", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--metric-samples", type=int, default=2000)
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--geometric-chunk-size", type=int, default=16)
    evaluate.add_argument(
        "--geometric-protocol",
        choices=("auto", "exact", "joint", "sequential", "legacy-topk"),
        default="auto",
        help=(
            "Particle-alignment protocol. auto uses exact permutation search for "
            "small systems and symmetry-consistent joint alignment for larger systems."
        ),
    )
    evaluate.add_argument(
        "--geometric-workers",
        type=int,
        default=1,
        help="Parallel workers for the full joint-alignment cost matrix.",
    )
    evaluate.add_argument(
        "--geometric-parallel-backend",
        choices=("process", "thread"),
        default="process",
        help="Use processes for formal CPU runs or threads for notebook safety.",
    )
    evaluate.add_argument("--geometric-max-iterations", type=int, default=50)
    evaluate.add_argument(
        "--geometric-refine-top-k",
        type=int,
        help="Deprecated top-k refinement count, valid only with --geometric-protocol legacy-topk.",
    )
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=evaluate_stage)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
