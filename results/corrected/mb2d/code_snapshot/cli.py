from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from adj_thermo.am import train_am
from adj_thermo.fm import load_beta_datasets, train_fm
from adj_thermo.langevin import beta_file_name, generate_langevin_dataset, sample_langevin
from adj_thermo.lj13_jax import generate_lj13_jax_dataset, sample_lj13_jax
from adj_thermo.metrics import dem_eq_emd2, energy_stats, energy_w2, energy_w2_n, geometric_w2, marginal_w2, pairwise_distance_w2
from adj_thermo.metrics_ala2 import ala2_phi_psi, ala2_summary_metrics, energy_values, js_divergence
from adj_thermo.problem import make_problem
from adj_thermo.sampler import make_sample_ode_fn, make_sample_ode_with_logq_fn, sample_ode
from adj_thermo.utils import ensure_dir, load_pickle
from adj_thermo.visualize import (
    plot_fm_md_energy_overlay,
    plot_fm_md_marginals_overlay,
    plot_fm_md_pairwise_combined_overlay,
    plot_fm_md_pairwise_overlay,
    plot_md_fm_am_energy_overlay,
    plot_md_fm_am_pairwise_combined_overlay,
    plot_energy_histogram,
    plot_loss_curves,
    plot_marginals,
    plot_pairwise_distances,
    plot_pairwise_distances_combined,
)
from adj_thermo.visualize_ala2 import (
    plot_ala2_energy_pair_overlay,
    plot_ala2_phi_psi_marginals,
    plot_ala2_phi_psi_marginals_three_way,
    plot_ala2_phi_psi_marginals_reweighted,
    plot_ala2_phi_psi_overlay,
    plot_ala2_phi_psi_reweighted_overlay,
    plot_ala2_phi_psi_three_way,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BETA_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]


def _fmt_beta(beta: float) -> str:
    return f"{float(beta):.2f}"


def _file_beta(beta: float) -> str:
    return _fmt_beta(beta).replace(".", "p")


def _run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        path = Path(args.run_dir)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "outputs" / f"{args.problem}_anchor_{_fmt_beta(args.beta0)}_target_{_fmt_beta(args.beta1)}"


def _data_dir(problem_name: str) -> Path:
    return ROOT / "data" / problem_name


def _data_name(args: argparse.Namespace, problem_name: str | None = None) -> str:
    name = getattr(args, "data_name", None)
    if name:
        return str(name)
    return str(problem_name or args.problem)


def _data_dir_for_args(args: argparse.Namespace, problem_name: str | None = None) -> Path:
    return ROOT / "data" / _data_name(args, problem_name)


def _reference_data_name(args: argparse.Namespace, problem_name: str | None = None) -> str:
    """Return the held-out reference dataset name used only for evaluation.

    Keeping this separate from ``--data-name`` prevents training samples from
    being silently reused as the evaluation reference.  The legacy behavior is
    preserved when ``--reference-data-name`` is omitted.
    """

    name = getattr(args, "reference_data_name", None)
    if name:
        return str(name)
    return _data_name(args, problem_name)


def _reference_data_dir_for_args(args: argparse.Namespace, problem_name: str | None = None) -> Path:
    return ROOT / "data" / _reference_data_name(args, problem_name)


def _make_problem(args: argparse.Namespace):
    if str(args.problem).lower() == "ala2":
        from adj_thermo.problem.ala2 import make_problem as make_ala2_problem

        return make_ala2_problem(
            openmm_device=str(args.ala2_openmm_device),
            openmm_grad_clip=None if args.ala2_openmm_grad_clip is None else float(args.ala2_openmm_grad_clip),
            openmm_system=str(args.ala2_openmm_system),
        )
    return make_problem(args.problem)


def _finite_rows_np(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape((arr.shape[0], -1))
    return arr[np.isfinite(flat).all(axis=1)]


def _step_size(problem_name: str, requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    if problem_name == "lj13":
        return 1.0e-5
    if problem_name == "dw4":
        return 5.0e-5
    if problem_name == "mb2d":
        return 1.0e-4
    return 5.0e-3


def _effective(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vars(args).copy()
    if cfg.get("prior_scale") is None:
        cfg["prior_scale"] = 0.15 if args.problem == "ala2" else 1.0
    if cfg.get("grad_clip") is None:
        cfg["grad_clip"] = 0.5 if args.problem == "ala2" and args.model in {"painn", "ala2_cpainn", "ala2_chiro_cpainn", "ala2_chiro_painn_jax"} else 1.0
    if args.problem == "ala2" and int(cfg.get("fm_eval_chunk_size") or 0) <= 0:
        cfg["fm_eval_chunk_size"] = 256
    if cfg.get("ode_method") is None:
        cfg["ode_method"] = "euler"
    if cfg.get("fm_eval_method") is None:
        cfg["fm_eval_method"] = cfg["ode_method"]
    if str(cfg.get("painn_atom_identity", "auto")).lower() == "auto":
        if args.problem == "ala2" and args.model == "ala2_cpainn":
            cfg["effective_painn_atom_identity"] = "learned_atom_index_ti"
        elif args.problem == "ala2" and args.model == "ala2_chiro_cpainn":
            cfg["effective_painn_atom_identity"] = "learned_atom_index_ti_chiral"
        elif args.problem == "ala2" and args.model == "ala2_chiro_painn_jax":
            cfg["effective_painn_atom_identity"] = "learned_atom_index_ti_chiral_painn_jax"
        elif args.problem == "ala2" and args.model == "painn" and str(getattr(args, "painn_backend", "jax")) == "amboltz_aligned":
            cfg["effective_painn_atom_identity"] = "learned_atom_index_hidden_state"
        else:
            cfg["effective_painn_atom_identity"] = "learned" if args.problem == "ala2" and args.model == "painn" else "none"
    else:
        cfg["effective_painn_atom_identity"] = str(cfg["painn_atom_identity"]).lower()
    if args.smoke:
        cfg["n_per_beta"] = min(int(args.n_per_beta), 2000)
        cfg["fm_steps"] = min(int(args.fm_steps), 500)
        cfg["am_steps"] = min(int(args.am_steps), 200)
        cfg["n_eval"] = min(int(args.n_eval), 1000)
        cfg["batch_size"] = min(int(args.batch_size), 128)
        cfg["am_batch_size"] = min(int(args.am_batch_size), 32)
        cfg["langevin_steps"] = min(int(args.langevin_steps), 500)
        cfg["ode_steps"] = min(int(args.ode_steps), 80)
        cfg["K"] = min(int(args.K), 20)
        cfg["am_num_loss_steps"] = min(int(args.am_num_loss_steps), 8)
        cfg["am_keep_last_steps"] = min(int(args.am_keep_last_steps), 4)
        cfg["lj13_warmup_steps"] = min(int(args.lj13_warmup_steps), 20)
        if args.problem == "lj13":
            cfg["n_per_beta"] = min(int(args.n_per_beta), 64)
            cfg["n_eval"] = min(int(args.n_eval), 64)
        if args.problem == "ala2":
            cfg["n_per_beta"] = min(int(args.n_per_beta), 64)
            cfg["n_eval"] = min(int(args.n_eval), 16)
            cfg["batch_size"] = min(int(args.batch_size), 8)
            cfg["am_batch_size"] = min(int(args.am_batch_size), 2)
            cfg["ode_steps"] = min(int(args.ode_steps), 4)
            cfg["K"] = min(int(args.K), 4)
    return cfg


def _config(args: argparse.Namespace, eff: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": args.problem,
        "data_name": _data_name(args),
        "reference_data_name": _reference_data_name(args),
        "beta_grid": args.beta_grid,
        "beta0": float(args.beta0),
        "beta1": float(args.beta1),
        "seed": int(args.seed),
        "effective": eff,
        "model_type": args.model,
        "model": {
            "type": args.model,
            "painn_backend": str(args.painn_backend),
            "hidden_dim": int(args.hidden_dim),
            "n_layers": int(args.n_layers),
            "t_embed_dim": int(args.t_embed_dim),
            "beta_embed_dim": int(args.beta_embed_dim),
            "painn_atom_identity": str(args.painn_atom_identity),
            "effective_painn_atom_identity": str(eff.get("effective_painn_atom_identity", args.painn_atom_identity)),
            "painn_atom_embed_dim": int(args.painn_atom_embed_dim),
            "ala2_temp_embed_min": float(args.ala2_temp_embed_min),
            "ala2_temp_embed_max": float(args.ala2_temp_embed_max),
            "ala2_temp_embed_l0": float(args.ala2_temp_embed_l0),
        },
        "sampling": {
            "ode_method": str(eff.get("ode_method", "euler")),
            "fm_eval_method": str(eff.get("fm_eval_method", eff.get("ode_method", "euler"))),
        },
        "ala2": {
            "openmm_device": str(args.ala2_openmm_device),
            "openmm_system": str(args.ala2_openmm_system),
            "openmm_grad_clip": None if args.ala2_openmm_grad_clip is None else float(args.ala2_openmm_grad_clip),
        },
    }


def generate_data(args: argparse.Namespace, eff: dict[str, Any]) -> None:
    problem = _make_problem(args)
    key = jax.random.PRNGKey(int(args.seed))
    out = _data_dir_for_args(args, problem.name)
    if problem.name == "lj13":
        generate_lj13_jax_dataset(
            args.beta_grid,
            int(eff["n_per_beta"]),
            int(eff["lj13_warmup_steps"]),
            out,
            seed=int(args.seed),
            kernel=str(args.lj13_sampler_kernel),
            step_size=float(args.lj13_init_step_size),
            init_file=args.lj13_init_file,
            n_chains=int(args.lj13_n_chains),
        )
        print(f"[data] wrote JAX LJ13 beta datasets to {out}", flush=True)
        return
    if problem.name == "ala2":
        from adj_thermo.problem.ala2 import TEMP_TO_BETA, data_file_for_beta

        for beta in args.beta_grid:
            print(f"[data] ala2 beta={float(beta):.8f} uses external MD dataset {data_file_for_beta(float(beta))}", flush=True)
        print(f"[data] available Ala2 beta map: {TEMP_TO_BETA}", flush=True)
        return
    generate_langevin_dataset(
        problem,
        args.beta_grid,
        int(eff["n_per_beta"]),
        int(eff["langevin_steps"]),
        _step_size(problem.name, args.langevin_step_size),
        float(args.clip_range or problem.default_clip_range),
        key,
        args.init_mode,
        out,
    )
    print(f"[data] wrote beta datasets to {out}", flush=True)


def train_fm_stage(args: argparse.Namespace, eff: dict[str, Any]) -> Any:
    problem = _make_problem(args)
    if args.model in {"ala2_cpainn", "ala2_chiro_cpainn", "ala2_chiro_painn_jax"} and problem.name != "ala2":
        raise ValueError(f"--model {args.model} is only supported for --problem ala2.")
    if args.model == "painn" and problem.name not in {"lj13", "ala2"}:
        raise ValueError("--model painn is currently supported only for molecular problems lj13 and ala2.")
    if args.model == "painn" and args.painn_backend == "amboltz_aligned" and (problem.n_particles is None or problem.spatial_dim != 3):
        raise ValueError("--painn-backend amboltz_aligned requires a 3D molecular particle problem.")
    if args.model == "egnn" and (problem.n_particles is None or problem.spatial_dim is None):
        raise ValueError("--model egnn requires a particle problem with n_particles and spatial_dim.")
    datasets = load_beta_datasets(
        _data_dir_for_args(args, problem.name),
        args.beta_grid,
        samples_per_beta=int(args.fm_samples_per_beta),
        seed=int(args.seed),
    )
    run_dir = ensure_dir(_run_dir(args))
    params, _ = train_fm(
        problem,
        datasets,
        args.beta_grid,
        run_dir,
        jax.random.PRNGKey(int(args.seed) + 101),
        steps=int(eff["fm_steps"]),
        batch_size=int(eff["batch_size"]),
        lr=float(args.fm_lr),
        lr_min=float(args.fm_lr_min),
        prior_scale=float(args.prior_scale),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        t_embed_dim=int(args.t_embed_dim),
        beta_embed_dim=int(args.beta_embed_dim),
        grad_clip=float(args.grad_clip),
        log_every=int(args.log_every),
        config=_config(args, eff),
        model_type=str(args.model),
        painn_backend=str(args.painn_backend),
        fm_eval_every=int(args.fm_eval_every),
        fm_eval_samples=int(args.fm_eval_samples),
        fm_eval_steps=int(args.fm_eval_steps),
        fm_eval_chunk_size=int(args.fm_eval_chunk_size),
        fm_eval_betas=args.fm_eval_betas,
        fm_eval_method=str(eff["fm_eval_method"]),
        painn_atom_identity=str(args.painn_atom_identity),
        painn_atom_embed_dim=int(args.painn_atom_embed_dim),
        ala2_temp_embed_min=float(args.ala2_temp_embed_min),
        ala2_temp_embed_max=float(args.ala2_temp_embed_max),
        ala2_temp_embed_l0=float(args.ala2_temp_embed_l0),
        save_checkpoints=args.save_fm_checkpoints,
    )
    print(f"[FM] checkpoint written to {run_dir / 'fm_params.pkl'}", flush=True)
    return params


def train_am_stage(args: argparse.Namespace, eff: dict[str, Any], base_params: Any | None = None) -> Any:
    problem = _make_problem(args)
    if args.model in {"ala2_cpainn", "ala2_chiro_cpainn", "ala2_chiro_painn_jax"} and problem.name != "ala2":
        raise ValueError(f"--model {args.model} is only supported for --problem ala2.")
    if args.model == "painn" and problem.name not in {"lj13", "ala2"}:
        raise ValueError("--model painn is currently supported only for molecular problems lj13 and ala2.")
    if args.model == "egnn" and (problem.n_particles is None or problem.spatial_dim is None):
        raise ValueError("--model egnn requires a particle problem with n_particles and spatial_dim.")
    run_dir = ensure_dir(_run_dir(args))
    if base_params is None:
        base_params = load_pickle(run_dir / "fm_params.pkl")
    params, _ = train_am(
        problem,
        base_params,
        run_dir,
        jax.random.PRNGKey(int(args.seed) + 202),
        beta0=float(args.beta0),
        beta1=float(args.beta1),
        steps=int(eff["am_steps"]),
        batch_size=int(eff["am_batch_size"]),
        lr=float(args.am_lr),
        lr_min=float(args.am_lr_min),
        K=int(eff["K"]),
        prior_scale=float(args.prior_scale),
        am_num_loss_steps=int(eff["am_num_loss_steps"]),
        am_keep_last_steps=int(eff["am_keep_last_steps"]),
        max_sigma=float(args.max_sigma),
        energy_grad_scale=float(args.energy_grad_scale),
        t_embed_dim=int(args.t_embed_dim),
        beta_embed_dim=int(args.beta_embed_dim),
        grad_clip=float(args.grad_clip),
        adam_b1=float(args.adam_b1),
        weight_decay=float(args.weight_decay),
        log_every=int(args.log_every),
        model_type=str(args.model),
        save_checkpoints=args.save_am_checkpoints,
    )
    print(f"[AM] checkpoint written to {run_dir / 'am_params.pkl'}", flush=True)
    return params


def _reference(problem, beta: float, args: argparse.Namespace, eff: dict[str, Any], offset: int) -> np.ndarray:
    n_eval = int(eff["n_eval"])
    if problem.name == "lj13":
        samples, stats = sample_lj13_jax(
            beta,
            n_eval,
            warmup_steps=int(eff["lj13_warmup_steps"]),
            kernel=str(args.lj13_sampler_kernel),
            seed=int(args.seed) + offset,
            step_size=float(args.lj13_init_step_size),
            init_file=args.lj13_init_file,
            n_chains=int(args.lj13_n_chains),
        )
        print(
            f"[reference] lj13-jax beta={float(beta):.2f} "
            f"warmup_accept={stats['warmup_accept_rate']:.3f} sample_accept={stats['sample_accept_rate']:.3f}",
            flush=True,
        )
        return samples
    pieces: list[np.ndarray] = []
    total = 0
    attempt = 0
    while total < n_eval and attempt < 8:
        need = n_eval - total
        draw = max(need + max(256, need // 20), need)
        arr = sample_langevin(
            problem,
            beta,
            draw,
            int(eff["langevin_steps"]),
            _step_size(problem.name, args.langevin_step_size),
            float(args.clip_range or problem.default_clip_range),
            jax.random.PRNGKey(int(args.seed) + offset + attempt * 1009),
            args.init_mode,
        )
        finite = _finite_rows_np(np.asarray(jax.device_get(arr), dtype=np.float32))
        if finite.shape[0]:
            pieces.append(finite[:need])
            total += min(need, finite.shape[0])
        attempt += 1
    if not pieces:
        raise ValueError(f"No finite reference samples for {problem.name} beta={float(beta):.2f}")
    out = np.concatenate(pieces, axis=0)[:n_eval]
    if out.shape[0] < n_eval:
        print(
            f"[reference] beta={float(beta):.2f} only collected {out.shape[0]}/{n_eval} finite samples",
            flush=True,
        )
    return out.astype(np.float32, copy=False)


def _reference_from_dataset_or_langevin(problem, beta: float, args: argparse.Namespace, eff: dict[str, Any], offset: int) -> tuple[np.ndarray, str]:
    n_eval = int(eff["n_eval"])
    explicit_reference = bool(getattr(args, "reference_data_name", None))
    path = _reference_data_dir_for_args(args, problem.name) / beta_file_name(beta)
    if path.exists():
        arr = _finite_rows_np(np.load(path))
        if arr.shape[0] >= n_eval:
            rng = np.random.default_rng(int(args.seed) + offset)
            idx = rng.choice(arr.shape[0], size=n_eval, replace=False)
            return np.asarray(arr[idx], dtype=np.float32), f"dataset:{path}"
        if explicit_reference:
            raise ValueError(
                f"Explicit reference dataset beta={float(beta):.2f} has only "
                f"{arr.shape[0]}/{n_eval} finite rows: {path}"
            )
        print(
            f"[reference] beta={float(beta):.2f} dataset has {arr.shape[0]}/{n_eval} finite rows; sampling extra",
            flush=True,
        )
    elif explicit_reference:
        raise FileNotFoundError(
            f"Missing explicit reference dataset for beta={float(beta):.2f}: {path}"
        )
    if problem.name == "ala2":
        from adj_thermo.problem.ala2 import data_file_for_beta, load_ala2_positions_for_beta

        arr = _finite_rows_np(load_ala2_positions_for_beta(float(beta), center=True))
        if arr.shape[0] <= 0:
            raise ValueError(f"No finite Ala2 reference samples for beta={float(beta):.8f}")
        rng = np.random.default_rng(int(args.seed) + offset)
        replace = arr.shape[0] < n_eval
        idx = rng.choice(arr.shape[0], size=n_eval, replace=replace)
        return np.asarray(arr[idx], dtype=np.float32), f"ala2-external:{data_file_for_beta(float(beta))}"
    arr = _reference(problem, beta, args, eff, offset)
    if problem.name == "lj13":
        return arr, "jax:on-the-fly"
    return arr, "langevin:on-the-fly"


def _sample_ode_chunked(
    params: Any,
    key: jax.Array,
    problem,
    beta: float,
    n_eval: int,
    chunk_size: int,
    args: argparse.Namespace,
    eff: dict[str, Any],
    log_prefix: str = "eval-fm-betas",
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    sample_fns: dict[int, Any] = {}
    done = 0
    chunk_id = 0
    while done < int(n_eval):
        count = min(int(chunk_size), int(n_eval) - done)
        if count not in sample_fns:
            sample_fns[count] = make_sample_ode_fn(
                params,
                problem,
                count,
                int(eff["ode_steps"]),
                float(args.prior_scale),
                int(args.t_embed_dim),
                int(args.beta_embed_dim),
                method=str(eff["ode_method"]),
            )
        subkey = jax.random.fold_in(key, chunk_id)
        samples = sample_fns[count](subkey, float(beta))
        chunks.append(np.asarray(jax.device_get(samples), dtype=np.float32))
        done += count
        chunk_id += 1
        print(f"[{log_prefix}] beta={float(beta):.2f} method={eff['ode_method']} sampled {done}/{int(n_eval)}", flush=True)
    return np.concatenate(chunks, axis=0)


def _sample_ode_with_logq_chunked(
    params: Any,
    key: jax.Array,
    problem,
    beta: float,
    n_eval: int,
    chunk_size: int,
    args: argparse.Namespace,
    eff: dict[str, Any],
    log_prefix: str = "eval-fm-betas",
) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    logq_chunks: list[np.ndarray] = []
    sample_fns: dict[int, Any] = {}
    done = 0
    chunk_id = 0
    while done < int(n_eval):
        count = min(int(chunk_size), int(n_eval) - done)
        if count not in sample_fns:
            sample_fns[count] = make_sample_ode_with_logq_fn(
                params,
                problem,
                count,
                int(eff["ode_steps"]),
                prior_scale=float(args.prior_scale),
                t_embed_dim=int(args.t_embed_dim),
                beta_embed_dim=int(args.beta_embed_dim),
                method=str(eff["ode_method"]),
                density_mode=str(args.density_mode),
                hutchinson_probes=int(args.density_hutchinson_probes),
            )
        subkey = jax.random.fold_in(key, chunk_id)
        samples, logq = sample_fns[count](subkey, float(beta))
        chunks.append(np.asarray(jax.device_get(samples), dtype=np.float32))
        logq_chunks.append(np.asarray(jax.device_get(logq), dtype=np.float64))
        done += count
        chunk_id += 1
        print(
            f"[{log_prefix}] beta={float(beta):.2f} method={eff['ode_method']} "
            f"density={args.density_mode}/{int(args.density_hutchinson_probes)} sampled {done}/{int(n_eval)}",
            flush=True,
        )
    return np.concatenate(chunks, axis=0), np.concatenate(logq_chunks, axis=0)


def _logsumexp_np(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    m = float(np.max(arr))
    return float(m + np.log(np.sum(np.exp(arr - m))))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return float(np.interp(float(q), cdf, values))


def _weighted_hist_prob(values: np.ndarray, weights: np.ndarray | None, bins: int, range_: tuple[tuple[float, float], ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    hist, _ = np.histogramdd(values, bins=int(bins), range=range_, weights=weights_arr)
    p = hist.astype(np.float64).reshape(-1)
    s = float(np.sum(p))
    return p / s if s > 0.0 else p


def _compute_and_save_reweighting(
    samples: np.ndarray,
    ref: np.ndarray,
    logq: np.ndarray,
    beta: float,
    problem,
    out_dir: Path,
    params_kind: str,
    beta_tag: str,
    beta_fmt: str,
    seed: int,
) -> dict[str, Any]:
    print(f"[reweight] beta={float(beta):.2f} computing proposal energies", flush=True)
    energy = energy_values(samples, problem, chunk_size=256)
    logq = np.asarray(logq, dtype=np.float64).reshape(-1)
    n = min(samples.shape[0], logq.shape[0], energy.shape[0])
    samples = np.asarray(samples[:n], dtype=np.float32)
    ref = np.asarray(ref[: min(ref.shape[0], n)], dtype=np.float32)
    logq = logq[:n]
    energy = np.asarray(energy[:n], dtype=np.float64)
    raw_logw = -float(beta) * energy - logq
    finite = np.isfinite(raw_logw)
    logw = np.full((n,), -np.inf, dtype=np.float64)
    weights = np.zeros((n,), dtype=np.float64)
    if np.any(finite):
        logz = _logsumexp_np(raw_logw[finite])
        logw[finite] = raw_logw[finite] - logz
        weights[finite] = np.exp(logw[finite])
        weights = weights / np.sum(weights)
    ess = float(1.0 / np.sum(weights * weights)) if np.sum(weights) > 0.0 else 0.0
    ess_fraction = float(ess / float(n)) if n > 0 else float("nan")

    np.save(out_dir / f"{params_kind}_logq_beta_{beta_fmt}.npy", logq)
    np.save(out_dir / f"{params_kind}_energy_beta_{beta_fmt}.npy", energy)
    np.save(out_dir / f"{params_kind}_logw_beta_{beta_fmt}.npy", logw)
    np.save(out_dir / f"{params_kind}_weights_beta_{beta_fmt}.npy", weights)

    out: dict[str, Any] = {
        "enabled": True,
        "density_mode": "hutchinson",
        "n": int(n),
        "finite_fraction": float(np.mean(finite)) if n > 0 else float("nan"),
        "ess": ess,
        "ess_fraction": ess_fraction,
        "logq_mean": float(np.nanmean(logq)),
        "logq_std": float(np.nanstd(logq)),
        "raw_logw_finite_min": float(np.min(raw_logw[finite])) if np.any(finite) else float("nan"),
        "raw_logw_finite_max": float(np.max(raw_logw[finite])) if np.any(finite) else float("nan"),
        "raw_logw_finite_std": float(np.std(raw_logw[finite])) if np.any(finite) else float("nan"),
        "weighted_energy_mean": float(np.sum(weights * energy)) if n > 0 else float("nan"),
        "weighted_energy_q50": _weighted_quantile(energy, weights, 0.50),
        "weighted_energy_q90": _weighted_quantile(energy, weights, 0.90),
        "weighted_energy_q99": _weighted_quantile(energy, weights, 0.99),
    }
    if problem.name == "ala2" and n > 0:
        sample_phi_psi = ala2_phi_psi(samples)
        ref_phi_psi = ala2_phi_psi(ref)
        rama_range = ((-np.pi, np.pi), (-np.pi, np.pi))
        out["weighted_rama_js"] = js_divergence(
            _weighted_hist_prob(sample_phi_psi, weights, 80, rama_range),
            _weighted_hist_prob(ref_phi_psi, None, 80, rama_range),
        )
        plot_ala2_phi_psi_reweighted_overlay(
            ref,
            samples,
            weights,
            beta,
            out_dir / f"reweighted_rama_overlay_beta_{beta_tag}.png",
            sample_label=params_kind.upper(),
        )
        plot_ala2_phi_psi_marginals_reweighted(
            ref,
            samples,
            weights,
            beta,
            out_dir / f"reweighted_torsion_marginals_beta_{beta_tag}.png",
            sample_label=params_kind.upper(),
        )
    return out


def sample_eval_stage(args: argparse.Namespace, eff: dict[str, Any], fm_params: Any | None = None, am_params: Any | None = None) -> dict[str, Any]:
    problem = _make_problem(args)
    run_dir = ensure_dir(_run_dir(args))
    if fm_params is None:
        fm_params = load_pickle(run_dir / "fm_params.pkl")
    if am_params is None:
        am_params = load_pickle(run_dir / "am_params.pkl")

    n_eval = int(eff["n_eval"])
    fm0 = sample_ode(fm_params, jax.random.PRNGKey(int(args.seed) + 301), problem, float(args.beta0), n_eval, int(eff["ode_steps"]), float(args.prior_scale), int(args.t_embed_dim), int(args.beta_embed_dim), method=str(eff["ode_method"]))
    fm1 = sample_ode(fm_params, jax.random.PRNGKey(int(args.seed) + 302), problem, float(args.beta1), n_eval, int(eff["ode_steps"]), float(args.prior_scale), int(args.t_embed_dim), int(args.beta_embed_dim), method=str(eff["ode_method"]))
    am1 = sample_ode(am_params, jax.random.PRNGKey(int(args.seed) + 303), problem, float(args.beta1), n_eval, int(eff["ode_steps"]), float(args.prior_scale), int(args.t_embed_dim), int(args.beta_embed_dim), method=str(eff["ode_method"]))
    ref0, ref0_source = _reference_from_dataset_or_langevin(
        problem, float(args.beta0), args, eff, 401
    )
    ref1, ref1_source = _reference_from_dataset_or_langevin(
        problem, float(args.beta1), args, eff, 402
    )

    np.save(run_dir / f"fm_samples_beta_{_fmt_beta(args.beta0)}.npy", fm0)
    np.save(run_dir / f"fm_samples_beta_{_fmt_beta(args.beta1)}.npy", fm1)
    np.save(run_dir / f"am_samples_beta_{_fmt_beta(args.beta1)}.npy", am1)
    np.save(run_dir / f"ref_samples_beta_{_fmt_beta(args.beta0)}.npy", ref0)
    np.save(run_dir / f"ref_samples_beta_{_fmt_beta(args.beta1)}.npy", ref1)

    energy_w2_samples = int(args.energy_w2_samples)
    metrics = {
        "reference_data_name": _reference_data_name(args, problem.name),
        "ref_beta0_source": ref0_source,
        "ref_beta1_source": ref1_source,
        "energy_w2_samples": energy_w2_samples,
        "fm_beta0_energy_w2": energy_w2(fm0, ref0, problem),
        "fm_beta0_energy_w2_2k": energy_w2_n(fm0, ref0, problem, n_samples=energy_w2_samples, seed=int(args.seed) + 410),
        "fm_beta1_energy_w2": energy_w2(fm1, ref1, problem),
        "fm_beta1_energy_w2_2k": energy_w2_n(fm1, ref1, problem, n_samples=energy_w2_samples, seed=int(args.seed) + 411),
        "am_beta1_energy_w2": energy_w2(am1, ref1, problem),
        "am_beta1_energy_w2_2k": energy_w2_n(am1, ref1, problem, n_samples=energy_w2_samples, seed=int(args.seed) + 412),
        "fm_beta0_energy_stats": energy_stats(fm0, problem),
        "fm_beta1_energy_stats": energy_stats(fm1, problem),
        "am_beta1_energy_stats": energy_stats(am1, problem),
        "ref_beta0_energy_stats": energy_stats(ref0, problem),
        "ref_beta1_energy_stats": energy_stats(ref1, problem),
        "fm_beta1_marginal_w2": marginal_w2(fm1, ref1),
        "am_beta1_marginal_w2": marginal_w2(am1, ref1),
    }
    pair_fm1 = pairwise_distance_w2(fm1, ref1, problem)
    pair_am1 = pairwise_distance_w2(am1, ref1, problem)
    if pair_fm1 is not None:
        metrics["fm_beta1_pairwise_distance_w2"] = pair_fm1
        metrics["am_beta1_pairwise_distance_w2"] = pair_am1

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    plot_energy_histogram(ref0, fm0, ref1, fm1, am1, problem, run_dir / "energy_histogram.png")
    plot_marginals(ref0, fm0, ref1, fm1, am1, problem, run_dir / "marginals.png")
    fm_loss_path = run_dir / "fm_loss_history.npy"
    am_loss_path = run_dir / "am_loss_history.npy"
    plot_loss_curves(
        np.load(fm_loss_path) if fm_loss_path.exists() else np.asarray([]),
        np.load(am_loss_path) if am_loss_path.exists() else np.asarray([]),
        run_dir / "loss_curves.png",
    )
    if problem.n_particles and problem.spatial_dim:
        plot_pairwise_distances(ref0, fm0, ref1, fm1, am1, problem, run_dir / "pairwise_distances.png")
        plot_pairwise_distances_combined(ref0, fm0, ref1, fm1, am1, problem, run_dir / "pairwise_distances_combined.png")

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"[eval] outputs written to {run_dir}", flush=True)
    return metrics


def eval_fm_betas_stage(
    args: argparse.Namespace,
    eff: dict[str, Any],
    fm_params: Any | None = None,
    params_kind: str = "fm",
) -> dict[str, Any]:
    problem = _make_problem(args)
    run_dir = ensure_dir(_run_dir(args))
    params_kind = str(params_kind).lower()
    if params_kind not in {"fm", "am"}:
        raise ValueError("params_kind must be 'fm' or 'am'.")
    out_dir = ensure_dir(run_dir / f"{params_kind}_beta_sweep")
    sample_label = params_kind.upper()
    if fm_params is None:
        fm_params = load_pickle(run_dir / f"{params_kind}_params.pkl")

    betas = args.eval_betas if args.eval_betas is not None else args.beta_grid
    n_eval = int(eff["n_eval"])
    chunk_size = max(1, int(args.eval_chunk_size))
    energy_w2_samples = int(args.energy_w2_samples)
    metrics_path = out_dir / f"{params_kind}_beta_sweep_metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                all_metrics = json.load(f)
            if not isinstance(all_metrics.get("per_beta"), dict):
                all_metrics["per_beta"] = {}
        except Exception:
            all_metrics = {"per_beta": {}}
    else:
        all_metrics = {"per_beta": {}}
    all_metrics.update(
        {
            "problem": problem.name,
            "n_eval": n_eval,
            "ode_steps": int(eff["ode_steps"]),
            "ode_method": str(eff["ode_method"]),
            "eval_betas": sorted(set([float(beta) for beta in all_metrics.get("eval_betas", [])] + [float(beta) for beta in betas])),
            "energy_w2_samples": energy_w2_samples,
            "track_density": bool(getattr(args, "track_density", False) or getattr(args, "reweight", False)),
            "reweight": bool(getattr(args, "reweight", False)),
            "density_mode": str(getattr(args, "density_mode", "hutchinson")),
            "density_hutchinson_probes": int(getattr(args, "density_hutchinson_probes", 1)),
        }
    )

    for i, beta in enumerate(betas):
        beta = float(beta)
        beta_tag = _file_beta(beta)
        print(f"[eval-{params_kind}-betas] beta={beta:.2f} n_eval={n_eval}", flush=True)
        track_density = bool(getattr(args, "track_density", False) or getattr(args, "reweight", False))
        logq = None
        if track_density:
            fm, logq = _sample_ode_with_logq_chunked(
                fm_params,
                jax.random.PRNGKey(int(args.seed) + 7000 + i),
                problem,
                beta,
                n_eval,
                chunk_size,
                args,
                eff,
                log_prefix=f"eval-{params_kind}-betas",
            )
        else:
            fm = _sample_ode_chunked(
                fm_params,
                jax.random.PRNGKey(int(args.seed) + 7000 + i),
                problem,
                beta,
                n_eval,
                chunk_size,
                args,
                eff,
                log_prefix=f"eval-{params_kind}-betas",
            )
        print(f"[eval-{params_kind}-betas] beta={beta:.2f} loading/generating MD reference", flush=True)
        ref, ref_source = _reference_from_dataset_or_langevin(problem, beta, args, eff, 8000 + i)
        print(f"[eval-{params_kind}-betas] beta={beta:.2f} reference source: {ref_source}", flush=True)

        np.save(out_dir / f"{params_kind}_samples_beta_{_fmt_beta(beta)}.npy", fm)
        np.save(out_dir / f"md_samples_beta_{_fmt_beta(beta)}.npy", ref)
        if logq is not None:
            np.save(out_dir / f"{params_kind}_logq_beta_{_fmt_beta(beta)}.npy", logq)
        print(f"[eval-{params_kind}-betas] beta={beta:.2f} samples saved", flush=True)

        print(f"[eval-{params_kind}-betas] beta={beta:.2f} computing metrics", flush=True)
        pair = pairwise_distance_w2(fm, ref, problem)
        geom = None
        dem_eq = None
        if problem.name in {"dw4", "lj13"} and problem.n_particles and problem.spatial_dim and int(args.geometric_w2_samples) > 0:
            geom = geometric_w2(
                fm,
                ref,
                problem,
                n_samples=int(args.geometric_w2_samples),
                seed=int(args.seed) + 9000 + i,
                cost_chunk_size=int(args.geometric_w2_chunk_size),
            )
        if problem.name in {"dw4", "lj13"} and problem.n_particles and problem.spatial_dim and int(args.dem_eq_emd2_samples) > 0:
            dem_eq = dem_eq_emd2(
                fm,
                ref,
                problem,
                n_samples=int(args.dem_eq_emd2_samples),
                seed=int(args.seed) + 9300 + i,
            )
        if problem.name == "ala2":
            item = {"beta": beta, "reference_source": ref_source}
            item.update(ala2_summary_metrics(fm, ref, problem, energy_w2_samples=energy_w2_samples, seed=int(args.seed) + 9100 + i))
            if geom is not None:
                item["geometric_w2"] = geom
        elif problem.name == "lj13":
            item = {
                "beta": beta,
                "reference_source": ref_source,
                "ew2": energy_w2(fm, ref, problem),
                "ew2_2k": energy_w2_n(fm, ref, problem, n_samples=energy_w2_samples, seed=int(args.seed) + 9100 + i),
                "w2": pair,
                "geometric_w2": geom,
            }
            if dem_eq is not None:
                item["dem_eq_emd2"] = dem_eq
        else:
            item = {
                "beta": beta,
                "reference_source": ref_source,
                f"{params_kind}_energy_w2": energy_w2(fm, ref, problem),
                f"{params_kind}_energy_w2_2k": energy_w2_n(fm, ref, problem, n_samples=energy_w2_samples, seed=int(args.seed) + 9100 + i),
                f"{params_kind}_energy_stats": energy_stats(fm, problem),
                "md_energy_stats": energy_stats(ref, problem),
                f"{params_kind}_marginal_w2": marginal_w2(fm, ref),
            }
            if pair is not None:
                item[f"{params_kind}_pairwise_distance_w2"] = pair
            if geom is not None:
                item[f"{params_kind}_geometric_w2"] = geom
            if dem_eq is not None:
                item[f"{params_kind}_dem_eq_emd2"] = dem_eq

        if bool(getattr(args, "reweight", False)):
            if logq is None:
                raise ValueError("--reweight requires density tracking logq; use --track-density or --reweight.")
            item["reweighting"] = _compute_and_save_reweighting(
                fm,
                ref,
                logq,
                beta,
                problem,
                out_dir,
                params_kind,
                beta_tag,
                _fmt_beta(beta),
                int(args.seed) + 10000 + i,
            )

        all_metrics["per_beta"][_fmt_beta(beta)] = item
        with (out_dir / f"{params_kind}_beta_sweep_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, sort_keys=True)
        print(f"[eval-{params_kind}-betas] beta={beta:.2f} metrics saved", flush=True)

        fm_compare = None
        if params_kind == "am":
            fm_compare_path = run_dir / "fm_beta_sweep" / f"fm_samples_beta_{_fmt_beta(beta)}.npy"
            if fm_compare_path.exists():
                fm_compare = np.load(fm_compare_path)
                if fm_compare.shape[0] != fm.shape[0]:
                    n = min(fm_compare.shape[0], fm.shape[0], ref.shape[0])
                    fm_compare = fm_compare[:n]
                    fm = fm[:n]
                    ref = ref[:n]
                    all_metrics["n_eval_plot"] = int(n)
                print(f"[eval-{params_kind}-betas] beta={beta:.2f} loaded FM comparison from {fm_compare_path}", flush=True)
            else:
                print(f"[eval-{params_kind}-betas] beta={beta:.2f} no FM comparison found at {fm_compare_path}", flush=True)

        print(f"[eval-{params_kind}-betas] beta={beta:.2f} plotting energy overlay", flush=True)
        if fm_compare is not None:
            plot_md_fm_am_energy_overlay(ref, fm_compare, fm, beta, problem, out_dir / f"energy_overlay_beta_{beta_tag}.png")
        else:
            plot_fm_md_energy_overlay(ref, fm, beta, problem, out_dir / f"energy_overlay_beta_{beta_tag}.png", sample_label=sample_label)
        if problem.n_particles and problem.spatial_dim:
            if args.eval_plot_pairwise_individual and problem.name != "lj13":
                print(f"[eval-{params_kind}-betas] beta={beta:.2f} plotting individual pairwise overlays", flush=True)
                plot_fm_md_pairwise_overlay(ref, fm, beta, problem, out_dir / f"pairwise_overlay_beta_{beta_tag}.png")
            print(f"[eval-{params_kind}-betas] beta={beta:.2f} plotting combined pairwise overlay", flush=True)
            if fm_compare is not None:
                plot_md_fm_am_pairwise_combined_overlay(
                    ref,
                    fm_compare,
                    fm,
                    beta,
                    problem,
                    out_dir / f"pairwise_combined_overlay_beta_{beta_tag}.png",
                )
            else:
                plot_fm_md_pairwise_combined_overlay(
                    ref,
                    fm,
                    beta,
                    problem,
                    out_dir / f"pairwise_combined_overlay_beta_{beta_tag}.png",
                    sample_label=sample_label,
                )
        if problem.name == "ala2":
            print(f"[eval-{params_kind}-betas] beta={beta:.2f} plotting Ala2 torsion overlays", flush=True)
            if fm_compare is not None:
                plot_ala2_phi_psi_three_way(ref, fm_compare, fm, beta, out_dir / f"rama_overlay_beta_{beta_tag}.png")
                plot_ala2_phi_psi_marginals_three_way(ref, fm_compare, fm, beta, out_dir / f"torsion_marginals_beta_{beta_tag}.png", am_label=sample_label)
            else:
                plot_ala2_phi_psi_overlay(ref, fm, beta, out_dir / f"rama_overlay_beta_{beta_tag}.png", sample_label=sample_label)
                plot_ala2_phi_psi_marginals(ref, fm, beta, out_dir / f"torsion_marginals_beta_{beta_tag}.png", sample_label=sample_label)
            plot_ala2_energy_pair_overlay(ref, fm, beta, problem, out_dir / f"ala2_energy_pair_overlay_beta_{beta_tag}.png", sample_label=sample_label)
        elif problem.name != "lj13":
            print(f"[eval-{params_kind}-betas] beta={beta:.2f} plotting marginal overlays", flush=True)
            plot_fm_md_marginals_overlay(ref, fm, beta, problem, out_dir / f"marginals_overlay_beta_{beta_tag}.png")
        print(f"[eval-{params_kind}-betas] beta={beta:.2f} done", flush=True)

    with (out_dir / f"{params_kind}_beta_sweep_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)
    print(json.dumps(all_metrics, indent=2, sort_keys=True), flush=True)
    print(f"[eval-{params_kind}-betas] outputs written to {out_dir}", flush=True)
    return all_metrics


def verify_lj13_reference_stage(args: argparse.Namespace, eff: dict[str, Any]) -> dict[str, Any]:
    if args.problem != "lj13":
        raise ValueError("verify-lj13-reference is only defined for --problem lj13.")
    problem = _make_problem(args)
    data_dir = _data_dir_for_args(args, problem.name)
    beta = float(args.beta1 if args.eval_betas is None else args.eval_betas[0])
    path = data_dir / beta_file_name(beta)
    if not path.exists():
        raise FileNotFoundError(f"Missing LJ13 reference dataset: {path}")
    arr = _finite_rows_np(np.load(path))
    n = min(int(args.reference_check_samples), arr.shape[0])
    if n <= 0:
        raise ValueError(f"No finite samples found in {path}")
    rng = np.random.default_rng(int(args.seed) + 17001)
    samples = arr[rng.choice(arr.shape[0], size=n, replace=False)]

    e_jax = np.asarray(jax.device_get(problem.energy_fn(samples)), dtype=np.float64)

    # Cross-check the closed-form positive energy U = 2 * LJ + oscillator.
    try:
        import torch

        x = torch.tensor(samples.reshape((n, 13, 3)), dtype=torch.float64)
        dists = torch.vmap(torch.pdist)(x)
        lj = ((1.0 / dists) ** 12 - 2.0 * (1.0 / dists) ** 6).sum(dim=-1)
        centered = x - torch.mean(x, dim=1, keepdim=True)
        osc = 0.5 * centered.pow(2).sum(dim=(-2, -1))
        e_reference = (2.0 * lj + osc).detach().cpu().numpy()
        backend = "torch"
    except Exception as exc:
        coords = samples.reshape((n, 13, 3)).astype(np.float64)
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dmat = np.sqrt(np.sum(diff * diff, axis=-1))
        iu = np.triu_indices(13, k=1)
        dists = dmat[:, iu[0], iu[1]]
        lj = ((1.0 / dists) ** 12 - 2.0 * (1.0 / dists) ** 6).sum(axis=-1)
        centered = coords - np.mean(coords, axis=1, keepdims=True)
        osc = 0.5 * np.sum(centered * centered, axis=(1, 2))
        e_reference = 2.0 * lj + osc
        backend = f"numpy_fallback:{type(exc).__name__}"

    diff = e_jax - np.asarray(e_reference, dtype=np.float64)
    out = {
        "problem": problem.name,
        "data_name": _data_name(args, problem.name),
        "beta": beta,
        "dataset": str(path),
        "n_samples": int(n),
        "reference_energy_backend": backend,
        "jax_energy_mean": float(np.mean(e_jax)),
        "reference_energy_mean": float(np.mean(e_reference)),
        "max_abs_energy_diff": float(np.max(np.abs(diff))),
        "mean_abs_energy_diff": float(np.mean(np.abs(diff))),
        "rmse_energy_diff": float(np.sqrt(np.mean(diff * diff))),
    }
    out_path = data_dir / f"energy_check_beta_{_fmt_beta(beta)}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    print(f"[verify-lj13-reference] wrote {out_path}", flush=True)
    return out


def reference_geow2_stage(args: argparse.Namespace, eff: dict[str, Any]) -> dict[str, Any]:
    problem = _make_problem(args)
    if not problem.n_particles or not problem.spatial_dim:
        raise ValueError("reference-geow2 requires a particle problem with n_particles and spatial_dim.")
    data_dir = _data_dir_for_args(args, problem.name)
    betas = args.eval_betas if args.eval_betas is not None else args.beta_grid
    n = int(args.reference_check_samples)
    repeats = int(args.ref_split_repeats)
    results: dict[str, Any] = {
        "problem": problem.name,
        "data_name": _data_name(args, problem.name),
        "n_samples": n,
        "repeats": repeats,
        "per_beta": {},
    }
    for beta in betas:
        beta = float(beta)
        path = data_dir / beta_file_name(beta)
        if not path.exists():
            raise FileNotFoundError(f"Missing reference dataset: {path}")
        arr = _finite_rows_np(np.load(path))
        if arr.shape[0] < 2 * n:
            raise ValueError(f"{path} has {arr.shape[0]} finite rows, need at least {2 * n} for disjoint random split.")
        beta_rows = []
        for rep in range(repeats):
            rng = np.random.default_rng(int(args.seed) + 19001 + rep * 1009 + int(round(beta * 1000)))
            idx = rng.choice(arr.shape[0], size=2 * n, replace=False)
            ref_a = np.asarray(arr[idx[:n]], dtype=np.float32)
            ref_b = np.asarray(arr[idx[n:]], dtype=np.float32)
            top_k = 1 if problem.name == "lj13" else 32
            geo = geometric_w2(
                ref_a,
                ref_b,
                problem,
                n_samples=n,
                seed=int(args.seed) + 20001 + rep,
                cost_chunk_size=int(args.geometric_w2_chunk_size),
                dem_refine_top_k=top_k,
            )
            ew = energy_w2_n(ref_a, ref_b, problem, n_samples=n, seed=int(args.seed) + 21001 + rep)
            pair = pairwise_distance_w2(ref_a, ref_b, problem)
            row = {
                "repeat": rep,
                "geometric_w2": geo,
                "energy_w2": ew,
                "pairwise_distance_w2": pair,
            }
            beta_rows.append(row)
            print(
                f"[reference-geow2] beta={beta:.2f} repeat={rep} "
                f"geow2={geo:.6g} ew2={ew:.6g} pairw2={pair if pair is None else f'{pair:.6g}'}",
                flush=True,
            )
        geos = [float(r["geometric_w2"]) for r in beta_rows if r["geometric_w2"] is not None]
        ews = [float(r["energy_w2"]) for r in beta_rows if r["energy_w2"] is not None]
        pairs = [float(r["pairwise_distance_w2"]) for r in beta_rows if r["pairwise_distance_w2"] is not None]
        results["per_beta"][_fmt_beta(beta)] = {
            "dataset": str(path),
            "finite_rows": int(arr.shape[0]),
            "runs": beta_rows,
            "geometric_w2_mean": float(np.mean(geos)) if geos else None,
            "geometric_w2_std": float(np.std(geos)) if geos else None,
            "energy_w2_mean": float(np.mean(ews)) if ews else None,
            "pairwise_distance_w2_mean": float(np.mean(pairs)) if pairs else None,
        }
    out_path = data_dir / f"reference_geow2_random_split_n{n}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    print(f"[reference-geow2] wrote {out_path}", flush=True)
    return results


def merge_lj13_shards_stage(args: argparse.Namespace, eff: dict[str, Any]) -> dict[str, Any]:
    if args.problem != "lj13":
        raise ValueError("merge-lj13-shards is only defined for --problem lj13.")
    problem = _make_problem(args)
    out_dir = ensure_dir(_data_dir_for_args(args, problem.name))
    shard_dir = Path(args.shard_dir) if args.shard_dir else ROOT / "data" / f"{_data_name(args, problem.name)}_shards"
    if not shard_dir.is_absolute():
        shard_dir = ROOT / shard_dir
    if not shard_dir.exists():
        raise FileNotFoundError(f"Shard directory does not exist: {shard_dir}")

    betas = args.eval_betas if args.eval_betas is not None else args.beta_grid
    results: dict[str, Any] = {
        "problem": problem.name,
        "data_name": _data_name(args, problem.name),
        "output_dir": str(out_dir),
        "shard_dir": str(shard_dir),
        "per_beta": {},
    }

    for beta in betas:
        beta = float(beta)
        tag = _fmt_beta(beta)
        file_tag = _file_beta(beta)
        patterns = []
        if args.shard_glob:
            patterns.append(str(args.shard_glob).format(beta=tag, beta_file=file_tag))
        patterns.extend(
            [
                f"*{tag}*.npy",
                f"*{file_tag}*.npy",
                f"samples_beta_{tag}_*.npy",
                f"samples_beta_{tag}.npy",
            ]
        )
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(sorted(shard_dir.glob(pattern)))
        # Preserve order while removing duplicates.
        seen: set[Path] = set()
        unique_paths = []
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                unique_paths.append(path)
                seen.add(resolved)
        if not unique_paths and len(list(shard_dir.glob("*.npy"))) == 1:
            unique_paths = list(shard_dir.glob("*.npy"))
        if not unique_paths:
            raise FileNotFoundError(
                f"No shard .npy files found for beta={tag} in {shard_dir}. "
                "Use --shard-glob to specify a pattern."
            )

        pieces = []
        shard_rows = []
        for path in unique_paths:
            raw = np.load(path, allow_pickle=True)
            before = int(raw.reshape((raw.shape[0], -1)).shape[0])
            arr = np.asarray(raw, dtype=np.float32).reshape((-1, problem.dim))
            arr = _finite_rows_np(arr)
            finite = int(arr.shape[0])
            if finite:
                coords = arr.reshape((-1, problem.n_particles, problem.spatial_dim))
                coords = coords - np.mean(coords, axis=1, keepdims=True)
                arr = coords.reshape((-1, problem.dim)).astype(np.float32, copy=False)
                pieces.append(arr)
            shard_rows.append({"file": str(path), "raw_rows": before, "finite_rows": finite})
            print(f"[merge-lj13-shards] beta={tag} {path} raw={before} finite={finite}", flush=True)

        if not pieces:
            raise ValueError(f"No finite rows found in shards for beta={tag}")
        merged = np.concatenate(pieces, axis=0)
        rng = np.random.default_rng(int(args.seed) + int(round(beta * 1000)) + 23001)
        order = rng.permutation(merged.shape[0])
        merged = merged[order]
        max_samples = int(args.merge_max_samples)
        if max_samples > 0:
            merged = merged[:max_samples]
        out_path = out_dir / beta_file_name(beta)
        np.save(out_path, merged.astype(np.float32, copy=False))
        item = {
            "beta": beta,
            "output": str(out_path),
            "output_shape": list(merged.shape),
            "n_shards": len(unique_paths),
            "raw_rows": int(sum(row["raw_rows"] for row in shard_rows)),
            "finite_rows_before_cap": int(sum(row["finite_rows"] for row in shard_rows)),
            "max_samples": max_samples,
            "shards": shard_rows,
        }
        results["per_beta"][tag] = item
        print(f"[merge-lj13-shards] beta={tag} wrote {out_path} shape={merged.shape}", flush=True)

    manifest_path = out_dir / "merge_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    print(f"[merge-lj13-shards] wrote {manifest_path}", flush=True)
    return results


def _load_reference_dataset_for_score(problem, beta: float, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    path = _reference_data_dir_for_args(args, problem.name) / beta_file_name(beta)
    if not path.exists():
        raise FileNotFoundError(f"Missing reference dataset for beta={float(beta):.2f}: {path}")
    arr = _finite_rows_np(np.load(path))
    if arr.shape[0] <= 0:
        raise ValueError(f"No finite reference rows in {path}")
    return arr, str(path)


def _candidate_sample_paths(run_dir: Path, kind: str, beta: float) -> list[Path]:
    tag = _fmt_beta(beta)
    return [
        run_dir / f"{kind}_beta_sweep" / f"{kind}_samples_beta_{tag}.npy",
        run_dir / f"{kind}_samples_beta_{tag}.npy",
    ]


def _score_reference_baseline(problem, ref: np.ndarray, beta: float, args: argparse.Namespace, beta_i: int) -> dict[str, Any] | None:
    repeats = int(args.score_ref_baseline_repeats)
    if repeats <= 0:
        return None
    n = min(int(args.geometric_w2_samples), int(args.energy_w2_samples))
    if n <= 0:
        raise ValueError("--score-ref-baseline-repeats requires positive --geometric-w2-samples and --energy-w2-samples.")
    if ref.shape[0] < 2 * n:
        raise ValueError(
            f"Reference beta={float(beta):.2f} has {ref.shape[0]} rows, need at least {2 * n} "
            "for disjoint reference baseline."
        )

    rows = []
    for rep in range(repeats):
        rng = np.random.default_rng(int(args.seed) + 34001 + beta_i * 1009 + rep)
        idx = rng.choice(ref.shape[0], size=2 * n, replace=False)
        ref_a = np.asarray(ref[idx[:n]], dtype=np.float32)
        ref_b = np.asarray(ref[idx[n:]], dtype=np.float32)
        geom = None
        if problem.name in {"dw4", "lj13"} and problem.n_particles and problem.spatial_dim:
            top_k = 1 if problem.name == "lj13" else 32
            geom = geometric_w2(
                ref_a,
                ref_b,
                problem,
                n_samples=n,
                seed=int(args.seed) + 35001 + beta_i * 1009 + rep,
                cost_chunk_size=int(args.geometric_w2_chunk_size),
                dem_refine_top_k=top_k,
            )
        ew = energy_w2_n(ref_a, ref_b, problem, n_samples=n, seed=int(args.seed) + 36001 + beta_i * 1009 + rep)
        pair = pairwise_distance_w2(ref_a, ref_b, problem)
        rows.append(
            {
                "repeat": rep,
                "n_samples": int(n),
                "geometric_w2": geom,
                "ew2_n": ew,
                "pairwise_distance_w2": pair,
            }
        )
        print(
            f"[score-samples] beta={beta:.2f} ref-baseline repeat={rep} "
            f"geow2={geom} ew2_n={ew:.6g} pairw2={pair}",
            flush=True,
        )

    geos = [float(row["geometric_w2"]) for row in rows if row["geometric_w2"] is not None]
    ews = [float(row["ew2_n"]) for row in rows if row["ew2_n"] is not None]
    pairs = [float(row["pairwise_distance_w2"]) for row in rows if row["pairwise_distance_w2"] is not None]
    return {
        "n_samples": int(n),
        "repeats": repeats,
        "runs": rows,
        "geometric_w2_mean": float(np.mean(geos)) if geos else None,
        "geometric_w2_std": float(np.std(geos)) if geos else None,
        "ew2_n_mean": float(np.mean(ews)) if ews else None,
        "ew2_n_std": float(np.std(ews)) if ews else None,
        "pairwise_distance_w2_mean": float(np.mean(pairs)) if pairs else None,
    }


def score_samples_stage(args: argparse.Namespace, eff: dict[str, Any]) -> dict[str, Any]:
    """Score already-saved model samples against reference datasets.

    This avoids re-running ODE sampling when only the reference set or metric
    implementation changes.  It is the preferred path for fast benchmark
    re-scoring after generating aligned LJ13/DW4 reference data.
    """
    problem = _make_problem(args)
    run_dir = ensure_dir(_run_dir(args))
    betas = args.eval_betas if args.eval_betas is not None else args.beta_grid
    kinds = [str(k).lower() for k in args.score_kinds]
    invalid = [k for k in kinds if k not in {"fm", "am"}]
    if invalid:
        raise ValueError(f"--score-kinds only supports fm/am, got {invalid}")

    out: dict[str, Any] = {
        "problem": problem.name,
        "data_name": _data_name(args, problem.name),
        "reference_data_name": _reference_data_name(args, problem.name),
        "run_dir": str(run_dir),
        "eval_betas": [float(beta) for beta in betas],
        "energy_w2_samples": int(args.energy_w2_samples),
        "geometric_w2_samples": int(args.geometric_w2_samples),
        "score_kinds": kinds,
        "per_beta": {},
    }

    for beta_i, beta_raw in enumerate(betas):
        beta = float(beta_raw)
        beta_key = _fmt_beta(beta)
        ref, ref_source = _load_reference_dataset_for_score(problem, beta, args)
        beta_item: dict[str, Any] = {
            "beta": beta,
            "reference_source": ref_source,
            "reference_rows": int(ref.shape[0]),
        }
        baseline = _score_reference_baseline(problem, ref, beta, args, beta_i)
        if baseline is not None:
            beta_item["reference_baseline"] = baseline
        for kind in kinds:
            sample_path = next((p for p in _candidate_sample_paths(run_dir, kind, beta) if p.exists()), None)
            if sample_path is None:
                beta_item[kind] = {
                    "status": "missing",
                    "candidate_paths": [str(p) for p in _candidate_sample_paths(run_dir, kind, beta)],
                }
                print(f"[score-samples] beta={beta:.2f} kind={kind} missing saved samples", flush=True)
                continue

            samples = _finite_rows_np(np.load(sample_path))
            if samples.shape[0] <= 0:
                beta_item[kind] = {"status": "empty", "sample_path": str(sample_path)}
                print(f"[score-samples] beta={beta:.2f} kind={kind} no finite rows", flush=True)
                continue

            print(
                f"[score-samples] beta={beta:.2f} kind={kind} "
                f"samples={samples.shape[0]} ref={ref.shape[0]}",
                flush=True,
            )
            pair = pairwise_distance_w2(samples, ref, problem)
            geom = None
            dem_eq = None
            if problem.name in {"dw4", "lj13"} and problem.n_particles and problem.spatial_dim and int(args.geometric_w2_samples) > 0:
                top_k = 1 if problem.name == "lj13" else 32
                geom = geometric_w2(
                    samples,
                    ref,
                    problem,
                    n_samples=int(args.geometric_w2_samples),
                    seed=int(args.seed) + 31001 + beta_i,
                    cost_chunk_size=int(args.geometric_w2_chunk_size),
                    dem_refine_top_k=top_k,
                )
            if problem.name in {"dw4", "lj13"} and problem.n_particles and problem.spatial_dim and int(args.dem_eq_emd2_samples) > 0:
                dem_eq = dem_eq_emd2(
                    samples,
                    ref,
                    problem,
                    n_samples=int(args.dem_eq_emd2_samples),
                    seed=int(args.seed) + 32001 + beta_i,
                )

            item = {
                "status": "ok",
                "sample_path": str(sample_path),
                "sample_rows": int(samples.shape[0]),
                "ew2": energy_w2(samples, ref, problem),
                "ew2_n": energy_w2_n(
                    samples,
                    ref,
                    problem,
                    n_samples=int(args.energy_w2_samples),
                    seed=int(args.seed) + 33001 + beta_i,
                ),
                "pairwise_distance_w2": pair,
                "geometric_w2": geom,
            }
            if dem_eq is not None:
                item["dem_eq_emd2"] = dem_eq
            beta_item[kind] = item
            print(
                f"[score-samples] beta={beta:.2f} kind={kind} "
                f"geow2={geom} ew2_n={item['ew2_n']:.6g} pairw2={pair}",
                flush=True,
            )
        out["per_beta"][beta_key] = beta_item

    safe_data_name = _reference_data_name(args, problem.name).replace("/", "_")
    safe_kinds = "_".join(kinds)
    beta_tag = "_".join(_file_beta(float(beta)) for beta in betas)
    out_path = run_dir / (
        f"score_samples_metrics_{safe_data_name}_{safe_kinds}_"
        f"betas_{beta_tag}_geo{int(args.geometric_w2_samples)}_ew{int(args.energy_w2_samples)}.json"
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    print(f"[score-samples] wrote {out_path}", flush=True)
    return out


def run_all(args: argparse.Namespace, eff: dict[str, Any]) -> None:
    generate_data(args, eff)
    fm_params = train_fm_stage(args, eff)
    am_params = train_am_stage(args, eff, base_params=fm_params)
    sample_eval_stage(args, eff, fm_params=fm_params, am_params=am_params)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-beta Flow Matching + Adjoint Matching toy suite")
    parser.add_argument(
        "stage",
        choices=[
            "generate-data",
            "train-fm",
            "train-am",
            "sample-eval",
            "eval-fm-betas",
            "eval-am-betas",
            "verify-lj13-reference",
            "reference-geow2",
            "merge-lj13-shards",
            "score-samples",
            "run-all",
        ],
    )
    parser.add_argument("--problem", choices=["dw1d", "dw2d", "mb2d", "dw4", "lj13", "ala2"], default="dw1d")
    parser.add_argument("--data-name", type=str, default=None, help="Training dataset directory under data/. Defaults to --problem.")
    parser.add_argument(
        "--reference-data-name",
        type=str,
        default=None,
        help=(
            "Held-out evaluation dataset directory under data/. Defaults to "
            "--data-name. When set explicitly, missing or undersized reference "
            "files are fatal instead of falling back to on-the-fly Langevin."
        ),
    )
    parser.add_argument("--model", choices=["mlp", "painn", "egnn", "ala2_cpainn", "ala2_chiro_cpainn", "ala2_chiro_painn_jax"], default="mlp")
    parser.add_argument(
        "--painn-backend",
        choices=["jax", "amboltz_aligned"],
        default="jax",
        help="PaiNN implementation backend. 'jax' keeps existing painn_jax; 'amboltz_aligned' uses the old amboltz hidden-state PaiNN with beta embedding.",
    )
    parser.add_argument("--beta-grid", type=float, nargs="+", default=DEFAULT_BETA_GRID)
    parser.add_argument("--eval-betas", type=float, nargs="+", default=None)
    parser.add_argument("--beta0", type=float, default=1.25)
    parser.add_argument("--beta1", type=float, default=1.30)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-per-beta", type=int, default=100000)
    parser.add_argument("--n-eval", type=int, default=5000)
    parser.add_argument("--langevin-steps", type=int, default=5000)
    parser.add_argument("--langevin-step-size", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--init-mode", choices=["normal", "uniform"], default="normal")
    parser.add_argument("--lj13-sampler-kernel", choices=["mala", "ula", "langevin"], default="mala")
    parser.add_argument("--lj13-warmup-steps", type=int, default=2000)
    parser.add_argument("--lj13-init-file", type=str, default="${AMBOLTZ_ROOT}/data/LJ13/train_split_LJ13-1000.npy")
    parser.add_argument("--lj13-n-chains", type=int, default=1024)
    parser.add_argument("--lj13-init-step-size", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--am-batch-size", type=int, default=512)
    parser.add_argument("--fm-steps", type=int, default=30000)
    parser.add_argument("--fm-lr", type=float, default=1.0e-3)
    parser.add_argument("--fm-lr-min", type=float, default=1.0e-5)
    parser.add_argument("--fm-samples-per-beta", type=int, default=0, help="Subsample this many finite training rows per beta before stacking FM datasets. 0 keeps all rows up to common length.")
    parser.add_argument("--fm-eval-every", type=int, default=0, help="Online FM quick eval interval. 0 disables it.")
    parser.add_argument("--fm-eval-samples", type=int, default=256, help="Samples per beta for online FM quick eval.")
    parser.add_argument("--fm-eval-steps", type=int, default=150, help="ODE steps for online FM quick eval.")
    parser.add_argument("--fm-eval-chunk-size", type=int, default=0, help="Chunk size for online FM quick eval sampling. 0 uses a safe default for heavy systems.")
    parser.add_argument("--fm-eval-betas", type=float, nargs="+", default=None, help="Betas for online FM quick eval. Defaults to beta-grid.")
    parser.add_argument("--fm-eval-method", choices=["euler", "heun"], default=None, help="ODE method for online FM quick eval. Defaults to --ode-method, which defaults to euler.")
    parser.add_argument("--save-fm-checkpoints", type=int, nargs="*", default=None)
    parser.add_argument("--am-steps", type=int, default=10000)
    parser.add_argument("--am-lr", type=float, default=2.0e-5)
    parser.add_argument("--am-lr-min", type=float, default=2.0e-7)
    parser.add_argument("--K", type=int, default=40)
    parser.add_argument("--am-num-loss-steps", type=int, default=20)
    parser.add_argument("--am-keep-last-steps", type=int, default=10)
    parser.add_argument("--save-am-checkpoints", type=int, nargs="*", default=None)
    parser.add_argument("--ode-steps", type=int, default=150)
    parser.add_argument("--ode-method", choices=["euler", "heun"], default=None, help="ODE sampler method. Defaults to euler.")
    parser.add_argument("--eval-chunk-size", type=int, default=10000)
    parser.add_argument("--track-density", action="store_true", help="Track proposal log density during eval ODE rollout and save logq arrays.")
    parser.add_argument("--density-mode", choices=["hutchinson"], default="hutchinson", help="Divergence estimator for density tracking.")
    parser.add_argument("--density-hutchinson-probes", type=int, default=1, help="Number of Hutchinson probes per ODE step for density tracking.")
    parser.add_argument("--reweight", action="store_true", help="Compute OpenMM/Boltzmann importance weights from tracked logq. Implies --track-density.")
    parser.add_argument("--eval-plot-pairwise-individual", action="store_true")
    parser.add_argument("--energy-w2-samples", type=int, default=2000)
    parser.add_argument("--geometric-w2-samples", type=int, default=2000)
    parser.add_argument("--geometric-w2-chunk-size", type=int, default=16)
    parser.add_argument("--dem-eq-emd2-samples", type=int, default=0)
    parser.add_argument("--reference-check-samples", type=int, default=2000)
    parser.add_argument("--ref-split-repeats", type=int, default=3)
    parser.add_argument("--shard-dir", type=str, default=None)
    parser.add_argument("--shard-glob", type=str, default=None)
    parser.add_argument("--merge-max-samples", type=int, default=0)
    parser.add_argument("--score-kinds", type=str, nargs="+", default=["fm", "am"], help="Saved sample kinds to score: fm and/or am.")
    parser.add_argument("--score-ref-baseline-repeats", type=int, default=0, help="Optional disjoint reference-vs-reference baseline repeats for score-samples.")
    parser.add_argument("--prior-scale", type=float, default=None)
    parser.add_argument("--max-sigma", type=float, default=50.0)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--adam-b1", type=float, default=0.9, help="Adam beta1 for AM training. Defaults to the existing ADTM value 0.9.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW-style weight decay for AM training. Defaults to the existing ADTM value 0.0.")
    parser.add_argument("--energy-grad-scale", type=float, default=1.0)
    parser.add_argument("--ala2-openmm-device", choices=["cpu", "cuda"], default=os.environ.get("ADTM_ALA2_OPENMM_DEVICE", "cpu"))
    parser.add_argument("--ala2-openmm-system", choices=["alanine_dipeptide_implicit", "amber99sbildn_obc1_xml"], default=os.environ.get("ADTM_ALA2_OPENMM_SYSTEM", "amber99sbildn_obc1_xml"))
    parser.add_argument("--ala2-openmm-grad-clip", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--t-embed-dim", type=int, default=16)
    parser.add_argument("--beta-embed-dim", type=int, default=8)
    parser.add_argument("--painn-atom-identity", choices=["auto", "learned", "none"], default="auto")
    parser.add_argument("--painn-atom-embed-dim", type=int, default=16)
    parser.add_argument("--ala2-temp-embed-min", type=float, default=300.0, help="Minimum Kelvin temperature for Ala2 TI-style normalized temperature conditioning.")
    parser.add_argument("--ala2-temp-embed-max", type=float, default=1000.0, help="Maximum Kelvin temperature for Ala2 TI-style normalized temperature conditioning.")
    parser.add_argument("--ala2-temp-embed-l0", type=float, default=75.0, help="Thermodynamic Interpolation positional-embedding length scale for Ala2 cPaiNN models.")
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    eff = _effective(args)
    args.prior_scale = float(eff["prior_scale"])
    args.grad_clip = float(eff["grad_clip"])
    args.fm_eval_chunk_size = int(eff["fm_eval_chunk_size"])
    args.ode_method = str(eff["ode_method"])
    args.fm_eval_method = str(eff["fm_eval_method"])
    if args.stage == "generate-data":
        generate_data(args, eff)
    elif args.stage == "train-fm":
        train_fm_stage(args, eff)
    elif args.stage == "train-am":
        train_am_stage(args, eff)
    elif args.stage == "sample-eval":
        sample_eval_stage(args, eff)
    elif args.stage == "eval-fm-betas":
        eval_fm_betas_stage(args, eff)
    elif args.stage == "eval-am-betas":
        eval_fm_betas_stage(args, eff, params_kind="am")
    elif args.stage == "verify-lj13-reference":
        verify_lj13_reference_stage(args, eff)
    elif args.stage == "reference-geow2":
        reference_geow2_stage(args, eff)
    elif args.stage == "merge-lj13-shards":
        merge_lj13_shards_stage(args, eff)
    elif args.stage == "score-samples":
        score_samples_stage(args, eff)
    elif args.stage == "run-all":
        run_all(args, eff)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
