from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.langevin import beta_file_name
from adj_thermo.model import apply_mlp, init_mlp
from adj_thermo.model.egnn import apply_egnn_velocity_params, init_egnn_velocity
from adj_thermo.model.painn_velocity import (
    apply_painn_velocity_params,
    init_painn_velocity,
    painn_trainable_params_from_package,
)
from adj_thermo.model.painn_ala2_chiro_jax import (
    ala2_chiro_painn_jax_trainable_params_from_package,
    apply_ala2_chiro_painn_jax_velocity_params,
    init_ala2_chiro_painn_jax_velocity,
    split_ala2_chiro_painn_jax_trainable_params,
)
from adj_thermo.optim import AdamState, adam_update, clip_grads, init_adam
from adj_thermo.problem.base import ProblemSpec
from adj_thermo.utils import ensure_dir, save_pickle

Params = Any
FmTrainStep = Callable[[Params, AdamState, jax.Array, jax.Array, jax.Array, jax.Array], tuple[Params, AdamState, jax.Array]]


def _stack_datasets(datasets: dict[float, Any], beta_grid: list[float] | tuple[float, ...]) -> tuple[jax.Array, jax.Array]:
    betas = jnp.asarray([float(b) for b in beta_grid], dtype=jnp.float32)
    arrays = [jnp.asarray(datasets[float(b)], dtype=jnp.float32) for b in beta_grid]
    return betas, jnp.stack(arrays, axis=0)


def load_beta_datasets(
    data_dir: str | Path,
    beta_grid: list[float] | tuple[float, ...],
    samples_per_beta: int = 0,
    seed: int = 0,
) -> dict[float, np.ndarray]:
    data_dir = Path(data_dir)
    samples_per_beta = int(samples_per_beta or 0)
    out: dict[float, np.ndarray] = {}
    for beta_i, beta in enumerate(beta_grid):
        path = data_dir / beta_file_name(beta)
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset for beta={beta}: {path}")
        arr = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        flat = arr.reshape((arr.shape[0], -1))
        finite = np.isfinite(flat).all(axis=1)
        if not bool(np.all(finite)):
            dropped = int(arr.shape[0] - int(np.sum(finite)))
            print(f"[FM] beta={float(beta):.2f} dropped {dropped} non-finite rows from {path}", flush=True)
            arr = arr[finite]
        if arr.shape[0] == 0:
            raise ValueError(f"No finite rows in dataset for beta={beta}: {path}")
        if samples_per_beta > 0:
            if arr.shape[0] < samples_per_beta:
                raise ValueError(
                    f"Requested --fm-samples-per-beta={samples_per_beta}, "
                    f"but beta={float(beta):.8f} only has {arr.shape[0]} finite rows."
                )
            rng = np.random.default_rng(int(seed) + 1009 * int(beta_i) + 17)
            idx = rng.choice(arr.shape[0], size=samples_per_beta, replace=False)
            arr = arr[np.sort(idx)]
            print(
                f"[FM] beta={float(beta):.8f} subsampled {samples_per_beta}/{flat.shape[0]} finite rows "
                f"with seed={int(seed) + 1009 * int(beta_i) + 17}",
                flush=True,
            )
        out[float(beta)] = arr
    common_n = min(arr.shape[0] for arr in out.values())
    if any(arr.shape[0] != common_n for arr in out.values()):
        print(f"[FM] truncating beta datasets to common finite length {common_n}", flush=True)
        out = {beta: arr[:common_n] for beta, arr in out.items()}
    return out


def _fm_loss_from_stacked(
    params: Params,
    key: jax.Array,
    betas: jax.Array,
    stacked_data: jax.Array,
    problem: ProblemSpec,
    batch_size: int,
    prior_scale: float | jax.Array,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    model_type: str = "mlp",
    painn_backend: str = "jax",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
) -> jax.Array:
    key_beta, key_idx, key_z, key_t = jax.random.split(key, 4)
    beta_idx = jax.random.randint(key_beta, (int(batch_size),), minval=0, maxval=stacked_data.shape[0])
    beta = betas[beta_idx][:, None]
    idx = jax.random.randint(key_idx, (int(batch_size),), minval=0, maxval=stacked_data.shape[1])
    x1 = problem.project_fn(stacked_data[beta_idx, idx, :])
    z = jnp.asarray(prior_scale, dtype=x1.dtype) * jax.random.normal(key_z, x1.shape, dtype=x1.dtype)
    z = problem.project_fn(z)
    t = jax.random.uniform(key_t, (int(batch_size), 1), dtype=x1.dtype)
    x_t = problem.project_fn((1.0 - t) * z + t * x1)
    u_t = problem.project_fn(x1 - z)
    if model_type == "ala2_chiro_painn_jax":
        if painn_model is None or painn_state is None or painn_config is None:
            raise ValueError("Ala2 chiral PaiNN FM loss requires model, state, and config.")
        pred = apply_ala2_chiro_painn_jax_velocity_params(
            painn_model, params, painn_state, x_t, t, beta, problem, painn_config
        )
    elif model_type == "painn":
        if painn_config is None:
            raise ValueError("PaiNN FM loss requires painn_config.")
        if painn_model is None or painn_state is None:
            raise ValueError("JAX PaiNN FM loss requires painn_model and painn_state.")
        pred = apply_painn_velocity_params(painn_model, params, painn_state, x_t, t, beta, problem, painn_config)
    elif model_type == "egnn":
        if egnn_config is None:
            raise ValueError("EGNN FM loss requires egnn_config.")
        pred = apply_egnn_velocity_params(params, x_t, t, beta, problem, egnn_config)
    else:
        pred = apply_mlp(params, x_t, t, beta, problem, t_embed_dim=t_embed_dim, beta_embed_dim=beta_embed_dim)
    return jnp.mean((pred - u_t) ** 2)


def fm_loss_fn(
    params: Params,
    key: jax.Array,
    datasets: dict[float, Any],
    beta_grid: list[float] | tuple[float, ...],
    problem: ProblemSpec,
    batch_size: int,
    prior_scale: float,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
) -> jax.Array:
    betas, stacked_data = _stack_datasets(datasets, beta_grid)
    return _fm_loss_from_stacked(
        params,
        key,
        betas,
        stacked_data,
        problem,
        batch_size,
        prior_scale,
        t_embed_dim=t_embed_dim,
        beta_embed_dim=beta_embed_dim,
    )


def _make_fm_train_step(
    problem: ProblemSpec,
    batch_size: int,
    prior_scale: float,
    t_embed_dim: int,
    beta_embed_dim: int,
    grad_clip: float,
    model_type: str = "mlp",
    painn_backend: str = "jax",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1.0e-8,
    weight_decay: float = 0.0,
) -> FmTrainStep:
    def train_step(
        params: Params,
        opt_state: AdamState,
        key: jax.Array,
        betas: jax.Array,
        stacked_data: jax.Array,
        lr: jax.Array,
    ) -> tuple[Params, AdamState, jax.Array]:
        loss, grads = jax.value_and_grad(
            lambda p: _fm_loss_from_stacked(
                p,
                key,
                betas,
                stacked_data,
                problem,
                batch_size,
                prior_scale,
                t_embed_dim=t_embed_dim,
                beta_embed_dim=beta_embed_dim,
                model_type=model_type,
                painn_backend=painn_backend,
                painn_model=painn_model,
                painn_state=painn_state,
                painn_config=painn_config,
                egnn_config=egnn_config,
            ),
            allow_int=True,
        )(params)
        grads = clip_grads(grads, grad_clip)
        params, opt_state = adam_update(
            params,
            grads,
            opt_state,
            lr=lr,
            beta1=float(adam_beta1),
            beta2=float(adam_beta2),
            eps=float(adam_eps),
            weight_decay=float(weight_decay),
        )
        return params, opt_state, loss

    return jax.jit(train_step)


def linear_decay_lr(step: int, total_steps: int, lr: float, min_lr: float) -> float:
    if total_steps <= 1:
        return float(min_lr)
    frac = min(1.0, max(0.0, float(step - 1) / float(total_steps - 1)))
    return float(min_lr) + (float(lr) - float(min_lr)) * (1.0 - frac)


def train_fm(
    problem: ProblemSpec,
    datasets: dict[float, Any],
    beta_grid: list[float] | tuple[float, ...],
    run_dir: str | Path,
    key: jax.Array,
    steps: int = 30000,
    batch_size: int = 512,
    lr: float = 1.0e-3,
    lr_min: float = 1.0e-5,
    prior_scale: float = 1.0,
    hidden_dim: int = 128,
    n_layers: int = 3,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    grad_clip: float = 1.0,
    log_every: int = 100,
    config: dict[str, Any] | None = None,
    model_type: str = "mlp",
    fm_eval_every: int = 0,
    fm_eval_samples: int = 256,
    fm_eval_steps: int = 150,
    fm_eval_chunk_size: int = 0,
    fm_eval_betas: list[float] | tuple[float, ...] | None = None,
    fm_eval_method: str = "euler",
    painn_backend: str = "jax",
    painn_atom_identity: str = "auto",
    painn_atom_embed_dim: int = 16,
    ala2_temp_embed_min: float = 300.0,
    ala2_temp_embed_max: float = 1000.0,
    ala2_temp_embed_l0: float = 75.0,
    save_checkpoints: list[int] | tuple[int, ...] | None = None,
) -> tuple[Params, np.ndarray]:
    run_dir = ensure_dir(run_dir)
    key, init_key = jax.random.split(key)
    painn_model = None
    painn_state = None
    painn_config = None
    egnn_config = None
    if model_type == "ala2_chiro_painn_jax":
        package, painn_model = init_ala2_chiro_painn_jax_velocity(
            init_key,
            problem,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            t_embed_dim=t_embed_dim,
            temp_embed_dim=beta_embed_dim,
            atom_embed_dim=painn_atom_embed_dim,
            temp_min=ala2_temp_embed_min,
            temp_max=ala2_temp_embed_max,
            temp_embed_l0=ala2_temp_embed_l0,
        )
        params = ala2_chiro_painn_jax_trainable_params_from_package(package)
        painn_state = package["state"]
        painn_config = package["config"]
    elif model_type == "painn":
        if str(painn_backend) != "jax":
            raise ValueError("The TAM release supports only the JAX PaiNN backend.")
        package, painn_model = init_painn_velocity(
            init_key,
            problem,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            t_embed_dim=t_embed_dim,
            beta_embed_dim=beta_embed_dim,
            atom_identity=painn_atom_identity,
            atom_embed_dim=painn_atom_embed_dim,
        )
        params = painn_trainable_params_from_package(package)
        painn_state = package["state"]
        painn_config = package["config"]
    elif model_type == "egnn":
        package = init_egnn_velocity(
            init_key,
            problem,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            t_embed_dim=t_embed_dim,
            beta_embed_dim=beta_embed_dim,
        )
        params = package["params"]
        egnn_config = package["config"]
    else:
        input_dim = int(problem.dim) + int(t_embed_dim) + int(beta_embed_dim)
        params = init_mlp(init_key, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=problem.dim, n_layers=n_layers)
    opt_state = init_adam(params)
    history: list[float] = []
    fm_eval_method = str(fm_eval_method).lower()
    if fm_eval_method not in {"euler", "heun"}:
        raise ValueError(f"Unsupported FM eval ODE method {fm_eval_method!r}; use 'euler' or 'heun'.")
    ckpt_steps = {int(s) for s in (save_checkpoints or []) if 0 < int(s) <= int(steps)}
    if ckpt_steps:
        print(f"[FM] will save checkpoints at steps={sorted(ckpt_steps)}", flush=True)
    eval_log_path = run_dir / "fm_online_eval.jsonl"

    betas, stacked_data = _stack_datasets(datasets, beta_grid)
    betas = jax.device_put(betas)
    stacked_data = jax.device_put(stacked_data)
    fm_optimizer = {
        "adam_beta1": 0.95 if model_type == "ala2_chiro_painn_jax" else 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1.0e-8,
        "weight_decay": 0.0,
        "grad_clip": float(grad_clip),
    }
    if config is not None:
        config.setdefault("optimizer", {})
        config["optimizer"].update(fm_optimizer)
        config.setdefault("fm_eval", {})
        config["fm_eval"].update(
            {
                "every": int(fm_eval_every),
                "samples": int(fm_eval_samples),
                "steps": int(fm_eval_steps),
                "chunk_size": int(fm_eval_chunk_size),
                "method": str(fm_eval_method),
                "betas": None if fm_eval_betas is None else [float(b) for b in fm_eval_betas],
            }
        )
        config.setdefault("checkpoint", {})
        config["checkpoint"]["save_fm_checkpoints"] = sorted(ckpt_steps)
        if painn_config is not None:
            config.setdefault("model", {})
            config["model"].update(
                {
                    "painn_backend": painn_config.get("painn_backend", painn_backend),
                    "atom_identity": painn_config.get("atom_identity", "none"),
                    "atom_embed_dim": int(painn_config.get("atom_embed_dim", 0)),
                    "node_feature_dim": int(painn_config.get("node_feature_dim", 0)),
                    "radius": float(painn_config.get("radius", 0.0)),
                    "n_rbf": int(painn_config.get("n_rbf", 0)),
                    "radial_basis": painn_config.get("radial_basis"),
                    "cutoff": painn_config.get("cutoff"),
                }
            )

    train_step = _make_fm_train_step(
        problem=problem,
        batch_size=int(batch_size),
        prior_scale=float(prior_scale),
        t_embed_dim=int(t_embed_dim),
        beta_embed_dim=int(beta_embed_dim),
        grad_clip=float(grad_clip),
        model_type=model_type,
        painn_backend=str(painn_backend),
        painn_model=painn_model,
        painn_state=painn_state,
        painn_config=painn_config,
        egnn_config=egnn_config,
        adam_beta1=float(fm_optimizer["adam_beta1"]),
        adam_beta2=float(fm_optimizer["adam_beta2"]),
        adam_eps=float(fm_optimizer["adam_eps"]),
        weight_decay=float(fm_optimizer["weight_decay"]),
    )
    print(
        f"[FM] prepared device datasets shape={tuple(stacked_data.shape)} "
        f"betas={tuple(float(b) for b in beta_grid)}; model={model_type} "
        f"painn_backend={str(painn_config.get('painn_backend', painn_backend)) if painn_config else 'n/a'}; "
        f"fm_eval_method={fm_eval_method}; "
        "mixed-beta batches; first step includes JIT compile",
        flush=True,
    )

    def fm_package(current_params: Params) -> Any:
        if model_type == "ala2_chiro_painn_jax":
            model_params, extra_params = split_ala2_chiro_painn_jax_trainable_params(current_params)
            return {
                "model_type": "ala2_chiro_painn_jax",
                "config": painn_config,
                "params": model_params,
                "state": painn_state,
                "extra_params": extra_params,
            }
        if model_type == "painn":
            payload = {
                "model_type": "painn",
                "painn_backend": "jax",
                "config": painn_config,
                "params": current_params["model"] if isinstance(current_params, dict) and "model" in current_params else current_params,
                "state": painn_state,
            }
            if isinstance(current_params, dict) and "extra" in current_params:
                payload["extra_params"] = current_params["extra"]
            return payload
        if model_type == "egnn":
            return {"model_type": "egnn", "config": egnn_config, "params": current_params}
        return current_params

    def jsonable(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): jsonable(v) for k, v in value.items()}
        try:
            if isinstance(value, float) and not np.isfinite(value):
                return None
        except Exception:
            pass
        return value

    def write_fm_eval_row(
        step: int,
        beta: float,
        metrics: dict[str, Any],
        status: str,
        error: str | None,
        n_samples: int,
        chunk_size: int,
    ) -> None:
        row: dict[str, Any] = {
            "status": str(status),
            "step": int(step),
            "beta": float(beta),
            "problem": str(problem.name),
            "model_type": str(model_type),
            "painn_backend": str(painn_config.get("painn_backend", painn_backend)) if painn_config else None,
            "n_samples": int(n_samples),
            "ode_steps": int(fm_eval_steps),
            "ode_method": str(fm_eval_method),
            "prior_scale": float(prior_scale),
            "chunk_size": int(chunk_size),
        }
        if error is not None:
            row["error"] = str(error)
        row.update({str(k): jsonable(v) for k, v in metrics.items()})
        with eval_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def quick_eval(current_params: Params, step: int) -> None:
        if int(fm_eval_every) <= 0:
            return
        from adj_thermo.sampler import sample_ode

        eval_betas = tuple(float(b) for b in (fm_eval_betas or beta_grid))
        n_eval_samples = int(fm_eval_samples)
        if n_eval_samples <= 0:
            return
        if int(fm_eval_chunk_size) > 0:
            eval_chunk_size = int(fm_eval_chunk_size)
        else:
            eval_chunk_size = n_eval_samples
        eval_chunk_size = max(1, min(eval_chunk_size, n_eval_samples))

        if model_type == "ala2_chiro_painn_jax":
            eval_params = fm_package(current_params)
        elif model_type == "painn":
            eval_params = {
                "model_type": "painn",
                "painn_backend": "jax",
                "config": painn_config,
                "params": current_params,
                "state": painn_state,
            }
        elif model_type == "egnn":
            eval_params = {"model_type": "egnn", "config": egnn_config, "params": current_params}
        else:
            eval_params = current_params

        for j, beta in enumerate(eval_betas):
            key_eval = jax.random.PRNGKey(950000 + int(step) * 17 + j)
            sample_chunks: list[np.ndarray] = []
            done = 0
            chunk_index = 0
            while done < n_eval_samples:
                n_chunk = min(eval_chunk_size, n_eval_samples - done)
                sample_chunks.append(
                    sample_ode(
                        eval_params,
                        jax.random.fold_in(key_eval, chunk_index),
                        problem,
                        beta,
                        n_chunk,
                        int(fm_eval_steps),
                        float(prior_scale),
                        int(t_embed_dim),
                        int(beta_embed_dim),
                        method=fm_eval_method,
                    )
                )
                done += n_chunk
                chunk_index += 1
            samples = np.concatenate(sample_chunks, axis=0)
            try:
                from adj_thermo.metrics import energy_stats, energy_w2_n

                samples_np = np.asarray(jax.device_get(samples), dtype=np.float32)
                ref = datasets.get(float(beta))
                if ref is None:
                    data_name = None
                    if isinstance(config, dict):
                        data_name = config.get("data_name")
                    if not data_name:
                        data_name = problem.name
                    ref_path = Path("data") / str(data_name) / beta_file_name(float(beta))
                    if ref_path.exists():
                        ref = np.load(ref_path, allow_pickle=False).astype(np.float32, copy=False)
                stats = energy_stats(samples_np, problem)
                if ref is not None:
                    ref_np = np.asarray(ref, dtype=np.float32)
                    ref_stats = energy_stats(ref_np, problem)
                    ew2 = energy_w2_n(
                        samples_np,
                        ref_np,
                        problem,
                        n_samples=min(256, int(fm_eval_samples)),
                        seed=970000 + int(step) + j,
                    )
                    item = {
                        "ew2": float(ew2),
                        "energy_mean": float(stats.get("mean", float("nan"))),
                        "energy_std": float(stats.get("std", float("nan"))),
                        "md_energy_mean": float(ref_stats.get("mean", float("nan"))),
                        "md_energy_std": float(ref_stats.get("std", float("nan"))),
                    }
                    print(
                        "[FM eval] "
                        f"step={int(step)} beta={float(beta):.8f} method={fm_eval_method} "
                        f"ew2={ew2:.6g} "
                        f"energy_mean={stats.get('mean', float('nan')):.6g} "
                        f"energy_std={stats.get('std', float('nan')):.6g} "
                        f"md_energy_mean={ref_stats.get('mean', float('nan')):.6g} "
                        f"md_energy_std={ref_stats.get('std', float('nan')):.6g}",
                        flush=True,
                    )
                    write_fm_eval_row(step, beta, item, "ok", None, n_eval_samples, eval_chunk_size)
                else:
                    item = {
                        "reference_source": "missing",
                        "energy_mean": float(stats.get("mean", float("nan"))),
                        "energy_std": float(stats.get("std", float("nan"))),
                    }
                    print(
                        "[FM eval] "
                        f"step={int(step)} beta={float(beta):.8f} method={fm_eval_method} "
                        "ref=missing "
                        f"energy_mean={stats.get('mean', float('nan')):.6g} "
                        f"energy_std={stats.get('std', float('nan')):.6g}",
                        flush=True,
                    )
                    write_fm_eval_row(step, beta, item, "ok", None, n_eval_samples, eval_chunk_size)
            except Exception as exc:
                print(f"[FM eval] step={int(step)} beta={float(beta):.8f} method={fm_eval_method} failed: {type(exc).__name__}: {exc}", flush=True)
                write_fm_eval_row(step, beta, {}, "error", f"{type(exc).__name__}: {exc}", n_eval_samples, eval_chunk_size)

    for step in range(1, int(steps) + 1):
        key, step_key = jax.random.split(key)
        current_lr = linear_decay_lr(step, int(steps), lr, lr_min)
        params, opt_state, loss = train_step(
            params,
            opt_state,
            step_key,
            betas,
            stacked_data,
            jnp.asarray(current_lr, dtype=jnp.float32),
        )
        loss_f = float(loss)
        history.append(loss_f)
        if log_every > 0 and (step == 1 or step % int(log_every) == 0 or step == int(steps)):
            print(f"[FM] step {step:6d} loss={loss_f:.6f} lr={current_lr:.3g}", flush=True)
        if int(fm_eval_every) > 0 and (step % int(fm_eval_every) == 0 or step == int(steps)):
            quick_eval(params, step)
        if step in ckpt_steps:
            save_pickle(run_dir / f"fm_params_step_{step}.pkl", fm_package(params))
            np.save(run_dir / f"fm_loss_history_step_{step}.npy", np.asarray(history, dtype=np.float32))
            print(f"[FM] checkpoint written to {run_dir / f'fm_params_step_{step}.pkl'}", flush=True)

    hist = np.asarray(history, dtype=np.float32)
    save_pickle(run_dir / "fm_params.pkl", fm_package(params))
    np.save(run_dir / "fm_loss_history.npy", hist)
    if config is not None:
        try:
            import yaml
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=True)
        except Exception:
            pass
    return params, hist
