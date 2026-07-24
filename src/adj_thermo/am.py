from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.am_velocity import (
    ANCHOR_RESIDUAL,
    TARGET_REFINEMENT,
    is_anchor_residual_checkpoint,
    make_anchor_residual_checkpoint,
    normalize_am_parameterization,
    total_and_correction_velocity,
)
from adj_thermo.model import apply_mlp
from adj_thermo.model.egnn import apply_egnn_velocity_params, is_egnn_velocity_params
from adj_thermo.model.painn_velocity import (
    apply_painn_velocity_params,
    build_painn_velocity_model,
    is_painn_velocity_params,
    painn_trainable_params_from_package,
    split_painn_trainable_params,
)
from adj_thermo.model.painn_ala2_chiro_jax import (
    ala2_chiro_painn_jax_trainable_params_from_package,
    apply_ala2_chiro_painn_jax_velocity_params,
    build_ala2_chiro_painn_jax_velocity_model,
    is_ala2_chiro_painn_jax_velocity_params,
    split_ala2_chiro_painn_jax_trainable_params,
)
from adj_thermo.optim import AdamState, adam_update, clip_grads, init_adam
from adj_thermo.problem.base import ProblemSpec
from adj_thermo.utils import ensure_dir, save_pickle

Params = Any
AmTrainStep = Callable[[Params, Params, AdamState, jax.Array, jax.Array], tuple[Params, AdamState, jax.Array]]


def linear_decay_lr(step: int, total_steps: int, lr: float, min_lr: float) -> float:
    if total_steps <= 1:
        return float(min_lr)
    frac = min(1.0, max(0.0, float(step - 1) / float(total_steps - 1)))
    return float(min_lr) + (float(lr) - float(min_lr)) * (1.0 - frac)


def memoryless_sigma(t: jax.Array, h: float, max_sigma: float = 50.0) -> jax.Array:
    sigma = jnp.sqrt(2.0 * (1.0 - t + float(h)) / (t + float(h)))
    if float(max_sigma) > 0.0:
        sigma = jnp.minimum(sigma, float(max_sigma))
    return sigma


def _apply_velocity(
    params: Params,
    x: jax.Array,
    t: jax.Array,
    beta: float | jax.Array,
    problem: ProblemSpec,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    model_type: str = "mlp",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
) -> jax.Array:
    if model_type == "ala2_chiro_painn_jax":
        if painn_model is None or painn_state is None or painn_config is None:
            raise ValueError("Ala2 chiral PaiNN velocity call requires model, state, and config.")
        return apply_ala2_chiro_painn_jax_velocity_params(
            painn_model, params, painn_state, x, t, beta, problem, painn_config
        )
    if model_type == "painn":
        if painn_config is None:
            raise ValueError("PaiNN velocity call requires painn_config.")
        if painn_model is None or painn_state is None:
            raise ValueError("JAX PaiNN velocity call requires painn_model and painn_state.")
        return apply_painn_velocity_params(painn_model, params, painn_state, x, t, beta, problem, painn_config)
    if model_type == "egnn":
        if egnn_config is None:
            raise ValueError("EGNN velocity call requires egnn_config.")
        return apply_egnn_velocity_params(params, x, t, beta, problem, egnn_config)
    return apply_mlp(params, x, t, beta, problem, t_embed_dim=t_embed_dim, beta_embed_dim=beta_embed_dim)


def memoryless_drift(
    params: Params,
    x: jax.Array,
    t: jax.Array,
    beta: float | jax.Array,
    problem: ProblemSpec,
    h: float,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    model_type: str = "mlp",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
) -> jax.Array:
    v = _apply_velocity(
        params,
        x,
        t,
        beta,
        problem,
        t_embed_dim=t_embed_dim,
        beta_embed_dim=beta_embed_dim,
        model_type=model_type,
        painn_model=painn_model,
        painn_state=painn_state,
        painn_config=painn_config,
        egnn_config=egnn_config,
    )
    return problem.project_fn(2.0 * v - x / (t + float(h)))


def sample_memoryless_sde(
    params: Params,
    key: jax.Array,
    beta: float,
    problem: ProblemSpec,
    batch_size: int,
    K: int,
    prior_scale: float,
    max_sigma: float = 50.0,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    model_type: str = "mlp",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
    base_params: Params | None = None,
    beta0: float | None = None,
    am_parameterization: str = TARGET_REFINEMENT,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    parameterization = normalize_am_parameterization(am_parameterization)
    if parameterization == ANCHOR_RESIDUAL and (base_params is None or beta0 is None):
        raise ValueError("Anchor-residual SDE sampling requires base_params and beta0.")
    h = 1.0 / float(K)
    sqrt_h = jnp.sqrt(jnp.asarray(h, dtype=jnp.float32))
    key_x, key_noise = jax.random.split(key)
    x0 = float(prior_scale) * jax.random.normal(key_x, (int(batch_size), int(problem.dim)), dtype=jnp.float32)
    x0 = jnp.asarray(problem.project_fn(x0), dtype=x0.dtype)

    def body(x: jax.Array, k: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        noise_key = jax.random.fold_in(key_noise, k)
        t_scalar = k.astype(jnp.float32) / float(K)
        t_after_scalar = (k.astype(jnp.float32) + 1.0) / float(K)
        t_col = jnp.full((int(batch_size), 1), t_scalar, dtype=x.dtype)
        t_after_col = jnp.full((int(batch_size), 1), t_after_scalar, dtype=x.dtype)
        if parameterization == TARGET_REFINEMENT:
            velocity = _apply_velocity(
                params,
                x,
                t_col,
                beta,
                problem,
                t_embed_dim=t_embed_dim,
                beta_embed_dim=beta_embed_dim,
                model_type=model_type,
                painn_model=painn_model,
                painn_state=painn_state,
                painn_config=painn_config,
                egnn_config=egnn_config,
            )
        else:
            def apply_velocity(p: Params, y: jax.Array, s: jax.Array, b: float) -> jax.Array:
                return _apply_velocity(
                    p,
                    y,
                    s,
                    b,
                    problem,
                    t_embed_dim=t_embed_dim,
                    beta_embed_dim=beta_embed_dim,
                    model_type=model_type,
                    painn_model=painn_model,
                    painn_state=painn_state,
                    painn_config=painn_config,
                    egnn_config=egnn_config,
                )

            velocity, _ = total_and_correction_velocity(
                apply_velocity,
                params,
                base_params,
                x,
                t_col,
                float(beta0),
                float(beta),
                parameterization,
            )
        drift = problem.project_fn(2.0 * velocity - x / (t_col + float(h)))
        sigma = float(prior_scale) * memoryless_sigma(t_col, h, max_sigma=max_sigma)
        noise = jnp.asarray(problem.project_fn(jax.random.normal(noise_key, x.shape, dtype=x.dtype)), dtype=x.dtype)
        x_next = jnp.asarray(problem.project_fn(x + h * drift + sqrt_h * sigma * noise), dtype=x.dtype)
        return x_next, (x, t_col, x_next, t_after_col)

    x_final, (xs, ts, xs_after, ts_after) = jax.lax.scan(body, x0, jnp.arange(int(K)))
    return xs, ts, xs_after, ts_after, x_final


def terminal_state_with_base(
    base_params: Params,
    x_last: jax.Array,
    t_last: jax.Array,
    beta0: float,
    problem: ProblemSpec,
    h: float,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    model_type: str = "mlp",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
) -> jax.Array:
    v_base = _apply_velocity(
        base_params,
        x_last,
        t_last,
        beta0,
        problem,
        t_embed_dim=t_embed_dim,
        beta_embed_dim=beta_embed_dim,
        model_type=model_type,
        painn_model=painn_model,
        painn_state=painn_state,
        painn_config=painn_config,
        egnn_config=egnn_config,
    )
    out = problem.project_fn(x_last + float(h) * v_base)
    return jnp.asarray(out, dtype=x_last.dtype)


def terminal_adjoint_delta_beta(
    x_terminal: jax.Array,
    beta0: float,
    beta1: float,
    problem: ProblemSpec,
    energy_grad_scale: float = 1.0,
) -> jax.Array:
    out = problem.project_fn(float(energy_grad_scale) * (float(beta1) - float(beta0)) * problem.grad_energy_fn(x_terminal))
    return jnp.asarray(out, dtype=x_terminal.dtype)


def _base_drift_vjp(
    base_params: Params,
    x: jax.Array,
    t: jax.Array,
    cotangent: jax.Array,
    beta0: float,
    problem: ProblemSpec,
    h: float,
    t_embed_dim: int,
    beta_embed_dim: int,
    model_type: str,
    painn_model: Any | None,
    painn_state: Any | None,
    painn_config: dict[str, Any] | None,
    egnn_config: dict[str, Any] | None,
) -> jax.Array:
    def one(x_single: jax.Array, t_single: jax.Array, a_single: jax.Array) -> jax.Array:
        def drift_y(y: jax.Array) -> jax.Array:
            return memoryless_drift(
                base_params,
                y[None, :],
                t_single[None, :],
                beta0,
                problem,
                h,
                t_embed_dim,
                beta_embed_dim,
                model_type=model_type,
                painn_model=painn_model,
                painn_state=painn_state,
                painn_config=painn_config,
                egnn_config=egnn_config,
            )[0]

        _, pullback = jax.vjp(drift_y, x_single)
        (jt_a,) = pullback(a_single)
        return jt_a

    return jax.vmap(one)(x, t, cotangent)


def backward_lean_adjoint(
    base_params: Params,
    xs_after: jax.Array,
    ts_after: jax.Array,
    a_T: jax.Array,
    beta0: float,
    problem: ProblemSpec,
    h: float,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    model_type: str = "mlp",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
) -> jax.Array:
    def reverse_body(a_next: jax.Array, inputs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x_after, t_after = inputs
        jt_a = _base_drift_vjp(
            base_params,
            x_after,
            t_after,
            a_next,
            beta0,
            problem,
            h,
            t_embed_dim,
            beta_embed_dim,
            model_type,
            painn_model,
            painn_state,
            painn_config,
            egnn_config,
        )
        a_prev = problem.project_fn(a_next + float(h) * jt_a)
        a_prev = jax.lax.stop_gradient(a_prev)
        return a_prev, a_prev

    _, rev = jax.lax.scan(reverse_body, a_T, (xs_after[::-1], ts_after[::-1]))
    return jax.lax.stop_gradient(rev[::-1])


def _select_loss_timesteps(
    xs: jax.Array,
    ts: jax.Array,
    adjs: jax.Array,
    key: jax.Array,
    num_loss_steps: int,
    keep_last_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    total = int(xs.shape[0])
    n_loss = min(max(int(num_loss_steps), 1), total)
    n_keep = min(max(int(keep_last_steps), 0), n_loss, total)
    n_random = n_loss - n_keep
    last_idx = jnp.arange(total - n_keep, total, dtype=jnp.int32) if n_keep > 0 else jnp.zeros((0,), dtype=jnp.int32)
    if n_random > 0:
        prefix = total - n_keep
        rand_idx = jax.random.choice(key, prefix, shape=(n_random,), replace=False).astype(jnp.int32)
        idx = jnp.sort(jnp.concatenate([rand_idx, last_idx]))
    else:
        idx = last_idx
    return xs[idx], ts[idx], adjs[idx]


def am_loss_fn(
    params: Params,
    base_params: Params,
    key: jax.Array,
    beta0: float,
    beta1: float,
    problem: ProblemSpec,
    batch_size: int,
    K: int,
    prior_scale: float,
    am_num_loss_steps: int,
    am_keep_last_steps: int,
    max_sigma: float = 50.0,
    energy_grad_scale: float = 1.0,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    model_type: str = "mlp",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
    am_parameterization: str = TARGET_REFINEMENT,
) -> jax.Array:
    parameterization = normalize_am_parameterization(am_parameterization)
    h = 1.0 / float(K)
    key_path, key_steps = jax.random.split(key)
    xs, ts, xs_after, ts_after, _ = sample_memoryless_sde(
        params,
        key_path,
        beta1,
        problem,
        batch_size,
        K,
        prior_scale,
        max_sigma,
        t_embed_dim,
        beta_embed_dim,
        model_type=model_type,
        painn_model=painn_model,
        painn_state=painn_state,
        painn_config=painn_config,
        egnn_config=egnn_config,
        base_params=base_params,
        beta0=beta0,
        am_parameterization=parameterization,
    )
    xs = jax.lax.stop_gradient(xs)
    ts = jax.lax.stop_gradient(ts)
    xs_after = jax.lax.stop_gradient(xs_after)
    ts_after = jax.lax.stop_gradient(ts_after)

    x_terminal = terminal_state_with_base(
        base_params,
        xs[-1],
        ts[-1],
        beta0,
        problem,
        h,
        t_embed_dim,
        beta_embed_dim,
        model_type=model_type,
        painn_model=painn_model,
        painn_state=painn_state,
        painn_config=painn_config,
        egnn_config=egnn_config,
    )
    x_terminal = jax.lax.stop_gradient(x_terminal)
    a_T = jax.lax.stop_gradient(terminal_adjoint_delta_beta(x_terminal, beta0, beta1, problem, energy_grad_scale))
    xs_after_for_adj = xs_after.at[-1].set(x_terminal)
    adjs = backward_lean_adjoint(
        base_params,
        xs_after_for_adj,
        ts_after,
        a_T,
        beta0,
        problem,
        h,
        t_embed_dim,
        beta_embed_dim,
        model_type=model_type,
        painn_model=painn_model,
        painn_state=painn_state,
        painn_config=painn_config,
        egnn_config=egnn_config,
    )
    adjs = jax.lax.stop_gradient(adjs)

    xs_sel, ts_sel, adjs_sel = _select_loss_timesteps(xs, ts, adjs, key_steps, am_num_loss_steps, am_keep_last_steps)
    flat_x = jax.lax.stop_gradient(xs_sel.reshape((-1, int(problem.dim))))
    flat_t = jax.lax.stop_gradient(ts_sel.reshape((-1, 1)))
    flat_a = jax.lax.stop_gradient(adjs_sel.reshape((-1, int(problem.dim))))
    def apply_velocity(p: Params, y: jax.Array, s: jax.Array, b: float) -> jax.Array:
        return _apply_velocity(
            p,
            y,
            s,
            b,
            problem,
            t_embed_dim=t_embed_dim,
            beta_embed_dim=beta_embed_dim,
            model_type=model_type,
            painn_model=painn_model,
            painn_state=painn_state,
            painn_config=painn_config,
            egnn_config=egnn_config,
        )

    _, correction = total_and_correction_velocity(
        apply_velocity,
        params,
        base_params,
        flat_x,
        flat_t,
        beta0,
        beta1,
        parameterization,
    )
    sigma = float(prior_scale) * memoryless_sigma(flat_t, h, max_sigma=max_sigma)
    term = (2.0 / sigma) * correction + sigma * flat_a
    return 0.5 * jnp.mean(jnp.sum(term * term, axis=-1))


def _make_am_train_step(
    problem: ProblemSpec,
    beta0: float,
    beta1: float,
    batch_size: int,
    K: int,
    prior_scale: float,
    am_num_loss_steps: int,
    am_keep_last_steps: int,
    max_sigma: float,
    energy_grad_scale: float,
    t_embed_dim: int,
    beta_embed_dim: int,
    grad_clip: float,
    adam_b1: float,
    weight_decay: float,
    model_type: str = "mlp",
    painn_model: Any | None = None,
    painn_state: Any | None = None,
    painn_config: dict[str, Any] | None = None,
    egnn_config: dict[str, Any] | None = None,
    am_parameterization: str = TARGET_REFINEMENT,
) -> AmTrainStep:
    def train_step(
        params: Params,
        base_params: Params,
        opt_state: AdamState,
        key: jax.Array,
        lr: jax.Array,
    ) -> tuple[Params, AdamState, jax.Array]:
        loss, grads = jax.value_and_grad(
            lambda p: am_loss_fn(
                p,
                base_params,
                key,
                beta0,
                beta1,
                problem,
                batch_size,
                K,
                prior_scale,
                am_num_loss_steps,
                am_keep_last_steps,
                max_sigma=max_sigma,
                energy_grad_scale=energy_grad_scale,
                t_embed_dim=t_embed_dim,
                beta_embed_dim=beta_embed_dim,
                model_type=model_type,
                painn_model=painn_model,
                painn_state=painn_state,
                painn_config=painn_config,
                egnn_config=egnn_config,
                am_parameterization=am_parameterization,
            )
            ,
            allow_int=True,
        )(params)
        grads = clip_grads(grads, grad_clip)
        params, opt_state = adam_update(params, grads, opt_state, lr=lr, beta1=float(adam_b1), weight_decay=float(weight_decay))
        return params, opt_state, loss

    return jax.jit(train_step)


def train_am(
    problem: ProblemSpec,
    base_params: Params,
    run_dir: str | Path,
    key: jax.Array,
    beta0: float,
    beta1: float,
    steps: int = 10000,
    batch_size: int = 512,
    lr: float = 2.0e-5,
    lr_min: float | None = None,
    K: int = 40,
    prior_scale: float = 1.0,
    am_num_loss_steps: int = 20,
    am_keep_last_steps: int = 10,
    max_sigma: float = 50.0,
    energy_grad_scale: float = 1.0,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    grad_clip: float = 1.0,
    adam_b1: float = 0.9,
    weight_decay: float = 0.0,
    log_every: int = 100,
    model_type: str = "mlp",
    save_checkpoints: list[int] | tuple[int, ...] | None = None,
    am_parameterization: str = TARGET_REFINEMENT,
) -> tuple[Params, np.ndarray]:
    run_dir = ensure_dir(run_dir)
    parameterization = normalize_am_parameterization(am_parameterization)
    if is_anchor_residual_checkpoint(base_params):
        raise ValueError("AM training requires a base FM checkpoint, not an anchor-residual AM checkpoint.")
    painn_model = None
    painn_state = None
    painn_config = None
    egnn_config = None
    save_as_ala2 = is_ala2_chiro_painn_jax_velocity_params(base_params) or str(model_type) == "ala2_chiro_painn_jax"
    save_as_painn = is_painn_velocity_params(base_params) or str(model_type) == "painn"
    save_as_egnn = is_egnn_velocity_params(base_params) or str(model_type) == "egnn"
    if save_as_ala2:
        if not is_ala2_chiro_painn_jax_velocity_params(base_params):
            raise ValueError("Ala2 AM requires an ala2_chiro_painn_jax FM package.")
        model_type = "ala2_chiro_painn_jax"
        painn_config = base_params["config"]
        painn_model = build_ala2_chiro_painn_jax_velocity_model(painn_config)
        painn_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x), base_params["state"])
        base_ala2_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x), ala2_chiro_painn_jax_trainable_params_from_package(base_params)
        )
        params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), base_ala2_params)
        base_params_train = base_ala2_params
    elif save_as_painn:
        if not is_painn_velocity_params(base_params):
            raise ValueError("PaiNN AM requires a PaiNN FM package with model_type/config/params/state.")
        model_type = "painn"
        painn_config = base_params["config"]
        if str(base_params.get("painn_backend", "jax")) != "jax":
            raise ValueError("The TAM release supports only the JAX PaiNN backend.")
        painn_model = build_painn_velocity_model(painn_config)
        painn_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x), base_params["state"])
        base_painn_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), painn_trainable_params_from_package(base_params))
        params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), base_painn_params)
        base_params_train = base_painn_params
    elif save_as_egnn:
        if not is_egnn_velocity_params(base_params):
            raise ValueError("EGNN AM requires an EGNN FM package with model_type/config/params.")
        model_type = "egnn"
        egnn_config = base_params["config"]
        base_egnn_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), base_params["params"])
        params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), base_egnn_params)
        base_params_train = base_egnn_params
    else:
        model_type = "mlp"
        base_params_train = jax.tree_util.tree_map(lambda x: jnp.asarray(x), base_params)
        params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), base_params_train)
    opt_state = init_adam(params)
    history: list[float] = []
    train_step = _make_am_train_step(
        problem=problem,
        beta0=float(beta0),
        beta1=float(beta1),
        batch_size=int(batch_size),
        K=int(K),
        prior_scale=float(prior_scale),
        am_num_loss_steps=int(am_num_loss_steps),
        am_keep_last_steps=int(am_keep_last_steps),
        max_sigma=float(max_sigma),
        energy_grad_scale=float(energy_grad_scale),
        t_embed_dim=int(t_embed_dim),
        beta_embed_dim=int(beta_embed_dim),
        grad_clip=float(grad_clip),
        adam_b1=float(adam_b1),
        weight_decay=float(weight_decay),
        model_type=model_type,
        painn_model=painn_model,
        painn_state=painn_state,
        painn_config=painn_config,
        egnn_config=egnn_config,
        am_parameterization=parameterization,
    )
    ckpt_steps = {int(s) for s in (save_checkpoints or []) if 0 < int(s) <= int(steps)}
    if ckpt_steps:
        print(f"[AM] will save checkpoints at steps={sorted(ckpt_steps)}", flush=True)

    def build_trainable_package(current_params: Params) -> Params:
        if save_as_ala2:
            model_params, extra_params = split_ala2_chiro_painn_jax_trainable_params(current_params)
            return {
                "model_type": "ala2_chiro_painn_jax",
                "config": painn_config,
                "params": model_params,
                "state": painn_state,
                "extra_params": extra_params,
            }
        if save_as_painn:
            model_params, extra_params = split_painn_trainable_params(current_params)
            payload = {"model_type": "painn", "painn_backend": "jax", "config": painn_config, "params": model_params, "state": painn_state}
            if extra_params:
                payload["extra_params"] = extra_params
            return payload
        if save_as_egnn:
            return {"model_type": "egnn", "config": egnn_config, "params": current_params}
        return current_params

    def save_am_package(path: Path, current_params: Params) -> None:
        trainable_package = build_trainable_package(current_params)
        if parameterization == ANCHOR_RESIDUAL:
            payload = make_anchor_residual_checkpoint(base_params, trainable_package, beta0, beta1)
        else:
            payload = trainable_package
        save_pickle(path, payload)

    print(
        f"[AM] prepared jitted train step K={int(K)} batch={int(batch_size)} "
        f"loss_steps={int(am_num_loss_steps)} keep_last={int(am_keep_last_steps)} "
        f"parameterization={parameterization}; first step includes JIT compile",
        flush=True,
    )

    for step in range(1, int(steps) + 1):
        key, step_key = jax.random.split(key)
        current_lr = float(lr) if lr_min is None else linear_decay_lr(step, int(steps), float(lr), float(lr_min))
        params, opt_state, loss = train_step(
            params,
            base_params_train,
            opt_state,
            step_key,
            jnp.asarray(current_lr, dtype=jnp.float32),
        )
        loss_f = float(loss)
        history.append(loss_f)
        if log_every > 0 and (step == 1 or step % int(log_every) == 0 or step == int(steps)):
            print(f"[AM] step {step:6d} loss={loss_f:.6f} lr={current_lr:.3g}", flush=True)
        if step in ckpt_steps:
            save_am_package(run_dir / f"am_params_step_{step}.pkl", params)
            np.save(run_dir / f"am_loss_history_step_{step}.npy", np.asarray(history, dtype=np.float32))
            print(f"[AM] checkpoint written to {run_dir / f'am_params_step_{step}.pkl'}", flush=True)

    hist = np.asarray(history, dtype=np.float32)
    save_am_package(run_dir / "am_params.pkl", params)
    np.save(run_dir / "am_loss_history.npy", hist)
    return params, hist
