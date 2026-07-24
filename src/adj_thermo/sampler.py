from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.am_velocity import is_anchor_residual_checkpoint
from adj_thermo.model import apply_mlp
from adj_thermo.model.painn_ala2_chiro_jax import (
    ala2_chiro_painn_jax_trainable_params_from_package,
    apply_ala2_chiro_painn_jax_velocity_params,
    build_ala2_chiro_painn_jax_velocity_model,
    is_ala2_chiro_painn_jax_velocity_params,
)
from adj_thermo.model.egnn import apply_egnn_velocity_params, is_egnn_velocity_params
from adj_thermo.model.painn_velocity import (
    apply_painn_velocity_params,
    build_painn_velocity_model,
    is_painn_velocity_params,
    painn_trainable_params_from_package,
)
from adj_thermo.problem.base import ProblemSpec

VelocityApply = Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array]


def _prepare_velocity_apply(
    params: Any,
    problem: ProblemSpec,
    t_embed_dim: int,
    beta_embed_dim: int,
) -> tuple[Any, VelocityApply]:
    if is_anchor_residual_checkpoint(params):
        base_model_params, apply_base = _prepare_velocity_apply(
            params["base_package"], problem, t_embed_dim, beta_embed_dim
        )
        correction_model_params, apply_correction = _prepare_velocity_apply(
            params["correction_package"], problem, t_embed_dim, beta_embed_dim
        )
        beta0 = float(params["beta0"])
        beta1 = float(params["beta1"])

        def apply_fn(model_params: Any, x: jax.Array, t: jax.Array, beta: jax.Array) -> jax.Array:
            base_params, correction_params = model_params
            v_base_beta0 = apply_base(base_params, x, t, jnp.asarray(beta0, dtype=x.dtype))
            v_base_beta1 = apply_base(base_params, x, t, jnp.asarray(beta1, dtype=x.dtype))
            v_correction_beta1 = apply_correction(
                correction_params, x, t, jnp.asarray(beta1, dtype=x.dtype)
            )
            return v_base_beta0 + v_correction_beta1 - v_base_beta1

        return (base_model_params, correction_model_params), apply_fn

    if is_ala2_chiro_painn_jax_velocity_params(params):
        ala2_config = params["config"]
        ala2_model = build_ala2_chiro_painn_jax_velocity_model(ala2_config)
        ala2_params = ala2_chiro_painn_jax_trainable_params_from_package(params)
        ala2_state = params["state"]

        def apply_fn(model_params: Any, x: jax.Array, t: jax.Array, beta: jax.Array) -> jax.Array:
            return apply_ala2_chiro_painn_jax_velocity_params(
                ala2_model, model_params, ala2_state, x, t, beta, problem, ala2_config
            )

        return ala2_params, apply_fn

    if is_egnn_velocity_params(params):
        egnn_params = params["params"]
        egnn_config = params["config"]

        def apply_fn(model_params: Any, x: jax.Array, t: jax.Array, beta: jax.Array) -> jax.Array:
            return apply_egnn_velocity_params(model_params, x, t, beta, problem, egnn_config)

        return egnn_params, apply_fn

    if is_painn_velocity_params(params):
        painn_config = params["config"]
        painn_model = build_painn_velocity_model(painn_config)
        painn_params = painn_trainable_params_from_package(params)
        painn_state = params["state"]

        def apply_fn(model_params: Any, x: jax.Array, t: jax.Array, beta: jax.Array) -> jax.Array:
            return apply_painn_velocity_params(painn_model, model_params, painn_state, x, t, beta, problem, painn_config)

        return painn_params, apply_fn

    def apply_fn(model_params: Any, x: jax.Array, t: jax.Array, beta: jax.Array) -> jax.Array:
        return apply_mlp(model_params, x, t, beta, problem, t_embed_dim=t_embed_dim, beta_embed_dim=beta_embed_dim)

    return params, apply_fn


def make_sample_ode_fn(
    params: Any,
    problem: ProblemSpec,
    n_samples: int,
    n_steps: int,
    prior_scale: float = 1.0,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    method: str = "euler",
) -> Any:
    n_samples = int(n_samples)
    n_steps = int(n_steps)
    h = 1.0 / float(n_steps)
    method = str(method).lower()
    if method not in {"euler", "heun"}:
        raise ValueError(f"Unsupported ODE sampler method {method!r}; use 'euler' or 'heun'.")
    checkpoint_beta = float(params["beta1"]) if is_anchor_residual_checkpoint(params) else None

    model_params, apply_velocity = _prepare_velocity_apply(params, problem, t_embed_dim, beta_embed_dim)

    @jax.jit
    def sample_once(model_params: Any, key: jax.Array, beta: jax.Array) -> jax.Array:
        x0 = float(prior_scale) * jax.random.normal(key, (n_samples, int(problem.dim)), dtype=jnp.float32)
        x0 = jnp.asarray(problem.project_fn(x0), dtype=x0.dtype)

        def step(x: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
            t = jnp.full((n_samples, 1), k.astype(x.dtype) / float(n_steps), dtype=x.dtype)
            v1 = apply_velocity(model_params, x, t, beta)
            x_euler = jnp.asarray(problem.project_fn(x + h * v1), dtype=x.dtype)
            if method == "euler":
                x_next = x_euler
            else:
                t_next = jnp.full((n_samples, 1), (k.astype(x.dtype) + 1.0) / float(n_steps), dtype=x.dtype)
                v2 = apply_velocity(model_params, x_euler, t_next, beta)
                x_heun = jnp.asarray(problem.project_fn(x + 0.5 * h * (v1 + v2)), dtype=x.dtype)
                x_next = jax.lax.cond(k < n_steps - 1, lambda _: x_heun, lambda _: x_euler, operand=None)
            return jnp.asarray(x_next, dtype=x.dtype), None

        x, _ = jax.lax.scan(step, x0, jnp.arange(n_steps, dtype=jnp.int32))
        return x

    def wrapped(key: jax.Array, beta: float | jax.Array) -> jax.Array:
        requested_beta = float(beta)
        if checkpoint_beta is not None and not np.isclose(requested_beta, checkpoint_beta, rtol=0.0, atol=1.0e-7):
            raise ValueError(
                f"Anchor-residual checkpoint targets beta={checkpoint_beta:.8g}, "
                f"but sampling requested beta={requested_beta:.8g}."
            )
        effective_beta = checkpoint_beta if checkpoint_beta is not None else requested_beta
        return sample_once(model_params, key, jnp.asarray(effective_beta, dtype=jnp.float32))

    return wrapped


def sample_ode(
    params: Any,
    key: jax.Array,
    problem: ProblemSpec,
    beta: float,
    n_samples: int,
    n_steps: int,
    prior_scale: float = 1.0,
    t_embed_dim: int = 16,
    beta_embed_dim: int = 8,
    method: str = "euler",
) -> np.ndarray:
    sample_fn = make_sample_ode_fn(
        params,
        problem,
        n_samples,
        n_steps,
        prior_scale=prior_scale,
        t_embed_dim=t_embed_dim,
        beta_embed_dim=beta_embed_dim,
        method=method,
    )
    x = sample_fn(key, beta)
    return np.asarray(jax.device_get(x), dtype=np.float32)
