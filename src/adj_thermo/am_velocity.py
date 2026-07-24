from __future__ import annotations

from typing import Any, Callable

import jax

Params = Any
VelocityApply = Callable[[Params, Any, Any, float], Any]

TARGET_REFINEMENT = "target-refinement"
ANCHOR_RESIDUAL = "anchor-residual"
AM_PARAMETERIZATIONS = (TARGET_REFINEMENT, ANCHOR_RESIDUAL)


def normalize_am_parameterization(value: str) -> str:
    mode = str(value).strip().lower().replace("_", "-")
    if mode not in AM_PARAMETERIZATIONS:
        choices = ", ".join(AM_PARAMETERIZATIONS)
        raise ValueError(f"Unsupported AM parameterization {value!r}; choose one of: {choices}.")
    return mode


def total_and_correction_velocity(
    apply_velocity: VelocityApply,
    params: Params,
    base_params: Params,
    x: Any,
    t: Any,
    beta0: float,
    beta1: float,
    parameterization: str = TARGET_REFINEMENT,
) -> tuple[Any, Any]:
    mode = normalize_am_parameterization(parameterization)
    v_trainable_beta1 = apply_velocity(params, x, t, beta1)
    v_base_beta0 = jax.lax.stop_gradient(apply_velocity(base_params, x, t, beta0))
    if mode == TARGET_REFINEMENT:
        return v_trainable_beta1, v_trainable_beta1 - v_base_beta0

    v_base_beta1 = jax.lax.stop_gradient(apply_velocity(base_params, x, t, beta1))
    correction = v_trainable_beta1 - v_base_beta1
    return v_base_beta0 + correction, correction


def make_anchor_residual_checkpoint(
    base_package: Params,
    correction_package: Params,
    beta0: float,
    beta1: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_type": "tam_anchor_residual",
        "parameterization": ANCHOR_RESIDUAL,
        "beta0": float(beta0),
        "beta1": float(beta1),
        "base_package": base_package,
        "correction_package": correction_package,
    }


def is_anchor_residual_checkpoint(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("model_type") == "tam_anchor_residual"
        and value.get("parameterization") == ANCHOR_RESIDUAL
        and "base_package" in value
        and "correction_package" in value
        and "beta0" in value
        and "beta1" in value
    )
