"""Auditable equilibrium references for the four-particle DW benchmark.

The legacy DW4 datasets were produced by taking only the endpoint of a short
unadjusted Langevin trajectory.  That procedure has both finite-time and
finite-step bias.  This module instead works in an orthonormal six-dimensional
centre-of-mass-free coordinate system and uses annealed sequential Monte Carlo
with a Metropolis-adjusted Langevin (MALA) rejuvenation kernel.

The construction starts from an exactly sampled isotropic Gaussian and bridges
to ``exp(-beta * U)``.  Every rejuvenation proposal has a Metropolis correction;
random rotations, reflections and particle permutations are exact symmetries of
both the target and Lebesgue measure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np


PAIR_I = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int32)
PAIR_J = np.asarray([1, 2, 3, 2, 3, 3], dtype=np.int32)


def helmert_basis_np() -> np.ndarray:
    """Return the fixed 4x3 orthonormal COM-free Helmert basis."""

    return np.asarray(
        [
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
            [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
            [0.0, -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
            [0.0, 0.0, -3.0 / np.sqrt(12.0)],
        ],
        dtype=np.float64,
    )


def coordinates_to_reduced_np(x: np.ndarray) -> np.ndarray:
    """Map ``(..., 4, 2)`` centred coordinates to ``(..., 3, 2)``."""

    values = np.asarray(x, dtype=np.float64).reshape((*np.shape(x)[:-2], 4, 2))
    values = values - np.mean(values, axis=-2, keepdims=True)
    return np.einsum("nk,...nd->...kd", helmert_basis_np(), values)


def reduced_to_coordinates_np(y: np.ndarray) -> np.ndarray:
    """Map ``(..., 3, 2)`` reduced coordinates to centred ``(..., 4, 2)``."""

    values = np.asarray(y, dtype=np.float64).reshape((*np.shape(y)[:-2], 3, 2))
    return np.einsum("nk,...kd->...nd", helmert_basis_np(), values)


def pair_distances_np(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).reshape((-1, 4, 2))
    return np.linalg.norm(values[:, PAIR_I] - values[:, PAIR_J], axis=-1)


def dw4_energy_np(x: np.ndarray) -> np.ndarray:
    """ADTM DW4 energy, identical to ``problem/dw4.py`` defaults."""

    distances = pair_distances_np(x)
    z = distances - 1.0
    return np.sum(-4.0 * z * z + 0.9 * z**4, axis=-1)


def _helmert_basis_jax(dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(helmert_basis_np(), dtype=dtype)


def reduced_to_coordinates_jax(y: jax.Array) -> jax.Array:
    values = jnp.asarray(y).reshape((-1, 3, 2))
    return jnp.einsum("nk,bkd->bnd", _helmert_basis_jax(values.dtype), values)


def dw4_energy_reduced_jax(y: jax.Array) -> jax.Array:
    x = reduced_to_coordinates_jax(y)
    pair_i = jnp.asarray(PAIR_I)
    pair_j = jnp.asarray(PAIR_J)
    diff = x[:, pair_i] - x[:, pair_j]
    distance = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1.0e-12)
    z = distance - jnp.asarray(1.0, dtype=x.dtype)
    return jnp.sum(-4.0 * z * z + 0.9 * z**4, axis=-1)


def log_standard_source_jax(y: jax.Array, sigma: float) -> jax.Array:
    values = jnp.asarray(y).reshape((y.shape[0], 6))
    variance = jnp.asarray(sigma, dtype=values.dtype) ** 2
    return -0.5 * jnp.sum(values * values, axis=-1) / variance - 3.0 * jnp.log(
        2.0 * jnp.pi * variance
    )


def _bridge_logp_single(y_flat: jax.Array, t: jax.Array, beta: jax.Array, sigma: jax.Array) -> jax.Array:
    y = y_flat.reshape((1, 3, 2))
    source = log_standard_source_jax(y, sigma)[0]
    target = -beta * dw4_energy_reduced_jax(y)[0]
    return (1.0 - t) * source + t * target


_BRIDGE_VALUE_AND_GRAD = jax.vmap(
    jax.value_and_grad(_bridge_logp_single),
    in_axes=(0, None, None, None),
)


@partial(jax.jit, static_argnames=("n_steps",))
def _mala_rejuvenate(
    y: jax.Array,
    key: jax.Array,
    *,
    t: jax.Array,
    beta: jax.Array,
    sigma: jax.Array,
    step_size: jax.Array,
    n_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Apply a fixed-step MALA kernel and return per-step acceptance rates."""

    flat = y.reshape((y.shape[0], 6))

    def body(state: tuple[jax.Array, jax.Array], _: jax.Array) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        current, current_key = state
        current_key, noise_key, accept_key = jax.random.split(current_key, 3)
        logp, grad = _BRIDGE_VALUE_AND_GRAD(current, t, beta, sigma)
        noise = jax.random.normal(noise_key, current.shape, dtype=current.dtype)
        forward_mean = current + step_size * grad
        proposal = forward_mean + jnp.sqrt(2.0 * step_size) * noise
        proposal_logp, proposal_grad = _BRIDGE_VALUE_AND_GRAD(proposal, t, beta, sigma)
        reverse_mean = proposal + step_size * proposal_grad
        log_q_forward = -jnp.sum((proposal - forward_mean) ** 2, axis=-1) / (4.0 * step_size)
        log_q_reverse = -jnp.sum((current - reverse_mean) ** 2, axis=-1) / (4.0 * step_size)
        log_alpha = proposal_logp - logp + log_q_reverse - log_q_forward
        finite = jnp.isfinite(log_alpha) & jnp.all(jnp.isfinite(proposal), axis=-1)
        accept = finite & (jnp.log(jax.random.uniform(accept_key, (current.shape[0],))) < log_alpha)
        updated = jnp.where(accept[:, None], proposal, current)
        return (updated, current_key), jnp.mean(accept.astype(current.dtype))

    (flat, _), acceptance = jax.lax.scan(
        body,
        (flat, key),
        jnp.arange(int(n_steps), dtype=jnp.int32),
    )
    return flat.reshape((-1, 3, 2)), acceptance


_PERMUTATIONS = np.asarray(
    [
        [a, b, c, d]
        for a in range(4)
        for b in range(4)
        for c in range(4)
        for d in range(4)
        if len({a, b, c, d}) == 4
    ],
    dtype=np.int32,
)


@jax.jit
def _symmetry_refresh(y: jax.Array, key: jax.Array) -> jax.Array:
    """Apply independent exact O(2) and S4 symmetry moves."""

    key_perm, key_angle, key_reflect = jax.random.split(key, 3)
    x = reduced_to_coordinates_jax(y)
    permutation_index = jax.random.randint(
        key_perm,
        (x.shape[0],),
        minval=0,
        maxval=int(_PERMUTATIONS.shape[0]),
    )
    permutations = jnp.asarray(_PERMUTATIONS)[permutation_index]
    x = jnp.take_along_axis(x, permutations[:, :, None], axis=1)
    angle = jax.random.uniform(
        key_angle,
        (x.shape[0],),
        minval=0.0,
        maxval=2.0 * jnp.pi,
        dtype=x.dtype,
    )
    reflect = jnp.where(
        jax.random.bernoulli(key_reflect, 0.5, (x.shape[0],)),
        jnp.asarray(1.0, x.dtype),
        jnp.asarray(-1.0, x.dtype),
    )
    x0 = x[..., 0]
    x1 = reflect[:, None] * x[..., 1]
    cosine = jnp.cos(angle)[:, None]
    sine = jnp.sin(angle)[:, None]
    rotated = jnp.stack(
        (cosine * x0 - sine * x1, sine * x0 + cosine * x1),
        axis=-1,
    )
    basis = _helmert_basis_jax(rotated.dtype)
    return jnp.einsum("nk,bnd->bkd", basis, rotated)


def _conditional_ess(log_increment: np.ndarray) -> float:
    values = np.asarray(log_increment, dtype=np.float64)
    values = values - np.max(values)
    weights = np.exp(values)
    return float(np.sum(weights) ** 2 / np.sum(weights * weights))


def _choose_temperature(
    t: float,
    log_ratio: np.ndarray,
    target_cess: float,
    *,
    iterations: int = 48,
) -> float:
    remaining = 1.0 - float(t)
    n = int(log_ratio.shape[0])
    if _conditional_ess(remaining * log_ratio) >= float(target_cess) * n:
        return 1.0
    low = 0.0
    high = remaining
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        if _conditional_ess(middle * log_ratio) < float(target_cess) * n:
            high = middle
        else:
            low = middle
    if low <= 1.0e-12:
        raise RuntimeError("Adaptive SMC temperature increment collapsed to zero.")
    return min(1.0, float(t) + low)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    values = values / np.sum(values)
    cumulative = np.cumsum(values)
    cumulative[-1] = 1.0
    positions = (float(rng.random()) + np.arange(values.shape[0], dtype=np.float64)) / values.shape[0]
    return np.searchsorted(cumulative, positions, side="left").astype(np.int64)


@dataclass(frozen=True)
class DW4SMCConfig:
    source_sigma: float = 1.0
    target_cess: float = 0.80
    mala_steps_per_stage: int = 8
    final_mala_steps: int = 64
    mala_step_size: float = 0.020
    mala_step_decay: float = 0.75
    max_stages: int = 256


def summarize_dw4(samples: np.ndarray) -> dict[str, Any]:
    x = np.asarray(samples, dtype=np.float64).reshape((-1, 4, 2))
    distances = pair_distances_np(x)
    energy = dw4_energy_np(x)
    minimum = np.min(distances, axis=-1)
    radius = np.sqrt(np.mean(np.sum(x * x, axis=-1), axis=-1))
    com = np.linalg.norm(np.mean(x, axis=1), axis=-1)
    return {
        "n": int(x.shape[0]),
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "energy_quantiles": [float(v) for v in np.quantile(energy, [0.001, 0.01, 0.5, 0.99, 0.999])],
        "pair_distance_mean": float(np.mean(distances)),
        "pair_distance_std": float(np.std(distances)),
        "min_pair_quantiles": [float(v) for v in np.quantile(minimum, [0.001, 0.01, 0.5, 0.99])],
        "radius_gyration_mean": float(np.mean(radius)),
        "radius_gyration_std": float(np.std(radius)),
        "com_abs_max": float(np.max(com)),
    }


def run_dw4_smc(
    *,
    beta: float,
    n_particles: int,
    seed: int,
    config: DW4SMCConfig | None = None,
    progress: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate an unweighted COM-free DW4 population at one beta."""

    if not bool(jax.config.x64_enabled):
        raise RuntimeError("DW4 reference-v2 requires JAX_ENABLE_X64=true.")
    cfg = config or DW4SMCConfig()
    if int(n_particles) < 32:
        raise ValueError("n_particles must be at least 32.")
    if not (0.0 < float(cfg.target_cess) < 1.0):
        raise ValueError("target_cess must lie in (0, 1).")
    if float(beta) <= 0.0:
        raise ValueError("beta must be positive.")

    rng = np.random.default_rng(int(seed) + 9173)
    key = jax.random.PRNGKey(int(seed))
    key, init_key = jax.random.split(key)
    y = float(cfg.source_sigma) * jax.random.normal(
        init_key,
        (int(n_particles), 3, 2),
        dtype=jnp.float64,
    )
    ancestry = np.arange(int(n_particles), dtype=np.int64)
    stages: list[dict[str, Any]] = []
    t = 0.0
    log_normalizer = 0.0

    for stage_index in range(int(cfg.max_stages)):
        energy = np.asarray(jax.device_get(dw4_energy_reduced_jax(y)), dtype=np.float64)
        source_logp = np.asarray(
            jax.device_get(log_standard_source_jax(y, float(cfg.source_sigma))),
            dtype=np.float64,
        )
        log_ratio = -float(beta) * energy - source_logp
        next_t = _choose_temperature(t, log_ratio, float(cfg.target_cess))
        delta = next_t - t
        log_increment = delta * log_ratio
        maximum = float(np.max(log_increment))
        unnormalized = np.exp(log_increment - maximum)
        normalizer = float(np.mean(unnormalized))
        log_normalizer += maximum + float(np.log(normalizer))
        weights = unnormalized / np.sum(unnormalized)
        cess = float(1.0 / np.sum(weights * weights))
        indices = _systematic_resample(weights, rng)
        y = y[jnp.asarray(indices)]
        ancestry = ancestry[indices]

        key, symmetry_key, mala_key = jax.random.split(key, 3)
        y = _symmetry_refresh(y, symmetry_key)
        deterministic_step = float(cfg.mala_step_size) / (
            1.0 + float(cfg.mala_step_decay) * float(next_t) * float(beta)
        )
        y, acceptance = _mala_rejuvenate(
            y,
            mala_key,
            t=jnp.asarray(next_t, dtype=y.dtype),
            beta=jnp.asarray(float(beta), dtype=y.dtype),
            sigma=jnp.asarray(float(cfg.source_sigma), dtype=y.dtype),
            step_size=jnp.asarray(deterministic_step, dtype=y.dtype),
            n_steps=int(cfg.mala_steps_per_stage),
        )
        acceptance_np = np.asarray(jax.device_get(acceptance), dtype=np.float64)
        finite = bool(np.isfinite(np.asarray(jax.device_get(y))).all())
        stages.append(
            {
                "stage": int(stage_index + 1),
                "t": float(next_t),
                "delta_t": float(delta),
                "conditional_ess": cess,
                "conditional_ess_fraction": cess / int(n_particles),
                "unique_parent_fraction": float(np.unique(indices).size / int(n_particles)),
                "unique_initial_ancestor_fraction": float(np.unique(ancestry).size / int(n_particles)),
                "mala_step_size": deterministic_step,
                "mala_acceptance_mean": float(np.mean(acceptance_np)),
                "mala_acceptance_min_step": float(np.min(acceptance_np)),
                "mala_acceptance_max_step": float(np.max(acceptance_np)),
                "finite": finite,
                "log_normalizer_estimate": float(log_normalizer),
            }
        )
        if progress:
            print(
                "[dw4-smc] "
                f"beta={float(beta):.3f} stage={stage_index + 1} "
                f"t={next_t:.6f} cess={cess / int(n_particles):.3f} "
                f"accept={float(np.mean(acceptance_np)):.3f} "
                f"lineages={np.unique(ancestry).size}/{int(n_particles)}",
                flush=True,
            )
        if not finite:
            raise FloatingPointError(f"Non-finite DW4 SMC state at stage {stage_index + 1}.")
        t = next_t
        if t >= 1.0 - 1.0e-12:
            break
    else:
        raise RuntimeError(f"DW4 SMC exceeded max_stages={cfg.max_stages}.")

    remaining = int(cfg.final_mala_steps)
    final_acceptance: list[float] = []
    while remaining > 0:
        block = min(16, remaining)
        key, symmetry_key, mala_key = jax.random.split(key, 3)
        y = _symmetry_refresh(y, symmetry_key)
        deterministic_step = float(cfg.mala_step_size) / (
            1.0 + float(cfg.mala_step_decay) * float(beta)
        )
        y, acceptance = _mala_rejuvenate(
            y,
            mala_key,
            t=jnp.asarray(1.0, dtype=y.dtype),
            beta=jnp.asarray(float(beta), dtype=y.dtype),
            sigma=jnp.asarray(float(cfg.source_sigma), dtype=y.dtype),
            step_size=jnp.asarray(deterministic_step, dtype=y.dtype),
            n_steps=block,
        )
        final_acceptance.extend(
            float(v) for v in np.asarray(jax.device_get(acceptance), dtype=np.float64)
        )
        remaining -= block

    x = np.asarray(jax.device_get(reduced_to_coordinates_jax(y)), dtype=np.float64)
    if not np.isfinite(x).all():
        raise FloatingPointError("Non-finite values in final DW4 population.")
    diagnostics = {
        "schema": "adtm.dw4_reference_v2.smc_run.v1",
        "method": "annealed_smc_systematic_resampling_mala",
        "beta": float(beta),
        "seed": int(seed),
        "n_particles": int(n_particles),
        "intrinsic_dimension": 6,
        "ambient_dimension": 8,
        "config": asdict(cfg),
        "n_stages": int(len(stages)),
        "log_normalizer_estimate": float(log_normalizer),
        "unique_initial_ancestor_fraction": float(np.unique(ancestry).size / int(n_particles)),
        "final_mala_acceptance_mean": float(np.mean(final_acceptance)),
        "final_mala_acceptance_min": float(np.min(final_acceptance)),
        "final_mala_acceptance_max": float(np.max(final_acceptance)),
        "stages": stages,
        "summary": summarize_dw4(x),
    }
    return np.asarray(x.reshape((int(n_particles), 8)), dtype=np.float32), diagnostics


def equal_rank_w2(a: np.ndarray, b: np.ndarray) -> float:
    left = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    right = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    n = min(left.size, right.size)
    if n <= 0:
        raise ValueError("Cannot compute W2 from empty arrays.")
    if left.size != n:
        left = left[np.rint(np.linspace(0, left.size - 1, n)).astype(np.int64)]
    if right.size != n:
        right = right[np.rint(np.linspace(0, right.size - 1, n)).astype(np.int64)]
    return float(np.sqrt(np.mean((left - right) ** 2)))


def compare_dw4_samples(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64).reshape((-1, 4, 2))
    y = np.asarray(right, dtype=np.float64).reshape((-1, 4, 2))
    dx = pair_distances_np(x)
    dy = pair_distances_np(y)
    ex = dw4_energy_np(x)
    ey = dw4_energy_np(y)
    min_x = np.min(dx, axis=-1)
    min_y = np.min(dy, axis=-1)
    radius_x = np.sqrt(np.mean(np.sum(x * x, axis=-1), axis=-1))
    radius_y = np.sqrt(np.mean(np.sum(y * y, axis=-1), axis=-1))
    return {
        "energy_w2": equal_rank_w2(ex, ey),
        "pair_distance_w2": equal_rank_w2(dx, dy),
        "min_pair_distance_w2": equal_rank_w2(min_x, min_y),
        "radius_gyration_w2": equal_rank_w2(radius_x, radius_y),
        "energy_mean_delta": float(np.mean(ex) - np.mean(ey)),
        "pair_distance_mean_delta": float(np.mean(dx) - np.mean(dy)),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_sha256_manifest(root: str | Path, manifest: str | Path) -> list[str]:
    base = Path(root)
    errors: list[str] = []
    for raw in Path(manifest).read_text(encoding="ascii").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = base / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"sha256 mismatch: {relative}")
    return errors


def write_sha256_manifest(root: str | Path, paths: Iterable[str | Path]) -> Path:
    base = Path(root)
    output = base / "SHA256SUMS"
    lines: list[str] = []
    for value in sorted((Path(path) for path in paths), key=lambda item: item.as_posix()):
        relative = value.relative_to(base).as_posix()
        lines.append(f"{sha256_file(value)}  {relative}")
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output


__all__ = [
    "DW4SMCConfig",
    "compare_dw4_samples",
    "coordinates_to_reduced_np",
    "dw4_energy_np",
    "dw4_energy_reduced_jax",
    "equal_rank_w2",
    "helmert_basis_np",
    "pair_distances_np",
    "reduced_to_coordinates_np",
    "run_dw4_smc",
    "sha256_file",
    "summarize_dw4",
    "verify_sha256_manifest",
    "write_json",
    "write_sha256_manifest",
]
