from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.problem.base import ProblemSpec
from adj_thermo.utils import ensure_dir


def beta_file_name(beta: float) -> str:
    return f"samples_beta_{float(beta):.2f}.npy"


def sample_langevin(
    problem: ProblemSpec,
    beta: float,
    n_samples: int,
    n_steps: int,
    step_size: float,
    clip_range: float,
    key: jax.Array,
    init_mode: str = "normal",
) -> jax.Array:
    key_init, key_noise = jax.random.split(key)
    if str(init_mode).lower() == "uniform":
        x = jax.random.uniform(
            key_init,
            (int(n_samples), int(problem.dim)),
            minval=-float(clip_range),
            maxval=float(clip_range),
            dtype=jnp.float32,
        )
    else:
        x = jax.random.normal(key_init, (int(n_samples), int(problem.dim)), dtype=jnp.float32)
    x = problem.project_fn(x)
    sqrt_2h = jnp.sqrt(2.0 * jnp.asarray(step_size, dtype=jnp.float32))
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)

    def body(state: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
        noise_key = jax.random.fold_in(key_noise, k)
        grad = problem.project_fn(problem.grad_energy_fn(state))
        noise = problem.project_fn(jax.random.normal(noise_key, state.shape, dtype=state.dtype))
        nxt = state - float(step_size) * beta_arr * grad + sqrt_2h * noise
        nxt = problem.project_fn(nxt)
        if problem.name != "mb2d" and float(clip_range) > 0.0:
            nxt = jnp.clip(nxt, -float(clip_range), float(clip_range))
            nxt = problem.project_fn(nxt)
        return nxt, None

    out, _ = jax.lax.scan(body, x, jnp.arange(int(n_steps)))
    return out.astype(jnp.float32)


def generate_langevin_dataset(
    problem: ProblemSpec,
    beta_grid: list[float] | tuple[float, ...],
    n_samples: int,
    n_steps: int,
    step_size: float,
    clip_range: float,
    key: jax.Array,
    init_mode: str,
    output_dir: str | Path,
) -> dict[float, np.ndarray]:
    output = ensure_dir(output_dir)
    datasets: dict[float, np.ndarray] = {}
    for i, beta in enumerate(beta_grid):
        subkey = jax.random.fold_in(key, i)
        samples = sample_langevin(problem, beta, n_samples, n_steps, step_size, clip_range, subkey, init_mode)
        arr = np.asarray(jax.device_get(samples), dtype=np.float32)
        np.save(output / beta_file_name(beta), arr)
        datasets[float(beta)] = arr
    return datasets
