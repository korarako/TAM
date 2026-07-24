"""Batched JAX Langevin sampler for the center-of-mass-free LJ13 system."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.langevin import beta_file_name
from adj_thermo.problem import make_problem
from adj_thermo.utils import ensure_dir

N_PARTICLES = 13
SPATIAL_DIM = 3
DIM = N_PARTICLES * SPATIAL_DIM


def _load_initial_positions(path: str | Path | None, n_chains: int, seed: int) -> np.ndarray | None:
    if path is None or str(path) == "":
        return None
    arr = np.load(path, allow_pickle=True).astype(np.float32).reshape((-1, DIM))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(arr.shape[0], size=int(n_chains), replace=arr.shape[0] < int(n_chains))
    return arr[idx]


def _initial_state(key: jax.Array, init_file: str | Path | None, n_chains: int, seed: int) -> jax.Array:
    initial = _load_initial_positions(init_file, n_chains, seed)
    if initial is None:
        initial = jax.random.normal(key, (int(n_chains), DIM), dtype=jnp.float32)
    return make_problem("lj13").project_fn(jnp.asarray(initial, dtype=jnp.float32))


def sample_lj13_langevin(
    beta: float,
    n_samples: int,
    warmup_steps: int = 2000,
    step_size: float = 1.0e-4,
    seed: int = 0,
    init_file: str | Path | None = None,
    n_chains: int = 1024,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample LJ13 with unadjusted Langevin dynamics and mean-free noise."""

    problem = make_problem("lj13")
    n_samples = int(n_samples)
    n_chains = max(1, int(n_chains))
    n_sample_steps = int(math.ceil(n_samples / n_chains))
    beta_arr = jnp.asarray(float(beta), dtype=jnp.float32)
    h = jnp.asarray(float(step_size), dtype=jnp.float32)
    sqrt_2h = jnp.sqrt(2.0 * h)

    def step(x: jax.Array, key: jax.Array) -> jax.Array:
        noise = problem.project_fn(jax.random.normal(key, x.shape, dtype=x.dtype))
        proposal = x - h * beta_arr * problem.grad_energy_fn(x) + sqrt_2h * noise
        return problem.project_fn(proposal)

    @jax.jit
    def run(key: jax.Array, x0: jax.Array) -> jax.Array:
        def warm_step(x: jax.Array, index: jax.Array) -> tuple[jax.Array, None]:
            return step(x, jax.random.fold_in(key, index)), None

        def sample_step(x: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
            x_next = step(x, jax.random.fold_in(key, index))
            return x_next, x_next

        x_warm, _ = jax.lax.scan(warm_step, x0, jnp.arange(int(warmup_steps), dtype=jnp.int32))
        _, samples = jax.lax.scan(
            sample_step,
            x_warm,
            jnp.arange(int(warmup_steps), int(warmup_steps) + n_sample_steps, dtype=jnp.int32),
        )
        return samples

    key = jax.random.PRNGKey(int(seed))
    key_init, key_run = jax.random.split(key)
    samples = run(key_run, _initial_state(key_init, init_file, n_chains, int(seed)))
    array = np.asarray(jax.device_get(samples), dtype=np.float32).reshape((-1, DIM))[:n_samples]
    return array, {
        "step_size": float(step_size),
        "n_chains": float(n_chains),
        "warmup_steps": float(warmup_steps),
    }


def generate_lj13_langevin_dataset(
    beta_grid: list[float] | tuple[float, ...],
    n_samples: int,
    warmup_steps: int,
    output_dir: str | Path,
    seed: int = 0,
    step_size: float = 1.0e-4,
    init_file: str | Path | None = None,
    n_chains: int = 1024,
) -> dict[float, np.ndarray]:
    output = ensure_dir(output_dir)
    datasets: dict[float, np.ndarray] = {}
    for index, beta_value in enumerate(beta_grid):
        beta = float(beta_value)
        print(
            f"[lj13-langevin] beta={beta:.2f} n_samples={int(n_samples)} "
            f"chains={int(n_chains)} step_size={float(step_size):.3g}",
            flush=True,
        )
        samples, _ = sample_lj13_langevin(
            beta,
            int(n_samples),
            warmup_steps=int(warmup_steps),
            step_size=float(step_size),
            seed=int(seed) + index * 1009,
            init_file=init_file,
            n_chains=int(n_chains),
        )
        np.save(output / beta_file_name(beta), samples)
        datasets[beta] = samples
    return datasets
