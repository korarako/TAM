from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.utils import pairwise_distances, project_mean_free

ALA2_TORSION_INDICES = ((4, 6, 7, 8), (6, 7, 8, 16))


def wasserstein2_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def wasserstein2_1d_n(a: np.ndarray, b: np.ndarray, n_samples: int = 2000, seed: int = 0) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(int(n_samples), a.size, b.size)
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if a.size > n:
        a = a[rng.choice(a.size, size=n, replace=False)]
    if b.size > n:
        b = b[np.random.default_rng(int(seed) + 1).choice(b.size, size=n, replace=False)]
    return wasserstein2_1d(a, b)


def pair_distance_values(samples: np.ndarray, n_particles: int = 22, spatial_dim: int = 3) -> np.ndarray:
    x = jnp.asarray(samples, dtype=jnp.float32)
    d = pairwise_distances(x, n_particles=int(n_particles), spatial_dim=int(spatial_dim))
    return np.asarray(jax.device_get(d), dtype=np.float64)


def torsion_angles(samples: np.ndarray, indices: tuple[tuple[int, int, int, int], ...] = ALA2_TORSION_INDICES) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64).reshape((len(samples), -1, 3))
    out = []
    for i, j, k, l in indices:
        p0, p1, p2, p3 = x[:, i], x[:, j], x[:, k], x[:, l]
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1 = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1.0e-12)
        v = b0 - b1 * np.sum(b0 * b1, axis=-1, keepdims=True)
        w = b2 - b1 * np.sum(b2 * b1, axis=-1, keepdims=True)
        y = np.sum(np.cross(b1, v) * w, axis=-1)
        xdot = np.sum(v * w, axis=-1)
        out.append(np.arctan2(y, xdot))
    return np.stack(out, axis=-1)


def ala2_phi_psi(samples: np.ndarray) -> np.ndarray:
    return torsion_angles(samples)


def circular_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    z = np.exp(1j * np.asarray(values, dtype=np.float64))
    mean_z = np.mean(z)
    return {f"{prefix}_circ_mean": float(np.angle(mean_z)), f"{prefix}_circ_r": float(np.abs(mean_z))}


def _hist_prob(values: np.ndarray, bins: int, range_: tuple[tuple[float, float], ...]) -> np.ndarray:
    hist, _ = np.histogramdd(values, bins=bins, range=range_)
    p = hist.astype(np.float64).reshape(-1)
    s = float(np.sum(p))
    return p / s if s > 0.0 else p


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1.0e-12) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0.0, p * np.log((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0.0, q * np.log((q + eps) / (m + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def energy_values(samples: np.ndarray, problem: Any, chunk_size: int = 256) -> np.ndarray:
    vals = []
    arr = np.asarray(samples, dtype=np.float32)
    for i in range(0, arr.shape[0], int(chunk_size)):
        e = problem.energy_fn(jnp.asarray(arr[i : i + int(chunk_size)], dtype=jnp.float32))
        vals.append(np.asarray(jax.device_get(e), dtype=np.float64))
    return np.concatenate(vals, axis=0) if vals else np.zeros((0,), dtype=np.float64)


def ala2_summary_metrics(
    samples: np.ndarray,
    reference_samples: np.ndarray,
    problem: Any,
    energy_chunk_size: int = 256,
    energy_w2_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    n_particles = int(problem.n_particles or 22)
    spatial_dim = int(problem.spatial_dim or 3)
    samples = np.asarray(jax.device_get(project_mean_free(jnp.asarray(samples, dtype=jnp.float32), n_particles, spatial_dim)), dtype=np.float32)
    reference_samples = np.asarray(jax.device_get(project_mean_free(jnp.asarray(reference_samples, dtype=jnp.float32), n_particles, spatial_dim)), dtype=np.float32)

    sample_energy = energy_values(samples, problem, chunk_size=energy_chunk_size)
    ref_energy = energy_values(reference_samples, problem, chunk_size=energy_chunk_size)
    sample_dist_values = pair_distance_values(samples, n_particles, spatial_dim)
    ref_dist_values = pair_distance_values(reference_samples, n_particles, spatial_dim)
    sample_dist = sample_dist_values.reshape(-1)
    ref_dist = ref_dist_values.reshape(-1)
    sample_phi_psi = ala2_phi_psi(samples)
    ref_phi_psi = ala2_phi_psi(reference_samples)
    sample_energy_finite = sample_energy[np.isfinite(sample_energy)]
    ref_energy_finite = ref_energy[np.isfinite(ref_energy)]
    sample_min_dist = np.min(sample_dist_values, axis=1) if sample_dist_values.ndim == 2 else sample_dist_values
    ref_min_dist = np.min(ref_dist_values, axis=1) if ref_dist_values.ndim == 2 else ref_dist_values

    def finite_stat(values: np.ndarray, fn: Any, default: float = float("nan")) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return default if arr.size == 0 else float(fn(arr))

    rama_range = ((-np.pi, np.pi), (-np.pi, np.pi))
    out: dict[str, float] = {
        "ew2": wasserstein2_1d(sample_energy, ref_energy),
        "ew2_2k": wasserstein2_1d_n(sample_energy, ref_energy, n_samples=energy_w2_samples, seed=seed),
        "energy_w2_samples": int(energy_w2_samples),
        "w2": wasserstein2_1d(sample_dist, ref_dist),
        "min_pair_distance_w2": wasserstein2_1d(sample_min_dist, ref_min_dist),
        "phi_w2": wasserstein2_1d(sample_phi_psi[:, 0], ref_phi_psi[:, 0]),
        "psi_w2": wasserstein2_1d(sample_phi_psi[:, 1], ref_phi_psi[:, 1]),
        "rama_js": js_divergence(_hist_prob(sample_phi_psi, 80, rama_range), _hist_prob(ref_phi_psi, 80, rama_range)),
        "energy_mean": float(np.mean(sample_energy)),
        "energy_std": float(np.std(sample_energy)),
        "md_energy_mean": float(np.mean(ref_energy)),
        "md_energy_std": float(np.std(ref_energy)),
        "energy_finite_fraction": float(np.mean(np.isfinite(sample_energy))),
        "md_energy_finite_fraction": float(np.mean(np.isfinite(ref_energy))),
        "energy_finite_mean": finite_stat(sample_energy_finite, np.mean),
        "energy_finite_q50": finite_stat(sample_energy_finite, lambda x: np.quantile(x, 0.50)),
        "energy_finite_q90": finite_stat(sample_energy_finite, lambda x: np.quantile(x, 0.90)),
        "energy_finite_q99": finite_stat(sample_energy_finite, lambda x: np.quantile(x, 0.99)),
        "md_energy_finite_q50": finite_stat(ref_energy_finite, lambda x: np.quantile(x, 0.50)),
        "md_energy_finite_q90": finite_stat(ref_energy_finite, lambda x: np.quantile(x, 0.90)),
        "md_energy_finite_q99": finite_stat(ref_energy_finite, lambda x: np.quantile(x, 0.99)),
        "min_pair_distance_q01": float(np.quantile(sample_min_dist, 0.01)),
        "min_pair_distance_q05": float(np.quantile(sample_min_dist, 0.05)),
        "min_pair_distance_q50": float(np.quantile(sample_min_dist, 0.50)),
        "md_min_pair_distance_q01": float(np.quantile(ref_min_dist, 0.01)),
        "md_min_pair_distance_q05": float(np.quantile(ref_min_dist, 0.05)),
        "md_min_pair_distance_q50": float(np.quantile(ref_min_dist, 0.50)),
    }
    out.update(circular_stats(sample_phi_psi[:, 0], "phi"))
    out.update(circular_stats(sample_phi_psi[:, 1], "psi"))
    com = samples.reshape((-1, n_particles, spatial_dim)).mean(axis=1)
    out["com_norm_mean"] = float(np.mean(np.linalg.norm(com, axis=1)))
    return out
