from __future__ import annotations

import itertools

import numpy as np

from adj_thermo.problem.base import ProblemSpec
from adj_thermo.utils import pairwise_distances


def empirical_wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    a = np.sort(a)
    b = np.sort(b)
    if a.size != n:
        a = a[np.linspace(0, a.size - 1, n).round().astype(np.int64)]
    if b.size != n:
        b = b[np.linspace(0, b.size - 1, n).round().astype(np.int64)]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def energy_stats(samples: np.ndarray, problem: ProblemSpec) -> dict[str, float]:
    import jax
    e = np.asarray(jax.device_get(problem.energy_fn(samples)), dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(np.mean(e)), "std": float(np.std(e)), "min": float(np.min(e)), "max": float(np.max(e))}


def energy_w2(samples: np.ndarray, ref: np.ndarray, problem: ProblemSpec) -> float:
    import jax
    e_samples = np.asarray(jax.device_get(problem.energy_fn(samples)))
    e_ref = np.asarray(jax.device_get(problem.energy_fn(ref)))
    return empirical_wasserstein_1d(e_samples, e_ref)


def _subsample_rows(x: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    x = np.asarray(x)
    n = min(int(n_samples), int(x.shape[0]))
    if n <= 0 or x.shape[0] <= n:
        return x
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return x[idx]


def energy_w2_n(samples: np.ndarray, ref: np.ndarray, problem: ProblemSpec, n_samples: int = 2000, seed: int = 0) -> float:
    """Paper-style energy W2 on a fixed number of generated/reference samples."""
    samples_n = _subsample_rows(samples, n_samples, seed)
    ref_n = _subsample_rows(ref, n_samples, seed + 1)
    return energy_w2(samples_n, ref_n, problem)


def energy_w2_2k(samples: np.ndarray, ref: np.ndarray, problem: ProblemSpec, seed: int = 0) -> float:
    return energy_w2_n(samples, ref, problem, n_samples=2000, seed=seed)


def marginal_w2(samples: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    samples = np.asarray(samples)
    ref = np.asarray(ref)
    out = {f"x{i}_w2": empirical_wasserstein_1d(samples[:, i], ref[:, i]) for i in range(samples.shape[1])}
    out["mean_marginal_w2"] = float(np.mean(list(out.values()))) if out else 0.0
    return out


def pairwise_distance_w2(samples: np.ndarray, ref: np.ndarray, problem: ProblemSpec) -> float | None:
    if not problem.n_particles or not problem.spatial_dim:
        return None
    import jax
    d_samples = np.asarray(jax.device_get(pairwise_distances(samples, problem.n_particles, problem.spatial_dim))).reshape(-1)
    d_ref = np.asarray(jax.device_get(pairwise_distances(ref, problem.n_particles, problem.spatial_dim))).reshape(-1)
    return empirical_wasserstein_1d(d_samples, d_ref)



def _center_coords(samples: np.ndarray, n_particles: int, spatial_dim: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64).reshape((-1, int(n_particles), int(spatial_dim)))
    return x - np.mean(x, axis=1, keepdims=True)


def _procrustes_cost_matrix(x: np.ndarray, y: np.ndarray, chunk_size: int = 64) -> np.ndarray:
    """Squared O(s)-aligned distances between two centered molecular sample sets."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_norm = np.sum(x * x, axis=(1, 2))
    y_norm = np.sum(y * y, axis=(1, 2))
    out = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
    for start in range(0, x.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), x.shape[0])
        cross = np.einsum("jna,inb->ijab", y, x[start:stop], optimize=True)
        singular = np.linalg.svd(cross.reshape((-1, x.shape[2], x.shape[2])), compute_uv=False)
        trace = np.sum(singular, axis=-1).reshape((stop - start, y.shape[0]))
        cost = x_norm[start:stop, None] + y_norm[None, :] - 2.0 * trace
        out[start:stop] = np.maximum(cost, 0.0)
    return out



def _procrustes_cost_matrix_jax(x: np.ndarray, y: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spatial_dim = int(x.shape[2])
    y_j = jnp.asarray(y, dtype=jnp.float32)

    @jax.jit
    def block_cost(x_block: jax.Array) -> jax.Array:
        cross = jnp.einsum("jna,inb->ijab", y_j, x_block, optimize=True)
        singular = jnp.linalg.svd(jnp.reshape(cross, (-1, spatial_dim, spatial_dim)), compute_uv=False)
        trace = jnp.reshape(jnp.sum(singular, axis=-1), (x_block.shape[0], y_j.shape[0]))
        x_norm = jnp.sum(x_block * x_block, axis=(1, 2))
        y_norm = jnp.sum(y_j * y_j, axis=(1, 2))
        cost = x_norm[:, None] + y_norm[None, :] - 2.0 * trace
        return jnp.maximum(cost, 0.0)

    chunks = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        x_block = np.asarray(x[start:stop], dtype=np.float64)
        chunks.append(np.asarray(jax.device_get(block_cost(jnp.asarray(x_block)))))
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


def _exact_permutation_geometric_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int = 16,
) -> np.ndarray:
    """Squared geometric ground costs with exact particle permutation search."""
    n_particles = int(x.shape[1])
    x_norm = np.sum(x * x, axis=(1, 2))
    y_norm = np.sum(y * y, axis=(1, 2))
    out = np.full((x.shape[0], y.shape[0]), np.inf, dtype=np.float64)
    perms = list(itertools.permutations(range(n_particles)))
    for start in range(0, x.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), x.shape[0])
        best = np.full((stop - start, y.shape[0]), np.inf, dtype=np.float64)
        x_chunk = x[start:stop]
        for perm in perms:
            yp = y[:, perm, :]
            cross = np.einsum("jna,inb->ijab", yp, x_chunk, optimize=True)
            singular = np.linalg.svd(cross.reshape((-1, x.shape[2], x.shape[2])), compute_uv=False)
            trace = np.sum(singular, axis=-1).reshape((stop - start, y.shape[0]))
            cost = x_norm[start:stop, None] + y_norm[None, :] - 2.0 * trace
            best = np.minimum(best, cost)
        out[start:stop] = np.maximum(best, 0.0)
    return out



def _exact_permutation_geometric_cost_matrix_jax(
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    """JIT-accelerated exact permutation + O(s) Procrustes ground costs."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    n_particles = int(x.shape[1])
    spatial_dim = int(x.shape[2])
    perms = np.asarray(list(itertools.permutations(range(n_particles))), dtype=np.int32)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    perms_j = jnp.asarray(perms, dtype=jnp.int32)

    @jax.jit
    def block_cost(x_block: jax.Array) -> jax.Array:
        yp = jnp.take(y_j, perms_j, axis=1)
        cross = jnp.einsum("jpna,bnc->bjpac", yp, x_block, optimize=True)
        if spatial_dim == 2:
            frob_sq = jnp.sum(cross * cross, axis=(-2, -1))
            det = cross[..., 0, 0] * cross[..., 1, 1] - cross[..., 0, 1] * cross[..., 1, 0]
            trace = jnp.sqrt(jnp.maximum(frob_sq + 2.0 * jnp.abs(det), 0.0))
        else:
            singular = jnp.linalg.svd(jnp.reshape(cross, (-1, spatial_dim, spatial_dim)), compute_uv=False)
            trace = jnp.reshape(jnp.sum(singular, axis=-1), (x_block.shape[0], y_j.shape[0], perms_j.shape[0]))
        x_norm = jnp.sum(x_block * x_block, axis=(1, 2))
        y_norm = jnp.sum(y_j * y_j, axis=(1, 2))
        cost = x_norm[:, None, None] + y_norm[None, :, None] - 2.0 * trace
        return jnp.maximum(jnp.min(cost, axis=2), 0.0)

    chunks = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        x_block = np.asarray(x[start:stop], dtype=np.float64)
        chunks.append(np.asarray(jax.device_get(block_cost(jnp.asarray(x_block)))))
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


def _canonicalize_by_distance_signature(coords: np.ndarray, atom_species: tuple[int, ...] | None = None) -> np.ndarray:
    """Approximate permutation handling for larger identical-particle systems.

    Each particle is assigned a signature given by its sorted distances to all
    other particles.  Sorting these signatures gives a deterministic particle
    order before the orthogonal Procrustes alignment.
    """
    coords = np.asarray(coords, dtype=np.float64)
    out = np.empty_like(coords)
    species = None if atom_species is None else np.asarray(atom_species, dtype=np.int64)
    for i, x in enumerate(coords):
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        sig = np.sort(dist, axis=-1)
        keys = [sig[:, k] for k in range(sig.shape[1] - 1, -1, -1)]
        if species is not None and species.shape[0] == x.shape[0]:
            keys.append(species)
        order = np.lexsort(tuple(keys))
        out[i] = x[order]
    return out


def _orthogonal_procrustes_sq_cost(x: np.ndarray, y: np.ndarray) -> float:
    cross = y.T @ x
    trace = float(np.sum(np.linalg.svd(cross, compute_uv=False)))
    cost = float(np.sum(x * x) + np.sum(y * y) - 2.0 * trace)
    return max(cost, 0.0)


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except Exception:
        # Fallback keeps the metric available in minimal environments.  It is a
        # greedy one-to-one matching, not exact OT.
        remaining = set(range(cost.shape[1]))
        rows = []
        cols = []
        for i in range(cost.shape[0]):
            if not remaining:
                break
            j = min(remaining, key=lambda c: cost[i, c])
            rows.append(i)
            cols.append(j)
            remaining.remove(j)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _hungarian_reorder_to_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reorder particles of y to match x using a per-configuration assignment."""
    particle_cost = np.sqrt(np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    rows, cols = _linear_sum_assignment(particle_cost)
    y_reordered = np.empty_like(y)
    y_reordered[rows] = y[cols]
    return y_reordered


def _hungarian_procrustes_pair_cost(x: np.ndarray, y: np.ndarray) -> float:
    y_reordered = _hungarian_reorder_to_reference(x, y)
    return _orthogonal_procrustes_sq_cost(x, y_reordered)


def _dem_hungarian_procrustes_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Strict DEM-style ground distances for all sample pairs.

    For each pair of structures, particles in y are assigned to x by Hungarian
    matching, followed by orthogonal Procrustes alignment.  The returned matrix
    contains L2 distances, matching DEM's `eot` cost matrix convention.
    """
    out = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            out[i, j] = np.sqrt(_hungarian_procrustes_pair_cost(xi, yj))
    return out


def _dem_style_geometric_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    approx_cost: np.ndarray | None = None,
    refine_top_k: int = 32,
    exact_all_max_samples: int = 256,
) -> np.ndarray:
    """DEM-style particle matching + O(s) Procrustes sample costs.

    DEM computes, for each pair of structures, a Hungarian particle assignment
    followed by a Kabsch/Procrustes rigid alignment.  Doing this for every pair
    of 2000 x 2000 LJ13 structures is expensive, so for larger evaluations we
    refine only the nearest candidates under the fast distance-signature
    Procrustes approximation and keep the approximation elsewhere.  The final
    sample-level OT still runs on the full cost matrix.
    """
    n, m = int(x.shape[0]), int(y.shape[0])
    if approx_cost is None:
        species = None
        xc = _canonicalize_by_distance_signature(x, species)
        yc = _canonicalize_by_distance_signature(y, species)
        approx_cost = _procrustes_cost_matrix(xc, yc)

    cost = np.asarray(approx_cost, dtype=np.float64).copy()
    if n <= 0 or m <= 0:
        return cost

    if max(n, m) <= int(exact_all_max_samples):
        candidate_cols = [np.arange(m, dtype=np.int64) for _ in range(n)]
    else:
        k = max(1, min(int(refine_top_k), m))
        candidate_cols = []
        for i in range(n):
            if k == m:
                cols = np.arange(m, dtype=np.int64)
            else:
                cols = np.argpartition(cost[i], k - 1)[:k]
            candidate_cols.append(cols.astype(np.int64, copy=False))

    for i, cols in enumerate(candidate_cols):
        xi = x[i]
        for j in cols:
            cost[i, int(j)] = _hungarian_procrustes_pair_cost(xi, y[int(j)])
    return cost


def dem_eq_emd2(
    samples: np.ndarray,
    ref: np.ndarray,
    problem: ProblemSpec,
    n_samples: int = 500,
    seed: int = 0,
) -> float | None:
    """DEM-style equivariant EMD2 approximation.

    This intentionally mirrors `jarridrb/DEM`'s `eot`: each structure-pair cost
    is computed by Hungarian particle assignment plus Kabsch/Procrustes rigid
    alignment, and sample-level OT is solved on the resulting L2-distance cost
    matrix.  This is not the same as the paper-style sqrt(OT over D^2) used by
    `geometric_w2`, so it is reported as a separate metric.
    """
    if not problem.n_particles or not problem.spatial_dim:
        return None
    samples = np.asarray(samples, dtype=np.float64).reshape((len(samples), -1))
    ref = np.asarray(ref, dtype=np.float64).reshape((len(ref), -1))
    samples = samples[np.isfinite(samples).all(axis=1)]
    ref = ref[np.isfinite(ref).all(axis=1)]
    n = min(int(n_samples), samples.shape[0], ref.shape[0])
    if n <= 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if samples.shape[0] > n:
        samples = samples[rng.choice(samples.shape[0], size=n, replace=False)]
    else:
        samples = samples[:n]
    if ref.shape[0] > n:
        ref = ref[rng.choice(ref.shape[0], size=n, replace=False)]
    else:
        ref = ref[:n]

    x = _center_coords(samples, int(problem.n_particles), int(problem.spatial_dim))
    y = _center_coords(ref, int(problem.n_particles), int(problem.spatial_dim))
    cost = _dem_hungarian_procrustes_distance_matrix(x, y)
    rows, cols = _linear_sum_assignment(cost)
    if len(rows) == 0:
        return float("nan")
    return float(np.mean(cost[rows, cols]))


def geometric_w2(
    samples: np.ndarray,
    ref: np.ndarray,
    problem: ProblemSpec,
    n_samples: int = 2000,
    seed: int = 0,
    exact_permutation_max_particles: int = 6,
    cost_chunk_size: int = 16,
    dem_refine_top_k: int = 32,
) -> float | None:
    """Geometric Wasserstein-2 over samples.

    The ground distance is invariant to translations and orthogonal transforms
    through Procrustes alignment.  For small systems such as dw4, particle
    permutations are minimized exactly.  For larger systems such as lj13, we
    follow the DEM-style approximation: a fast distance-signature Procrustes
    cost matrix is refined with per-pair Hungarian particle matching + rigid
    alignment for the nearest candidate pairs before solving sample-level OT.
    """
    samples = np.asarray(samples, dtype=np.float64).reshape((len(samples), -1))
    ref = np.asarray(ref, dtype=np.float64).reshape((len(ref), -1))
    finite_samples = np.isfinite(samples).all(axis=1)
    finite_ref = np.isfinite(ref).all(axis=1)
    samples = samples[finite_samples]
    ref = ref[finite_ref]
    n = min(int(n_samples), samples.shape[0], ref.shape[0])
    if n <= 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if samples.shape[0] > n:
        samples = samples[rng.choice(samples.shape[0], size=n, replace=False)]
    else:
        samples = samples[:n]
    if ref.shape[0] > n:
        ref = ref[rng.choice(ref.shape[0], size=n, replace=False)]
    else:
        ref = ref[:n]

    if not problem.n_particles or not problem.spatial_dim:
        x_norm = np.sum(samples * samples, axis=1)
        y_norm = np.sum(ref * ref, axis=1)
        cost = x_norm[:, None] + y_norm[None, :] - 2.0 * samples @ ref.T
        rows, cols = _linear_sum_assignment(np.maximum(cost, 0.0))
        if len(rows) == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.maximum(cost[rows, cols], 0.0))))

    x = _center_coords(samples, int(problem.n_particles), int(problem.spatial_dim))
    y = _center_coords(ref, int(problem.n_particles), int(problem.spatial_dim))
    if int(problem.n_particles) <= int(exact_permutation_max_particles):
        try:
            cost = _exact_permutation_geometric_cost_matrix_jax(x, y, chunk_size=max(16, int(cost_chunk_size) * 8))
        except Exception as exc:
            print(f"[geometric_w2] JAX exact cost failed, falling back to NumPy: {exc}", flush=True)
            cost = _exact_permutation_geometric_cost_matrix(x, y, chunk_size=max(1, int(cost_chunk_size)))
    else:
        species = problem.atom_species if getattr(problem, "atom_species", None) else None
        x_sig = _canonicalize_by_distance_signature(x, species)
        y_sig = _canonicalize_by_distance_signature(y, species)
        try:
            approx_cost = _procrustes_cost_matrix_jax(x_sig, y_sig, chunk_size=max(64, int(cost_chunk_size) * 16))
        except Exception as exc:
            print(f"[geometric_w2] JAX Procrustes cost failed, falling back to NumPy: {exc}", flush=True)
            approx_cost = _procrustes_cost_matrix(x_sig, y_sig, chunk_size=max(1, int(cost_chunk_size)))
        cost = _dem_style_geometric_cost_matrix(
            x,
            y,
            approx_cost=approx_cost,
            refine_top_k=int(dem_refine_top_k),
        )
    rows, cols = _linear_sum_assignment(cost)
    if len(rows) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(cost[rows, cols])))
