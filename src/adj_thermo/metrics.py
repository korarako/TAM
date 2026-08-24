from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from adj_thermo.geometric_alignment import joint_alignment_cost_matrix
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



def _sample_matrix(samples: np.ndarray) -> np.ndarray:
    value = np.asarray(samples, dtype=np.float64)
    if value.ndim == 0:
        raise ValueError("samples must have a leading sample dimension")
    feature_size = int(np.prod(value.shape[1:])) if value.ndim > 1 else 1
    return value.reshape((value.shape[0], feature_size))


def _reshape_particle_coords(
    samples: np.ndarray,
    n_particles: int,
    spatial_dim: int,
) -> np.ndarray:
    matrix = _sample_matrix(samples)
    expected = int(n_particles) * int(spatial_dim)
    if matrix.shape[1] != expected:
        raise ValueError(
            "particle configurations have the wrong flattened dimension: "
            f"expected {expected}, received {matrix.shape[1]}"
        )
    return matrix.reshape((matrix.shape[0], int(n_particles), int(spatial_dim)))


def _center_coords(samples: np.ndarray, n_particles: int, spatial_dim: int) -> np.ndarray:
    x = _reshape_particle_coords(samples, n_particles, spatial_dim)
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
    """Solve the exact linear assignment used by empirical equal-weight OT."""

    return linear_sum_assignment(np.asarray(cost, dtype=np.float64))


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


def _sequential_geometric_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Full squared-cost matrix for the literature sequential alignment.

    This reproduces the convention that first fixes a raw-coordinate Hungarian
    particle assignment and then performs one O(s) Procrustes alignment.  It is
    intentionally available for protocol comparison, but it is not invariant
    to independent rotations of the two input configurations.
    """

    out = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            out[i, j] = _hungarian_procrustes_pair_cost(xi, yj)
    return out


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


GEOMETRIC_PROTOCOLS = ("auto", "exact", "joint", "sequential", "legacy-topk")


@dataclass(frozen=True)
class GeometricW2Result:
    value: float | None
    requested_protocol: str
    resolved_protocol: str
    sample_count: int
    n_particles: int | None
    spatial_dim: int | None
    ground_cost: str
    outer_transport: str
    subsampling: str
    seed: int
    alignment_parallel_backend: str | None
    alignment_workers: int | None
    joint_max_iterations: int | None
    candidate_truncation: int | None
    symmetry_consistent: bool
    approximate_particle_alignment: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_geometric_protocol(
    problem: ProblemSpec,
    requested: str,
    exact_permutation_max_particles: int,
) -> str:
    protocol = str(requested).lower()
    if protocol not in GEOMETRIC_PROTOCOLS:
        raise ValueError(
            f"unknown geometric protocol {requested!r}; expected one of {GEOMETRIC_PROTOCOLS}"
        )
    if not problem.n_particles or not problem.spatial_dim:
        if protocol != "auto":
            raise ValueError("non-particle systems support only geometric protocol 'auto'")
        return "euclidean"
    if protocol == "auto":
        return (
            "exact"
            if int(problem.n_particles) <= int(exact_permutation_max_particles)
            else "joint"
        )
    if protocol == "legacy-topk" and int(problem.n_particles) <= int(
        exact_permutation_max_particles
    ):
        # TAM <=0.2.1 already used exact permutation enumeration for small
        # systems; only its larger-system branch used proxy/top-k refinement.
        return "exact"
    if protocol == "exact" and int(problem.n_particles) > int(exact_permutation_max_particles):
        raise ValueError(
            "exact particle-permutation enumeration is disabled for "
            f"{problem.n_particles} particles; increase exact_permutation_max_particles "
            "only if the factorial cost is tractable"
        )
    return protocol


def geometric_w2_result(
    samples: np.ndarray,
    ref: np.ndarray,
    problem: ProblemSpec,
    n_samples: int = 2000,
    seed: int = 0,
    exact_permutation_max_particles: int = 6,
    cost_chunk_size: int = 16,
    dem_refine_top_k: int | None = None,
    *,
    protocol: str = "auto",
    joint_max_iterations: int = 50,
    joint_workers: int = 1,
    joint_parallel_backend: str = "process",
) -> GeometricW2Result:
    """Evaluate empirical geometric W2 with an explicit alignment protocol.

    ``auto`` uses exact permutation enumeration for small systems (including
    DW-4) and full all-pairs iterative joint alignment for larger systems
    (including LJ-13). ``sequential`` reproduces the published convention that
    performs raw-coordinate Hungarian matching before rigid alignment.
    ``legacy-topk`` reproduces the mixed proxy/refinement estimator from TAM
    v0.2.1 and should be used only to audit historical results.
    """

    requested_protocol = str(protocol).lower()
    if dem_refine_top_k is not None and requested_protocol == "auto":
        warnings.warn(
            "Passing dem_refine_top_k with protocol='auto' selects the deprecated "
            "legacy-topk estimator. Pass protocol='legacy-topk' explicitly when "
            "reproducing TAM <=0.2.1 results.",
            FutureWarning,
            stacklevel=2,
        )
        requested_protocol = "legacy-topk"
    elif dem_refine_top_k is not None and requested_protocol != "legacy-topk":
        raise ValueError("dem_refine_top_k is valid only for protocol='legacy-topk'")

    resolved_protocol = _resolve_geometric_protocol(
        problem,
        requested_protocol,
        exact_permutation_max_particles,
    )
    joint_parallel_backend = str(joint_parallel_backend).lower()
    if joint_parallel_backend not in {"process", "thread"}:
        raise ValueError("joint_parallel_backend must be 'process' or 'thread'")
    if int(joint_workers) < 1:
        raise ValueError("joint_workers must be positive")
    if int(joint_max_iterations) < 1:
        raise ValueError("joint_max_iterations must be positive")
    alignment_parallel_backend = (
        ("serial" if int(joint_workers) == 1 else joint_parallel_backend)
        if resolved_protocol == "joint"
        else None
    )
    alignment_workers = int(joint_workers) if resolved_protocol == "joint" else None
    recorded_max_iterations = (
        int(joint_max_iterations) if resolved_protocol == "joint" else None
    )
    species = getattr(problem, "atom_species", None)
    if resolved_protocol != "euclidean" and species is not None:
        if len(species) != int(problem.n_particles):
            raise ValueError("atom_species must contain one entry per particle")
        if len(set(species)) > 1:
            raise ValueError(
                "geometric W2 currently supports permutations of identical particles "
                "only; species-constrained particle assignment is not implemented"
            )

    samples = _sample_matrix(samples)
    ref = _sample_matrix(ref)
    if resolved_protocol == "euclidean" and samples.shape[1] != ref.shape[1]:
        raise ValueError(
            "generated and reference samples must have the same flattened dimension: "
            f"received {samples.shape[1]} and {ref.shape[1]}"
        )
    if resolved_protocol != "euclidean":
        expected_dim = int(problem.n_particles) * int(problem.spatial_dim)
        if samples.shape[1] != expected_dim or ref.shape[1] != expected_dim:
            raise ValueError(
                "particle configurations have the wrong flattened dimension: "
                f"expected {expected_dim}, received {samples.shape[1]} and {ref.shape[1]}"
            )
    samples = samples[np.isfinite(samples).all(axis=1)]
    ref = ref[np.isfinite(ref).all(axis=1)]
    n = min(int(n_samples), samples.shape[0], ref.shape[0])
    if n <= 0:
        return GeometricW2Result(
            value=float("nan"),
            requested_protocol=str(protocol).lower(),
            resolved_protocol=resolved_protocol,
            sample_count=0,
            n_particles=problem.n_particles,
            spatial_dim=problem.spatial_dim,
            ground_cost="squared Frobenius L2; no per-particle normalization",
            outer_transport="exact equal-weight linear assignment",
            subsampling="single deterministic RNG draw (generated, then reference)",
            seed=int(seed),
            alignment_parallel_backend=alignment_parallel_backend,
            alignment_workers=alignment_workers,
            joint_max_iterations=recorded_max_iterations,
            candidate_truncation=None,
            symmetry_consistent=resolved_protocol in {"euclidean", "exact", "joint"},
            approximate_particle_alignment=resolved_protocol in {"joint", "sequential", "legacy-topk"},
        )
    # Keep the TAM <=0.2.1 draw order so DW-4 and non-particle metrics do not
    # change merely because the LJ-13 alignment implementation was corrected.
    rng = np.random.default_rng(int(seed))
    samples = (
        samples[rng.choice(samples.shape[0], size=n, replace=False)]
        if samples.shape[0] > n
        else samples[:n]
    )
    ref = (
        ref[rng.choice(ref.shape[0], size=n, replace=False)]
        if ref.shape[0] > n
        else ref[:n]
    )
    subsampling = "single deterministic RNG draw (generated, then reference)"

    candidate_truncation: int | None = None
    if resolved_protocol == "euclidean":
        x_norm = np.sum(samples * samples, axis=1)
        y_norm = np.sum(ref * ref, axis=1)
        cost = np.maximum(x_norm[:, None] + y_norm[None, :] - 2.0 * samples @ ref.T, 0.0)
    else:
        raw_x = _reshape_particle_coords(
            samples,
            int(problem.n_particles),
            int(problem.spatial_dim),
        )
        raw_y = _reshape_particle_coords(
            ref,
            int(problem.n_particles),
            int(problem.spatial_dim),
        )
        if resolved_protocol == "joint":
            # The pair solver performs centering exactly once.
            cost = joint_alignment_cost_matrix(
                raw_x,
                raw_y,
                max_iterations=int(joint_max_iterations),
                workers=int(joint_workers),
                parallel_backend=joint_parallel_backend,
            )
        else:
            x = raw_x - np.mean(raw_x, axis=1, keepdims=True)
            y = raw_y - np.mean(raw_y, axis=1, keepdims=True)

        if resolved_protocol == "exact":
            try:
                cost = _exact_permutation_geometric_cost_matrix_jax(
                    x,
                    y,
                    chunk_size=max(16, int(cost_chunk_size) * 8),
                )
            except Exception as exc:
                print(
                    f"[geometric_w2] JAX exact cost failed, falling back to NumPy: {exc}",
                    flush=True,
                )
                cost = _exact_permutation_geometric_cost_matrix(
                    x,
                    y,
                    chunk_size=max(1, int(cost_chunk_size)),
                )
        elif resolved_protocol == "sequential":
            cost = _sequential_geometric_cost_matrix(x, y)
        elif resolved_protocol == "legacy-topk":
            requested_top_k = 32 if dem_refine_top_k is None else int(dem_refine_top_k)
            if requested_top_k < 1:
                raise ValueError("dem_refine_top_k must be positive")
            candidate_truncation = (
                None if max(x.shape[0], y.shape[0]) <= 256
                else min(requested_top_k, y.shape[0])
            )
            species = problem.atom_species if getattr(problem, "atom_species", None) else None
            x_sig = _canonicalize_by_distance_signature(x, species)
            y_sig = _canonicalize_by_distance_signature(y, species)
            try:
                approx_cost = _procrustes_cost_matrix_jax(
                    x_sig,
                    y_sig,
                    chunk_size=max(64, int(cost_chunk_size) * 16),
                )
            except Exception as exc:
                print(
                    f"[geometric_w2] JAX proxy cost failed, falling back to NumPy: {exc}",
                    flush=True,
                )
                approx_cost = _procrustes_cost_matrix(
                    x_sig,
                    y_sig,
                    chunk_size=max(1, int(cost_chunk_size)),
                )
            cost = _dem_style_geometric_cost_matrix(
                x,
                y,
                approx_cost=approx_cost,
                refine_top_k=requested_top_k,
            )

    rows, cols = _linear_sum_assignment(cost)
    value = float("nan") if len(rows) == 0 else float(np.sqrt(np.mean(cost[rows, cols])))
    return GeometricW2Result(
        value=value,
        requested_protocol=str(protocol).lower(),
        resolved_protocol=resolved_protocol,
        sample_count=n,
        n_particles=problem.n_particles,
        spatial_dim=problem.spatial_dim,
        ground_cost="squared Frobenius L2; no per-particle normalization",
        outer_transport="exact equal-weight linear assignment",
        subsampling=subsampling,
        seed=int(seed),
        alignment_parallel_backend=alignment_parallel_backend,
        alignment_workers=alignment_workers,
        joint_max_iterations=recorded_max_iterations,
        candidate_truncation=candidate_truncation,
        symmetry_consistent=resolved_protocol in {"euclidean", "exact", "joint"},
        approximate_particle_alignment=resolved_protocol in {"joint", "sequential", "legacy-topk"},
    )


def geometric_w2(
    samples: np.ndarray,
    ref: np.ndarray,
    problem: ProblemSpec,
    n_samples: int = 2000,
    seed: int = 0,
    exact_permutation_max_particles: int = 6,
    cost_chunk_size: int = 16,
    dem_refine_top_k: int | None = None,
    *,
    protocol: str = "auto",
    joint_max_iterations: int = 50,
    joint_workers: int = 1,
    joint_parallel_backend: str = "process",
) -> float | None:
    """Return only the value from :func:`geometric_w2_result`."""

    return geometric_w2_result(
        samples,
        ref,
        problem,
        n_samples=n_samples,
        seed=seed,
        exact_permutation_max_particles=exact_permutation_max_particles,
        cost_chunk_size=cost_chunk_size,
        dem_refine_top_k=dem_refine_top_k,
        protocol=protocol,
        joint_max_iterations=joint_max_iterations,
        joint_workers=joint_workers,
        joint_parallel_backend=joint_parallel_backend,
    ).value
