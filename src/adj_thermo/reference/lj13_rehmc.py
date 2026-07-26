"""Replica-exchange HMC references for the BMS LJ-13 benchmark.

The target is BMS Eq. (234), with all published constants set to one,

    U(x) = sum_{i<j} [(1/r_ij)^12 - 2 (1/r_ij)^6]
           + sum_i ||x_i - x_COM||^2.

Sampling is performed in a fixed 36-dimensional orthonormal COM-free
coordinate system.  Every molecular-dynamics trajectory and every
temperature exchange has a Metropolis correction.  The module deliberately
does not read the legacy LJ13 arrays: independent initial states are built
from compact structures, so a new reference cannot silently inherit the old
ULA bias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import math
from statistics import NormalDist
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np


N_PARTICLES = 13
SPATIAL_DIM = 3
AMBIENT_DIM = N_PARTICLES * SPATIAL_DIM
INTRINSIC_DIM = (N_PARTICLES - 1) * SPATIAL_DIM
PAIR_I, PAIR_J = np.triu_indices(N_PARTICLES, k=1)


def helmert_basis_np() -> np.ndarray:
    """Return a deterministic 13x12 orthonormal COM-free Helmert basis."""

    basis = np.zeros((N_PARTICLES, N_PARTICLES - 1), dtype=np.float64)
    for column in range(N_PARTICLES - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def reduced_to_coordinates_np(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64).reshape(
        (*np.shape(y)[:-1], N_PARTICLES - 1, SPATIAL_DIM)
    )
    return np.einsum("nk,...kd->...nd", helmert_basis_np(), values)


def coordinates_to_reduced_np(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).reshape(
        (*np.shape(x)[:-1], N_PARTICLES, SPATIAL_DIM)
    )
    values = values - np.mean(values, axis=-2, keepdims=True)
    reduced = np.einsum("nk,...nd->...kd", helmert_basis_np(), values)
    return reduced.reshape((*values.shape[:-2], INTRINSIC_DIM))


def _basis_jax(dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(helmert_basis_np(), dtype=dtype)


def reduced_to_coordinates_jax(y: jax.Array) -> jax.Array:
    values = jnp.asarray(y).reshape((-1, N_PARTICLES - 1, SPATIAL_DIM))
    return jnp.einsum("nk,bkd->bnd", _basis_jax(values.dtype), values)


def lj13_energy_np(x: np.ndarray, min_distance: float = 1.0e-6) -> np.ndarray:
    """Independent NumPy implementation of the exact ADTM/BMS target."""

    values = np.asarray(x, dtype=np.float64).reshape((-1, N_PARTICLES, SPATIAL_DIM))
    values = values - np.mean(values, axis=1, keepdims=True)
    delta = values[:, PAIR_I] - values[:, PAIR_J]
    distance = np.linalg.norm(delta, axis=-1)
    distance = np.maximum(distance, float(min_distance))
    inv6 = distance**-6
    pair_energy = np.sum(inv6 * inv6 - 2.0 * inv6, axis=-1)
    confinement = np.sum(values * values, axis=(1, 2))
    return pair_energy + confinement


def lj13_virial_np(x: np.ndarray, min_distance: float = 1.0e-6) -> np.ndarray:
    """Analytic ``x·grad U`` in the 36D COM-free support."""

    values = np.asarray(x, dtype=np.float64).reshape((-1, N_PARTICLES, SPATIAL_DIM))
    values = values - np.mean(values, axis=1, keepdims=True)
    delta = values[:, PAIR_I] - values[:, PAIR_J]
    distance = np.linalg.norm(delta, axis=-1)
    distance = np.maximum(distance, float(min_distance))
    inv6 = distance**-6
    pair_virial = np.sum(-12.0 * inv6 * inv6 + 12.0 * inv6, axis=-1)
    confinement_virial = 2.0 * np.sum(values * values, axis=(1, 2))
    return pair_virial + confinement_virial


def lj13_energy_reduced_jax(y: jax.Array, min_distance: float = 1.0e-6) -> jax.Array:
    values = jnp.asarray(y).reshape((-1, INTRINSIC_DIM))
    x = reduced_to_coordinates_jax(values)
    pair_i = jnp.asarray(PAIR_I)
    pair_j = jnp.asarray(PAIR_J)
    delta = x[:, pair_i] - x[:, pair_j]
    distance_squared = jnp.sum(delta * delta, axis=-1)
    distance_squared = jnp.maximum(
        distance_squared,
        jnp.asarray(float(min_distance) ** 2, dtype=x.dtype),
    )
    inv6 = distance_squared**-3
    pair_energy = jnp.sum(inv6 * inv6 - 2.0 * inv6, axis=-1)
    confinement = jnp.sum(x * x, axis=(1, 2))
    return pair_energy + confinement


def _single_energy(y: jax.Array) -> jax.Array:
    return lj13_energy_reduced_jax(y[None, :])[0]


_BATCH_VALUE_AND_GRAD = jax.vmap(jax.value_and_grad(_single_energy))


def energy_and_grad_reduced_jax(y: jax.Array) -> tuple[jax.Array, jax.Array]:
    values = jnp.asarray(y)
    leading = values.shape[:-1]
    flat = values.reshape((-1, INTRINSIC_DIM))
    energy, gradient = _BATCH_VALUE_AND_GRAD(flat)
    return energy.reshape(leading), gradient.reshape((*leading, INTRINSIC_DIM))


@dataclass(frozen=True)
class LJ13REHMCConfig:
    beta_min: float = 0.15
    beta_max: float = 1.20
    n_replicas: int = 16
    target_betas: tuple[float, ...] = (0.8, 1.0, 1.2)
    n_ensembles: int = 4
    warmup_rounds: int = 2_000
    settle_rounds: int = 500
    production_rounds: int = 5_000
    storage_stride: int = 1
    block_rounds: int = 50
    leapfrog_min: int = 5
    leapfrog_max: int = 15
    # A fixed-step scan on collision-free starts found that 5e-4--1e-2 gives
    # essentially unit acceptance and short moves, while ~3e-2 reaches the
    # useful 0.7--0.9 regime.  Start at 2e-2 and let warmup adapt each beta
    # slot independently before a frozen-step settle.
    initial_step_size: float = 2.0e-2
    target_accept: float = 0.80
    adaptation_rate: float = 0.20
    minimum_step_size: float = 1.0e-6
    # T1 with ceilings of 4e-2 and 2.7e-2 mixed well but produced respectively
    # ~1e-4 and ~1e-5 large-Hamiltonian-error proposals.  Use a conservative
    # 2e-2 ceiling for the pre-registered zero-divergence production gate.
    maximum_step_size: float = 2.0e-2
    seed: int = 61001


class REHMCState(NamedTuple):
    y: jax.Array
    energy: jax.Array
    labels: jax.Array
    key: jax.Array


def make_beta_ladder(config: LJ13REHMCConfig) -> np.ndarray:
    """Build a geometric ladder that contains every requested target beta."""

    if config.n_replicas < len(config.target_betas) + 2:
        raise ValueError("n_replicas is too small for the requested target betas")
    base_count = config.n_replicas - len(config.target_betas)
    base = np.geomspace(config.beta_min, config.beta_max, base_count)
    values = np.concatenate((base, np.asarray(config.target_betas, dtype=np.float64)))
    values = np.unique(np.round(values, decimals=14))
    if values.size != config.n_replicas:
        # A target may coincide with a geometric point.  Add midpoints in the
        # largest log-beta gaps until the requested size is recovered.
        while values.size < config.n_replicas:
            gaps = np.diff(np.log(values))
            index = int(np.argmax(gaps))
            midpoint = math.sqrt(float(values[index]) * float(values[index + 1]))
            values = np.unique(np.append(values, midpoint))
    if values.size != config.n_replicas:
        raise RuntimeError("failed to construct a unique beta ladder")
    values.sort()
    for target in config.target_betas:
        if not np.any(values == float(target)):
            raise RuntimeError(f"target beta {target} is absent from the ladder")
    return values


def _icosahedron_np() -> np.ndarray:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    shell: list[tuple[float, float, float]] = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            shell.append((0.0, a, b))
            shell.append((a, b, 0.0))
            shell.append((b, 0.0, a))
    x = np.concatenate((np.zeros((1, 3)), 0.5 * np.asarray(shell)), axis=0)
    return x - np.mean(x, axis=0, keepdims=True)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _random_compact_cluster(rng: np.random.Generator) -> np.ndarray:
    # A loose 0.55 collision check creates r^-12 energies around 1e5 and can
    # leave a replica permanently rejecting.  Build a broader Poisson cluster
    # whose pair distances are already in a numerically mobile regime.
    for _cluster_attempt in range(100):
        points: list[np.ndarray] = []
        for _ in range(N_PARTICLES):
            accepted = False
            for _attempt in range(20_000):
                direction = rng.normal(size=3)
                direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
                radius = 2.0 * rng.random() ** (1.0 / 3.0)
                candidate = radius * direction
                if not points or min(
                    np.linalg.norm(candidate - point) for point in points
                ) >= 0.85:
                    points.append(candidate)
                    accepted = True
                    break
            if not accepted:
                break
        if len(points) == N_PARTICLES:
            x = np.asarray(points)
            x = x - np.mean(x, axis=0, keepdims=True)
            if float(lj13_energy_np(x[None, ...])[0]) < 500.0:
                return x
    return _icosahedron_np() + rng.normal(scale=0.08, size=(N_PARTICLES, 3))


def initialize_ensembles_np(
    config: LJ13REHMCConfig,
    betas: np.ndarray,
) -> np.ndarray:
    """Create independent, collision-free starts without legacy data."""

    rng = np.random.default_rng(int(config.seed))
    output = np.empty(
        (config.n_ensembles, betas.size, INTRINSIC_DIM),
        dtype=np.float64,
    )
    for ensemble in range(config.n_ensembles):
        base = _icosahedron_np() if ensemble % 2 == 0 else _random_compact_cluster(rng)
        for replica in range(betas.size):
            for _attempt in range(1_000):
                x = base + rng.normal(
                    scale=0.025 + 0.075 * (1.0 - float(betas[replica] / betas[-1])),
                    size=base.shape,
                )
                x = x - np.mean(x, axis=0, keepdims=True)
                x = x[rng.permutation(N_PARTICLES)] @ _random_rotation(rng)
                distance = np.linalg.norm(x[PAIR_I] - x[PAIR_J], axis=-1)
                candidate_energy = float(lj13_energy_np(x[None, ...])[0])
                if float(np.min(distance)) > 0.70 and candidate_energy < 500.0:
                    output[ensemble, replica] = coordinates_to_reduced_np(
                        x.reshape((1, AMBIENT_DIM))
                    )[0]
                    break
            else:
                raise RuntimeError("failed to build a collision-free LJ13 initial state")
    return output


def _hmc_update(
    y: jax.Array,
    energy: jax.Array,
    beta: jax.Array,
    step_size: jax.Array,
    key: jax.Array,
    n_leapfrog: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key_momentum, key_accept = jax.random.split(key)
    momentum0 = jax.random.normal(key_momentum, y.shape, dtype=y.dtype)
    _, gradient0 = energy_and_grad_reduced_jax(y)
    beta_b = beta[None, :, None]
    eps = step_size[None, :, None]
    momentum = momentum0 - 0.5 * eps * beta_b * gradient0

    def leapfrog_body(
        index: int,
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        position, momenta, value, gradient = carry
        position = position + eps * momenta
        value, gradient = energy_and_grad_reduced_jax(position)
        coefficient = jnp.where(
            index == n_leapfrog - 1,
            jnp.asarray(0.5, dtype=position.dtype),
            jnp.asarray(1.0, dtype=position.dtype),
        )
        momenta = momenta - coefficient * eps * beta_b * gradient
        return position, momenta, value, gradient

    proposal, momentum, proposal_energy, _ = jax.lax.fori_loop(
        0,
        n_leapfrog,
        leapfrog_body,
        (y, momentum, energy, gradient0),
    )
    current_h = beta[None, :] * energy + 0.5 * jnp.sum(momentum0 * momentum0, axis=-1)
    proposal_h = beta[None, :] * proposal_energy + 0.5 * jnp.sum(
        momentum * momentum, axis=-1
    )
    energy_error = proposal_h - current_h
    finite = (
        jnp.isfinite(energy_error)
        & jnp.isfinite(proposal_energy)
        & jnp.all(jnp.isfinite(proposal), axis=-1)
    )
    log_uniform = jnp.log(
        jax.random.uniform(key_accept, energy_error.shape, dtype=y.dtype)
    )
    accept = finite & (log_uniform < jnp.minimum(-energy_error, 0.0))
    updated_y = jnp.where(accept[..., None], proposal, y)
    updated_energy = jnp.where(accept, proposal_energy, energy)
    divergence = (~finite) | (jnp.abs(energy_error) > 1.0e3)
    return updated_y, updated_energy, accept, divergence, energy_error


def _swap_phase(
    y: jax.Array,
    energy: jax.Array,
    labels: jax.Array,
    beta: jax.Array,
    key: jax.Array,
    parity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    left_np = np.arange(parity, int(beta.shape[0]) - 1, 2, dtype=np.int32)
    right_np = left_np + 1
    left = jnp.asarray(left_np)
    right = jnp.asarray(right_np)
    log_alpha = (beta[left] - beta[right])[None, :] * (
        energy[:, left] - energy[:, right]
    )
    accept = jnp.log(
        jax.random.uniform(key, log_alpha.shape, dtype=energy.dtype)
    ) < jnp.minimum(log_alpha, 0.0)

    left_y = y[:, left]
    right_y = y[:, right]
    y = y.at[:, left].set(jnp.where(accept[..., None], right_y, left_y))
    y = y.at[:, right].set(jnp.where(accept[..., None], left_y, right_y))
    left_e = energy[:, left]
    right_e = energy[:, right]
    energy = energy.at[:, left].set(jnp.where(accept, right_e, left_e))
    energy = energy.at[:, right].set(jnp.where(accept, left_e, right_e))
    left_label = labels[:, left]
    right_label = labels[:, right]
    labels = labels.at[:, left].set(jnp.where(accept, right_label, left_label))
    labels = labels.at[:, right].set(jnp.where(accept, left_label, right_label))

    pair_acceptance = jnp.zeros(
        (energy.shape[0], beta.shape[0] - 1),
        dtype=jnp.bool_,
    )
    pair_acceptance = pair_acceptance.at[:, left].set(accept)
    return y, energy, labels, pair_acceptance


def _symmetry_refresh(y: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Apply independent exact S13 and Haar-O(3) symmetry moves.

    Cartesian orientations and identical-particle labels are nuisance modes.
    Refreshing them prevents apparently stable molecular structures from
    retaining the handful of orientations/permutations used at initialization.
    The move is orthogonal, volume preserving and accepted with probability one.
    """

    leading = y.shape[:-1]
    flat_y = y.reshape((-1, INTRINSIC_DIM))
    x = reduced_to_coordinates_jax(flat_y)
    n_states = flat_y.shape[0]
    key_perm, key_quaternion, key_reflection = jax.random.split(key, 3)
    permutation_keys = jax.random.split(key_perm, n_states)
    permutations = jax.vmap(
        lambda value: jax.random.permutation(value, N_PARTICLES)
    )(permutation_keys)
    x = jnp.take_along_axis(x, permutations[..., None], axis=1)

    uniform = jax.random.uniform(
        key_quaternion,
        (n_states, 3),
        dtype=x.dtype,
    )
    u1, u2, u3 = uniform[:, 0], uniform[:, 1], uniform[:, 2]
    qx = jnp.sqrt(1.0 - u1) * jnp.sin(2.0 * jnp.pi * u2)
    qy = jnp.sqrt(1.0 - u1) * jnp.cos(2.0 * jnp.pi * u2)
    qz = jnp.sqrt(u1) * jnp.sin(2.0 * jnp.pi * u3)
    qw = jnp.sqrt(u1) * jnp.cos(2.0 * jnp.pi * u3)
    rotation = jnp.stack(
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
        axis=-1,
    ).reshape((n_states, 3, 3))
    x = jnp.einsum("bij,bnj->bni", rotation, x)
    reflection = jnp.where(
        jax.random.bernoulli(key_reflection, 0.5, (n_states,)),
        jnp.asarray(1.0, dtype=x.dtype),
        jnp.asarray(-1.0, dtype=x.dtype),
    )
    reflection_matrix = jnp.stack(
        (reflection, jnp.ones_like(reflection), jnp.ones_like(reflection)),
        axis=-1,
    )
    x = x * reflection_matrix[:, None, :]
    refreshed = jnp.einsum("nk,bnd->bkd", _basis_jax(x.dtype), x)
    refreshed = refreshed.reshape((*leading, INTRINSIC_DIM))
    energy = lj13_energy_reduced_jax(refreshed.reshape((-1, INTRINSIC_DIM)))
    return refreshed, energy.reshape(leading)


@partial(
    jax.jit,
    static_argnames=("n_rounds", "leapfrog_min", "leapfrog_max"),
)
def _run_block(
    state: REHMCState,
    beta: jax.Array,
    step_size: jax.Array,
    *,
    n_rounds: int,
    leapfrog_min: int,
    leapfrog_max: int,
) -> tuple[REHMCState, dict[str, jax.Array]]:
    def one_round(
        carry: REHMCState,
        _unused: jax.Array,
    ) -> tuple[REHMCState, dict[str, jax.Array]]:
        y, energy, labels, key = carry
        key, key_l, key_hmc, key_even, key_odd, key_symmetry = jax.random.split(
            key, 6
        )
        n_leapfrog = jax.random.randint(
            key_l,
            (),
            minval=int(leapfrog_min),
            maxval=int(leapfrog_max) + 1,
        )
        y, energy, hmc_accept, divergence, energy_error = _hmc_update(
            y,
            energy,
            beta,
            step_size,
            key_hmc,
            n_leapfrog,
        )
        y, energy, labels, swap_even = _swap_phase(
            y, energy, labels, beta, key_even, 0
        )
        y, energy, labels, swap_odd = _swap_phase(
            y, energy, labels, beta, key_odd, 1
        )
        y, energy = _symmetry_refresh(y, key_symmetry)
        diagnostics = {
            "hmc_accept": hmc_accept,
            "divergence": divergence,
            "energy_error": energy_error,
            "swap_accept": swap_even | swap_odd,
            "energy": energy,
            "labels": labels,
            "y": y,
            "n_leapfrog": n_leapfrog,
        }
        return REHMCState(y, energy, labels, key), diagnostics

    return jax.lax.scan(
        one_round,
        state,
        jnp.arange(int(n_rounds), dtype=jnp.int32),
    )


def _round_trip_counts(label_history: np.ndarray) -> np.ndarray:
    """Count completed hot->cold->hot trips for every walker."""

    history = np.asarray(label_history, dtype=np.int32)
    rounds, ensembles, replicas = history.shape
    counts = np.zeros((ensembles, replicas), dtype=np.int64)
    stage = np.zeros((ensembles, replicas), dtype=np.int8)
    for time in range(rounds):
        labels = history[time]
        for ensemble in range(ensembles):
            at_hot = int(labels[ensemble, 0])
            at_cold = int(labels[ensemble, -1])
            if stage[ensemble, at_hot] == 0:
                stage[ensemble, at_hot] = 1
            elif stage[ensemble, at_hot] == 2:
                counts[ensemble, at_hot] += 1
                stage[ensemble, at_hot] = 1
            if stage[ensemble, at_cold] == 1:
                stage[ensemble, at_cold] = 2
    return counts


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        stop = start + 1
        while stop < flat.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks.reshape(np.shape(values))


def _split_chains(chains: np.ndarray) -> np.ndarray:
    values = np.asarray(chains, dtype=np.float64)
    length = values.shape[1] // 2
    if length < 4:
        raise ValueError("at least eight draws per chain are required")
    return np.concatenate((values[:, :length], values[:, -length:]), axis=0)


def _standard_rhat(chains: np.ndarray) -> float:
    values = _split_chains(chains)
    n = values.shape[1]
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(n * np.var(np.mean(values, axis=1), ddof=1))
    variance = (n - 1.0) / n * within + between / n
    return float(np.sqrt(variance / within)) if within > 0.0 else float("nan")


def _rank_normalized_rhat(chains: np.ndarray) -> float:
    values = np.asarray(chains, dtype=np.float64)
    ranks = _rankdata_average(values)
    n = ranks.size
    probabilities = (ranks - 0.375) / (n + 0.25)
    normal = np.vectorize(NormalDist().inv_cdf, otypes=[np.float64])(probabilities)
    folded = np.abs(values - np.median(values))
    folded_ranks = _rankdata_average(folded)
    folded_probabilities = (folded_ranks - 0.375) / (n + 0.25)
    folded_normal = np.vectorize(NormalDist().inv_cdf, otypes=[np.float64])(
        folded_probabilities
    )
    return max(_standard_rhat(normal), _standard_rhat(folded_normal))


def _autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    padded = 1 << int(math.ceil(math.log2(max(2, 2 * n))))
    transform = np.fft.rfft(x, n=padded)
    autocov = np.fft.irfft(transform * np.conjugate(transform), n=padded)[:n]
    autocov /= np.arange(n, 0, -1)
    if autocov[0] <= 0.0:
        result = np.zeros(n, dtype=np.float64)
        result[0] = 1.0
        return result
    return autocov / autocov[0]


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    ranks = _rankdata_average(values)
    n = ranks.size
    probability = (ranks - 0.375) / (n + 0.25)
    return np.vectorize(NormalDist().inv_cdf, otypes=[np.float64])(probability)


def _multi_chain_ess(chains: np.ndarray) -> float:
    """Stan-style split-chain ESS with positive/monotone paired lags."""

    values = _split_chains(chains)
    m, n = values.shape
    chain_variance = np.var(values, axis=1, ddof=1)
    within = float(np.mean(chain_variance))
    between = float(n * np.var(np.mean(values, axis=1), ddof=1))
    variance_plus = (n - 1.0) / n * within + between / n
    if not np.isfinite(variance_plus) or variance_plus <= 0.0:
        return float(m * n) if np.all(values == values.flat[0]) else float("nan")

    autocovariance = []
    for chain in values:
        centered = chain - np.mean(chain)
        padded = 1 << int(math.ceil(math.log2(max(2, 2 * n))))
        transform = np.fft.rfft(centered, n=padded)
        covariance = np.fft.irfft(
            transform * np.conjugate(transform),
            n=padded,
        )[:n]
        covariance /= np.arange(n, 0, -1)
        autocovariance.append(covariance)
    mean_autocovariance = np.mean(np.stack(autocovariance), axis=0)
    rho = 1.0 - (within - mean_autocovariance) / variance_plus
    rho[0] = 1.0
    pair_sums: list[float] = []
    lag = 0
    previous = float("inf")
    while lag + 1 < rho.size:
        pair = float(rho[lag] + rho[lag + 1])
        if pair < 0.0:
            break
        pair = min(pair, previous)
        pair_sums.append(pair)
        previous = pair
        lag += 2
    tau = max(1.0, -1.0 + 2.0 * float(np.sum(pair_sums)))
    return float(min(m * n, m * n / tau))


def chain_diagnostics(chains: np.ndarray) -> dict[str, float]:
    values = np.asarray(chains, dtype=np.float64)
    lower = np.quantile(values, 0.05)
    upper = np.quantile(values, 0.95)
    lower_indicator = (values <= lower).astype(np.float64)
    upper_indicator = (values >= upper).astype(np.float64)
    return {
        "rank_normalized_split_rhat": _rank_normalized_rhat(values),
        "bulk_ess": _multi_chain_ess(_rank_normalize(values)),
        "tail_ess": min(
            _multi_chain_ess(lower_indicator),
            _multi_chain_ess(upper_indicator),
        ),
    }


def _sample_observables(x: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(x, dtype=np.float64).reshape((-1, N_PARTICLES, SPATIAL_DIM))
    raw_com_abs_max = np.max(np.abs(np.mean(raw, axis=1)), axis=1)
    values = raw
    values = values - np.mean(values, axis=1, keepdims=True)
    distance = np.linalg.norm(
        values[:, PAIR_I] - values[:, PAIR_J],
        axis=-1,
    )
    return {
        "energy": lj13_energy_np(values),
        "radius_of_gyration": np.sqrt(np.mean(np.sum(values * values, axis=-1), axis=1)),
        "minimum_pair_distance": np.min(distance, axis=1),
        "mean_pair_distance": np.mean(distance, axis=1),
        "com_abs_max": raw_com_abs_max,
    }


def _histogram_ratio_slope(
    energy_a: np.ndarray,
    energy_b: np.ndarray,
    beta_a: float,
    beta_b: float,
) -> dict[str, Any]:
    lo = max(float(np.quantile(energy_a, 0.01)), float(np.quantile(energy_b, 0.01)))
    hi = min(float(np.quantile(energy_a, 0.99)), float(np.quantile(energy_b, 0.99)))
    if not hi > lo:
        return {"status": "insufficient_overlap"}
    edges = np.linspace(lo, hi, 101)
    count_a, _ = np.histogram(energy_a, bins=edges)
    count_b, _ = np.histogram(energy_b, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    minimum_count = max(5, min(40, energy_a.size // 500))
    mask = (count_a >= minimum_count) & (count_b >= minimum_count)
    if int(np.sum(mask)) < 5:
        return {"status": "insufficient_overlap", "used_bins": int(np.sum(mask))}
    weight = np.minimum(count_a[mask], count_b[mask]).astype(np.float64)
    design = np.column_stack((np.ones(int(np.sum(mask))), centers[mask]))
    weighted_design = design * np.sqrt(weight)[:, None]
    weighted_target = np.log(count_b[mask] / count_a[mask]) * np.sqrt(weight)
    intercept, slope = np.linalg.lstsq(
        weighted_design,
        weighted_target,
        rcond=None,
    )[0]
    theory = -(float(beta_b) - float(beta_a))
    return {
        "status": "ok",
        "used_bins": int(np.sum(mask)),
        "intercept": float(intercept),
        "slope": float(slope),
        "theory": theory,
        "relative_error": abs(float(slope) - theory) / abs(theory),
        "energy_range": [lo, hi],
    }


def _reweight_closure(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    beta_source: float,
    beta_target: float,
) -> dict[str, float]:
    log_weight = -(float(beta_target) - float(beta_source)) * source["energy"]
    log_weight -= float(np.max(log_weight))
    weight = np.exp(log_weight)
    weight /= np.sum(weight)
    ess = 1.0 / np.sum(weight * weight)
    result: dict[str, float] = {
        "ess": float(ess),
        "ess_fraction": float(ess / weight.size),
        "maximum_weight": float(np.max(weight)),
    }
    for name in (
        "energy",
        "radius_of_gyration",
        "mean_pair_distance",
        "minimum_pair_distance",
    ):
        predicted = float(np.sum(weight * source[name]))
        observed = float(np.mean(target[name]))
        result[f"{name}_reweighted"] = predicted
        result[f"{name}_direct"] = observed
        result[f"{name}_difference"] = predicted - observed
    return result


def _virial_diagnostics(
    samples_by_chain: np.ndarray,
    beta: float,
    batch_size: int = 1024,
) -> dict[str, float]:
    del batch_size
    flat_x = np.asarray(samples_by_chain, dtype=np.float64).reshape((-1, AMBIENT_DIM))
    virial = lj13_virial_np(flat_x)
    chains = virial.reshape(samples_by_chain.shape[:2])
    diagnostics = chain_diagnostics(chains)
    ess = max(float(diagnostics["bulk_ess"]), 1.0)
    scaled = float(beta) * virial
    mean_scaled = float(np.mean(scaled))
    mcse = float(np.std(scaled, ddof=1) / math.sqrt(ess))
    return {
        "expected_intrinsic_dimension": float(INTRINSIC_DIM),
        "beta_times_mean_virial": mean_scaled,
        "relative_error": abs(mean_scaled - INTRINSIC_DIM) / INTRINSIC_DIM,
        "mcse": mcse,
        "absolute_z": abs(mean_scaled - INTRINSIC_DIM) / max(mcse, 1.0e-15),
        **diagnostics,
    }


def run_lj13_rehmc(
    config: LJ13REHMCConfig,
    *,
    progress: bool = True,
) -> tuple[dict[float, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Run one independent PT pool.

    Returns
    -------
    samples
        Target-beta samples shaped ``(ensemble, draw, 39)``.
    diagnostics
        JSON-serializable sampler and equilibrium diagnostics.
    traces
        Compact NumPy arrays useful for independent auditing.
    """

    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError(
            "LJ13 reference-v2 requires JAX x64. Set JAX_ENABLE_X64=true "
            "before importing JAX."
        )
    if config.n_ensembles < 2:
        raise ValueError("at least two independent ensembles are required")
    if config.beta_min <= 0.0 or config.beta_max <= config.beta_min:
        raise ValueError("beta bounds must satisfy 0 < beta_min < beta_max")
    if not (0.0 < config.initial_step_size):
        raise ValueError("initial_step_size must be positive")
    if not (1 <= config.leapfrog_min <= config.leapfrog_max):
        raise ValueError("leapfrog bounds must satisfy 1 <= min <= max")
    if config.block_rounds <= 0 or config.storage_stride <= 0:
        raise ValueError("block_rounds and storage_stride must be positive")
    if config.warmup_rounds % config.block_rounds != 0:
        raise ValueError("warmup_rounds must be divisible by block_rounds")
    if config.settle_rounds % config.block_rounds != 0:
        raise ValueError("settle_rounds must be divisible by block_rounds")
    if config.production_rounds % config.block_rounds != 0:
        raise ValueError("production_rounds must be divisible by block_rounds")
    if config.production_rounds % config.storage_stride != 0:
        raise ValueError("production_rounds must be divisible by storage_stride")
    betas_np = make_beta_ladder(config)
    betas = jnp.asarray(betas_np, dtype=jnp.float64)
    y0_np = initialize_ensembles_np(config, betas_np)
    y0 = jnp.asarray(y0_np, dtype=jnp.float64)
    energy0, _ = energy_and_grad_reduced_jax(y0)
    labels0 = jnp.broadcast_to(
        jnp.arange(config.n_replicas, dtype=jnp.int32)[None, :],
        (config.n_ensembles, config.n_replicas),
    )
    state = REHMCState(
        y0,
        energy0,
        labels0,
        jax.random.PRNGKey(int(config.seed) + 7919),
    )
    step_size = np.full(
        config.n_replicas,
        float(config.initial_step_size),
        dtype=np.float64,
    )

    warm_accept_sum = np.zeros(config.n_replicas, dtype=np.float64)
    warm_divergence = np.zeros(config.n_replicas, dtype=np.int64)
    warm_count = 0
    warm_blocks = config.warmup_rounds // config.block_rounds
    for block in range(warm_blocks):
        state, block_output = _run_block(
            state,
            betas,
            jnp.asarray(step_size),
            n_rounds=config.block_rounds,
            leapfrog_min=config.leapfrog_min,
            leapfrog_max=config.leapfrog_max,
        )
        acceptance = np.asarray(jax.device_get(block_output["hmc_accept"]))
        divergence = np.asarray(jax.device_get(block_output["divergence"]))
        mean_acceptance = np.mean(acceptance, axis=(0, 1))
        warm_accept_sum += np.sum(acceptance, axis=(0, 1))
        warm_divergence += np.sum(divergence, axis=(0, 1))
        warm_count += acceptance.shape[0] * acceptance.shape[1]
        rate = float(config.adaptation_rate) / math.sqrt(block + 1.0)
        step_size *= np.exp(rate * (mean_acceptance - float(config.target_accept)))
        step_size = np.clip(
            step_size,
            config.minimum_step_size,
            config.maximum_step_size,
        )
        if progress and (block == 0 or (block + 1) % max(1, warm_blocks // 20) == 0):
            print(
                f"[lj13-rehmc:warmup] block={block + 1}/{warm_blocks} "
                f"accept={float(np.mean(mean_acceptance)):.3f} "
                f"eps=[{float(np.min(step_size)):.3g},{float(np.max(step_size)):.3g}] "
                f"divergences={int(np.sum(warm_divergence))}",
                flush=True,
            )

    settle_accept_sum = np.zeros(config.n_replicas, dtype=np.float64)
    settle_divergence = np.zeros(config.n_replicas, dtype=np.int64)
    settle_count = 0
    settle_blocks = config.settle_rounds // config.block_rounds
    for block in range(settle_blocks):
        state, block_output = _run_block(
            state,
            betas,
            jnp.asarray(step_size),
            n_rounds=config.block_rounds,
            leapfrog_min=config.leapfrog_min,
            leapfrog_max=config.leapfrog_max,
        )
        acceptance = np.asarray(jax.device_get(block_output["hmc_accept"]))
        divergence = np.asarray(jax.device_get(block_output["divergence"]))
        settle_accept_sum += np.sum(acceptance, axis=(0, 1))
        settle_divergence += np.sum(divergence, axis=(0, 1))
        settle_count += acceptance.shape[0] * acceptance.shape[1]
    if progress and settle_blocks:
        print(
            f"[lj13-rehmc:settle] rounds={config.settle_rounds} "
            f"accept={float(np.mean(settle_accept_sum / settle_count)):.3f} "
            f"divergences={int(np.sum(settle_divergence))}",
            flush=True,
        )

    saved_y: list[np.ndarray] = []
    saved_energy: list[np.ndarray] = []
    label_history: list[np.ndarray] = [
        np.asarray(jax.device_get(state.labels), dtype=np.int32)[None, ...]
    ]
    production_accept_sum = np.zeros(config.n_replicas, dtype=np.float64)
    production_divergence = np.zeros(config.n_replicas, dtype=np.int64)
    swap_accept_sum = np.zeros(config.n_replicas - 1, dtype=np.float64)
    production_count = 0
    production_blocks = config.production_rounds // config.block_rounds
    for block in range(production_blocks):
        state, block_output = _run_block(
            state,
            betas,
            jnp.asarray(step_size),
            n_rounds=config.block_rounds,
            leapfrog_min=config.leapfrog_min,
            leapfrog_max=config.leapfrog_max,
        )
        output = {
            name: np.asarray(jax.device_get(value))
            for name, value in block_output.items()
        }
        production_accept_sum += np.sum(output["hmc_accept"], axis=(0, 1))
        production_divergence += np.sum(output["divergence"], axis=(0, 1))
        swap_accept_sum += np.sum(output["swap_accept"], axis=(0, 1))
        production_count += output["hmc_accept"].shape[0] * output["hmc_accept"].shape[1]
        offset = block * config.block_rounds
        keep = (
            (np.arange(config.block_rounds, dtype=np.int64) + offset + 1)
            % config.storage_stride
            == 0
        )
        saved_y.append(output["y"][keep])
        saved_energy.append(output["energy"][keep])
        label_history.append(output["labels"])
        if progress and (
            block == 0 or (block + 1) % max(1, production_blocks // 20) == 0
        ):
            completed = (block + 1) * config.block_rounds
            print(
                f"[lj13-rehmc:production] rounds={completed}/{config.production_rounds} "
                f"accept={float(np.mean(production_accept_sum / production_count)):.3f} "
                f"swap={float(np.mean(swap_accept_sum / completed / config.n_ensembles)):.3f} "
                f"divergences={int(np.sum(production_divergence))}",
                flush=True,
            )

    y_history = np.concatenate(saved_y, axis=0)
    energy_history = np.concatenate(saved_energy, axis=0)
    labels = np.concatenate(label_history, axis=0)
    # scan output is (draw, ensemble, replica, dim); output contracts are
    # (ensemble, draw, dim).
    target_samples: dict[float, np.ndarray] = {}
    target_observables: dict[float, dict[str, np.ndarray]] = {}
    target_indices: dict[str, int] = {}
    for target in config.target_betas:
        index = int(np.flatnonzero(betas_np == float(target))[0])
        target_indices[f"{float(target):.2f}"] = index
        target_y = np.transpose(y_history[:, :, index, :], (1, 0, 2))
        target_x = reduced_to_coordinates_np(target_y).reshape(
            (config.n_ensembles, target_y.shape[1], AMBIENT_DIM)
        )
        target_samples[float(target)] = target_x.astype(np.float64)
        flat_observables = _sample_observables(target_x)
        target_observables[float(target)] = {
            name: values.reshape((config.n_ensembles, target_y.shape[1]))
            for name, values in flat_observables.items()
        }

    round_trips = _round_trip_counts(labels)
    diagnostics: dict[str, Any] = {
        "schema": "adtm.lj13_rehmc.pool.v1",
        "config": asdict(config),
        "beta_ladder": [float(value) for value in betas_np],
        "target_indices": target_indices,
        "initialization": {
            "legacy_reference_used": False,
            "even_ensembles": "perturbed_icosahedral_like",
            "odd_ensembles": "independent_random_compact_cluster",
        },
        "step_size_final": [float(value) for value in step_size],
        "warmup_hmc_acceptance": [
            float(value) for value in warm_accept_sum / max(warm_count, 1)
        ],
        "production_hmc_acceptance": [
            float(value) for value in production_accept_sum / max(production_count, 1)
        ],
        "settle_hmc_acceptance": [
            float(value) for value in settle_accept_sum / max(settle_count, 1)
        ],
        "settle_divergences": [int(value) for value in settle_divergence],
        "production_divergences": [
            int(value) for value in production_divergence
        ],
        "swap_acceptance": [
            float(value)
            for value in swap_accept_sum
            / max(config.production_rounds * config.n_ensembles, 1)
        ],
        "round_trip_counts": round_trips.tolist(),
        "round_trip_total": int(np.sum(round_trips)),
        "round_trip_per_ensemble": [
            int(value) for value in np.sum(round_trips, axis=1)
        ],
        "label_permutation_valid": bool(
            np.all(
                np.sort(labels, axis=-1)
                == np.arange(config.n_replicas, dtype=np.int32)[None, None, :]
            )
        ),
        "targets": {},
        "cross_beta": {},
    }
    for target, observables in target_observables.items():
        target_record: dict[str, Any] = {
            "n_chains": config.n_ensembles,
            "draws_per_chain": int(observables["energy"].shape[1]),
            "observables": {},
            "virial": _virial_diagnostics(target_samples[target], target),
        }
        for name in (
            "energy",
            "radius_of_gyration",
            "minimum_pair_distance",
            "mean_pair_distance",
        ):
            values = observables[name]
            target_record["observables"][name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                **chain_diagnostics(values),
            }
        target_record["com_abs_max"] = float(np.max(observables["com_abs_max"]))
        target_record["minimum_pair_distance_global"] = float(
            np.min(observables["minimum_pair_distance"])
        )
        diagnostics["targets"][f"{target:.2f}"] = target_record

    targets = sorted(target_observables)
    for beta_a, beta_b in zip(targets[:-1], targets[1:]):
        obs_a = {
            name: values.reshape(-1)
            for name, values in target_observables[beta_a].items()
        }
        obs_b = {
            name: values.reshape(-1)
            for name, values in target_observables[beta_b].items()
        }
        key = f"{beta_a:.2f}_to_{beta_b:.2f}"
        diagnostics["cross_beta"][key] = {
            "density_ratio": _histogram_ratio_slope(
                obs_a["energy"],
                obs_b["energy"],
                beta_a,
                beta_b,
            ),
            "forward_reweight_closure": _reweight_closure(
                obs_a,
                obs_b,
                beta_a,
                beta_b,
            ),
            "reverse_reweight_closure": _reweight_closure(
                obs_b,
                obs_a,
                beta_b,
                beta_a,
            ),
        }

    traces = {
        "beta_ladder": betas_np,
        "step_size_final": step_size,
        "stored_energy": energy_history,
        "labels": labels,
        "round_trip_counts": round_trips,
    }
    return target_samples, diagnostics, traces


def audit_pass_fail(diagnostics: dict[str, Any], *, pilot: bool) -> dict[str, Any]:
    """Apply explicit gates without pretending a short pilot is publishable."""

    thresholds = {
        "hmc_acceptance_min": 0.65,
        "hmc_acceptance_max": 0.95,
        "swap_acceptance_min": 0.10,
        "swap_acceptance_median_min": 0.20,
        "com_abs_max": 1.0e-9,
        "rhat_max": 1.01,
        "bulk_ess_min": 1_000.0,
        "tail_ess_min": 1_000.0,
        "virial_relative_error_max": 0.05,
        "virial_absolute_z_max": 3.0,
        "density_ratio_relative_error_max": 0.05,
        "round_trip_total_min": 10,
        "round_trip_per_ensemble_min": 10,
        "minimum_pair_distance_min": 0.20,
        "importance_ess_min": 400.0,
        "importance_ess_fraction_min": 0.01,
        "importance_maximum_weight_max": 0.02,
    }
    failures: list[str] = []
    warnings: list[str] = []
    acceptance = np.asarray(diagnostics["production_hmc_acceptance"])
    if not np.isfinite(acceptance).all():
        failures.append("HMC acceptance contains non-finite values")
    if float(np.min(acceptance)) < thresholds["hmc_acceptance_min"]:
        failures.append("minimum HMC acceptance is below 0.65")
    if float(np.max(acceptance)) > thresholds["hmc_acceptance_max"]:
        warnings.append("maximum HMC acceptance is above 0.95; trajectories may mix slowly")
    swap = np.asarray(diagnostics["swap_acceptance"])
    if not np.isfinite(swap).all():
        failures.append("swap acceptance contains non-finite values")
    if float(np.min(swap)) < thresholds["swap_acceptance_min"]:
        failures.append("minimum adjacent swap acceptance is below 0.10")
    if float(np.median(swap)) < thresholds["swap_acceptance_median_min"]:
        failures.append("median adjacent swap acceptance is below 0.20")
    if sum(diagnostics["production_divergences"]) != 0:
        failures.append("production contains HMC divergences")
    if not bool(diagnostics.get("label_permutation_valid", False)):
        failures.append("walker labels are not permutations")
    if int(diagnostics["round_trip_total"]) < thresholds["round_trip_total_min"]:
        (warnings if pilot else failures).append("fewer than ten full hot-cold-hot round trips")
    per_ensemble = np.asarray(diagnostics["round_trip_per_ensemble"], dtype=np.int64)
    if int(np.min(per_ensemble)) < thresholds["round_trip_per_ensemble_min"]:
        (warnings if pilot else failures).append(
            "at least one ensemble has fewer than ten full round trips"
        )
    for beta, record in diagnostics["targets"].items():
        if float(record["com_abs_max"]) > thresholds["com_abs_max"]:
            failures.append(f"beta={beta}: COM residual exceeds tolerance")
        if float(record["minimum_pair_distance_global"]) < thresholds[
            "minimum_pair_distance_min"
        ]:
            failures.append(f"beta={beta}: samples approach the LJ numerical guard")
        virial = record["virial"]
        virial_values = [
            float(virial["rank_normalized_split_rhat"]),
            float(virial["bulk_ess"]),
            float(virial["tail_ess"]),
            float(virial["absolute_z"]),
        ]
        if not np.isfinite(virial_values).all():
            failures.append(f"beta={beta}: virial diagnostics are non-finite")
        if float(virial["relative_error"]) > thresholds["virial_relative_error_max"]:
            failures.append(f"beta={beta}: virial identity fails")
        if float(virial["rank_normalized_split_rhat"]) > thresholds["rhat_max"]:
            failures.append(f"beta={beta}: virial R-hat exceeds 1.01")
        if float(virial["absolute_z"]) > thresholds["virial_absolute_z_max"]:
            failures.append(f"beta={beta}: virial discrepancy exceeds 3 MCSE")
        for name, observable in record["observables"].items():
            values = [
                float(observable["rank_normalized_split_rhat"]),
                float(observable["bulk_ess"]),
                float(observable["tail_ess"]),
            ]
            if not np.isfinite(values).all():
                failures.append(f"beta={beta} {name}: diagnostics are non-finite")
                continue
            if values[0] > thresholds["rhat_max"]:
                failures.append(f"beta={beta} {name}: R-hat exceeds 1.01")
            if values[1] < thresholds["bulk_ess_min"]:
                (warnings if pilot else failures).append(
                    f"beta={beta} {name}: bulk ESS below 1000"
                )
            if values[2] < thresholds["tail_ess_min"]:
                (warnings if pilot else failures).append(
                    f"beta={beta} {name}: tail ESS below 1000"
                )
    for pair, record in diagnostics["cross_beta"].items():
        slope = record["density_ratio"]
        if slope.get("status") != "ok":
            failures.append(f"{pair}: insufficient cross-beta overlap")
        else:
            slope_error = float(slope["relative_error"])
            if not np.isfinite(slope_error):
                failures.append(f"{pair}: density-ratio slope is non-finite")
            elif slope_error > thresholds["density_ratio_relative_error_max"]:
                failures.append(f"{pair}: density-ratio slope fails")
        for direction in ("forward_reweight_closure", "reverse_reweight_closure"):
            closure = record[direction]
            values = (
                float(closure["ess"]),
                float(closure["ess_fraction"]),
                float(closure["maximum_weight"]),
            )
            if not np.isfinite(values).all():
                failures.append(f"{pair} {direction}: non-finite importance diagnostics")
                continue
            if values[0] < thresholds["importance_ess_min"]:
                failures.append(f"{pair} {direction}: importance ESS below 400")
            if values[1] < thresholds["importance_ess_fraction_min"]:
                failures.append(f"{pair} {direction}: importance ESS fraction below 1%")
            if values[2] > thresholds["importance_maximum_weight_max"]:
                failures.append(f"{pair} {direction}: maximum weight exceeds 2%")
    return {
        "schema": "adtm.lj13_reference_v2.audit.v1",
        "scope": "pilot_diagnostic_only" if pilot else "formal_reference_candidate",
        "status": "pass" if not failures else "fail",
        "publishable": bool(not pilot and not failures),
        "thresholds": thresholds,
        "failures": failures,
        "warnings": warnings,
    }
