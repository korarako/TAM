from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import multiprocessing as mp
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def center_configuration(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    return value - np.mean(value, axis=-2, keepdims=True)


def squared_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    delta = x[:, None, :] - y[None, :, :]
    return np.sum(delta * delta, axis=-1)


def sorted_distance_signatures(x: np.ndarray) -> np.ndarray:
    distances = np.sqrt(np.maximum(squared_distance_matrix(x, x), 0.0))
    return np.sort(distances, axis=-1)


def assignment_from_cost(cost: np.ndarray) -> np.ndarray:
    value = np.asarray(cost, dtype=np.float64)
    rows, cols = linear_sum_assignment(value)
    expected = np.arange(value.shape[0], dtype=np.int64)
    if rows.size != value.shape[0] or not np.array_equal(rows, expected):
        raise RuntimeError("Hungarian assignment did not cover every particle")
    return np.asarray(cols, dtype=np.int64)


def signature_assignment(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sx = sorted_distance_signatures(x)
    sy = sorted_distance_signatures(y)
    return assignment_from_cost(squared_distance_matrix(sx, sy))


def exact_congruence_assignment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-9,
    maximum_search_nodes: int = 200_000,
) -> np.ndarray | None:
    """Recover an exact distance-matrix isomorphism when one exists.

    The fast path makes ``D(x, P x Q + t) = 0`` robust for structures with
    repeated distance signatures, including an ideal icosahedral LJ-13 cluster.
    Generic non-congruent pairs return before the backtracking search.
    """

    dx = np.sqrt(np.maximum(squared_distance_matrix(x, x), 0.0))
    dy = np.sqrt(np.maximum(squared_distance_matrix(y, y), 0.0))
    scale = max(1.0, float(np.max(dx)), float(np.max(dy)))
    tolerance = float(relative_tolerance) * scale
    triangle = np.triu_indices(x.shape[0], k=1)
    if not np.allclose(
        np.sort(dx[triangle]),
        np.sort(dy[triangle]),
        rtol=0.0,
        atol=tolerance,
    ):
        return None

    sx = np.sort(dx, axis=-1)
    sy = np.sort(dy, axis=-1)
    candidates = [
        np.asarray(
            [
                j
                for j in range(y.shape[0])
                if np.allclose(sx[i], sy[j], rtol=0.0, atol=tolerance)
            ],
            dtype=np.int64,
        )
        for i in range(x.shape[0])
    ]
    if any(values.size == 0 for values in candidates):
        return None

    order = sorted(
        range(x.shape[0]),
        key=lambda i: (int(candidates[i].size), -float(np.var(dx[i])), int(i)),
    )
    mapping = np.full(x.shape[0], -1, dtype=np.int64)
    used = np.zeros(y.shape[0], dtype=bool)
    nodes = 0

    def search(depth: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > int(maximum_search_nodes):
            return False
        if depth == len(order):
            return True
        i = order[depth]
        for candidate in candidates[i]:
            j = int(candidate)
            if used[j]:
                continue
            if any(
                abs(float(dx[i, order[k]] - dy[j, int(mapping[order[k]])])) > tolerance
                for k in range(depth)
            ):
                continue
            mapping[i] = j
            used[j] = True
            if search(depth + 1):
                return True
            used[j] = False
            mapping[i] = -1
        return False

    return mapping.copy() if search(0) else None


def procrustes_rotation(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``R in O(d)`` minimizing ``||x - y R||_F``."""

    u, _, vt = np.linalg.svd(y.T @ x, full_matrices=False)
    return u @ vt


def aligned_squared_cost(
    x: np.ndarray,
    y: np.ndarray,
    permutation: np.ndarray,
) -> tuple[float, np.ndarray]:
    matched = y[np.asarray(permutation, dtype=np.int64)]
    rotation = procrustes_rotation(matched, x)
    residual = x - matched @ rotation
    return float(np.sum(residual * residual)), rotation


def alternating_alignment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    initial_permutation: np.ndarray | None = None,
    initial_rotation: np.ndarray | None = None,
    max_iterations: int = 50,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    if (initial_permutation is None) == (initial_rotation is None):
        raise ValueError("provide exactly one initial permutation or rotation")

    if initial_permutation is not None:
        permutation = np.asarray(initial_permutation, dtype=np.int64)
        best_cost, rotation = aligned_squared_cost(x, y, permutation)
    else:
        rotation = np.asarray(initial_rotation, dtype=np.float64)
        permutation = assignment_from_cost(squared_distance_matrix(x, y @ rotation))
        best_cost, rotation = aligned_squared_cost(x, y, permutation)

    best_permutation = permutation.copy()
    best_rotation = rotation.copy()
    seen: set[tuple[int, ...]] = set()

    for iteration in range(1, int(max_iterations) + 1):
        key = tuple(int(i) for i in permutation)
        if key in seen:
            return best_cost, best_permutation, best_rotation, iteration - 1
        seen.add(key)

        candidate = assignment_from_cost(squared_distance_matrix(x, y @ rotation))
        cost, candidate_rotation = aligned_squared_cost(x, y, candidate)
        if cost < best_cost:
            best_cost = cost
            best_permutation = candidate.copy()
            best_rotation = candidate_rotation.copy()
        if np.array_equal(candidate, permutation):
            return best_cost, best_permutation, best_rotation, iteration
        permutation = candidate
        rotation = candidate_rotation

    return best_cost, best_permutation, best_rotation, int(max_iterations)


def pca_frame(x: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(x.T @ x)
    return eigenvectors[:, np.argsort(eigenvalues)[::-1]]


def sign_matrices(spatial_dimension: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.diag(np.asarray(signs, dtype=np.float64))
        for signs in itertools.product((-1.0, 1.0), repeat=int(spatial_dimension))
    )


def joint_alignment_squared_cost(
    x: np.ndarray,
    y: np.ndarray,
    *,
    pca_multistart: bool = True,
    max_iterations: int = 50,
) -> tuple[float, dict[str, Any]]:
    """Approximate ``min_{R in O(d), P in S(n)} ||x - P y R||_F^2``.

    This is a symmetry-consistent multi-start local solver, not a proof of the
    global minimum over all particle permutations.
    """

    x = center_configuration(np.asarray(x, dtype=np.float64))
    y = center_configuration(np.asarray(y, dtype=np.float64))
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(
            "joint alignment expects matching (particles, spatial_dim) arrays; "
            f"received {x.shape} and {y.shape}"
        )
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    congruence = exact_congruence_assignment(x, y)
    if congruence is not None:
        congruent_cost, congruent_rotation = aligned_squared_cost(x, y, congruence)
        numerical_scale = max(1.0, float(np.sum(x * x)), float(np.sum(y * y)))
        if congruent_cost <= 1.0e-16 * numerical_scale:
            return float(max(congruent_cost, 0.0)), {
                "start": "exact_distance_isomorphism",
                "iterations": 0,
                "permutation": congruence,
                "rotation": congruent_rotation,
            }

    starts: list[tuple[str, np.ndarray, str]] = [
        ("distance_signature", signature_assignment(x, y), "permutation")
    ]
    if pca_multistart:
        vx = pca_frame(x)
        vy = pca_frame(y)
        starts.extend(
            (f"pca_sign_{index}", vy @ signs @ vx.T, "rotation")
            for index, signs in enumerate(sign_matrices(x.shape[1]))
        )

    best_cost = np.inf
    best: dict[str, Any] = {}
    for name, value, kind in starts:
        if kind == "permutation":
            cost, permutation, rotation, iterations = alternating_alignment(
                x,
                y,
                initial_permutation=value,
                max_iterations=max_iterations,
            )
        else:
            cost, permutation, rotation, iterations = alternating_alignment(
                x,
                y,
                initial_rotation=value,
                max_iterations=max_iterations,
            )
        if cost < best_cost:
            best_cost = cost
            best = {
                "start": name,
                "iterations": int(iterations),
                "permutation": permutation,
                "rotation": rotation,
            }
    return float(max(best_cost, 0.0)), best


_PROCESS_LEFT: np.ndarray | None = None
_PROCESS_RIGHT: np.ndarray | None = None
_PROCESS_MAX_ITERATIONS = 50


def _initialize_process_worker(
    left: np.ndarray,
    right: np.ndarray,
    max_iterations: int,
) -> None:
    global _PROCESS_LEFT, _PROCESS_RIGHT, _PROCESS_MAX_ITERATIONS
    _PROCESS_LEFT = left
    _PROCESS_RIGHT = right
    _PROCESS_MAX_ITERATIONS = int(max_iterations)


def _process_row_block(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
    if _PROCESS_LEFT is None or _PROCESS_RIGHT is None:
        raise RuntimeError("joint-alignment process worker is not initialized")
    start, stop = bounds
    block = np.empty((stop - start, _PROCESS_RIGHT.shape[0]), dtype=np.float64)
    for local_index, index in enumerate(range(start, stop)):
        for column, y in enumerate(_PROCESS_RIGHT):
            block[local_index, column], _ = joint_alignment_squared_cost(
                _PROCESS_LEFT[index],
                y,
                max_iterations=_PROCESS_MAX_ITERATIONS,
            )
    return start, block


def joint_alignment_cost_matrix(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_iterations: int = 50,
    workers: int = 1,
    parallel_backend: str = "process",
) -> np.ndarray:
    """Compute the full squared joint-alignment ground-cost matrix.

    ``parallel_backend='process'`` is intended for the command-line evaluator
    and CPU-heavy formal runs. Calling it from a Python script with more than
    one worker requires the usual multiprocessing ``__main__`` guard.
    ``parallel_backend='thread'`` avoids that restriction and is convenient in
    notebooks, but can be slower for this mixed Python/NumPy workload.
    """

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]:
        raise ValueError(
            "expected sample arrays with matching (particles, spatial_dim); "
            f"received {left.shape} and {right.shape}"
        )
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    parallel_backend = str(parallel_backend).lower()
    if parallel_backend not in {"process", "thread"}:
        raise ValueError("parallel_backend must be 'process' or 'thread'")

    def build_row(index: int) -> tuple[int, np.ndarray]:
        row = np.empty(right.shape[0], dtype=np.float64)
        for column, y in enumerate(right):
            row[column], _ = joint_alignment_squared_cost(
                left[index],
                y,
                max_iterations=max_iterations,
            )
        return index, row

    out = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    if workers == 1:
        for index in range(left.shape[0]):
            _, out[index] = build_row(index)
        return out

    if parallel_backend == "thread":
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, row in executor.map(build_row, range(left.shape[0])):
                out[index] = row
        return out

    row_block_size = max(1, int(np.ceil(left.shape[0] / (workers * 8))))
    tasks = [
        (start, min(start + row_block_size, left.shape[0]))
        for start in range(0, left.shape[0], row_block_size)
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_process_worker,
        initargs=(left, right, int(max_iterations)),
    ) as executor:
        futures = [executor.submit(_process_row_block, task) for task in tasks]
        for future in as_completed(futures):
            start, block = future.result()
            out[start : start + block.shape[0]] = block
    return out


__all__ = [
    "joint_alignment_cost_matrix",
    "joint_alignment_squared_cost",
]
