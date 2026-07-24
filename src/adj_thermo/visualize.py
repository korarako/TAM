from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adj_thermo.problem.base import ProblemSpec
from adj_thermo.utils import pairwise_distances

BLUE = "#4C78A8"
GREEN = "#54A24B"
GREY = "#7F7F7F"
BLACK = "#1A1A1A"
GRID = "#D7E0EC"


def _energy_np(problem: ProblemSpec, samples: np.ndarray) -> np.ndarray:
    import jax
    return np.asarray(jax.device_get(problem.energy_fn(samples)))


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.9)


def _hist_range(arrays: list[np.ndarray], pad_frac: float = 0.03) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays if np.asarray(a).size])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(values)), float(np.max(values))
    pad = max(1.0e-6, float(hi - lo) * float(pad_frac))
    return float(lo - pad), float(hi + pad)


def _overlay_hist(ax, values: np.ndarray, bins: np.ndarray, label: str, color: str, *, filled: bool, dashed: bool = False) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    histtype = "stepfilled" if filled else "step"
    alpha = 0.22 if filled else 1.0
    linewidth = 2.0 if not filled else 1.9
    linestyle = "--" if dashed else "-"
    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype=histtype,
        color=color,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )
    if filled:
        ax.hist(values, bins=bins, density=True, histtype="step", color=color, linewidth=1.9, label="_nolegend_")


def plot_energy_histogram(ref0, fm0, ref1, fm1_optional, am1, problem: ProblemSpec, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    e_ref0 = _energy_np(problem, ref0)
    e_ref1 = _energy_np(problem, ref1)
    e_fm0 = _energy_np(problem, fm0)
    e_am1 = _energy_np(problem, am1)
    e_fm1 = None if fm1_optional is None else _energy_np(problem, fm1_optional)
    hist_inputs = [e_ref0, e_ref1, e_fm0, e_am1] + ([] if e_fm1 is None else [e_fm1])
    bins = np.linspace(*_hist_range(hist_inputs), 90)

    _overlay_hist(ax, e_ref0, bins, "true beta0", GREY, filled=False, dashed=True)
    _overlay_hist(ax, e_ref1, bins, "true beta1", BLACK, filled=False)
    if e_fm1 is not None:
        _overlay_hist(ax, e_fm1, bins, "FM samples", BLUE, filled=True)
    else:
        _overlay_hist(ax, e_fm0, bins, "FM samples", BLUE, filled=True)
    _overlay_hist(ax, e_am1, bins, "AM samples", GREEN, filled=True)

    _style_axes(ax)
    ax.set_xlabel("energy")
    ax.set_ylabel("density")
    ax.set_title(f"{problem.name}: energy histogram")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_marginals(ref0, fm0, ref1, fm1_optional, am1, problem: ProblemSpec, path: str | Path) -> None:
    dim = int(problem.dim)
    cols = min(2 if dim <= 2 else 4, dim)
    rows = int(np.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 4.7 * rows), squeeze=False)
    fm_for_overlay = fm1_optional if fm1_optional is not None else fm0

    for i in range(dim):
        ax = axes[i // cols][i % cols]
        arrays = [np.asarray(ref0)[:, i], np.asarray(ref1)[:, i], np.asarray(fm_for_overlay)[:, i], np.asarray(am1)[:, i]]
        bins = np.linspace(*_hist_range(arrays), 90)
        _overlay_hist(ax, arrays[0], bins, "true beta0", GREY, filled=False, dashed=True)
        _overlay_hist(ax, arrays[1], bins, "true beta1", BLACK, filled=False)
        _overlay_hist(ax, arrays[2], bins, "FM samples", BLUE, filled=True)
        _overlay_hist(ax, arrays[3], bins, "AM samples", GREEN, filled=True)
        _style_axes(ax)
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel("density")
        ax.set_title(f"{problem.name}: x{i} marginal")
        ax.legend(frameon=False, loc="upper right")

    for j in range(dim, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pairwise_distances(ref0, fm0, ref1, fm1_optional, am1, problem: ProblemSpec, path: str | Path) -> None:
    if not problem.n_particles or not problem.spatial_dim:
        return
    import jax
    fm_for_overlay = fm1_optional if fm1_optional is not None else fm0
    d_ref0 = np.asarray(jax.device_get(pairwise_distances(ref0, problem.n_particles, problem.spatial_dim)))
    d_ref1 = np.asarray(jax.device_get(pairwise_distances(ref1, problem.n_particles, problem.spatial_dim)))
    d_fm = np.asarray(jax.device_get(pairwise_distances(fm_for_overlay, problem.n_particles, problem.spatial_dim)))
    d_am = np.asarray(jax.device_get(pairwise_distances(am1, problem.n_particles, problem.spatial_dim)))
    n_pairs = d_ref0.shape[1]
    cols = 3
    rows = int(np.ceil(n_pairs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.4 * rows), squeeze=False)
    for i in range(n_pairs):
        ax = axes[i // cols][i % cols]
        arrays = [d_ref0[:, i], d_ref1[:, i], d_fm[:, i], d_am[:, i]]
        bins = np.linspace(*_hist_range(arrays), 70)
        _overlay_hist(ax, arrays[0], bins, "true beta0", GREY, filled=False, dashed=True)
        _overlay_hist(ax, arrays[1], bins, "true beta1", BLACK, filled=False)
        _overlay_hist(ax, arrays[2], bins, "FM samples", BLUE, filled=True)
        _overlay_hist(ax, arrays[3], bins, "AM samples", GREEN, filled=True)
        _style_axes(ax)
        ax.set_xlabel("distance")
        ax.set_ylabel("density")
        ax.set_title(f"pair {i}")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    for j in range(n_pairs, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pairwise_distances_combined(ref0, fm0, ref1, fm1_optional, am1, problem: ProblemSpec, path: str | Path) -> None:
    if not problem.n_particles or not problem.spatial_dim:
        return
    import jax
    fm_for_overlay = fm1_optional if fm1_optional is not None else fm0
    arrays = [
        np.asarray(jax.device_get(pairwise_distances(ref0, problem.n_particles, problem.spatial_dim))).reshape(-1),
        np.asarray(jax.device_get(pairwise_distances(ref1, problem.n_particles, problem.spatial_dim))).reshape(-1),
        np.asarray(jax.device_get(pairwise_distances(fm_for_overlay, problem.n_particles, problem.spatial_dim))).reshape(-1),
        np.asarray(jax.device_get(pairwise_distances(am1, problem.n_particles, problem.spatial_dim))).reshape(-1),
    ]
    bins = np.linspace(*_hist_range(arrays), 90)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    _overlay_hist(ax, arrays[0], bins, "true beta0", GREY, filled=False, dashed=True)
    _overlay_hist(ax, arrays[1], bins, "true beta1", BLACK, filled=False)
    _overlay_hist(ax, arrays[2], bins, "FM samples", BLUE, filled=True)
    _overlay_hist(ax, arrays[3], bins, "AM samples", GREEN, filled=True)
    _style_axes(ax)
    ax.set_xlabel("pair distance")
    ax.set_ylabel("density")
    ax.set_title(f"{problem.name}: pairwise distance histogram")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_loss_curves(fm_loss: np.ndarray, am_loss: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if fm_loss is not None and len(fm_loss):
        ax.plot(np.arange(1, len(fm_loss) + 1), fm_loss, color=BLUE, linewidth=2.0, label="FM")
    if am_loss is not None and len(am_loss):
        ax.plot(np.arange(1, len(am_loss) + 1), am_loss, color=GREEN, linewidth=2.0, label="AM")
    _style_axes(ax)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("loss curves")
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fm_md_energy_overlay(ref, fm, beta: float, problem: ProblemSpec, path: str | Path, sample_label: str = "FM") -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    e_ref = _energy_np(problem, ref)
    e_fm = _energy_np(problem, fm)
    bins = np.linspace(*_hist_range([e_ref, e_fm]), 90)
    _overlay_hist(ax, e_ref, bins, f"MD beta={float(beta):.2f}", BLACK, filled=False)
    _overlay_hist(ax, e_fm, bins, f"{sample_label} beta={float(beta):.2f}", BLUE, filled=True)
    _style_axes(ax)
    ax.set_xlabel("energy")
    ax.set_ylabel("density")
    ax.set_title(f"{problem.name}: {sample_label} vs MD energy beta={float(beta):.2f}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fm_md_marginals_overlay(ref, fm, beta: float, problem: ProblemSpec, path: str | Path) -> None:
    dim = int(problem.dim)
    cols = min(2 if dim <= 2 else 4, dim)
    rows = int(np.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 4.7 * rows), squeeze=False)
    for i in range(dim):
        ax = axes[i // cols][i % cols]
        arrays = [np.asarray(ref)[:, i], np.asarray(fm)[:, i]]
        bins = np.linspace(*_hist_range(arrays), 90)
        _overlay_hist(ax, arrays[0], bins, f"MD beta={float(beta):.2f}", BLACK, filled=False)
        _overlay_hist(ax, arrays[1], bins, f"FM beta={float(beta):.2f}", BLUE, filled=True)
        _style_axes(ax)
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel("density")
        ax.set_title(f"{problem.name}: x{i} beta={float(beta):.2f}")
        ax.legend(frameon=False, loc="upper right")
    for j in range(dim, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fm_md_pairwise_overlay(ref, fm, beta: float, problem: ProblemSpec, path: str | Path) -> None:
    if not problem.n_particles or not problem.spatial_dim:
        return
    import jax
    d_ref = np.asarray(jax.device_get(pairwise_distances(ref, problem.n_particles, problem.spatial_dim)))
    d_fm = np.asarray(jax.device_get(pairwise_distances(fm, problem.n_particles, problem.spatial_dim)))
    n_pairs = d_ref.shape[1]
    cols = 3
    rows = int(np.ceil(n_pairs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.4 * rows), squeeze=False)
    for i in range(n_pairs):
        ax = axes[i // cols][i % cols]
        arrays = [d_ref[:, i], d_fm[:, i]]
        bins = np.linspace(*_hist_range(arrays), 70)
        _overlay_hist(ax, arrays[0], bins, f"MD beta={float(beta):.2f}", BLACK, filled=False)
        _overlay_hist(ax, arrays[1], bins, f"FM beta={float(beta):.2f}", BLUE, filled=True)
        _style_axes(ax)
        ax.set_xlabel("distance")
        ax.set_ylabel("density")
        ax.set_title(f"pair {i}, beta={float(beta):.2f}")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    for j in range(n_pairs, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fm_md_pairwise_combined_overlay(ref, fm, beta: float, problem: ProblemSpec, path: str | Path, sample_label: str = "FM") -> None:
    if not problem.n_particles or not problem.spatial_dim:
        return
    import jax
    arrays = [
        np.asarray(jax.device_get(pairwise_distances(ref, problem.n_particles, problem.spatial_dim))).reshape(-1),
        np.asarray(jax.device_get(pairwise_distances(fm, problem.n_particles, problem.spatial_dim))).reshape(-1),
    ]
    bins = np.linspace(*_hist_range(arrays), 90)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    _overlay_hist(ax, arrays[0], bins, f"MD beta={float(beta):.2f}", BLACK, filled=False)
    _overlay_hist(ax, arrays[1], bins, f"{sample_label} beta={float(beta):.2f}", BLUE, filled=True)
    _style_axes(ax)
    ax.set_xlabel("pair distance")
    ax.set_ylabel("density")
    ax.set_title(f"{problem.name}: {sample_label} vs MD pairwise beta={float(beta):.2f}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_md_fm_am_energy_overlay(ref, fm, am, beta: float, problem: ProblemSpec, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    e_ref = _energy_np(problem, ref)
    e_fm = _energy_np(problem, fm)
    e_am = _energy_np(problem, am)
    bins = np.linspace(*_hist_range([e_ref, e_fm, e_am]), 90)
    _overlay_hist(ax, e_ref, bins, f"MD beta={float(beta):.2f}", BLACK, filled=False)
    _overlay_hist(ax, e_fm, bins, f"FM beta={float(beta):.2f}", BLUE, filled=True)
    _overlay_hist(ax, e_am, bins, f"AM beta={float(beta):.2f}", GREEN, filled=True)
    _style_axes(ax)
    ax.set_xlabel("energy")
    ax.set_ylabel("density")
    ax.set_title(f"{problem.name}: FM and AM vs MD energy beta={float(beta):.2f}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_md_fm_am_pairwise_combined_overlay(ref, fm, am, beta: float, problem: ProblemSpec, path: str | Path) -> None:
    if not problem.n_particles or not problem.spatial_dim:
        return
    import jax
    arrays = [
        np.asarray(jax.device_get(pairwise_distances(ref, problem.n_particles, problem.spatial_dim))).reshape(-1),
        np.asarray(jax.device_get(pairwise_distances(fm, problem.n_particles, problem.spatial_dim))).reshape(-1),
        np.asarray(jax.device_get(pairwise_distances(am, problem.n_particles, problem.spatial_dim))).reshape(-1),
    ]
    bins = np.linspace(*_hist_range(arrays), 90)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    _overlay_hist(ax, arrays[0], bins, f"MD beta={float(beta):.2f}", BLACK, filled=False)
    _overlay_hist(ax, arrays[1], bins, f"FM beta={float(beta):.2f}", BLUE, filled=True)
    _overlay_hist(ax, arrays[2], bins, f"AM beta={float(beta):.2f}", GREEN, filled=True)
    _style_axes(ax)
    ax.set_xlabel("pair distance")
    ax.set_ylabel("density")
    ax.set_title(f"{problem.name}: FM and AM vs MD pairwise beta={float(beta):.2f}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
