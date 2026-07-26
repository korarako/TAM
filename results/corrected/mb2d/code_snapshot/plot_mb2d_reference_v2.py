"""Generate standalone publication figures for the MB2D reference-v2 run.

The script deliberately has no DW4 or Ala2 code paths.  It consumes only the
audited equilibrium-reference, FM and AM arrays and writes one figure per file
in both PNG and PDF format, together with machine-readable plotting metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


TRUE = "#111111"
REFERENCE_COLOR = "#8FD18A"
FM = "#F2A270"
AM = "#4776CC"

TARGET_BETA = 1.20
REFERENCE_BETA = 1.00

# These bounds contain the physically relevant Mueller--Brown basins.  They are
# expanded automatically if the robust range of any audited sample set needs it.
CANONICAL_X_RANGE = (-1.8, 1.2)
CANONICAL_Y_RANGE = (-0.4, 2.2)

SERIES = {
    "True": {
        "label": "Exact target",
        "color": TRUE,
        "linestyle": "-",
        "file_tag": "true",
    },
    "Reference": {
        "label": "Equilibrium reference",
        "color": REFERENCE_COLOR,
        "linestyle": "--",
        "file_tag": "reference",
    },
    "FM": {
        "label": "Flow Matching",
        "color": FM,
        "linestyle": "-.",
        "file_tag": "fm",
    },
    "AM": {
        "label": "Adjoint Matching",
        "color": AM,
        "linestyle": ":",
        "file_tag": "am",
    },
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0)


def gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Return a normalized Gaussian kernel without requiring SciPy."""

    radius = max(1, int(math.ceil(float(truncate) * float(sigma))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / float(sigma)))
    return kernel / np.sum(kernel)


def gaussian_filter_axis(
    values: np.ndarray,
    sigma: float,
    axis: int,
    mode: str,
) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode=mode)
    result = np.zeros_like(values, dtype=np.float64)
    for index, weight in enumerate(kernel):
        source = [slice(None)] * values.ndim
        source[axis] = slice(index, index + values.shape[axis])
        result += float(weight) * padded[tuple(source)]
    return result


def gaussian_filter_2d(values: np.ndarray, sigma: float) -> np.ndarray:
    horizontal = gaussian_filter_axis(
        np.asarray(values, dtype=np.float64),
        sigma,
        axis=1,
        mode="constant",
    )
    return gaussian_filter_axis(horizontal, sigma, axis=0, mode="constant")


def gaussian_filter_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter_axis(
        np.asarray(values, dtype=np.float64),
        sigma,
        axis=0,
        mode="edge",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    fig.savefig(paths[0], facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


def mb_energy(points: np.ndarray) -> np.ndarray:
    """Return the scaled Mueller--Brown energy U(x)."""

    points = np.asarray(points, dtype=np.float64)
    coefficients = np.asarray([-200.0, -100.0, -170.0, 15.0])
    quadratic_x = np.asarray([-1.0, -1.0, -6.5, 0.7])
    cross_xy = np.asarray([0.0, 0.0, 11.0, 0.6])
    quadratic_y = np.asarray([-10.0, -10.0, -6.5, 0.7])
    centers_x = np.asarray([1.0, 0.0, -0.5, -1.0])
    centers_y = np.asarray([0.0, 0.5, 1.5, 1.0])

    dx = points[..., 0, None] - centers_x
    dy = points[..., 1, None] - centers_y
    exponent = quadratic_x * dx * dx + cross_xy * dx * dy + quadratic_y * dy * dy
    return 0.02 * np.sum(coefficients * np.exp(exponent), axis=-1)


def load_samples(mb_run: Path) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    source_paths = {
        "true_beta_1p00_samples": mb_run / "ref_samples_beta_1.00.npy",
        "reference_beta_1p20": mb_run / "ref_samples_beta_1.20.npy",
        "fm_beta_1p20": mb_run / "fm_samples_beta_1.20.npy",
        "am_beta_1p20": mb_run / "am_samples_beta_1.20.npy",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MB2D reference-v2 input(s): " + ", ".join(missing))

    samples = {
        "Reference": np.load(source_paths["reference_beta_1p20"]),
        "FM": np.load(source_paths["fm_beta_1p20"]),
        "AM": np.load(source_paths["am_beta_1p20"]),
    }
    beta_1_samples = np.load(source_paths["true_beta_1p00_samples"])
    samples_for_domain = {"beta1.00 reference": beta_1_samples, **samples}

    for label, values in samples_for_domain.items():
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"{label} must have shape (n, 2), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} contains non-finite values")
    return samples_for_domain, source_paths


def choose_plot_range(sample_sets: Iterable[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Choose one robust domain for every density and contour plot."""

    stacked = np.concatenate(tuple(sample_sets), axis=0)
    robust_low = np.quantile(stacked, 0.0005, axis=0)
    robust_high = np.quantile(stacked, 0.9995, axis=0)

    lows = np.minimum(robust_low, np.asarray([CANONICAL_X_RANGE[0], CANONICAL_Y_RANGE[0]]))
    highs = np.maximum(robust_high, np.asarray([CANONICAL_X_RANGE[1], CANONICAL_Y_RANGE[1]]))
    spans = highs - lows
    lows -= 0.025 * spans
    highs += 0.025 * spans

    # Outward rounding makes the recorded plotting range easy to reproduce.
    lows = np.floor(lows * 10.0) / 10.0
    highs = np.ceil(highs * 10.0) / 10.0
    return (float(lows[0]), float(highs[0])), (float(lows[1]), float(highs[1]))


def make_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(x_range[0], x_range[1], grid_size + 1)
    y_edges = np.linspace(y_range[0], y_range[1], grid_size + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.stack((xx, yy), axis=-1)
    energy = mb_energy(points)
    return x_edges, y_edges, x, y, energy, points


def exact_density(
    energy: np.ndarray,
    beta: float,
    cell_area: float,
) -> np.ndarray:
    log_density = -float(beta) * np.asarray(energy, dtype=np.float64)
    log_density -= float(np.max(log_density))
    density = np.exp(log_density)
    density /= float(np.sum(density) * cell_area)
    return density


def sample_density(
    samples: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma_bins: float,
) -> tuple[np.ndarray, float]:
    finite = np.asarray(samples[np.all(np.isfinite(samples), axis=1)], dtype=np.float64)
    inside = (
        (finite[:, 0] >= x_edges[0])
        & (finite[:, 0] <= x_edges[-1])
        & (finite[:, 1] >= y_edges[0])
        & (finite[:, 1] <= y_edges[-1])
    )
    histogram, _, _ = np.histogram2d(
        finite[inside, 0],
        finite[inside, 1],
        bins=(x_edges, y_edges),
        density=False,
    )
    smoothed = gaussian_filter_2d(
        histogram.T.astype(np.float64),
        sigma=float(sigma_bins),
    )
    cell_area = float(np.diff(x_edges)[0] * np.diff(y_edges)[0])
    normalizer = float(np.sum(smoothed) * cell_area)
    if normalizer <= 0.0:
        raise ValueError("The sample density histogram is empty")
    return smoothed / normalizer, float(np.mean(inside))


def plot_density(
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    title: str,
    color: str,
    common_norm: LogNorm,
    output_stem: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.25, 4.6), layout="constrained")
    del color  # Method colors are reserved for one-dimensional overlays.
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    levels = np.geomspace(float(common_norm.vmin), float(common_norm.vmax), 18)
    masked = np.ma.masked_less(
        np.asarray(density, dtype=np.float64),
        float(common_norm.vmin),
    )
    image = ax.contourf(
        x,
        y,
        masked,
        levels=levels,
        cmap="gist_earth",
        norm=common_norm,
        extend="max",
    )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    clean_axes(ax)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"Probability density $p(x)$")
    return save_figure(fig, output_stem)


def smooth_histogram(
    values: np.ndarray,
    edges: np.ndarray,
    sigma_bins: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    filtered_weights = None if weights is None else np.asarray(weights)[finite]
    counts, _ = np.histogram(
        np.asarray(values)[finite],
        bins=edges,
        weights=filtered_weights,
        density=False,
    )
    density = counts.astype(np.float64)
    if float(sigma_bins) > 0.0:
        density = gaussian_filter_1d(
            density,
            sigma=float(sigma_bins),
        )
    widths = np.diff(edges)
    normalizer = float(np.sum(density * widths))
    if normalizer > 0.0:
        density /= normalizer
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    sorted_indices = np.argsort(values[finite])
    sorted_values = np.asarray(values)[finite][sorted_indices]
    sorted_weights = np.asarray(weights)[finite][sorted_indices]
    cumulative = np.cumsum(sorted_weights)
    threshold = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def plot_distribution(
    x_values: dict[str, np.ndarray],
    y_values: dict[str, np.ndarray],
    xlabel: str,
    title: str,
    x_range: tuple[float, float],
    output_stem: Path,
) -> list[Path]:
    """Draw paper-style one-dimensional overlays.

    The empirical distributions follow the requested visual grammar: the
    equilibrium reference and FM are translucent filled histograms, while AM
    is a blue step outline.  The analytic stationary target remains black.
    This is intentionally different from the two-dimensional figures, where
    every distribution is rendered in its own panel with no overlay.
    """

    fig, ax = plt.subplots(figsize=(5.5, 4.4), layout="constrained")
    handles: dict[str, object] = {}

    handles["Reference"] = ax.fill_between(
        x_values["Reference"],
        0.0,
        y_values["Reference"],
        step="mid",
        color=SERIES["Reference"]["color"],
        alpha=0.78,
        linewidth=0.45,
        edgecolor=SERIES["Reference"]["color"],
        label=SERIES["Reference"]["label"],
        zorder=1,
    )
    handles["FM"] = ax.fill_between(
        x_values["FM"],
        0.0,
        y_values["FM"],
        step="mid",
        color=SERIES["FM"]["color"],
        alpha=0.70,
        linewidth=0.45,
        edgecolor=SERIES["FM"]["color"],
        label=SERIES["FM"]["label"],
        zorder=2,
    )
    (handles["AM"],) = ax.plot(
        x_values["AM"],
        y_values["AM"],
        drawstyle="steps-mid",
        color=SERIES["AM"]["color"],
        linestyle="-",
        lw=2.25,
        label=SERIES["AM"]["label"],
        zorder=4,
    )
    (handles["True"],) = ax.plot(
        x_values["True"],
        y_values["True"],
        color=SERIES["True"]["color"],
        linestyle="-",
        lw=2.15,
        label=SERIES["True"]["label"],
        zorder=5,
    )
    ax.set_xlim(x_range)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    clean_axes(ax)
    legend_order = ("True", "Reference", "FM", "AM")
    ax.legend(
        [handles[key] for key in legend_order],
        [SERIES[key]["label"] for key in legend_order],
        frameon=True,
        facecolor="white",
        edgecolor="#D6D6D6",
        framealpha=0.95,
    )
    return save_figure(fig, output_stem)


def make_energy_distribution(
    energy_grid: np.ndarray,
    true_density: np.ndarray,
    cell_area: float,
    samples: dict[str, np.ndarray],
    output_stem: Path,
    bins: int,
) -> tuple[list[Path], tuple[float, float]]:
    sample_energies = {key: mb_energy(values) for key, values in samples.items()}
    true_weights = true_density.ravel() * cell_area
    grid_energy = energy_grid.ravel()

    lower = [weighted_quantile(grid_energy, true_weights, 0.001)]
    upper = [weighted_quantile(grid_energy, true_weights, 0.999)]
    for values in sample_energies.values():
        lower.append(float(np.quantile(values, 0.001)))
        upper.append(float(np.quantile(values, 0.999)))
    lo = min(lower)
    hi = max(upper)
    padding = 0.03 * (hi - lo)
    energy_range = (float(lo - padding), float(hi + padding))
    edges = np.linspace(energy_range[0], energy_range[1], bins + 1)

    x_values: dict[str, np.ndarray] = {}
    y_values: dict[str, np.ndarray] = {}
    x_values["True"], y_values["True"] = smooth_histogram(
        grid_energy,
        edges,
        sigma_bins=1.0,
        weights=true_weights,
    )
    for key, values in sample_energies.items():
        x_values[key], y_values[key] = smooth_histogram(values, edges, sigma_bins=0.0)

    paths = plot_distribution(
        x_values,
        y_values,
        xlabel=r"$U(x)$",
        title=r"Energy distribution, $\beta=1.20$",
        x_range=energy_range,
        output_stem=output_stem,
    )
    return paths, energy_range


def make_coordinate_distribution(
    axis: int,
    true_density: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    samples: dict[str, np.ndarray],
    output_stem: Path,
    bins: int,
) -> list[Path]:
    coordinate = x if axis == 0 else y
    coordinate_range = (float(coordinate[0]), float(coordinate[-1]))
    integration_spacing = float(y[1] - y[0]) if axis == 0 else float(x[1] - x[0])
    true_marginal = (
        np.sum(true_density, axis=0 if axis == 0 else 1) * integration_spacing
    )
    edges = np.linspace(coordinate_range[0], coordinate_range[1], bins + 1)

    x_values: dict[str, np.ndarray] = {"True": coordinate}
    y_values: dict[str, np.ndarray] = {"True": true_marginal}
    for key, values in samples.items():
        x_values[key], y_values[key] = smooth_histogram(
            values[:, axis],
            edges,
            sigma_bins=0.0,
        )

    return plot_distribution(
        x_values,
        y_values,
        xlabel=rf"$x_{axis + 1}$",
        title=rf"$x_{axis + 1}$ distribution, $\beta=1.20$",
        x_range=coordinate_range,
        output_stem=output_stem,
    )


def file_record(path: Path, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def generate(
    mb_run: Path,
    output: Path,
    grid_size: int,
    density_sigma_bins: float,
) -> dict[str, object]:
    loaded_samples, source_paths = load_samples(mb_run)
    samples = {key: loaded_samples[key] for key in ("Reference", "FM", "AM")}
    x_range, y_range = choose_plot_range(loaded_samples.values())
    x_edges, y_edges, x, y, energy_grid, _ = make_grid(x_range, y_range, grid_size)
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    cell_area = dx * dy

    true_beta_1 = exact_density(energy_grid, REFERENCE_BETA, cell_area)
    true_beta_12 = exact_density(energy_grid, TARGET_BETA, cell_area)
    empirical: dict[str, np.ndarray] = {}
    coverages: dict[str, float] = {}
    for key, values in samples.items():
        empirical[key], coverages[key] = sample_density(
            values,
            x_edges,
            y_edges,
            sigma_bins=density_sigma_bins,
        )
    target_densities = {"True": true_beta_12, **empirical}

    all_heatmap_densities = [true_beta_1, true_beta_12, *empirical.values()]
    common_vmax = float(max(np.max(density) for density in all_heatmap_densities))
    common_vmin = common_vmax * 1.0e-3
    common_norm = LogNorm(vmin=common_vmin, vmax=common_vmax, clip=True)

    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    generated += plot_density(
        true_beta_1,
        x_edges,
        y_edges,
        x_range,
        y_range,
        r"Exact density, $\beta=1.00$",
        TRUE,
        common_norm,
        output / "mb2d_true_density_beta_1p00",
    )
    generated += plot_density(
        true_beta_12,
        x_edges,
        y_edges,
        x_range,
        y_range,
        r"Exact density, $\beta=1.20$",
        TRUE,
        common_norm,
        output / "mb2d_true_density_beta_1p20",
    )
    generated += plot_density(
        empirical["Reference"],
        x_edges,
        y_edges,
        x_range,
        y_range,
        r"Equilibrium reference, $\beta=1.20$",
        REFERENCE_COLOR,
        common_norm,
        output / "mb2d_reference_density_beta_1p20",
    )
    generated += plot_density(
        empirical["FM"],
        x_edges,
        y_edges,
        x_range,
        y_range,
        r"FM proposal, $\beta=1.20$",
        FM,
        common_norm,
        output / "mb2d_fm_density_beta_1p20",
    )
    generated += plot_density(
        empirical["AM"],
        x_edges,
        y_edges,
        x_range,
        y_range,
        r"AM result, $\beta=1.20$",
        AM,
        common_norm,
        output / "mb2d_am_density_beta_1p20",
    )
    energy_paths, energy_range = make_energy_distribution(
        energy_grid,
        true_beta_12,
        cell_area,
        samples,
        output / "mb2d_energy_distribution_beta_1p20",
        bins=120,
    )
    generated += energy_paths
    generated += make_coordinate_distribution(
        0,
        true_beta_12,
        x,
        y,
        samples,
        output / "mb2d_x1_distribution_beta_1p20",
        bins=140,
    )
    generated += make_coordinate_distribution(
        1,
        true_beta_12,
        x,
        y,
        samples,
        output / "mb2d_x2_distribution_beta_1p20",
        bins=140,
    )

    metadata: dict[str, object] = {
        "artifact": "MB2D reference-v2 standalone figures",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(mb_run),
        "output_directory": str(output),
        "density_definition": "p_beta(x) = exp(-beta * U(x)) / Z_beta",
        "energy_definition": "scaled Mueller--Brown potential with U(x) = 0.02 * sum_i A_i exp(...)",
        "betas": {
            "exact_reference": REFERENCE_BETA,
            "exact_target": TARGET_BETA,
            "sample_target": TARGET_BETA,
        },
        "grid": {
            "shape_yx": [int(grid_size), int(grid_size)],
            "x1_range_edges": [float(value) for value in x_range],
            "x2_range_edges": [float(value) for value in y_range],
            "cell_width": [dx, dy],
            "shared_across_all_2d_figures": True,
        },
        "density_rendering": {
            "normalization": "probability density, integral over plotting grid equals 1",
            "style": "standalone filled-contour heat map; no method overlay",
            "colormap": "gist_earth",
            "shared_log_color_scale": {
                "vmin": common_vmin,
                "vmax": common_vmax,
            },
            "empirical_histogram_gaussian_sigma_bins": float(density_sigma_bins),
        },
        "distribution_rendering": {
            "energy_bins": 120,
            "coordinate_bins": 140,
            "empirical_histogram_gaussian_sigma_bins": 0.0,
            "exact_energy_gaussian_sigma_bins": 1.0,
            "style": "equilibrium reference and FM translucent filled histograms; AM blue step outline; exact target black line",
            "energy_display_range": [float(value) for value in energy_range],
            "energy_range_rule": "outer 0.1%-99.9% quantiles across exact target and each sample set, plus 3% padding",
        },
        "palette": {
            "True": TRUE,
            "Reference": REFERENCE_COLOR,
            "FM": FM,
            "AM": AM,
        },
        "sample_counts": {
            key: int(len(values)) for key, values in samples.items()
        },
        "sample_fraction_inside_2d_domain": coverages,
        "source_files": {
            key: {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
            for key, path in source_paths.items()
        },
        "software": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "numpy": np.__version__,
            "matplotlib": mpl.__version__,
            "smoothing": "NumPy separable Gaussian convolution (truncate=4 sigma)",
        },
        "figure_count": len(generated) // 2,
        "files": [file_record(path, output) for path in sorted(generated)],
    }
    if Path(__file__).is_file():
        metadata["script"] = {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        }
    metadata_path = output / "mb2d_plot_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate standalone MB2D reference-v2 figures (PNG and PDF)."
    )
    parser.add_argument(
        "--mb-run",
        type=Path,
        required=True,
        help="Directory containing the audited MB2D reference/FM/AM .npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory used only for MB2D figures.",
    )
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--density-sigma-bins", type=float, default=3.0)
    args = parser.parse_args()

    if args.grid_size < 100:
        raise ValueError("--grid-size must be at least 100")
    if not math.isfinite(args.density_sigma_bins) or args.density_sigma_bins <= 0.0:
        raise ValueError("--density-sigma-bins must be positive and finite")

    configure_matplotlib()
    metadata = generate(
        args.mb_run.resolve(),
        args.output.resolve(),
        grid_size=args.grid_size,
        density_sigma_bins=args.density_sigma_bins,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "figure_count": metadata["figure_count"],
                "file_count": len(metadata["files"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
