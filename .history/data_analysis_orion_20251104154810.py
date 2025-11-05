"""Utility cells (Python script with # %% delimiters) for exploring ORION cores."""

# %% Imports & configuration
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


PAIRS_DIR = Path("core_patches_npy")  # update if your data lives elsewhere


# %% Core discovery helpers
def discover_core_basenames(pairs_dir: Path) -> List[str]:
    """Return sorted basenames (without suffix) that have both H&E and ORION files."""

    pairs_dir = Path(pairs_dir)
    cores = []
    for he_path in sorted(pairs_dir.glob("core_*_HE.npy")):
        base = he_path.stem.replace("_HE", "")
        orion_path = pairs_dir / f"{base}_ORION.npy"
        if orion_path.exists():
            cores.append(base)
    return cores


def get_orion_path(core_base: str, pairs_dir: Path = PAIRS_DIR) -> Path:
    """Convenience helper to build the ORION .npy path for a core."""

    return Path(pairs_dir) / f"{core_base}_ORION.npy"


def _np_to_float01(a: np.ndarray) -> np.ndarray:
    """Normalize raw ORION arrays to float32 in [0, 1] when possible."""

    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype in (np.uint16, np.int16):
        a = a.astype(np.float32)
        if a.max(initial=0.0) > 1.5:
            # heuristic scaling for high-dynamic-range uint16
            a = a / (np.percentile(a, 99.9) + 1e-6)
    elif a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.max(initial=0.0) > 1.5:
        a = a / 255.0
    return a


def load_orion_cube(core_base: str, pairs_dir: Path = PAIRS_DIR) -> np.ndarray:
    """
    Load an ORION cube and return as float32 array with shape (C, H, W).

    The stored array may be (H, W, C) or (C, H, W); we enforce (C, H, W).
    """

    path = get_orion_path(core_base, pairs_dir)
    data = np.load(path, mmap_mode="r")
    data = _np_to_float01(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D ORION array, got shape {data.shape} for core {core_base}")
    if data.shape[0] == 20:
        cube = data
    elif data.shape[-1] == 20:
        cube = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(
            f"Unexpected ORION shape {data.shape} for core {core_base}; expected channel dimension of 20"
        )
    return np.ascontiguousarray(cube, dtype=np.float32)


# %% Visualization utilities
def _resolve_percentiles(array: np.ndarray, percentiles: Sequence[float] | None) -> tuple[float, float]:
    if not percentiles:
        return float(array.min()), float(array.max())
    lo, hi = np.percentile(array, percentiles)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def plot_orion_channels(
    core_base: str,
    channels: Sequence[int] | None = None,
    pairs_dir: Path = PAIRS_DIR,
    col_wrap: int = 5,
    percentiles: Sequence[float] | None = (0.5, 99.5),
    cmap: str = "magma",
    apply_log: bool = False,
    figsize_per_col: float = 3.0,
    figsize_per_row: float = 3.0,
    suptitle: str | None = None,
) -> plt.Figure:
    """
    Visualize one ORION core's channels in a grid.

    Parameters
    ----------
    core_base : str
        Basename without suffix (e.g., 'core_00123').
    channels : sequence of int, optional
        Which channels to display. Defaults to all available channels.
    pairs_dir : Path
        Directory containing *_ORION.npy arrays.
    col_wrap : int
        Maximum number of columns per row.
    percentiles : (float, float) or None
        If provided, use these percentiles per channel for contrast stretching.
    cmap : str
        Matplotlib colormap.
    apply_log : bool
        If True, apply log1p scaling before plotting to highlight faint signal.
    figsize_per_col, figsize_per_row : float
        Size multipliers controlling the figure size.
    suptitle : str, optional
        Custom super-title; defaults to the core name.
    """

    cube = load_orion_cube(core_base, pairs_dir)
    total_channels, height, width = cube.shape
    if channels is None:
        channels = list(range(total_channels))
    if isinstance(channels, np.ndarray):
        channels = channels.tolist()
    if isinstance(channels, Iterable) and not isinstance(channels, (list, tuple)):
        channels = list(channels)

    n_channels = len(channels)
    n_cols = min(col_wrap, n_channels)
    n_rows = math.ceil(n_channels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_col * n_cols, figsize_per_row * n_rows),
        squeeze=False,
    )

    for idx, channel in enumerate(channels):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        if channel < 0 or channel >= total_channels:
            ax.set_axis_off()
            ax.set_title(f"ch {channel} (invalid)")
            continue
        channel_img = cube[channel]
        if apply_log:
            channel_img = np.log1p(np.maximum(channel_img, 0.0))
        vmin, vmax = _resolve_percentiles(channel_img, percentiles)
        ax.imshow(channel_img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"ch {channel}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots if channel count does not fill the grid
    for blank_idx in range(n_channels, n_rows * n_cols):
        row, col = divmod(blank_idx, n_cols)
        axes[row][col].set_axis_off()

    fig.suptitle(suptitle or f"{core_base} — ORION channels", y=0.995)
    fig.tight_layout()
    return fig


def show_cores(
    cores: Sequence[str],
    channels: Sequence[int] | None = None,
    **plot_kwargs,
) -> List[plt.Figure]:
    """Render multiple cores sequentially; returns the generated figures."""

    figures: List[plt.Figure] = []
    for core in cores:
        fig = plot_orion_channels(core, channels=channels, suptitle=core, **plot_kwargs)
        figures.append(fig)
        plt.show()
    return figures


# %% Example usage (uncomment to run in an interactive session)
if __name__ == "__main__":
    available_cores = discover_core_basenames(PAIRS_DIR)
    print(f"Discovered {len(available_cores)} cores under {PAIRS_DIR.resolve()}")
    if available_cores:
        example_core = available_cores[0]
        print(f"Displaying default visualization for {example_core}")
        plot_orion_channels(example_core)
        plt.show()

