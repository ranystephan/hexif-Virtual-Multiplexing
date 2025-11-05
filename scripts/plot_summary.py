#!/usr/bin/env python3
"""Create plots from `data_explorer.py summary` outputs.

Reads `channel_summary.json` (and optionally the per-core CSV) and
emits several diagnostic figures that highlight per-channel signal
levels, coverage, and speckle-like behaviour. The goal is to turn the
summary numbers into quick-glance visuals for model triage.

Example usage:

    python scripts/plot_summary.py \
        --summary_json output/data_summary/channel_summary.json \
        --output_dir output/data_summary/plots \
        --top_n 5

Figures generated:
  • channel_mean_intensity.png       – mean per-pixel intensity per channel
  • channel_coverage.png             – coverage (positive fraction) per channel
  • channel_speckle_vs_coverage.png  – scatter: coverage vs speckle score
  • channel_top_mean_heatmap.png     – heatmap of top-N cores by mean intensity
  • channel_speckle_bar.png          – average speckle score per channel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_summary(summary_path: Path) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, List[Dict[str, float]]]], List[float], Dict]:
    data = json.loads(summary_path.read_text())
    aggregated_raw = data.get("aggregated_channel_stats", {})
    rankings_raw = data.get("channel_rankings", {})

    def _coerce_keys(obj: Mapping[str, Dict]) -> Dict[int, Dict]:
        out: Dict[int, Dict] = {}
        for key, value in obj.items():
            try:
                out[int(key)] = value
            except (TypeError, ValueError):
                # fall back to best effort
                out[int(float(key))] = value
        return out

    aggregated = _coerce_keys(aggregated_raw)
    rankings = _coerce_keys(rankings_raw)
    thresholds = data.get("thresholds", [])
    return aggregated, rankings, thresholds, data


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _channel_labels(channels: Iterable[int]) -> List[str]:
    return [f"ch{ch:02d}" for ch in channels]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_bar(
    channels: Sequence[int],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    highlight_threshold: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(channels))
    bars = ax.bar(x, values, color="#3778bf")
    ax.set_xticks(x)
    ax.set_xticklabels(_channel_labels(channels), rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if highlight_threshold is not None:
        ax.axhline(highlight_threshold, color="red", linestyle="--", linewidth=1, label="threshold")
        ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scatter(
    x: Sequence[float],
    y: Sequence[float],
    channels: Sequence[int],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x, y, color="#2c7fb8", s=70, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    for xi, yi, ch in zip(x, y, channels):
        ax.annotate(f"ch{ch:02d}", (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#08306b")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    channels: Sequence[int],
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    fmt: str = "{x:.3f}",
) -> None:
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(0.9 * matrix.shape[1] + 4, 0.4 * matrix.shape[0] + 3))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(_channel_labels(channels))
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([f"rank {i+1}" for i in range(matrix.shape[1])], rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    for (i, j), val in np.ndenumerate(matrix):
        if np.isnan(val):
            label = "–"
        else:
            label = fmt.format(x=val)
        ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean intensity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------


def generate_plots(
    summary_json: Path,
    output_dir: Path,
    *,
    top_n: int = 5,
    min_channels: Optional[Sequence[int]] = None,
) -> None:
    aggregated, rankings, thresholds, raw_data = load_summary(summary_json)
    ensure_output_dir(output_dir)

    channels = sorted(aggregated.keys())
    if not channels:
        raise RuntimeError("No channels found in aggregated stats.")

    # Metric keys based on first threshold
    primary_thr = thresholds[0] if thresholds else 0.08
    frac_key = f"pos_fraction_{primary_thr:.3f}_mean"
    mean_intensity_key = "mean_mean"
    speckle_key = "speckle_score_mean"

    mean_values = [aggregated[ch].get(mean_intensity_key, np.nan) for ch in channels]
    coverage_values = [aggregated[ch].get(frac_key, np.nan) for ch in channels]
    speckle_values = [aggregated[ch].get(speckle_key, np.nan) for ch in channels]

    plot_bar(
        channels,
        mean_values,
        title="Mean per-pixel intensity per channel",
        ylabel="mean intensity (quantile/log space)",
        output_path=output_dir / "channel_mean_intensity.png",
    )

    plot_bar(
        channels,
        coverage_values,
        title=f"Positive coverage per channel (threshold {primary_thr:.3f})",
        ylabel="fraction of pixels over threshold",
        output_path=output_dir / "channel_coverage.png",
    )

    plot_bar(
        channels,
        speckle_values,
        title="Average speckle score per channel",
        ylabel="speckle score (higher = sparser, intense dots)",
        output_path=output_dir / "channel_speckle_bar.png",
    )

    plot_scatter(
        coverage_values,
        speckle_values,
        channels,
        title="Speckle vs coverage",
        xlabel="coverage over threshold",
        ylabel="speckle score",
        output_path=output_dir / "channel_speckle_vs_coverage.png",
    )

    # Heatmap of top cores by mean intensity
    matrix = []
    for ch in channels:
        ranking = rankings.get(ch, {}).get("mean_intensity_top", [])
        row = [entry.get("mean", np.nan) for entry in ranking[:top_n]]
        if len(row) < top_n:
            row.extend([np.nan] * (top_n - len(row)))
        matrix.append(row)
    matrix_np = np.array(matrix, dtype=float)

    plot_heatmap(
        matrix_np,
        channels,
        title="Top cores by channel mean intensity",
        xlabel="core rank",
        output_path=output_dir / "channel_top_mean_heatmap.png",
    )

    # Optional: write quick text report pointing to extremes
    text_report = output_dir / "quick_insights.txt"
    with text_report.open("w") as f:
        f.write(f"Summary source: {summary_json}\n")
        f.write(f"Channels analysed: {len(channels)}\n")
        f.write(f"Primary coverage threshold: {primary_thr:.3f}\n\n")

        def top_n_by(metric_values: Sequence[float], n: int, reverse: bool = True) -> List[Tuple[int, float]]:
            pairs = list(zip(channels, metric_values))
            pairs = [p for p in pairs if not np.isnan(p[1])]
            return sorted(pairs, key=lambda kv: kv[1], reverse=reverse)[:n]

        f.write("Highest average max intensity channels:\n")
        for ch, val in top_n_by(mean_values, 5):
            f.write(f"  ch{ch:02d}: mean ≈ {val:.4f}\n")

        f.write("\nLowest coverage channels:\n")
        for ch, val in top_n_by(coverage_values, 5, reverse=False):
            f.write(f"  ch{ch:02d}: coverage ≈ {val:.4f}\n")

        f.write("\nMost speckled channels:\n")
        for ch, val in top_n_by(speckle_values, 5):
            f.write(f"  ch{ch:02d}: speckle ≈ {val:.4f}\n")

    print(f"Plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot data_explorer summary outputs.")
    parser.add_argument("--summary_json", type=str, required=True, help="Path to channel_summary.json produced by data_explorer summary.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where plots should be saved.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top cores to display per channel in the heatmap.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary_json = Path(args.summary_json)
    output_dir = Path(args.output_dir)
    generate_plots(summary_json, output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()

