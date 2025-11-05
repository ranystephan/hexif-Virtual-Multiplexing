#!/usr/bin/env python3
"""Utility script for inspecting H&E → ORION training data.

Provides:
  • Dataset-wide channel statistics and rankings.
  • Per-channel per-core intensity tables (top/bottom lists).
  • Quick visualization helpers for ORION ground-truth channels.

Example usage:

    python scripts/data_explorer.py summary \
        --pairs_dir core_patches_npy --output_dir output/data_summary

    python scripts/data_explorer.py visualize \
        --pairs_dir core_patches_npy --cores core_0001 core_0007 \
        --include_he --stretch 99.5

    python scripts/data_explorer.py channel-report \
        --pairs_dir core_patches_npy --channel 12 --top 10 --bottom 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# --------------------------- Data helpers ---------------------------


def _np_to_float01(a: np.ndarray) -> np.ndarray:
    """Best-effort conversion of ORION arrays to float32 in [0, 1] range."""

    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype in (np.uint16, np.int16):
        a = a.astype(np.float32)
        if float(a.max(initial=0.0)) > 1.5:
            a = a / (np.percentile(a, 99.9) + 1e-6)
    elif a.dtype != np.float32:
        a = a.astype(np.float32)

    if float(a.max(initial=0.0)) > 1.5:
        a = a / 255.0
    return a


def discover_basenames(pairs_dir: Path) -> List[str]:
    bases: List[str] = []
    for hef in sorted(pairs_dir.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (pairs_dir / f"{base}_ORION.npy").exists():
            bases.append(base)
    return bases


def load_orion_core(pairs_dir: Path, basename: str) -> np.ndarray:
    op = pairs_dir / f"{basename}_ORION.npy"
    if not op.exists():
        raise FileNotFoundError(f"Missing ORION array for {basename}: {op}")
    arr = np.load(op, mmap_mode="r")
    if arr.ndim == 3 and arr.shape[0] == 20:
        arr = np.transpose(arr, (1, 2, 0))
    arr = _np_to_float01(np.asarray(arr))
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected ORION shape {arr.shape} for {basename}")
    return arr


def load_he_core(pairs_dir: Path, basename: str) -> Optional[np.ndarray]:
    hp = pairs_dir / f"{basename}_HE.npy"
    if not hp.exists():
        return None
    he = np.load(hp, mmap_mode="r")
    he = _np_to_float01(np.asarray(he))
    if he.ndim == 2:
        he = np.repeat(he[..., None], 3, axis=2)
    return he


# ----------------------- Channel metric helpers --------------------


def _compute_block_density(mask: np.ndarray, block: int) -> np.ndarray:
    """Return per-block positive density for a boolean mask."""

    if block <= 1:
        return mask.astype(np.float32)

    H, W = mask.shape
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)

    new_h = mask.shape[0] // block
    new_w = mask.shape[1] // block
    if new_h == 0 or new_w == 0:
        return np.zeros((0, 0), dtype=np.float32)

    reshaped = mask.reshape(new_h, block, new_w, block)
    block_sum = reshaped.sum(axis=(1, 3))
    return block_sum.astype(np.float32) / float(block * block)


def compute_channel_stats(
    channel: np.ndarray,
    thresholds: Sequence[float],
    block_size: int = 32,
    dense_threshold: float = 0.35,
) -> Dict[str, float]:
    """Compute summary metrics for a single ORION channel array."""

    stats: Dict[str, float] = {}
    flat = channel.reshape(-1)
    stats["mean"] = float(flat.mean())
    stats["median"] = float(np.median(flat))
    stats["std"] = float(flat.std())
    stats["max"] = float(flat.max(initial=0.0))
    stats["p99"] = float(np.percentile(flat, 99))

    for thr in thresholds:
        mask = channel > thr
        pos_fraction = float(mask.mean())
        key = f"pos_fraction_{thr:.3f}"
        stats[key] = pos_fraction
        if mask.any():
            values = channel[mask]
            stats[f"pos_mean_{thr:.3f}"] = float(values.mean())
            stats[f"pos_p95_{thr:.3f}"] = float(np.percentile(values, 95))
        else:
            stats[f"pos_mean_{thr:.3f}"] = 0.0
            stats[f"pos_p95_{thr:.3f}"] = 0.0

        block_density = _compute_block_density(mask, block_size)
        if block_density.size == 0:
            dense_block_fraction = float(mask.any())
            mean_block_density = float(mask.mean())
            max_block_density = float(mask.max(initial=0.0))
        else:
            dense_block_fraction = float((block_density > dense_threshold).mean())
            mean_block_density = float(block_density.mean())
            max_block_density = float(block_density.max(initial=0.0))

        stats[f"dense_block_fraction_{thr:.3f}"] = dense_block_fraction
        stats[f"mean_block_density_{thr:.3f}"] = mean_block_density
        stats[f"max_block_density_{thr:.3f}"] = max_block_density

    # A heuristic "speckle" score: high when positive intensities are strong but spatial support is sparse.
    primary_thr = thresholds[0]
    frac_key = f"pos_fraction_{primary_thr:.3f}"
    mean_key = f"pos_mean_{primary_thr:.3f}"
    dense_key = f"dense_block_fraction_{primary_thr:.3f}"
    pos_fraction = stats.get(frac_key, 0.0)
    pos_mean = stats.get(mean_key, 0.0)
    dense_fraction = stats.get(dense_key, 0.0)
    stats["speckle_index"] = float((pos_mean + 1e-6) / (dense_fraction + 1e-6))
    stats["speckle_score"] = float((pos_mean + 1e-3) * (1.0 - pos_fraction))

    return stats


def summarize_core(
    basename: str,
    orion: np.ndarray,
    thresholds: Sequence[float],
    block_size: int = 32,
    dense_threshold: float = 0.35,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    H, W, C = orion.shape
    summary: Dict[int, Dict[str, float]] = {}
    for c in range(C):
        channel = orion[..., c]
        summary[c] = compute_channel_stats(channel, thresholds, block_size, dense_threshold)
        summary[c]["mean_intensity_per_pixel"] = summary[c]["mean"]
        summary[c]["total_intensity"] = float(summary[c]["mean"] * H * W)

    return {"channels": summary, "shape": {"height": H, "width": W}}


def aggregate_dataset_stats(
    per_core: Mapping[str, Mapping[str, Mapping[int, Mapping[str, float]]]]
) -> Dict[int, Dict[str, float]]:
    aggregated: Dict[int, Dict[str, float]] = {}
    per_channel_values: Dict[int, Dict[str, List[float]]] = {}

    for core_stats in per_core.values():
        channel_stats = core_stats["channels"]
        for ch, metrics in channel_stats.items():
            channel_dict = per_channel_values.setdefault(ch, {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    channel_dict.setdefault(key, []).append(float(value))

    for ch, metric_lists in per_channel_values.items():
        aggregated[ch] = {}
        for key, values in metric_lists.items():
            arr = np.array(values, dtype=np.float64)
            aggregated[ch][f"{key}_mean"] = float(arr.mean())
            aggregated[ch][f"{key}_median"] = float(np.median(arr))
            aggregated[ch][f"{key}_std"] = float(arr.std())

    return aggregated


def build_channel_rankings(
    per_core: Mapping[str, Mapping[str, Mapping[int, Mapping[str, float]]]],
    *,
    top: int,
    bottom: int,
    thresholds: Sequence[float],
) -> Dict[int, Dict[str, List[Dict[str, float]]]]:
    rankings: Dict[int, Dict[str, List[Dict[str, float]]]] = {}
    primary_thr = thresholds[0]
    frac_key = f"pos_fraction_{primary_thr:.3f}"

    # Prepare per-channel per-core lists
    channel_entries: Dict[int, List[Tuple[str, Mapping[str, float]]]] = {}
    for core, stats in per_core.items():
        for ch, metrics in stats["channels"].items():
            channel_entries.setdefault(ch, []).append((core, metrics))

    for ch, entries in channel_entries.items():
        top_entries = sorted(entries, key=lambda kv: kv[1]["mean"], reverse=True)
        bottom_entries = list(reversed(top_entries))
        speckle_entries = sorted(entries, key=lambda kv: kv[1].get("speckle_score", 0.0), reverse=True)

        def _serialize(items: Sequence[Tuple[str, Mapping[str, float]]], count: int) -> List[Dict[str, float]]:
            out: List[Dict[str, float]] = []
            for core_name, metric in items[:count]:
                out.append({
                    "core": core_name,
                    "mean": float(metric.get("mean", 0.0)),
                    "total_intensity": float(metric.get("total_intensity", 0.0)),
                    frac_key: float(metric.get(frac_key, 0.0)),
                    "speckle_score": float(metric.get("speckle_score", 0.0)),
                })
            return out

        rankings[ch] = {
            "mean_intensity_top": _serialize(top_entries, top),
            "mean_intensity_bottom": _serialize(bottom_entries, bottom),
            "speckle_score_top": _serialize(speckle_entries, top),
        }

    return rankings


# ----------------------------- Commands -----------------------------


def run_summary(args: argparse.Namespace) -> None:
    pairs_dir = Path(args.pairs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basenames = discover_basenames(pairs_dir)
    if args.limit:
        basenames = basenames[: args.limit]

    if not basenames:
        print(f"No core pairs found under {pairs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(basenames)} cores. Computing statistics...", file=sys.stderr)

    per_core: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for idx, base in enumerate(basenames, start=1):
        orion = load_orion_core(pairs_dir, base)
        per_core[base] = summarize_core(
            base,
            orion,
            thresholds=args.thresholds,
            block_size=args.block_size,
            dense_threshold=args.dense_threshold,
        )
        if idx % 10 == 0 or idx == len(basenames):
            print(f"  processed {idx}/{len(basenames)} cores", file=sys.stderr)

    aggregated = aggregate_dataset_stats(per_core)
    rankings = build_channel_rankings(
        per_core,
        top=args.top,
        bottom=args.bottom,
        thresholds=args.thresholds,
    )

    # Save JSON summary
    summary_payload = {
        "pairs_dir": str(pairs_dir.resolve()),
        "n_cores": len(basenames),
        "thresholds": list(args.thresholds),
        "block_size": args.block_size,
        "dense_threshold": args.dense_threshold,
        "aggregated_channel_stats": aggregated,
        "channel_rankings": rankings,
    }
    json_path = output_dir / "channel_summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2))

    # Save per-core per-channel CSV
    per_core_csv = output_dir / "per_core_channel_metrics.csv"
    with per_core_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        first_core = next(iter(per_core.values()))
        first_channel_metrics = next(iter(first_core["channels"].values()))
        metric_keys = [k for k in first_channel_metrics.keys() if isinstance(first_channel_metrics[k], (int, float))]
        metric_keys.sort()
        writer.writerow(["core", "channel"] + metric_keys)
        for core_name, core_stats in per_core.items():
            for ch, metrics in core_stats["channels"].items():
                row = [core_name, ch] + [metrics.get(k, 0.0) for k in metric_keys]
                writer.writerow(row)

    # Save aggregated CSV
    agg_csv = output_dir / "aggregated_channel_stats.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        if aggregated:
            example_metrics = next(iter(aggregated.values()))
            agg_keys = sorted(example_metrics.keys())
            writer.writerow(["channel"] + agg_keys)
            for ch in sorted(aggregated.keys()):
                row = [ch] + [aggregated[ch].get(k, 0.0) for k in agg_keys]
                writer.writerow(row)

    print(f"Summary saved to {json_path}")
    print(f"Per-core metrics → {per_core_csv}")
    print(f"Aggregated stats → {agg_csv}")

    # Highlight channels with lowest coverage and highest speckle score.
    primary_thr = args.thresholds[0]
    frac_key = f"pos_fraction_{primary_thr:.3f}_mean"
    speckle_key = "speckle_score_mean"

    def channel_sort_key(key: str) -> List[Tuple[int, float]]:
        return sorted(
            (
                (ch, metrics.get(key, float("nan")))
                for ch, metrics in aggregated.items()
                if not math.isnan(metrics.get(key, float("nan")))
            ),
            key=lambda x: x[1],
        )

    sparse_channels = channel_sort_key(frac_key)[:5]
    speckle_channels = sorted(
        (
            (ch, metrics.get(speckle_key, 0.0))
            for ch, metrics in aggregated.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    print("\nChannels with lowest positive coverage (primary threshold):")
    for ch, value in sparse_channels:
        print(f"  ch{ch:02d}: coverage ≈ {value:.4f}")

    print("\nChannels with highest speckle score (potential small-dot patterns):")
    for ch, value in speckle_channels:
        print(f"  ch{ch:02d}: speckle_score ≈ {value:.4f}")


def run_visualize(args: argparse.Namespace) -> None:
    pairs_dir = Path(args.pairs_dir)
    output_dir = Path(args.save_dir or (Path(args.output_dir) / "visualizations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    basenames = discover_basenames(pairs_dir)
    requested = args.cores or []
    missing = [core for core in requested if core not in basenames]
    if missing:
        raise ValueError(f"Requested cores not found: {missing}")

    if not requested:
        raise ValueError("No cores provided for visualization.")

    if args.channels and "all" not in args.channels:
        try:
            channels = [int(ch) for ch in args.channels]
        except ValueError as exc:
            raise ValueError("Channels must be integers or 'all'.") from exc
    else:
        channels = None

    import matplotlib.pyplot as plt  # Lazy import

    for core in requested:
        orion = load_orion_core(pairs_dir, core)
        he_img = load_he_core(pairs_dir, core) if args.include_he else None

        n_channels = orion.shape[2]
        channel_indices = channels or list(range(n_channels))
        n_to_display = len(channel_indices)
        cols = args.cols or 5
        rows = int(math.ceil(n_to_display / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
        axes = np.atleast_2d(axes)

        vmax_percentile = args.stretch

        for idx, ch in enumerate(channel_indices):
            ax = axes[idx // cols, idx % cols]
            data = orion[..., ch]
            vmax = np.percentile(data, vmax_percentile)
            if vmax <= 0:
                vmax = data.max(initial=1e-6)
            im = ax.imshow(data, cmap=args.cmap, vmin=0, vmax=vmax)
            ax.set_title(f"ch{ch:02d}")
            ax.axis("off")
            if args.colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused axes
        for idx in range(n_to_display, rows * cols):
            ax = axes[idx // cols, idx % cols]
            ax.axis("off")

        fig.suptitle(f"{core} – ORION channels", fontsize=16)
        fig.tight_layout()

        out_path = output_dir / f"{core}_orion_channels.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        if he_img is not None:
            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
            ax2.imshow(he_img)
            ax2.set_title(f"{core} – H&E")
            ax2.axis("off")
            he_path = output_dir / f"{core}_he.png"
            fig2.savefig(he_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig2)

        print(f"Saved visualization for {core} → {out_path}")


def run_channel_report(args: argparse.Namespace) -> None:
    pairs_dir = Path(args.pairs_dir)
    basenames = discover_basenames(pairs_dir)
    if not basenames:
        print(f"No cores found under {pairs_dir}", file=sys.stderr)
        sys.exit(1)

    thresholds = [args.threshold]

    per_core: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    n_channels: Optional[int] = None
    for base in basenames:
        orion = load_orion_core(pairs_dir, base)
        if n_channels is None:
            n_channels = orion.shape[2]
        per_core[base] = summarize_core(
            base,
            orion,
            thresholds=thresholds,
            block_size=args.block_size,
            dense_threshold=args.dense_threshold,
        )

    channel_entries: List[Tuple[str, Dict[str, float]]] = []
    for core, stats in per_core.items():
        channel_metrics = stats["channels"].get(args.channel)
        if channel_metrics is not None:
            channel_entries.append((core, channel_metrics))

    if not channel_entries:
        max_channel = n_channels - 1 if n_channels is not None else "unknown"
        raise ValueError(f"Channel {args.channel} not found in dataset (expected up to {max_channel}).")

    primary_key = f"pos_fraction_{args.threshold:.3f}"

    sorted_by_mean = sorted(channel_entries, key=lambda kv: kv[1]["mean"], reverse=True)
    sorted_by_cov = sorted(channel_entries, key=lambda kv: kv[1].get(primary_key, 0.0), reverse=True)

    def _display(label: str, data: Sequence[Tuple[str, Dict[str, float]]], count: int) -> None:
        print(f"\n{label}")
        for core_name, metrics in data[:count]:
            mean_val = metrics.get("mean", 0.0)
            cov_val = metrics.get(primary_key, 0.0)
            speckle = metrics.get("speckle_score", 0.0)
            print(f"  {core_name:<20} mean={mean_val:.4f} coverage={cov_val:.4f} speckle={speckle:.4f}")

    print(f"Channel {args.channel} report (threshold {args.threshold:.3f})")
    _display("Top cores by mean intensity:", sorted_by_mean, args.top)
    _display("Bottom cores by mean intensity:", list(reversed(sorted_by_mean)), args.bottom)
    _display("Best coverage cores (fraction over threshold):", sorted_by_cov, args.top)


# ----------------------------- CLI setup ----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore ORION training data.")
    parser.add_argument("--pairs_dir", type=str, default="core_patches_npy", help="Directory containing *_HE.npy and *_ORION.npy pairs.")
    parser.add_argument("--output_dir", type=str, default="output/data_explorer", help="Base directory for reports/plots.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Compute dataset-wide statistics and rankings.")
    summary.add_argument("--thresholds", type=float, nargs="+", default=[0.08, 0.20], help="Positive-pixel intensity thresholds (log-space).")
    summary.add_argument("--block_size", type=int, default=32, help="Block size for spatial-density heuristics.")
    summary.add_argument("--dense_threshold", type=float, default=0.35, help="Density above which a block is considered dense.")
    summary.add_argument("--top", type=int, default=10, help="Top-N cores to save per channel.")
    summary.add_argument("--bottom", type=int, default=5, help="Bottom-N cores to save per channel.")
    summary.add_argument("--limit", type=int, default=0, help="Limit number of cores (for quick sanity checks).")

    visualize = subparsers.add_parser("visualize", help="Create channel grids for selected cores.")
    visualize.add_argument("--cores", nargs="+", help="List of core basenames to visualize.")
    visualize.add_argument("--channels", nargs="*", default=["all"], help="Subset of channels to display (e.g. 0 1 5) or 'all'.")
    visualize.add_argument("--cols", type=int, default=5, help="Number of columns in the grid.")
    visualize.add_argument("--stretch", type=float, default=99.0, help="Percentile for vmax when plotting.")
    visualize.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap name.")
    visualize.add_argument("--colorbar", action="store_true", help="Add per-panel colorbars.")
    visualize.add_argument("--dpi", type=int, default=160, help="Figure DPI when saving.")
    visualize.add_argument("--include_he", action="store_true", help="Also save the H&E reference image.")
    visualize.add_argument("--save_dir", type=str, default="", help="Optional explicit directory for outputs.")

    channel_report = subparsers.add_parser("channel-report", help="Inspect a specific channel across cores.")
    channel_report.add_argument("--channel", type=int, required=True, help="Channel index (0-based).")
    channel_report.add_argument("--threshold", type=float, default=0.08, help="Positive threshold for coverage stats.")
    channel_report.add_argument("--block_size", type=int, default=32, help="Block size for speckle heuristics.")
    channel_report.add_argument("--dense_threshold", type=float, default=0.35, help="Density threshold for dense blocks.")
    channel_report.add_argument("--top", type=int, default=10, help="Top-N cores to list.")
    channel_report.add_argument("--bottom", type=int, default=5, help="Bottom-N cores to list.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "summary":
        run_summary(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "channel-report":
        run_channel_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

