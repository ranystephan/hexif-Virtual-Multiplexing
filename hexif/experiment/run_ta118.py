#!/usr/bin/env python3
import sys
import logging
import json
import math
from pathlib import Path
import numpy as np
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from hexif.inference import HexifPredictor
from hexif.preprocessing import extract_patches, detect_cores

# --- CONFIGURATION ---
# PLEASE UPDATE THESE PATHS TO THE ACTUAL SLIDE LOCATIONS
HE_SLIDE_PATH = "path/to/TA118_HE.tiff" 
ORION_SLIDE_PATH = "path/to/TA118_Orion.tiff" # Only needed if patches are not yet extracted

# Data / Output Config
PATCHES_DIR = Path("data/TA118/TA118_core_patches_npy")
OUTPUT_DIR = Path("hexif/experiment/results_ta118")
CHANNEL_MAP_PATH = Path("hexif/experiment/ta118_channels.csv")

# Optional: cap number of cores (0 = all)
MAX_CORES = 0

# Model Config - Update this to point to your trained model checkpoint
MODEL_PATH = "runs/nov5/focal_l1_plateau/best_model.pth" 
SCALER_PATH = "runs/nov5/focal_l1_plateau/orion_scaler.json"

# Presence metrics thresholds (linear space, after inverse scaling)
PRESENCE_THRESHOLD = 0.10
PRESENCE_MIN_FRACTION = 0.002
CALIBRATION_THRESHOLDS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28, 0.32]

# Channel stats summary thresholds (linear space)
SUMMARY_THRESHOLDS = [0.08, 0.20]
SUMMARY_BLOCK_SIZE = 32
SUMMARY_DENSE_THRESHOLD = 0.35

def ensure_patches_exist():
    """
    Checks if patches exist. If not, attempts to generate them from source slides.
    """
    if not PATCHES_DIR.exists() or not any(PATCHES_DIR.glob("*_HE.npy")):
        logging.info(f"Patches not found in {PATCHES_DIR}. Attempting to extract...")
        
        # Check source slides
        if not Path(HE_SLIDE_PATH).exists():
            logging.error(f"H&E Slide not found at {HE_SLIDE_PATH}. Cannot generate patches.")
            logging.error("Please update HE_SLIDE_PATH in this script or place the pre-processed patches in data/TA118_core_patches_npy/")
            sys.exit(1)
            
        logging.info("Detecting cores in H&E slide...")
        try:
            bboxes, _ = detect_cores(HE_SLIDE_PATH)
            logging.info(f"Found {len(bboxes)} cores.")
            
            logging.info("Extracting patches...")
            # We pass orion_path if it exists, else None
            orion_arg = ORION_SLIDE_PATH if Path(ORION_SLIDE_PATH).exists() else None
            if not orion_arg:
                logging.warning(f"Orion slide not found at {ORION_SLIDE_PATH}. Extracting H&E only (No Ground Truth will be generated).")
            
            extract_patches(HE_SLIDE_PATH, orion_arg, bboxes, str(PATCHES_DIR), target_size=2048)
            logging.info("Extraction complete.")
            
        except Exception as e:
            logging.error(f"Failed to extract patches: {e}")
            sys.exit(1)
    else:
        logging.info(f"Found existing patches in {PATCHES_DIR}.")

def load_channel_names(path: str) -> list[str]:
    logging.info(f"Loading channel names from {path}")
    names = {}
    try:
        with open(path, mode='r') as f:
            reader = csv.DictReader(f)
            # Check headers
            if not reader.fieldnames or "Index" not in reader.fieldnames or "Name" not in reader.fieldnames:
                    f.seek(0)
                    line = f.readline()
                    if "Index,Name" not in line: 
                        f.seek(0)
                        return [line.strip() for line in f.readlines() if line.strip()]

            for row in reader:
                try:
                    idx = int(row["Index"])
                    names[idx] = row["Name"]
                except ValueError:
                    continue
        
        max_idx = max(names.keys()) if names else 19
        out_names = [names.get(i, f"Channel {i}") for i in range(max_idx + 1)]
        return out_names
    except Exception as e:
        logging.error(f"Error loading channel names: {e}")
        return None

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Computes PCC and RMSE between prediction and ground truth per channel.
    Input shapes: (C, H, W)
    """
    C = pred.shape[0]
    metrics = {"pearson": [], "rmse": []}
    
    for c in range(C):
        p_flat = pred[c].flatten()
        g_flat = gt[c].flatten()
        
        # Pearson
        try:
            r, _ = pearsonr(p_flat, g_flat)
            if np.isnan(r): r = 0.0
        except:
            r = 0.0
        metrics["pearson"].append(r)
        
        # RMSE
        mse = np.mean((p_flat - g_flat) ** 2)
        rmse = np.sqrt(mse)
        metrics["rmse"].append(rmse)
        
    return metrics

def _compute_block_density(mask: np.ndarray, block: int) -> np.ndarray:
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
    thresholds: list[float],
    block_size: int = SUMMARY_BLOCK_SIZE,
    dense_threshold: float = SUMMARY_DENSE_THRESHOLD,
) -> dict:
    stats: dict[str, float] = {}
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

def summarize_core(orion: np.ndarray, thresholds: list[float]) -> dict:
    H, W, C = orion.shape
    summary: dict[int, dict[str, float]] = {}
    for c in range(C):
        channel = orion[..., c]
        summary[c] = compute_channel_stats(channel, thresholds)
        summary[c]["mean_intensity_per_pixel"] = summary[c]["mean"]
        summary[c]["total_intensity"] = float(summary[c]["mean"] * H * W)
    return {"channels": summary, "shape": {"height": H, "width": W}}

def aggregate_dataset_stats(per_core: dict) -> dict:
    aggregated: dict[int, dict[str, float]] = {}
    per_channel_values: dict[int, dict[str, list[float]]] = {}

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

def update_presence_counts(pred: np.ndarray, gt: np.ndarray, counts: dict, threshold: float, min_fraction: float):
    C = pred.shape[0]
    for c in range(C):
        pred_mask = pred[c] > threshold
        gt_mask = gt[c] > threshold
        pred_pos = float(pred_mask.mean()) >= min_fraction
        gt_pos = float(gt_mask.mean()) >= min_fraction
        if pred_pos and gt_pos:
            counts["tp"][c] += 1
        elif pred_pos and not gt_pos:
            counts["fp"][c] += 1
        elif not pred_pos and gt_pos:
            counts["fn"][c] += 1
        else:
            counts["tn"][c] += 1

def save_comparison_plot(he_img: np.ndarray, pred: np.ndarray, gt: np.ndarray, 
                        output_path: str, channel_names: list[str], metrics: dict):
    """
    Saves a large diagnostic plot comparing Pred vs GT side-by-side.
    """
    C = pred.shape[0]
    
    # Normalize HE
    if he_img.dtype != np.uint8:
        he_disp = (np.clip(he_img, 0, 1) * 255).astype(np.uint8)
    else:
        he_disp = he_img
        
    # Layout: 1 row for HE, then rows for channels.
    # Grid of pairs. 4 columns (2 pairs of Pred/GT).
    
    pairs_per_row = 2
    n_rows = int(np.ceil(C / pairs_per_row)) + 1 # +1 for HE header
    
    fig, axes = plt.subplots(n_rows, pairs_per_row * 2, figsize=(16, 4 * n_rows))
    
    # Row 0: HE image spanning width
    ax_he = plt.subplot2grid((n_rows, pairs_per_row * 2), (0, 0), colspan=pairs_per_row * 2)
    ax_he.imshow(he_disp)
    ax_he.set_title("Input H&E")
    ax_he.axis("off")
    
    # Channels
    for c in range(C):
        # Determine grid position
        # pair_idx = 0 or 1
        pair_idx = c % pairs_per_row
        row_idx = (c // pairs_per_row) + 1
        
        col_start = pair_idx * 2
        
        # Robust normalization for display (shared min/max for fair comparison)
        # Using 99th percentile of GT or Pred (whichever is higher) to avoid washout
        p99 = max(np.percentile(pred[c], 99), np.percentile(gt[c], 99) if gt is not None else 0)
        p01 = min(np.percentile(pred[c], 1), np.percentile(gt[c], 1) if gt is not None else 0)
        rng = p99 - p01 + 1e-6
        
        def norm(x): return np.clip((x - p01) / rng, 0, 1)
        
        # Pred
        ax_p = axes[row_idx, col_start]
        ax_p.imshow(norm(pred[c]), cmap="magma")
        title = f"{channel_names[c] if channel_names else f'Ch {c}'}\nPred"
        if metrics:
            title += f" (r={metrics['pearson'][c]:.2f})"
        ax_p.set_title(title, fontsize=9)
        ax_p.axis("off")
        
        # GT
        ax_g = axes[row_idx, col_start + 1]
        if gt is not None:
            ax_g.imshow(norm(gt[c]), cmap="magma")
            ax_g.set_title("Ground Truth", fontsize=9)
        else:
            ax_g.text(0.5, 0.5, "No GT", ha='center')
        ax_g.axis("off")
        
    # Hide unused axes
    for i in range(pairs_per_row * 2):
        axes[0, i].axis("off") # Hide underlying grid for row 0
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def save_performance_summary(results_df: pd.DataFrame, output_dir: Path, channel_names: list[str]):
    """
    Saves summary plots of the metrics.
    """
    if results_df.empty: return
    
    # Pearson Boxplot
    plt.figure(figsize=(15, 6))
    if "pearson" in results_df.columns:
        pass

def run_experiment():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    ensure_patches_exist()
    
    if not Path(MODEL_PATH).exists():
        logging.warning(f"Model file not found at {MODEL_PATH}.")
        return 
    if not Path(SCALER_PATH).exists():
        logging.error("Scaler is required.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    predictor = HexifPredictor(MODEL_PATH, SCALER_PATH, device="cuda")
    channel_names = load_channel_names(str(CHANNEL_MAP_PATH)) if CHANNEL_MAP_PATH.exists() else [f"Ch{i}" for i in range(20)]

    he_files = sorted(list(PATCHES_DIR.glob("*_HE.npy")))
    if MAX_CORES and MAX_CORES > 0:
        he_files = he_files[:MAX_CORES]
    logging.info(f"Found {len(he_files)} patches.")
    
    # Results storage
    all_metrics = [] # List of dicts: {core, channel, channel_name, pearson, rmse}
    per_core_stats = {}
    presence_counts = {
        "tp": np.zeros(20, dtype=np.int64),
        "fp": np.zeros(20, dtype=np.int64),
        "fn": np.zeros(20, dtype=np.int64),
        "tn": np.zeros(20, dtype=np.int64),
    }
    presence_counts_by_thr = {
        thr: {
            "tp": np.zeros(20, dtype=np.int64),
            "fp": np.zeros(20, dtype=np.int64),
            "fn": np.zeros(20, dtype=np.int64),
            "tn": np.zeros(20, dtype=np.int64),
        } for thr in CALIBRATION_THRESHOLDS
    }
    
    for he_file in tqdm(he_files, desc="Processing"):
        core_name = he_file.stem.replace("_HE", "")
        
        try:
            he_img = np.load(he_file)
            pred_linear = predictor.predict_image(he_img)
            
            # Load GT if available
            gt_path = PATCHES_DIR / f"{core_name}_ORION.npy"
            gt_img = None
            metrics = None
            
            if gt_path.exists():
                gt_raw = np.load(gt_path)
                # Fix shape: extract_patches saves (H, W, C), model predicts (C, H, W)
                if gt_raw.ndim == 3 and gt_raw.shape[2] == 20:
                    gt_img = gt_raw.transpose(2, 0, 1)
                elif gt_raw.ndim == 3 and gt_raw.shape[0] == 20:
                    gt_img = gt_raw
                
                # Compute metrics
                if gt_img is not None:
                    metrics = compute_metrics(pred_linear, gt_img)
                    update_presence_counts(
                        pred_linear, gt_img, presence_counts,
                        threshold=PRESENCE_THRESHOLD,
                        min_fraction=PRESENCE_MIN_FRACTION,
                    )
                    for thr in CALIBRATION_THRESHOLDS:
                        for_counts = presence_counts_by_thr[thr]
                        update_presence_counts(
                            pred_linear, gt_img, for_counts,
                            threshold=thr,
                            min_fraction=PRESENCE_MIN_FRACTION,
                        )
                    per_core_stats[core_name] = summarize_core(
                        gt_img.transpose(1, 2, 0),
                        thresholds=SUMMARY_THRESHOLDS,
                    )
                    
                    # Store for summary
                    for c in range(20):
                        all_metrics.append({
                            "core": core_name,
                            "channel_idx": c,
                            "channel_name": channel_names[c] if c < len(channel_names) else str(c),
                            "pearson": metrics["pearson"][c],
                            "rmse": metrics["rmse"][c]
                        })

            # Save results
            np.save(OUTPUT_DIR / f"{core_name}_pred.npy", pred_linear)
            
            # Plot
            save_comparison_plot(
                he_img, 
                pred_linear, 
                gt_img, 
                OUTPUT_DIR / f"{core_name}_comparison.png",
                channel_names,
                metrics
            )
            
        except Exception as e:
            logging.error(f"Error processing {core_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary Analysis
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(OUTPUT_DIR / "metrics_detailed.csv", index=False)
        
        # Average per channel
        summary = df.groupby(["channel_idx", "channel_name"]).agg({
            "pearson": ["mean", "std"],
            "rmse": ["mean", "std"]
        }).reset_index()
        summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
        summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
        
        logging.info("Summary Metrics:")
        print(summary)
        
        # Plot Boxplot of Pearson
        plt.figure(figsize=(15, 8))
        df.boxplot(column="pearson", by="channel_name", rot=90, figsize=(15,8))
        plt.title("Pearson Correlation by Channel")
        plt.suptitle("") # Remove default title
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "boxplot_pearson.png")
        
        logging.info(f"Summary plots saved to {OUTPUT_DIR}")

    if per_core_stats:
        aggregated = aggregate_dataset_stats(per_core_stats)
        summary_payload = {
            "pairs_dir": str(PATCHES_DIR.resolve()),
            "n_cores": len(per_core_stats),
            "thresholds": SUMMARY_THRESHOLDS,
            "block_size": SUMMARY_BLOCK_SIZE,
            "dense_threshold": SUMMARY_DENSE_THRESHOLD,
            "aggregated_channel_stats": aggregated,
        }
        json_path = OUTPUT_DIR / "channel_summary.json"
        json_path.write_text(json.dumps(summary_payload, indent=2))
        logging.info(f"Channel summary saved to {json_path}")

    if all_metrics:
        presence_rows = []
        for c in range(20):
            tp = presence_counts["tp"][c]
            fp = presence_counts["fp"][c]
            fn = presence_counts["fn"][c]
            tn = presence_counts["tn"][c]
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            presence_rows.append({
                "channel_idx": c,
                "channel_name": channel_names[c] if c < len(channel_names) else str(c),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            })
        presence_df = pd.DataFrame(presence_rows)
        presence_df.to_csv(OUTPUT_DIR / "presence_metrics.csv", index=False)
        logging.info("Presence metrics saved.")

        calibration_rows = []
        for c in range(20):
            best = {"threshold": None, "f1": -1.0, "precision": 0.0, "recall": 0.0}
            for thr in CALIBRATION_THRESHOLDS:
                counts = presence_counts_by_thr[thr]
                tp = counts["tp"][c]
                fp = counts["fp"][c]
                fn = counts["fn"][c]
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-8, precision + recall)
                if f1 > best["f1"]:
                    best = {"threshold": thr, "f1": f1, "precision": precision, "recall": recall}
            calibration_rows.append({
                "channel_idx": c,
                "channel_name": channel_names[c] if c < len(channel_names) else str(c),
                "best_threshold": best["threshold"],
                "best_f1": best["f1"],
                "precision_at_best": best["precision"],
                "recall_at_best": best["recall"],
            })
        pd.DataFrame(calibration_rows).to_csv(OUTPUT_DIR / "presence_calibration.csv", index=False)
        logging.info("Presence calibration saved.")

        # Worst cores per channel by Pearson
        worst_rows = []
        for c in range(20):
            channel_df = df[df["channel_idx"] == c].sort_values("pearson", ascending=True).head(5)
            for _, row in channel_df.iterrows():
                worst_rows.append({
                    "channel_idx": c,
                    "channel_name": row["channel_name"],
                    "core": row["core"],
                    "pearson": row["pearson"],
                    "rmse": row["rmse"],
                })
        pd.DataFrame(worst_rows).to_csv(OUTPUT_DIR / "worst_cores_by_channel.csv", index=False)

if __name__ == "__main__":
    run_experiment()
