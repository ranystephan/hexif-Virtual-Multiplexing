#!/usr/bin/env python3
"""
ORION Data Analysis Script
===========================
Comprehensive analysis of 20-channel ORION ground truth data

Features:
1. Ground truth visualization for any core(s)
2. Channel intensity statistics across all cores  
3. Spatial pattern analysis (dots vs blobs)
4. Statistical insights for model improvement
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage.measure import label, regionprops
import json
from typing import List, Dict, Tuple
import warnings
import argparse
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# Configuration
NUM_CHANNELS = 20
CMAPS = [
    "viridis", "magma", "plasma", "cividis", "inferno",
    "Greens", "Blues", "Reds", "Purples", "Oranges",
    "Greys", "twilight", "turbo", "coolwarm", "seismic",
    "RdYlBu", "RdYlGn", "Spectral", "jet", "rainbow"
]


# ================== Helper Functions ==================

def discover_cores(pairs_dir: Path) -> List[str]:
    """Discover all core basenames with paired HE and ORION files."""
    cores = []
    for he_file in sorted(pairs_dir.glob("core_*_HE.npy")):
        base = he_file.stem.replace("_HE", "")
        orion_file = pairs_dir / f"{base}_ORION.npy"
        if orion_file.exists():
            cores.append(base)
    return cores


def load_orion(pairs_dir: Path, basename: str) -> np.ndarray:
    """Load ORION data and normalize to [0,1] with shape (H, W, C)."""
    path = pairs_dir / f"{basename}_ORION.npy"
    arr = np.load(path, mmap_mode='r').copy()
    
    # Handle channel-first format
    if arr.ndim == 3 and arr.shape[0] == NUM_CHANNELS:
        arr = np.transpose(arr, (1, 2, 0))
    
    # Normalize to [0,1]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype in (np.uint16, np.int16):
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / np.percentile(arr, 99.9)
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    
    if arr.max() > 1.5:
        arr = arr / 255.0
    
    return arr


def load_he(pairs_dir: Path, basename: str) -> np.ndarray:
    """Load H&E data and normalize to [0,1]."""
    path = pairs_dir / f"{basename}_HE.npy"
    arr = np.load(path, mmap_mode='r').copy()
    
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.max() > 1.5:
        arr = arr.astype(np.float32) / 255.0
    
    return arr


# ================== Visualization Functions ==================

def visualize_core_channels(pairs_dir: Path, basename: str, channels: List[int] = None, 
                            figsize_per_ch: float = 3.0, save_path: str = None):
    """
    Visualize H&E and ORION channels for a given core.
    
    Args:
        pairs_dir: Path to data directory
        basename: Core name (e.g., 'core_001')
        channels: List of channel indices to show (default: all 20)
        figsize_per_ch: Size per subplot
        save_path: Path to save figure (optional)
    """
    if channels is None:
        channels = list(range(NUM_CHANNELS))
    
    # Load data
    he = load_he(pairs_dir, basename)
    orion = load_orion(pairs_dir, basename)
    
    print(f"\nCore: {basename}")
    print(f"H&E shape: {he.shape}")
    print(f"ORION shape: {orion.shape}")
    
    # Create figure: H&E + all channels
    n_plots = len(channels) + 1
    ncols = 5
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per_ch, nrows * figsize_per_ch))
    axes = axes.flatten()
    
    # Plot H&E
    axes[0].imshow(he)
    axes[0].set_title("H&E", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot each channel
    for i, ch in enumerate(channels, start=1):
        ch_data = orion[:, :, ch]
        im = axes[i].imshow(ch_data, cmap=CMAPS[ch % len(CMAPS)], vmin=0, vmax=1)
        
        # Statistics
        mean_val = ch_data.mean()
        max_val = ch_data.max()
        nonzero_pct = (ch_data > 0.01).sum() / ch_data.size * 100
        
        axes[i].set_title(
            f"Ch {ch}\n"
            f"mean={mean_val:.3f} max={max_val:.3f}\n"
            f"coverage={nonzero_pct:.1f}%",
            fontsize=10
        )
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Ground Truth for {basename}", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return he, orion


def visualize_channel_across_cores(pairs_dir: Path, channel_idx: int, core_list: List[str], 
                                   max_cores: int = 10, save_path: str = None):
    """Visualize a specific channel across multiple cores."""
    core_list = core_list[:max_cores]
    
    n = len(core_list)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()
    
    for i, basename in enumerate(core_list):
        orion = load_orion(pairs_dir, basename)
        ch_data = orion[:, :, channel_idx]
        
        im = axes[i].imshow(ch_data, cmap=CMAPS[channel_idx % len(CMAPS)], vmin=0, vmax=1)
        axes[i].set_title(f"{basename}\nmean={ch_data.mean():.3f}", fontsize=9)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Channel {channel_idx} Across Cores", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


# ================== Statistics Functions ==================

def compute_channel_statistics(pairs_dir: Path, cores_list: List[str], 
                               max_pixels_per_core: int = 500000) -> pd.DataFrame:
    """
    Compute comprehensive statistics for each channel across all cores.
    
    Returns:
        DataFrame with statistics per core and channel
    """
    results = []
    
    for basename in cores_list:
        print(f"Processing {basename}...", end='\r')
        orion = load_orion(pairs_dir, basename)
        H, W, C = orion.shape
        
        for ch in range(C):
            ch_data = orion[:, :, ch].flatten()
            
            # Sample if too large
            if len(ch_data) > max_pixels_per_core:
                ch_data = np.random.choice(ch_data, max_pixels_per_core, replace=False)
            
            # Compute statistics
            results.append({
                'core': basename,
                'channel': ch,
                'mean': ch_data.mean(),
                'median': np.median(ch_data),
                'std': ch_data.std(),
                'min': ch_data.min(),
                'max': ch_data.max(),
                'q01': np.percentile(ch_data, 1),
                'q05': np.percentile(ch_data, 5),
                'q95': np.percentile(ch_data, 95),
                'q99': np.percentile(ch_data, 99),
                'coverage_0.01': (ch_data > 0.01).sum() / len(ch_data) * 100,
                'coverage_0.05': (ch_data > 0.05).sum() / len(ch_data) * 100,
                'coverage_0.10': (ch_data > 0.10).sum() / len(ch_data) * 100,
                'skewness': skew(ch_data),
                'kurtosis': kurtosis(ch_data),
            })
    
    print("\nDone!" + " " * 50)
    return pd.DataFrame(results)


def show_top_bottom_cores_per_channel(stats_df: pd.DataFrame, channel: int, 
                                      metric: str = 'mean', top_n: int = 10, 
                                      bottom_n: int = 5):
    """Show top N and bottom N cores for a specific channel based on a metric."""
    channel_data = stats_df[stats_df['channel'] == channel].copy()
    channel_data = channel_data.sort_values(metric, ascending=False)
    
    print(f"\n{'='*80}")
    print(f"CHANNEL {channel} - Sorted by {metric.upper()}")
    print(f"{'='*80}\n")
    
    print(f"TOP {top_n} CORES (highest {metric}):")
    print("─" * 80)
    top_cores = channel_data.head(top_n)
    for idx, row in top_cores.iterrows():
        print(f"{row['core']:30s} | {metric}={row[metric]:.4f} | max={row['max']:.3f} | "
              f"coverage@0.10={row['coverage_0.10']:.1f}%")
    
    print(f"\nBOTTOM {bottom_n} CORES (lowest {metric}):")
    print("─" * 80)
    bottom_cores = channel_data.tail(bottom_n)
    for idx, row in bottom_cores.iterrows():
        print(f"{row['core']:30s} | {metric}={row[metric]:.4f} | max={row['max']:.3f} | "
              f"coverage@0.10={row['coverage_0.10']:.1f}%")
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS FOR CHANNEL {channel}:")
    print("─" * 80)
    print(f"Mean {metric} across all cores: {channel_data[metric].mean():.4f}")
    print(f"Std {metric} across all cores: {channel_data[metric].std():.4f}")
    print(f"Min {metric}: {channel_data[metric].min():.4f}")
    print(f"Max {metric}: {channel_data[metric].max():.4f}")
    print(f"Median coverage@0.10: {channel_data['coverage_0.10'].median():.1f}%")
    
    return top_cores, bottom_cores


# ================== Spatial Analysis Functions ==================

def analyze_spatial_patterns(orion: np.ndarray, channel: int, threshold: float = 0.1) -> Dict:
    """
    Analyze spatial patterns in a channel to detect dots vs blobs.
    
    Returns:
        dict with spatial metrics
    """
    ch_data = orion[:, :, channel]
    
    # Binary mask
    binary = ch_data > threshold
    
    if binary.sum() < 10:  # Too sparse
        return {
            'num_regions': 0,
            'avg_region_size': 0,
            'median_region_size': 0,
            'largest_region_size': 0,
            'region_size_std': 0,
            'fragmentation': 1.0,
            'avg_eccentricity': 0,
            'pattern_type': 'empty',
        }
    
    # Label connected components
    labeled = label(binary)
    regions = regionprops(labeled)
    
    if len(regions) == 0:
        return {
            'num_regions': 0,
            'avg_region_size': 0,
            'median_region_size': 0,
            'largest_region_size': 0,
            'region_size_std': 0,
            'fragmentation': 1.0,
            'avg_eccentricity': 0,
            'pattern_type': 'empty',
        }
    
    # Compute region properties
    areas = [r.area for r in regions]
    eccentricities = [r.eccentricity for r in regions]
    
    avg_size = np.mean(areas)
    median_size = np.median(areas)
    largest_size = np.max(areas)
    size_std = np.std(areas)
    
    # Fragmentation: ratio of number of regions to total coverage
    total_coverage = binary.sum()
    fragmentation = len(regions) / total_coverage if total_coverage > 0 else 1.0
    
    # Pattern classification
    if avg_size < 50:  # Small regions
        pattern = 'dotty'
    elif avg_size < 500:  # Medium regions
        pattern = 'mixed'
    else:  # Large regions
        pattern = 'blob'
    
    # High fragmentation also suggests dotty
    if fragmentation > 0.01:
        pattern = 'dotty'
    elif fragmentation < 0.001:
        pattern = 'blob'
    
    return {
        'num_regions': len(regions),
        'avg_region_size': avg_size,
        'median_region_size': median_size,
        'largest_region_size': largest_size,
        'region_size_std': size_std,
        'fragmentation': fragmentation,
        'avg_eccentricity': np.mean(eccentricities),
        'pattern_type': pattern,
    }


def spatial_analysis_all_cores(pairs_dir: Path, cores_list: List[str], 
                               threshold: float = 0.1) -> pd.DataFrame:
    """Analyze spatial patterns for all cores and channels."""
    results = []
    
    for basename in cores_list:
        print(f"Analyzing spatial patterns: {basename}...", end='\r')
        orion = load_orion(pairs_dir, basename)
        
        for ch in range(NUM_CHANNELS):
            metrics = analyze_spatial_patterns(orion, ch, threshold)
            metrics['core'] = basename
            metrics['channel'] = ch
            results.append(metrics)
    
    print("\nDone!" + " " * 50)
    return pd.DataFrame(results)


# ================== Visualization of Analysis Results ==================

def plot_channel_summary(channel_summary: pd.DataFrame, save_path: str = None):
    """Visualize cross-channel comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    weak_threshold_mean = 0.01
    weak_threshold_coverage = 5.0
    
    # Mean intensity per channel
    axes[0, 0].bar(channel_summary['channel'], channel_summary['mean_mean'], 
                   yerr=channel_summary['mean_std'], capsize=3, alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(weak_threshold_mean, color='red', linestyle='--', 
                       label=f'Weak threshold={weak_threshold_mean}')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean Intensity')
    axes[0, 0].set_title('Average Mean Intensity per Channel')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')
    axes[0, 0].set_xticks(range(NUM_CHANNELS))
    
    # Max intensity per channel
    axes[0, 1].bar(channel_summary['channel'], channel_summary['max_mean'],
                   yerr=channel_summary['max_std'], capsize=3, alpha=0.7, 
                   color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Max Intensity')
    axes[0, 1].set_title('Average Max Intensity per Channel')
    axes[0, 1].grid(alpha=0.3, axis='y')
    axes[0, 1].set_xticks(range(NUM_CHANNELS))
    
    # Coverage per channel
    axes[1, 0].bar(channel_summary['channel'], channel_summary['coverage_0.10_mean'],
                   alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axhline(weak_threshold_coverage, color='red', linestyle='--', 
                       label=f'Weak threshold={weak_threshold_coverage}%')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Coverage % (@0.10 threshold)')
    axes[1, 0].set_title('Average Signal Coverage per Channel')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    axes[1, 0].set_xticks(range(NUM_CHANNELS))
    
    # Skewness per channel
    axes[1, 1].bar(channel_summary['channel'], channel_summary['skewness_mean'],
                   alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Average Skewness')
    axes[1, 1].set_title('Skewness per Channel (higher = more sparse/dotty)')
    axes[1, 1].grid(alpha=0.3, axis='y')
    axes[1, 1].set_xticks(range(NUM_CHANNELS))
    
    plt.suptitle('Cross-Channel Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def plot_spatial_analysis(spatial_df: pd.DataFrame, pattern_summary: pd.DataFrame, 
                         save_path: str = None):
    """Visualize spatial pattern analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pattern type distribution per channel
    pattern_pcts = pattern_summary[['dotty_pct', 'mixed_pct', 'blob_pct']]
    pattern_pcts.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                      color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Percentage')
    axes[0, 0].set_title('Pattern Type Distribution per Channel')
    axes[0, 0].legend(['Dotty', 'Mixed', 'Blob'])
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # Average fragmentation per channel
    frag_by_channel = spatial_df.groupby('channel')['fragmentation'].mean()
    axes[0, 1].bar(frag_by_channel.index, frag_by_channel.values, alpha=0.7, 
                   color='purple', edgecolor='black')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Fragmentation (higher = more dotty)')
    axes[0, 1].set_title('Average Fragmentation per Channel')
    axes[0, 1].set_xticks(range(NUM_CHANNELS))
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Average region size per channel
    size_by_channel = spatial_df.groupby('channel')['avg_region_size'].mean()
    axes[1, 0].bar(size_by_channel.index, size_by_channel.values, alpha=0.7, 
                   color='teal', edgecolor='black')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Average Region Size (pixels)')
    axes[1, 0].set_title('Average Region Size per Channel')
    axes[1, 0].set_xticks(range(NUM_CHANNELS))
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Number of regions per channel
    regions_by_channel = spatial_df.groupby('channel')['num_regions'].mean()
    axes[1, 1].bar(regions_by_channel.index, regions_by_channel.values, alpha=0.7, 
                   color='coral', edgecolor='black')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Average Number of Regions')
    axes[1, 1].set_title('Average Number of Regions per Channel')
    axes[1, 1].set_xticks(range(NUM_CHANNELS))
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.suptitle('Spatial Pattern Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


# ================== Main Analysis Pipeline ==================

def generate_recommendations(channel_summary: pd.DataFrame, pattern_summary: pd.DataFrame, 
                            spatial_df: pd.DataFrame) -> str:
    """Generate modeling recommendations based on data analysis."""
    weak_threshold_mean = 0.01
    weak_threshold_coverage = 5.0
    
    # Identify problematic channels
    weak_chs = channel_summary[
        (channel_summary['mean_mean'] < weak_threshold_mean) | 
        (channel_summary['coverage_0.10_mean'] < weak_threshold_coverage)
    ]['channel'].tolist()
    
    dotty_chs = pattern_summary[pattern_summary['dotty_pct'] > 50].index.tolist()
    
    blob_chs = pattern_summary[pattern_summary['blob_pct'] > 50].index.tolist()
    
    high_var_chs = channel_summary[
        channel_summary['mean_std'] / (channel_summary['mean_mean'] + 1e-6) > 1.0
    ]['channel'].tolist()
    
    avg_frag = spatial_df.groupby('channel')['fragmentation'].mean()
    high_frag_chs = avg_frag[avg_frag > avg_frag.median()].index.tolist()
    
    # Generate report
    report = f"""
{'#'*100}
# MODELING RECOMMENDATIONS BASED ON DATA ANALYSIS
{'#'*100}

1. WEAK CHANNELS (Low Signal):
{'─'*100}
   Channels with very low signal: {weak_chs if weak_chs else 'None'}
   
   → Consider: Lower learning rate for these channels
   → Consider: Channel-specific loss weighting (reduce weight for weak channels)
   → Consider: Remove from training if consistently empty across cores

2. DOTTY PATTERN CHANNELS (Small Scattered Regions):
{'─'*100}
   Channels with predominantly dotty patterns: {dotty_chs if dotty_chs else 'None'}
   
   → Consider: Increase center_window size to capture more context
   → Consider: Add focal loss to emphasize small positive regions
   → Consider: Augment with random crops focused on positive regions
   → Consider: Multi-scale training (different patch sizes)
   → Consider: Add detection/segmentation head specifically for small objects
   → Consider: Increase receptive field size in decoder
   → Consider: Add attention mechanisms to focus on sparse regions

3. BLOB PATTERN CHANNELS (Large Contiguous Regions):
{'─'*100}
   Channels with predominantly blob patterns: {blob_chs if blob_chs else 'None'}
   
   → These channels should be easier to train
   → Consider using as"anchor channels" for multi-task learning

4. HIGH VARIANCE CHANNELS (Inconsistent Across Cores):
{'─'*100}
   Channels with high variance: {high_var_chs if high_var_chs else 'None'}
   
   → Consider: Batch normalization or instance normalization
   → Consider: Per-core or per-batch normalization
   → Consider: Stratified sampling to ensure diverse examples

5. SPATIAL PATTERN INSIGHTS:
{'─'*100}
   Channels with high fragmentation (>median): {high_frag_chs}
   
   → These channels have many small disconnected regions
   → Consider: Perceptual loss or feature matching loss
   → Consider: Adversarial training to improve spatial structure

6. GENERAL RECOMMENDATIONS:
{'─'*100}
   → Current model uses center_window=12 for loss weighting
   → For dotty channels, consider INCREASING center_window or using full image
   → Consider channel-specific center_window sizes based on pattern type
   → Consider adding perceptual/LPIPS loss for spatial structure
   → Current pos_threshold=0.10 may be too high for very sparse channels
   → Consider dynamic pos_threshold based on channel statistics
   → For dotty patterns: Add auxiliary task (e.g., counting bright spots)
   → Monitor per-channel validation loss to identify problematic channels
   → Consider using different architectures for dotty vs blob channels
   → For dotty patterns: Try U-Net with more decoder resolution
   → For blob patterns: Current architecture should work well

7. SPECIFIC CHANNEL STRATEGIES:
{'─'*100}
"""
    
    # Add channel-specific recommendations
    for ch in range(NUM_CHANNELS):
        ch_stats = channel_summary[channel_summary['channel'] == ch].iloc[0]
        ch_pattern = pattern_summary.loc[ch]
        
        strategy = "STANDARD"
        if ch in weak_chs:
            strategy = "WEAK - reduce loss weight or remove"
        elif ch in dotty_chs:
            strategy = "DOTTY - increase receptive field, focal loss"
        elif ch in blob_chs:
            strategy = "BLOB - standard training should work"
        
        report += f"   Ch {ch:2d}: {strategy:40s} | mean={ch_stats['mean_mean']:.4f} | "
        report += f"pattern={ch_pattern['pattern_type'] if 'pattern_type' in ch_pattern else 'N/A'}\n"
    
    report += f"\n{'#'*100}\n"
    
    return report


def run_full_analysis(pairs_dir: str = "core_patches_npy", output_dir: str = "analysis_results"):
    """Run complete data analysis pipeline."""
    pairs_dir = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*100)
    print("ORION DATA ANALYSIS")
    print("="*100)
    
    # Discover cores
    print("\n1. Discovering cores...")
    cores = discover_cores(pairs_dir)
    print(f"   Found {len(cores)} cores with paired H&E and ORION data")
    
    if len(cores) == 0:
        print("ERROR: No cores found!")
        return
    
    # Compute statistics
    print("\n2. Computing channel statistics...")
    stats_df = compute_channel_statistics(pairs_dir, cores)
    stats_df.to_csv(output_dir / "orion_channel_statistics.csv", index=False)
    print(f"   Saved to: {output_dir / 'orion_channel_statistics.csv'}")
    
    # Aggregate statistics
    print("\n3. Aggregating channel summary...")
    channel_summary = stats_df.groupby('channel').agg({
        'mean': ['mean', 'std', 'min', 'max'],
        'max': ['mean', 'std'],
        'coverage_0.01': 'mean',
        'coverage_0.05': 'mean',
        'coverage_0.10': 'mean',
        'skewness': 'mean',
        'kurtosis': 'mean',
    }).round(4)
    channel_summary.columns = ['_'.join(col).strip() for col in channel_summary.columns.values]
    channel_summary = channel_summary.reset_index()
    channel_summary.to_csv(output_dir / "orion_channel_summary.csv", index=False)
    print(f"   Saved to: {output_dir / 'orion_channel_summary.csv'}")
    
    # Spatial analysis
    print("\n4. Running spatial pattern analysis...")
    spatial_df = spatial_analysis_all_cores(pairs_dir, cores)
    spatial_df.to_csv(output_dir / "orion_spatial_analysis.csv", index=False)
    print(f"   Saved to: {output_dir / 'orion_spatial_analysis.csv'}")
    
    # Pattern summary
    print("\n5. Summarizing spatial patterns...")
    pattern_summary = spatial_df.groupby(['channel', 'pattern_type']).size().unstack(fill_value=0)
    pattern_summary['total'] = pattern_summary.sum(axis=1)
    pattern_summary['dotty_pct'] = (pattern_summary.get('dotty', 0) / pattern_summary['total'] * 100).round(1)
    pattern_summary['blob_pct'] = (pattern_summary.get('blob', 0) / pattern_summary['total'] * 100).round(1)
    pattern_summary['mixed_pct'] = (pattern_summary.get('mixed', 0) / pattern_summary['total'] * 100).round(1)
    pattern_summary.to_csv(output_dir / "orion_pattern_summary.csv")
    print(f"   Saved to: {output_dir / 'orion_pattern_summary.csv'}")
    
    # Visualizations
    print("\n6. Generating visualizations...")
    print("   - Channel summary plot...")
    plot_channel_summary(channel_summary, save_path=output_dir / "channel_summary.png")
    print("   - Spatial analysis plot...")
    plot_spatial_analysis(spatial_df, pattern_summary, save_path=output_dir / "spatial_analysis.png")
    
    # Example visualizations
    print("   - Example core visualization...")
    visualize_core_channels(pairs_dir, cores[0], save_path=output_dir / f"{cores[0]}_all_channels.png")
    
    # Generate recommendations
    print("\n7. Generating recommendations...")
    recommendations = generate_recommendations(channel_summary, pattern_summary, spatial_df)
    with open(output_dir / "modeling_recommendations.txt", 'w') as f:
        f.write(recommendations)
    print(recommendations)
    print(f"   Saved to: {output_dir / 'modeling_recommendations.txt'}")
    
    # Summary report
    weak_chs = channel_summary[
        (channel_summary['mean_mean'] < 0.01) | 
        (channel_summary['coverage_0.10_mean'] < 5.0)
    ]['channel'].tolist()
    
    dotty_chs = pattern_summary[pattern_summary['dotty_pct'] > 50].index.tolist()
    blob_chs = pattern_summary[pattern_summary['blob_pct'] > 50].index.tolist()
    
    summary = f"""
{'='*100}
ORION DATA ANALYSIS SUMMARY REPORT
{'='*100}

DATASET OVERVIEW:
  - Total cores: {len(cores)}
  - Total channels: {NUM_CHANNELS}
  - Data directory: {pairs_dir}

CHANNEL STATISTICS:
  - Weak channels (mean < 0.01 or coverage < 5%): {len(weak_chs)}
    {weak_chs if weak_chs else 'None'}
  
  - High skewness channels (sparse/dotty): {channel_summary.nlargest(5, 'skewness_mean')['channel'].tolist()}

SPATIAL PATTERNS:
  - Predominantly dotty channels: {len(dotty_chs)}
    {dotty_chs if dotty_chs else 'None'}
  
  - Predominantly blob channels: {len(blob_chs)}
    {blob_chs if blob_chs else 'None'}

FILES GENERATED:
  - orion_channel_statistics.csv: Per-core, per-channel intensity statistics
  - orion_channel_summary.csv: Aggregated statistics per channel
  - orion_spatial_analysis.csv: Spatial pattern metrics
  - orion_pattern_summary.csv: Pattern type distribution
  - channel_summary.png: Cross-channel comparison plots
  - spatial_analysis.png: Spatial pattern analysis plots
  - modeling_recommendations.txt: Detailed recommendations for improving model

{'='*100}
"""
    
    print(summary)
    with open(output_dir / "summary_report.txt", 'w') as f:
        f.write(summary)
    
    print(f"\n✓ Analysis complete! All results saved to: {output_dir.resolve()}")
    
    return {
        'cores': cores,
        'stats_df': stats_df,
        'channel_summary': channel_summary,
        'spatial_df': spatial_df,
        'pattern_summary': pattern_summary,
    }


# ================== Command Line Interface ==================

def main():
    parser = argparse.ArgumentParser(description='Analyze ORION 20-channel ground truth data')
    parser.add_argument('--pairs_dir', type=str, default='core_patches_npy',
                       help='Directory containing paired H&E and ORION .npy files')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--visualize_core', type=str, default=None,
                       help='Specific core to visualize (e.g., core_001)')
    parser.add_argument('--analyze_channel', type=int, default=None,
                       help='Specific channel to analyze in detail')
    
    args = parser.parse_args()
    
    if args.visualize_core:
        # Quick visualization mode
        pairs_dir = Path(args.pairs_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Visualizing {args.visualize_core}...")
        visualize_core_channels(pairs_dir, args.visualize_core, 
                               save_path=output_dir / f"{args.visualize_core}_visualization.png")
    
    elif args.analyze_channel is not None:
        # Channel-specific analysis
        pairs_dir = Path(args.pairs_dir)
        cores = discover_cores(pairs_dir)
        print(f"Analyzing channel {args.analyze_channel} across {len(cores)} cores...")
        
        stats_df = compute_channel_statistics(pairs_dir, cores)
        top, bottom = show_top_bottom_cores_per_channel(stats_df, args.analyze_channel, 
                                                        metric='mean')
    
    else:
        # Full analysis
        run_full_analysis(args.pairs_dir, args.output_dir)


if __name__ == "__main__":
    main()

