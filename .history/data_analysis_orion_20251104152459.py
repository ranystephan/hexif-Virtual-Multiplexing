#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Analysis for H&E → ORION Training Data

This script provides comprehensive analysis and visualization of the ORION dataset
to help understand channel characteristics, intensity distributions, and potential
training issues.

Run cells individually in VS Code or similar IDE that supports # %% cell delimiters.
"""

# %% Imports and Setup
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict
import seaborn as sns
from scipy import ndimage
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %% Configuration
PAIRS_DIR = Path("core_patches_npy")
OUTPUT_DIR = Path("data_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ORION channel names (adjust these to match your actual markers)
CHANNEL_NAMES = [
    f"Channel_{i:02d}" for i in range(20)
]

# Better colormaps for each channel (cycling through distinct colormaps)
CHANNEL_CMAPS = [
    "viridis", "magma", "plasma", "cividis", "inferno",
    "Greens", "Blues", "Reds", "Purples", "Oranges",
    "YlOrRd", "YlGnBu", "RdPu", "BuPu", "GnBu",
    "PuRd", "OrRd", "BuGn", "YlOrBr", "PuBu"
]

print(f"Data directory: {PAIRS_DIR.resolve()}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

# %% Utility Functions

def discover_cores(pairs_dir: Path) -> List[str]:
    """Discover all available core basenames."""
    cores = []
    for he_file in sorted(pairs_dir.glob("core_*_HE.npy")):
        base = he_file.stem.replace("_HE", "")
        orion_file = pairs_dir / f"{base}_ORION.npy"
        if orion_file.exists():
            cores.append(base)
    return cores

def load_he(pairs_dir: Path, basename: str) -> np.ndarray:
    """Load H&E image and normalize to [0, 1]."""
    he_path = pairs_dir / f"{basename}_HE.npy"
    he = np.load(he_path)
    
    # Normalize to float [0, 1]
    if he.dtype == np.uint8:
        he = he.astype(np.float32) / 255.0
    elif he.dtype in (np.uint16, np.int16):
        he = he.astype(np.float32)
        if he.max() > 1.5:
            he = he / np.percentile(he, 99.9)
    
    return he

def load_orion(pairs_dir: Path, basename: str) -> np.ndarray:
    """Load ORION image and normalize to [0, 1]."""
    orion_path = pairs_dir / f"{basename}_ORION.npy"
    orion = np.load(orion_path)
    
    # Handle channel-first format
    if orion.ndim == 3 and orion.shape[0] == 20:
        orion = np.transpose(orion, (1, 2, 0))  # C, H, W -> H, W, C
    
    # Normalize to float [0, 1]
    if orion.dtype == np.uint8:
        orion = orion.astype(np.float32) / 255.0
    elif orion.dtype in (np.uint16, np.int16):
        orion = orion.astype(np.float32)
        if orion.max() > 1.5:
            orion = orion / np.percentile(orion, 99.9)
    
    return orion

def get_channel_statistics(channel_data: np.ndarray) -> Dict:
    """Compute comprehensive statistics for a channel."""
    flat = channel_data.flatten()
    
    # Basic statistics
    stats = {
        'mean': float(np.mean(flat)),
        'median': float(np.median(flat)),
        'std': float(np.std(flat)),
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
        'q01': float(np.percentile(flat, 1)),
        'q05': float(np.percentile(flat, 5)),
        'q25': float(np.percentile(flat, 25)),
        'q75': float(np.percentile(flat, 75)),
        'q95': float(np.percentile(flat, 95)),
        'q99': float(np.percentile(flat, 99)),
    }
    
    # Signal coverage (% of pixels above various thresholds)
    stats['pct_above_0.01'] = float(np.mean(flat > 0.01) * 100)
    stats['pct_above_0.05'] = float(np.mean(flat > 0.05) * 100)
    stats['pct_above_0.10'] = float(np.mean(flat > 0.10) * 100)
    stats['pct_above_0.20'] = float(np.mean(flat > 0.20) * 100)
    
    # Sparsity measure
    stats['sparsity'] = float(np.mean(flat < 0.01) * 100)
    
    return stats

# %% Discover Available Cores
cores = discover_cores(PAIRS_DIR)
print(f"\nFound {len(cores)} core pairs:")
for i, core in enumerate(cores[:10]):  # Show first 10
    print(f"  {i+1}. {core}")
if len(cores) > 10:
    print(f"  ... and {len(cores) - 10} more")

# %% [CELL 1] Visualize Ground Truth for Specific Cores

def visualize_core_channels(
    pairs_dir: Path,
    basenames: List[str],
    channels_to_show: Optional[List[int]] = None,
    save_path: Optional[Path] = None
):
    """
    Visualize ground truth ORION channels for one or more cores.
    
    Parameters:
    -----------
    pairs_dir : Path
        Directory containing the core pairs
    basenames : List[str]
        List of core basenames to visualize
    channels_to_show : Optional[List[int]]
        List of channel indices to show (0-19). If None, shows all 20 channels.
    save_path : Optional[Path]
        Path to save the figure
    """
    if channels_to_show is None:
        channels_to_show = list(range(20))
    
    n_channels = len(channels_to_show)
    n_cores = len(basenames)
    
    # Create figure: columns for each core, rows for H&E + each channel
    fig, axes = plt.subplots(
        nrows=n_channels + 1,  # +1 for H&E row
        ncols=n_cores,
        figsize=(5 * n_cores, 3 * (n_channels + 1))
    )
    
    # Handle single core case
    if n_cores == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, basename in enumerate(basenames):
        # Load data
        he = load_he(pairs_dir, basename)
        orion = load_orion(pairs_dir, basename)
        
        # Show H&E in first row
        axes[0, col_idx].imshow(he)
        axes[0, col_idx].set_title(f"{basename}\nH&E", fontsize=10)
        axes[0, col_idx].axis('off')
        
        # Show each channel
        for row_idx, ch in enumerate(channels_to_show, start=1):
            ch_data = orion[:, :, ch]
            ch_name = CHANNEL_NAMES[ch]
            cmap = CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)]
            
            im = axes[row_idx, col_idx].imshow(ch_data, cmap=cmap, vmin=0, vmax=1)
            
            # Add statistics to title
            mean_val = np.mean(ch_data)
            max_val = np.max(ch_data)
            pct_signal = np.mean(ch_data > 0.05) * 100
            
            title = f"{ch_name}\nμ={mean_val:.3f} max={max_val:.3f}\n{pct_signal:.1f}% signal"
            axes[row_idx, col_idx].set_title(title, fontsize=8)
            axes[row_idx, col_idx].axis('off')
            
            # Add colorbar on the last column
            if col_idx == n_cores - 1:
                plt.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    return fig

# Example: Visualize first 2 cores, all channels
if len(cores) >= 2:
    visualize_core_channels(
        PAIRS_DIR,
        basenames=cores[:2],
        save_path=OUTPUT_DIR / "ground_truth_visualization.png"
    )

# %% [CELL 2] Visualize Specific Channels Across Multiple Cores

def visualize_channel_across_cores(
    pairs_dir: Path,
    channel_idx: int,
    basenames: Optional[List[str]] = None,
    max_cores: int = 8,
    save_path: Optional[Path] = None
):
    """
    Visualize a single channel across multiple cores to see variation.
    
    Parameters:
    -----------
    pairs_dir : Path
        Directory containing the core pairs
    channel_idx : int
        Channel index to visualize (0-19)
    basenames : Optional[List[str]]
        List of cores to show. If None, uses first max_cores available.
    max_cores : int
        Maximum number of cores to show
    save_path : Optional[Path]
        Path to save the figure
    """
    if basenames is None:
        basenames = cores[:max_cores]
    else:
        basenames = basenames[:max_cores]
    
    n_cores = len(basenames)
    ncols = min(4, n_cores)
    nrows = (n_cores + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = np.array(axes).flatten()
    
    ch_name = CHANNEL_NAMES[channel_idx]
    cmap = CHANNEL_CMAPS[channel_idx % len(CHANNEL_CMAPS)]
    
    for idx, basename in enumerate(basenames):
        orion = load_orion(pairs_dir, basename)
        ch_data = orion[:, :, channel_idx]
        
        im = axes[idx].imshow(ch_data, cmap=cmap, vmin=0, vmax=1)
        
        # Statistics
        mean_val = np.mean(ch_data)
        max_val = np.max(ch_data)
        pct_signal = np.mean(ch_data > 0.05) * 100
        
        title = f"{basename}\nμ={mean_val:.3f} max={max_val:.3f}\n{pct_signal:.1f}% signal"
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # Hide extra subplots
    for idx in range(n_cores, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"{ch_name} (Channel {channel_idx}) Across Cores", fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    return fig

# Example: Visualize channel 0 across first 8 cores
visualize_channel_across_cores(
    PAIRS_DIR,
    channel_idx=0,
    max_cores=8,
    save_path=OUTPUT_DIR / "channel_00_across_cores.png"
)

# %% [CELL 3] Channel Intensity Analysis Across All Cores

def analyze_channel_intensities(pairs_dir: Path, cores: List[str]) -> Dict:
    """
    Compute intensity statistics for each channel across all cores.
    
    Returns:
    --------
    Dict with structure:
        {
            channel_idx: {
                'global_stats': {...},
                'per_core': {basename: {...}, ...}
            }
        }
    """
    channel_data = defaultdict(lambda: {'per_core': {}})
    
    print("Analyzing channel intensities across all cores...")
    for i, basename in enumerate(cores):
        if (i + 1) % 10 == 0:
            print(f"  Processing core {i+1}/{len(cores)}: {basename}")
        
        orion = load_orion(pairs_dir, basename)
        
        for ch in range(20):
            ch_data = orion[:, :, ch]
            stats = get_channel_statistics(ch_data)
            channel_data[ch]['per_core'][basename] = stats
    
    # Compute global statistics per channel
    for ch in range(20):
        per_core = channel_data[ch]['per_core']
        
        global_stats = {
            'mean_of_means': np.mean([s['mean'] for s in per_core.values()]),
            'std_of_means': np.std([s['mean'] for s in per_core.values()]),
            'mean_of_maxs': np.mean([s['max'] for s in per_core.values()]),
            'mean_sparsity': np.mean([s['sparsity'] for s in per_core.values()]),
            'mean_pct_signal': np.mean([s['pct_above_0.05'] for s in per_core.values()]),
        }
        channel_data[ch]['global_stats'] = global_stats
    
    print("Analysis complete!")
    return dict(channel_data)

# Run analysis
channel_analysis = analyze_channel_intensities(PAIRS_DIR, cores)

# Save results
with open(OUTPUT_DIR / "channel_intensity_analysis.json", "w") as f:
    json.dump(channel_analysis, f, indent=2)
print(f"Saved analysis to {OUTPUT_DIR / 'channel_intensity_analysis.json'}")

# %% [CELL 4] Top and Bottom Cores by Channel Intensity

def print_intensity_rankings(channel_analysis: Dict, channel_idx: int, top_n: int = 10, bottom_n: int = 5):
    """Print cores ranked by mean intensity for a specific channel."""
    
    ch_name = CHANNEL_NAMES[channel_idx]
    per_core = channel_analysis[channel_idx]['per_core']
    
    # Sort by mean intensity
    sorted_cores = sorted(per_core.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"Channel {channel_idx}: {ch_name}")
    print(f"{'='*80}")
    
    global_stats = channel_analysis[channel_idx]['global_stats']
    print(f"\nGlobal Statistics:")
    print(f"  Mean intensity across cores: {global_stats['mean_of_means']:.4f} ± {global_stats['std_of_means']:.4f}")
    print(f"  Mean of max values: {global_stats['mean_of_maxs']:.4f}")
    print(f"  Mean sparsity (% pixels < 0.01): {global_stats['mean_sparsity']:.2f}%")
    print(f"  Mean signal coverage (% pixels > 0.05): {global_stats['mean_pct_signal']:.2f}%")
    
    print(f"\n🔥 TOP {top_n} cores by mean intensity:")
    print(f"{'Rank':<6} {'Core Name':<30} {'Mean':<10} {'Max':<10} {'% Signal':<12}")
    print("-" * 80)
    for rank, (basename, stats) in enumerate(sorted_cores[:top_n], 1):
        print(f"{rank:<6} {basename:<30} {stats['mean']:<10.4f} {stats['max']:<10.4f} {stats['pct_above_0.05']:<12.2f}")
    
    print(f"\n❄️  BOTTOM {bottom_n} cores by mean intensity:")
    print(f"{'Rank':<6} {'Core Name':<30} {'Mean':<10} {'Max':<10} {'% Signal':<12}")
    print("-" * 80)
    for rank, (basename, stats) in enumerate(sorted_cores[-bottom_n:][::-1], 1):
        actual_rank = len(sorted_cores) - bottom_n + rank
        print(f"{actual_rank:<6} {basename:<30} {stats['mean']:<10.4f} {stats['max']:<10.4f} {stats['pct_above_0.05']:<12.2f}")

# Example: Show rankings for channels 0-5
for ch in range(min(6, 20)):
    print_intensity_rankings(channel_analysis, ch, top_n=10, bottom_n=5)

# %% [CELL 5] Visualize Channel Intensity Distribution Summary

def plot_channel_intensity_summary(channel_analysis: Dict, save_path: Optional[Path] = None):
    """Create comprehensive visualization of channel statistics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect data
    channels = list(range(20))
    mean_intensities = [channel_analysis[ch]['global_stats']['mean_of_means'] for ch in channels]
    std_intensities = [channel_analysis[ch]['global_stats']['std_of_means'] for ch in channels]
    mean_maxs = [channel_analysis[ch]['global_stats']['mean_of_maxs'] for ch in channels]
    mean_sparsity = [channel_analysis[ch]['global_stats']['mean_sparsity'] for ch in channels]
    mean_signal = [channel_analysis[ch]['global_stats']['mean_pct_signal'] for ch in channels]
    
    # Plot 1: Mean intensity per channel
    ax = axes[0, 0]
    bars = ax.bar(channels, mean_intensities, yerr=std_intensities, capsize=5, alpha=0.7)
    ax.set_xlabel("Channel Index", fontsize=12)
    ax.set_ylabel("Mean Intensity", fontsize=12)
    ax.set_title("Average Intensity per Channel (± std across cores)", fontsize=14)
    ax.set_xticks(channels)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Mean of max values
    ax = axes[0, 1]
    bars = ax.bar(channels, mean_maxs, alpha=0.7, color='orange')
    ax.set_xlabel("Channel Index", fontsize=12)
    ax.set_ylabel("Mean of Max Values", fontsize=12)
    ax.set_title("Average Maximum Intensity per Channel", fontsize=14)
    ax.set_xticks(channels)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Sparsity
    ax = axes[1, 0]
    bars = ax.bar(channels, mean_sparsity, alpha=0.7, color='red')
    ax.set_xlabel("Channel Index", fontsize=12)
    ax.set_ylabel("Sparsity (%)", fontsize=12)
    ax.set_title("Channel Sparsity (% pixels < 0.01)", fontsize=14)
    ax.set_xticks(channels)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=90, color='darkred', linestyle='--', label='90% sparsity', alpha=0.5)
    ax.legend()
    
    # Plot 4: Signal coverage
    ax = axes[1, 1]
    bars = ax.bar(channels, mean_signal, alpha=0.7, color='green')
    ax.set_xlabel("Channel Index", fontsize=12)
    ax.set_ylabel("Signal Coverage (%)", fontsize=12)
    ax.set_title("Channel Signal Coverage (% pixels > 0.05)", fontsize=14)
    ax.set_xticks(channels)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    return fig

# Plot summary
plot_channel_intensity_summary(
    channel_analysis,
    save_path=OUTPUT_DIR / "channel_intensity_summary.png"
)

# %% [CELL 6] Identify Problematic Channels

def identify_problematic_channels(channel_analysis: Dict, 
                                   low_intensity_threshold: float = 0.02,
                                   high_sparsity_threshold: float = 95.0):
    """
    Identify channels that might be problematic for training.
    """
    print(f"\n{'='*80}")
    print("PROBLEMATIC CHANNEL ANALYSIS")
    print(f"{'='*80}\n")
    
    low_intensity_channels = []
    high_sparsity_channels = []
    low_variance_channels = []
    
    for ch in range(20):
        global_stats = channel_analysis[ch]['global_stats']
        ch_name = CHANNEL_NAMES[ch]
        
        # Check low intensity
        if global_stats['mean_of_means'] < low_intensity_threshold:
            low_intensity_channels.append((ch, global_stats['mean_of_means']))
        
        # Check high sparsity
        if global_stats['mean_sparsity'] > high_sparsity_threshold:
            high_sparsity_channels.append((ch, global_stats['mean_sparsity']))
        
        # Check low variance across cores
        if global_stats['std_of_means'] < 0.001:
            low_variance_channels.append((ch, global_stats['std_of_means']))
    
    print("⚠️  LOW INTENSITY CHANNELS (might be undertrained):")
    if low_intensity_channels:
        for ch, intensity in low_intensity_channels:
            print(f"  Channel {ch:2d} ({CHANNEL_NAMES[ch]}): mean intensity = {intensity:.5f}")
    else:
        print("  None found ✓")
    
    print(f"\n⚠️  HIGH SPARSITY CHANNELS (>{high_sparsity_threshold}% pixels near zero):")
    if high_sparsity_channels:
        for ch, sparsity in high_sparsity_channels:
            print(f"  Channel {ch:2d} ({CHANNEL_NAMES[ch]}): {sparsity:.2f}% sparse")
    else:
        print("  None found ✓")
    
    print("\n⚠️  LOW VARIANCE CHANNELS (similar across all cores):")
    if low_variance_channels:
        for ch, variance in low_variance_channels:
            print(f"  Channel {ch:2d} ({CHANNEL_NAMES[ch]}): std = {variance:.6f}")
    else:
        print("  None found ✓")
    
    return {
        'low_intensity': low_intensity_channels,
        'high_sparsity': high_sparsity_channels,
        'low_variance': low_variance_channels
    }

# Identify problematic channels
problematic = identify_problematic_channels(channel_analysis)

# %% [CELL 7] Distribution of Intensities (Histogram)

def plot_intensity_distributions(pairs_dir: Path, cores: List[str], 
                                  channels_to_plot: Optional[List[int]] = None,
                                  sample_cores: int = 10,
                                  save_path: Optional[Path] = None):
    """
    Plot intensity distributions (histograms) for selected channels.
    """
    if channels_to_plot is None:
        channels_to_plot = list(range(20))
    
    n_channels = len(channels_to_plot)
    fig, axes = plt.subplots(
        nrows=(n_channels + 3) // 4,
        ncols=4,
        figsize=(20, 5 * ((n_channels + 3) // 4))
    )
    axes = axes.flatten()
    
    # Sample cores for efficiency
    sampled_cores = cores[::max(1, len(cores) // sample_cores)][:sample_cores]
    
    print(f"Computing histograms for {len(sampled_cores)} cores...")
    
    for idx, ch in enumerate(channels_to_plot):
        ax = axes[idx]
        ch_name = CHANNEL_NAMES[ch]
        
        # Collect data from multiple cores
        all_values = []
        for basename in sampled_cores:
            orion = load_orion(pairs_dir, basename)
            ch_data = orion[:, :, ch].flatten()
            all_values.extend(ch_data[::10])  # Subsample for efficiency
        
        all_values = np.array(all_values)
        
        # Plot histogram
        ax.hist(all_values, bins=100, alpha=0.7, color=CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)].lower() 
                if CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)].lower() in plt.colormaps() else 'blue', 
                edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Intensity", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{ch_name}\nμ={np.mean(all_values):.3f}, med={np.median(all_values):.3f}", fontsize=10)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    return fig

# Plot distributions for all channels
plot_intensity_distributions(
    PAIRS_DIR,
    cores,
    sample_cores=10,
    save_path=OUTPUT_DIR / "intensity_distributions.png"
)

# %% Summary Report

print(f"\n{'='*80}")
print("DATA ANALYSIS SUMMARY")
print(f"{'='*80}\n")

print(f"Total cores analyzed: {len(cores)}")
print(f"Total channels: 20")
print(f"\nOutput files saved to: {OUTPUT_DIR.resolve()}")
print("  - ground_truth_visualization.png")
print("  - channel_00_across_cores.png")
print("  - channel_intensity_summary.png")
print("  - intensity_distributions.png")
print("  - channel_intensity_analysis.json")

print("\n✅ Visualization cells complete!")
print("\nNext steps:")
print("  1. Review the generated visualizations")
print("  2. Identify channels with low signal or unusual patterns")
print("  3. Use insights to adjust training parameters")
print("  4. Consider the statistical analysis cells for deeper insights")

