#!/usr/bin/env python3
"""
Visualization Tool for Training Pairs

This script visualizes H&E and Orion training pairs to verify:
1. Proper image alignment after registration
2. H&E image quality and appearance
3. Orion multiplex channel content
4. Spatial correspondence between modalities
"""

import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from typing import List, Tuple, Optional
from tifffile import imread
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_pair(he_path: str, orion_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a training pair from .npy files or image files."""
    try:
        if he_path.endswith('.npy'):
            he_img = np.load(he_path)
            orion_img = np.load(orion_path)
        else:
            he_img = imread(he_path)
            orion_img = imread(orion_path)
        
        return he_img, orion_img
    except Exception as e:
        logger.error(f"Failed to load pair {he_path}, {orion_path}: {e}")
        return None, None


def normalize_for_display(img: np.ndarray, percentile_clip: bool = True) -> np.ndarray:
    """Normalize image for display."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    
    img_float = img.astype(np.float32)
    
    if percentile_clip:
        # Clip to 1st-99th percentile to handle outliers
        p1, p99 = np.percentile(img_float, [1, 99])
        img_float = np.clip(img_float, p1, p99)
    
    # Normalize to [0, 1]
    img_min, img_max = img_float.min(), img_float.max()
    if img_max > img_min:
        img_float = (img_float - img_min) / (img_max - img_min)
    
    return img_float


def visualize_training_pair(he_img: np.ndarray, orion_img: np.ndarray, 
                          pair_id: str, channels_to_show: List[int] = [0, 1, 2, 3],
                          save_path: Optional[str] = None) -> None:
    """Visualize a single training pair with multiple Orion channels."""
    
    # Prepare H&E image
    if he_img.ndim == 3 and he_img.shape[2] == 3:
        he_display = normalize_for_display(he_img)
    elif he_img.ndim == 2:
        he_display = normalize_for_display(he_img)
        he_display = np.stack([he_display] * 3, axis=-1)  # Convert to RGB
    else:
        # Handle other formats
        he_display = normalize_for_display(he_img[:, :, 0] if he_img.ndim == 3 else he_img)
        he_display = np.stack([he_display] * 3, axis=-1)
    
    # Determine Orion format and channels to display
    if orion_img.ndim == 3 and orion_img.shape[0] > 10:  # Multi-channel format (C, H, W)
        num_orion_channels = orion_img.shape[0]
        orion_format = f"{num_orion_channels}-channel multiplex"
    elif orion_img.ndim == 3:  # RGB-like format
        num_orion_channels = orion_img.shape[2]
        orion_format = f"{num_orion_channels}-channel image"
    else:  # 2D image
        num_orion_channels = 1
        orion_format = "Single channel"
    
    # Limit channels to show based on available channels
    channels_to_show = [ch for ch in channels_to_show if ch < num_orion_channels]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Training Pair: {pair_id}\nH&E: {he_img.shape} | Orion: {orion_img.shape} ({orion_format})', 
                 fontsize=14, fontweight='bold')
    
    # H&E image (top left, spanning 2 columns)
    ax_he = fig.add_subplot(gs[0, :2])
    ax_he.imshow(he_display)
    ax_he.set_title('H&E Image', fontsize=12, fontweight='bold')
    ax_he.axis('off')
    
    # H&E histogram
    ax_he_hist = fig.add_subplot(gs[0, 2:])
    if he_img.ndim == 3:
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            if i < he_img.shape[2]:
                ax_he_hist.hist(he_img[:, :, i].flatten(), bins=50, alpha=0.6, 
                              color=color, label=f'Channel {i}')
    else:
        ax_he_hist.hist(he_img.flatten(), bins=50, alpha=0.7, color='gray')
    ax_he_hist.set_title('H&E Intensity Distribution')
    ax_he_hist.set_xlabel('Intensity')
    ax_he_hist.set_ylabel('Count')
    ax_he_hist.legend()
    
    # Orion channels
    for idx, channel in enumerate(channels_to_show):
        row = 1 + idx // 2
        col = idx % 2
        
        if row >= 3:  # Don't exceed subplot grid
            break
        
        ax = fig.add_subplot(gs[row, col])
        
        # Extract channel
        if orion_img.ndim == 3 and orion_img.shape[0] > 10:  # Multi-channel format (C, H, W)
            channel_img = orion_img[channel]
        elif orion_img.ndim == 3:  # RGB-like format (H, W, C)
            channel_img = orion_img[:, :, channel]
        else:  # 2D image
            channel_img = orion_img
        
        # Display channel
        channel_display = normalize_for_display(channel_img)
        im = ax.imshow(channel_display, cmap='hot')
        ax.set_title(f'Orion Channel {channel}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Channel histogram
        ax_hist = fig.add_subplot(gs[row, col + 2])
        ax_hist.hist(channel_img.flatten(), bins=50, alpha=0.7, color='orange')
        ax_hist.set_title(f'Ch{channel} Distribution')
        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Count')
    
    # Add alignment overlay in bottom row
    if len(channels_to_show) > 0:
        ax_overlay = fig.add_subplot(gs[2, :2])
        
        # Create overlay: H&E grayscale + first Orion channel
        he_gray = np.mean(he_display, axis=2) if he_display.ndim == 3 else he_display
        
        if orion_img.ndim == 3 and orion_img.shape[0] > 10:
            orion_ch0 = normalize_for_display(orion_img[channels_to_show[0]])
        elif orion_img.ndim == 3:
            orion_ch0 = normalize_for_display(orion_img[:, :, channels_to_show[0]])
        else:
            orion_ch0 = normalize_for_display(orion_img)
        
        # Resize if needed
        if he_gray.shape != orion_ch0.shape:
            from skimage.transform import resize
            orion_ch0 = resize(orion_ch0, he_gray.shape, preserve_range=True)
        
        # Create RGB overlay (R=H&E, G=Orion, B=0)
        overlay = np.zeros((*he_gray.shape, 3))
        overlay[:, :, 0] = he_gray  # Red channel
        overlay[:, :, 1] = orion_ch0  # Green channel
        
        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f'Alignment Check\n(Red=H&E, Green=Orion Ch{channels_to_show[0]})')
        ax_overlay.axis('off')
        
        # Alignment quality metrics
        ax_metrics = fig.add_subplot(gs[2, 2:])
        
        # Calculate basic alignment metrics
        correlation = np.corrcoef(he_gray.flatten(), orion_ch0.flatten())[0, 1]
        
        metrics_text = f"""Alignment Metrics:
        
Correlation: {correlation:.3f}
H&E mean: {np.mean(he_gray):.3f}
Orion mean: {np.mean(orion_ch0):.3f}
        
Shape match: {he_gray.shape == orion_ch0.shape}
H&E shape: {he_gray.shape}
Orion shape: {orion_ch0.shape}"""
        
        ax_metrics.text(0.1, 0.5, metrics_text, transform=ax_metrics.transAxes, 
                       fontsize=10, verticalalignment='center', fontfamily='monospace')
        ax_metrics.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def find_training_pairs(pairs_dir: str) -> List[Tuple[str, str, str]]:
    """Find all training pairs in the directory."""
    pairs_path = pathlib.Path(pairs_dir)
    
    if not pairs_path.exists():
        raise ValueError(f"Training pairs directory not found: {pairs_dir}")
    
    # Look for .npy files first
    he_files = list(pairs_path.glob("*_HE.npy"))
    
    if not he_files:
        # Look for image files
        he_files = list(pairs_path.glob("*_HE.tif")) + list(pairs_path.glob("*_HE.png"))
    
    pairs = []
    for he_file in he_files:
        # Extract base name
        if he_file.stem.endswith("_HE"):
            base_name = he_file.stem[:-3]
        else:
            base_name = he_file.stem.replace("_HE", "")
        
        # Find corresponding Orion file
        orion_extensions = [".npy", ".tif", ".png"]
        orion_file = None
        
        for ext in orion_extensions:
            potential_orion = pairs_path / f"{base_name}_ORION{ext}"
            if potential_orion.exists():
                orion_file = potential_orion
                break
        
        if orion_file:
            pairs.append((str(he_file), str(orion_file), base_name))
        else:
            logger.warning(f"No Orion file found for {he_file}")
    
    return pairs


def visualize_multiple_pairs(pairs_dir: str, num_pairs: int = 5, 
                           channels_to_show: List[int] = [0, 1, 2, 3],
                           output_dir: Optional[str] = None,
                           random_sample: bool = True) -> None:
    """Visualize multiple training pairs."""
    
    # Find all pairs
    pairs = find_training_pairs(pairs_dir)
    logger.info(f"Found {len(pairs)} training pairs")
    
    if len(pairs) == 0:
        logger.error("No training pairs found!")
        return
    
    # Sample pairs to visualize
    if random_sample:
        pairs_to_show = random.sample(pairs, min(num_pairs, len(pairs)))
    else:
        pairs_to_show = pairs[:num_pairs]
    
    # Create output directory if specified
    if output_dir:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Visualize each pair
    for i, (he_path, orion_path, pair_id) in enumerate(pairs_to_show):
        logger.info(f"Visualizing pair {i+1}/{len(pairs_to_show)}: {pair_id}")
        
        # Load images
        he_img, orion_img = load_training_pair(he_path, orion_path)
        
        if he_img is None or orion_img is None:
            logger.error(f"Failed to load pair {pair_id}")
            continue
        
        # Set save path
        save_path = None
        if output_dir:
            save_path = output_path / f"training_pair_{pair_id}.png"
        
        # Visualize
        try:
            visualize_training_pair(he_img, orion_img, pair_id, 
                                  channels_to_show, str(save_path) if save_path else None)
        except Exception as e:
            logger.error(f"Failed to visualize pair {pair_id}: {e}")
            continue
        
        # Pause between visualizations
        if i < len(pairs_to_show) - 1:
            input("Press Enter to continue to next pair...")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize H&E to Orion training pairs")
    parser.add_argument("--pairs_dir", type=str, required=True, 
                       help="Directory containing training pairs")
    parser.add_argument("--num_pairs", type=int, default=5, 
                       help="Number of pairs to visualize")
    parser.add_argument("--channels", type=int, nargs='+', default=[0, 1, 2, 3],
                       help="Orion channels to display")
    parser.add_argument("--output_dir", type=str, 
                       help="Directory to save visualizations")
    parser.add_argument("--sequential", action="store_true", 
                       help="Show pairs sequentially instead of random sampling")
    
    args = parser.parse_args()
    
    try:
        visualize_multiple_pairs(
            pairs_dir=args.pairs_dir,
            num_pairs=args.num_pairs,
            channels_to_show=args.channels,
            output_dir=args.output_dir,
            random_sample=not args.sequential
        )
        
        print("\n✅ Visualization completed!")
        print("\nWhat to look for:")
        print("- H&E images should show clear tissue morphology")
        print("- Orion channels should show different protein distributions")
        print("- Alignment overlay should show good spatial correspondence")
        print("- Correlation values > 0.3 indicate reasonable alignment")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 