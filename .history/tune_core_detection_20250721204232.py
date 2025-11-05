#!/usr/bin/env python3
"""
Core Detection Parameter Tuning Script

This script helps you interactively tune core detection parameters by visualizing
the results of different parameter combinations on your TMA images.

Usage:
    python tune_core_detection.py --he_wsi input_he.tif --orion_wsi input_orion.tif
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters
from scipy import ndimage
from tifffile import imread
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_wsi_image(image_path, channel=None, max_dim=4000):
    """Load WSI image, optionally extracting a specific channel."""
    logger.info(f"Loading image: {image_path}")
    
    img = imread(image_path)
    logger.info(f"Original shape: {img.shape}, dtype: {img.dtype}")
    
    # Handle multi-channel images
    if img.ndim == 3:
        if channel is not None and img.shape[0] <= 50:  # Likely (C, H, W) format
            logger.info(f"Extracting channel {channel} from {img.shape[0]} channels")
            img = img[channel]
        elif img.shape[2] == 3:  # RGB image
            logger.info("Converting RGB to grayscale")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif channel is not None and img.shape[2] <= 50:  # Likely (H, W, C) format
            logger.info(f"Extracting channel {channel} from {img.shape[2]} channels")
            img = img[:, :, channel]
    
    # Resize if too large
    if max(img.shape) > max_dim:
        scale = max_dim / max(img.shape)
        new_height = int(img.shape[0] * scale)
        new_width = int(img.shape[1] * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized to {img.shape} (scale: {scale:.3f})")
    
    logger.info(f"Final image shape: {img.shape}, range: {img.min()}-{img.max()}")
    return img


def detect_cores(image, min_area=50000, max_area=500000, circularity_threshold=0.4, 
                 expected_diameter=400, gaussian_sigma=2, morphology_disk_size=10):
    """
    Detect tissue cores in an image with tunable parameters.
    
    Returns:
        contours, filtered_contours, stats
    """
    # Step 1: Preprocessing
    if gaussian_sigma > 0:
        smoothed = filters.gaussian(image, sigma=gaussian_sigma, preserve_range=True).astype(image.dtype)
    else:
        smoothed = image.copy()
    
    # Step 2: Thresholding
    threshold = filters.threshold_otsu(smoothed)
    binary = smoothed > threshold
    
    # Step 3: Morphological operations
    if morphology_disk_size > 0:
        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=min_area//10)
        # Fill holes
        binary = ndimage.binary_fill_holes(binary)
        # Additional morphological operations
        selem = morphology.disk(morphology_disk_size)
        binary = morphology.opening(binary, selem)
        binary = morphology.closing(binary, selem)
    
    # Step 4: Find contours
    binary_uint8 = (binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 5: Filter contours
    filtered_contours = []
    stats = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < circularity_threshold:
            continue
        
        # Calculate equivalent diameter
        equiv_diameter = np.sqrt(4 * area / np.pi)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        filtered_contours.append(contour)
        stats.append({
            'area': area,
            'circularity': circularity,
            'equiv_diameter': equiv_diameter,
            'perimeter': perimeter,
            'bbox': (x, y, w, h),
            'centroid': (x + w//2, y + h//2)
        })
    
    return contours, filtered_contours, stats


def visualize_detection_results(image, all_contours, filtered_contours, stats, title="Core Detection"):
    """Visualize core detection results."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original Image\nShape: {image.shape}')
    axes[0].axis('off')
    
    # All detected contours
    img_all = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()
    if img_all.dtype != np.uint8:
        img_all = ((img_all - img_all.min()) / (img_all.max() - img_all.min()) * 255).astype(np.uint8)
    cv2.drawContours(img_all, all_contours, -1, (255, 0, 0), 2)
    axes[1].imshow(img_all)
    axes[1].set_title(f'All Contours\nFound: {len(all_contours)}')
    axes[1].axis('off')
    
    # Filtered contours
    img_filtered = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()
    if img_filtered.dtype != np.uint8:
        img_filtered = ((img_filtered - img_filtered.min()) / (img_filtered.max() - img_filtered.min()) * 255).astype(np.uint8)
    cv2.drawContours(img_filtered, filtered_contours, -1, (0, 255, 0), 3)
    
    # Add core numbers
    for i, stat in enumerate(stats):
        cx, cy = stat['centroid']
        cv2.putText(img_filtered, str(i+1), (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 0), 2)
    
    axes[2].imshow(img_filtered)
    axes[2].set_title(f'Filtered Cores\nFound: {len(filtered_contours)}')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def print_detection_stats(stats, title="Detection Statistics"):
    """Print statistics about detected cores."""
    print(f"\n{title}")
    print("=" * len(title))
    
    if not stats:
        print("No cores detected!")
        return
    
    areas = [s['area'] for s in stats]
    circularities = [s['circularity'] for s in stats]
    diameters = [s['equiv_diameter'] for s in stats]
    
    print(f"Cores detected: {len(stats)}")
    print(f"Area range: {min(areas):.0f} - {max(areas):.0f} (mean: {np.mean(areas):.0f})")
    print(f"Circularity range: {min(circularities):.3f} - {max(circularities):.3f} (mean: {np.mean(circularities):.3f})")
    print(f"Diameter range: {min(diameters):.1f} - {max(diameters):.1f} (mean: {np.mean(diameters):.1f})")


def run_parameter_sweep(image, title="Parameter Sweep"):
    """Run a parameter sweep to help tune detection parameters."""
    print(f"\n🔍 Running parameter sweep for {title}")
    print("=" * 50)
    
    # Test different parameter combinations
    param_sets = [
        # (min_area, max_area, circularity, expected_diameter, gaussian_sigma, morph_size)
        (10000, 1000000, 0.2, 400, 2, 5),   # Very permissive
        (30000, 800000, 0.3, 400, 2, 10),   # Moderately permissive  
        (50000, 500000, 0.4, 400, 2, 10),   # Default parameters
        (75000, 400000, 0.5, 350, 3, 15),   # Strict parameters
        (100000, 300000, 0.6, 300, 3, 20),  # Very strict
    ]
    
    param_names = ["Very Permissive", "Moderately Permissive", "Default", "Strict", "Very Strict"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (params, name) in enumerate(zip(param_sets, param_names)):
        min_area, max_area, circularity, expected_diameter, gaussian_sigma, morph_size = params
        
        all_contours, filtered_contours, stats = detect_cores(
            image, min_area=min_area, max_area=max_area, 
            circularity_threshold=circularity, expected_diameter=expected_diameter,
            gaussian_sigma=gaussian_sigma, morphology_disk_size=morph_size
        )
        
        print(f"{name:20}: {len(filtered_contours):3d} cores (min_area={min_area}, max_area={max_area}, circ={circularity})")
        
        # Visualize on subplot
        if i < 5:  # Only plot first 5
            img_viz = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()
            if img_viz.dtype != np.uint8:
                img_viz = ((img_viz - img_viz.min()) / (img_viz.max() - img_viz.min()) * 255).astype(np.uint8)
            cv2.drawContours(img_viz, filtered_contours, -1, (0, 255, 0), 2)
            axes[i].imshow(img_viz)
            axes[i].set_title(f'{name}\n{len(filtered_contours)} cores')
            axes[i].axis('off')
    
    # Hide the last subplot
    axes[5].axis('off')
    
    plt.suptitle(f'Parameter Sweep - {title}', fontsize=16)
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Tune core detection parameters")
    
    parser.add_argument("--he_wsi", required=True, help="Path to H&E WSI")
    parser.add_argument("--orion_wsi", help="Path to Orion WSI (optional)")
    parser.add_argument("--output", default="./core_detection_tuning", help="Output directory")
    parser.add_argument("--orion_channel", type=int, default=0, help="Channel to use for Orion (default: 0, typically DAPI)")
    
    # Detection parameters
    parser.add_argument("--min_area", type=int, default=50000, help="Minimum core area")
    parser.add_argument("--max_area", type=int, default=500000, help="Maximum core area")
    parser.add_argument("--circularity", type=float, default=0.4, help="Minimum circularity")
    parser.add_argument("--gaussian_sigma", type=float, default=2.0, help="Gaussian smoothing sigma")
    parser.add_argument("--morphology_size", type=int, default=10, help="Morphological operation disk size")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔬 CORE DETECTION PARAMETER TUNING")
    print("=" * 50)
    print(f"H&E WSI: {args.he_wsi}")
    if args.orion_wsi:
        print(f"Orion WSI: {args.orion_wsi}")
    print(f"Output: {args.output}")
    print()
    
    # Load H&E image
    print("📂 Loading H&E image...")
    he_img = load_wsi_image(args.he_wsi)
    
    # Run parameter sweep on H&E
    print("🎯 Running parameter sweep on H&E...")
    fig_he_sweep = run_parameter_sweep(he_img, "H&E Image")
    fig_he_sweep.savefig(output_path / "he_parameter_sweep.png", dpi=150, bbox_inches='tight')
    
    # Test current parameters on H&E
    print(f"\n🔍 Testing current parameters on H&E...")
    all_contours_he, filtered_contours_he, stats_he = detect_cores(
        he_img, 
        min_area=args.min_area,
        max_area=args.max_area,
        circularity_threshold=args.circularity,
        gaussian_sigma=args.gaussian_sigma,
        morphology_disk_size=args.morphology_size
    )
    
    print_detection_stats(stats_he, "H&E Detection Results")
    
    fig_he = visualize_detection_results(he_img, all_contours_he, filtered_contours_he, stats_he, "H&E Core Detection")
    fig_he.savefig(output_path / "he_detection.png", dpi=150, bbox_inches='tight')
    
    # Process Orion if provided
    if args.orion_wsi:
        print("\n📂 Loading Orion image...")
        orion_img = load_wsi_image(args.orion_wsi, channel=args.orion_channel)
        
        print("🎯 Running parameter sweep on Orion...")
        fig_orion_sweep = run_parameter_sweep(orion_img, f"Orion Image (Channel {args.orion_channel})")
        fig_orion_sweep.savefig(output_path / "orion_parameter_sweep.png", dpi=150, bbox_inches='tight')
        
        print(f"\n🔍 Testing current parameters on Orion...")
        all_contours_orion, filtered_contours_orion, stats_orion = detect_cores(
            orion_img,
            min_area=args.min_area,
            max_area=args.max_area,
            circularity_threshold=args.circularity,
            gaussian_sigma=args.gaussian_sigma,
            morphology_disk_size=args.morphology_size
        )
        
        print_detection_stats(stats_orion, f"Orion Detection Results (Channel {args.orion_channel})")
        
        fig_orion = visualize_detection_results(
            orion_img, all_contours_orion, filtered_contours_orion, stats_orion, 
            f"Orion Core Detection (Channel {args.orion_channel})"
        )
        fig_orion.savefig(output_path / "orion_detection.png", dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n💡 Recommendations:")
    print("1. Look at the parameter sweep plots to see which settings detect ~273 cores")
    print("2. Adjust min_area and max_area based on the actual core sizes in your images")
    print("3. Try different circularity thresholds if cores are not perfectly round")
    print("4. For Orion, try different channels (0=DAPI, 1=marker1, etc.)")
    print("5. Use the parameters that work best in your main registration workflow")


if __name__ == "__main__":
    main() 