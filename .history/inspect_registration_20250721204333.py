#!/usr/bin/env python3
"""
Registration Quality Inspection Script

This script helps you visually inspect the quality of VALIS registration results
by creating overlay images and calculating basic alignment metrics.

Usage:
    python inspect_registration.py --registered_dir ./registration_output/registered_wsi
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imread
from skimage.metrics import structural_similarity as ssim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_registered_image(image_path, max_dim=2000):
    """Load a registered image, handling different formats."""
    logger.info(f"Loading registered image: {image_path}")
    
    img = imread(image_path)
    logger.info(f"Original shape: {img.shape}, dtype: {img.dtype}")
    
    # Handle multi-channel images - convert to grayscale for visualization
    if img.ndim == 3:
        if img.shape[0] <= 50 and img.shape[0] < min(img.shape[1], img.shape[2]):
            # Likely (C, H, W) format
            if img.shape[0] >= 3:
                # Use RGB channels if available
                img = img[:3].transpose(1, 2, 0)
                logger.info("Using first 3 channels as RGB")
            else:
                # Use first channel
                img = img[0]
                logger.info("Using first channel")
        elif img.shape[2] == 3:
            # Already RGB
            logger.info("Image is RGB format")
        elif img.shape[2] <= 50:
            # Likely (H, W, C) format - use first channel
            img = img[:, :, 0]
            logger.info("Using first channel from (H,W,C) format")
    
    # Convert to grayscale if RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    # Normalize to 0-255 range
    if img_gray.dtype != np.uint8:
        img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min()) * 255).astype(np.uint8)
    
    # Resize if too large for visualization
    if max(img_gray.shape) > max_dim:
        scale = max_dim / max(img_gray.shape)
        new_height = int(img_gray.shape[0] * scale)
        new_width = int(img_gray.shape[1] * scale)
        img_gray = cv2.resize(img_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized for visualization to {img_gray.shape} (scale: {scale:.3f})")
    
    logger.info(f"Final processed shape: {img_gray.shape}")
    return img_gray


def create_overlay_visualization(he_img, orion_img, title="Registration Overlay"):
    """Create overlay visualization of registered images."""
    # Ensure both images are the same size
    min_height = min(he_img.shape[0], orion_img.shape[0])
    min_width = min(he_img.shape[1], orion_img.shape[1])
    
    he_cropped = he_img[:min_height, :min_width]
    orion_cropped = orion_img[:min_height, :min_width]
    
    # Create different overlay visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original H&E
    axes[0, 0].imshow(he_cropped, cmap='gray')
    axes[0, 0].set_title('H&E (Reference)')
    axes[0, 0].axis('off')
    
    # Original Orion
    axes[0, 1].imshow(orion_cropped, cmap='gray')
    axes[0, 1].set_title('Orion (Registered)')
    axes[0, 1].axis('off')
    
    # Side-by-side comparison
    comparison = np.hstack([he_cropped, orion_cropped])
    axes[0, 2].imshow(comparison, cmap='gray')
    axes[0, 2].set_title('Side-by-Side')
    axes[0, 2].axis('off')
    
    # Color overlay (H&E in red, Orion in green)
    overlay_rgb = np.zeros((min_height, min_width, 3), dtype=np.uint8)
    overlay_rgb[:, :, 0] = he_cropped  # Red channel
    overlay_rgb[:, :, 1] = orion_cropped  # Green channel
    axes[1, 0].imshow(overlay_rgb)
    axes[1, 0].set_title('Color Overlay\n(Red=H&E, Green=Orion)')
    axes[1, 0].axis('off')
    
    # Difference image
    diff = cv2.absdiff(he_cropped, orion_cropped)
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Absolute Difference')
    axes[1, 1].axis('off')
    
    # Blend overlay
    alpha = 0.5
    blended = cv2.addWeighted(he_cropped, alpha, orion_cropped, 1-alpha, 0)
    axes[1, 2].imshow(blended, cmap='gray')
    axes[1, 2].set_title(f'Blended Overlay (α={alpha})')
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def calculate_registration_metrics(he_img, orion_img):
    """Calculate registration quality metrics."""
    # Ensure both images are the same size
    min_height = min(he_img.shape[0], orion_img.shape[0])
    min_width = min(he_img.shape[1], orion_img.shape[1])
    
    he_cropped = he_img[:min_height, :min_width]
    orion_cropped = orion_img[:min_height, :min_width]
    
    # SSIM (Structural Similarity Index)
    ssim_val = ssim(he_cropped, orion_cropped, data_range=255)
    
    # Normalized Cross Correlation
    he_norm = he_cropped.astype(np.float64)
    orion_norm = orion_cropped.astype(np.float64)
    
    he_mean = np.mean(he_norm)
    orion_mean = np.mean(orion_norm)
    
    numerator = np.sum((he_norm - he_mean) * (orion_norm - orion_mean))
    denominator = np.sqrt(np.sum((he_norm - he_mean)**2) * np.sum((orion_norm - orion_mean)**2))
    
    if denominator > 0:
        ncc_val = numerator / denominator
    else:
        ncc_val = 0.0
    
    # Mean Absolute Error
    mae_val = np.mean(np.abs(he_cropped.astype(np.float64) - orion_cropped.astype(np.float64)))
    
    # Root Mean Square Error
    rmse_val = np.sqrt(np.mean((he_cropped.astype(np.float64) - orion_cropped.astype(np.float64))**2))
    
    return {
        'ssim': ssim_val,
        'ncc': ncc_val,
        'mae': mae_val,
        'rmse': rmse_val,
        'image_shape': (min_height, min_width)
    }


def create_metrics_plot(metrics):
    """Create a plot showing registration quality metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Similarity metrics (higher is better)
    similarity_metrics = ['SSIM', 'NCC']
    similarity_values = [metrics['ssim'], metrics['ncc']]
    
    bars1 = axes[0].bar(similarity_metrics, similarity_values, 
                       color=['skyblue', 'lightgreen'], alpha=0.7)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Similarity Score')
    axes[0].set_title('Similarity Metrics (Higher = Better)')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, similarity_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Error metrics (lower is better)
    error_metrics = ['MAE', 'RMSE']
    error_values = [metrics['mae'], metrics['rmse']]
    
    bars2 = axes[1].bar(error_metrics, error_values, 
                       color=['salmon', 'orange'], alpha=0.7)
    axes[1].set_ylabel('Error Value')
    axes[1].set_title('Error Metrics (Lower = Better)')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, error_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + max(error_values)*0.02, 
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def print_registration_assessment(metrics):
    """Print an assessment of registration quality."""
    print("\n📊 REGISTRATION QUALITY ASSESSMENT")
    print("=" * 40)
    
    # Print metrics
    print(f"SSIM (Structural Similarity): {metrics['ssim']:.3f}")
    print(f"NCC (Normalized Cross Correlation): {metrics['ncc']:.3f}")
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.1f}")
    print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.1f}")
    print(f"Image dimensions: {metrics['image_shape']}")
    
    print("\n🎯 QUALITY ASSESSMENT:")
    
    # SSIM assessment
    if metrics['ssim'] > 0.8:
        ssim_quality = "Excellent"
        ssim_emoji = "🟢"
    elif metrics['ssim'] > 0.6:
        ssim_quality = "Good"
        ssim_emoji = "🟡"
    elif metrics['ssim'] > 0.4:
        ssim_quality = "Fair"
        ssim_emoji = "🟠"
    else:
        ssim_quality = "Poor"
        ssim_emoji = "🔴"
    
    print(f"{ssim_emoji} SSIM Quality: {ssim_quality} ({metrics['ssim']:.3f})")
    
    # NCC assessment
    if metrics['ncc'] > 0.7:
        ncc_quality = "Excellent"
        ncc_emoji = "🟢"
    elif metrics['ncc'] > 0.5:
        ncc_quality = "Good"
        ncc_emoji = "🟡"
    elif metrics['ncc'] > 0.3:
        ncc_quality = "Fair"
        ncc_emoji = "🟠"
    else:
        ncc_quality = "Poor"
        ncc_emoji = "🔴"
    
    print(f"{ncc_emoji} NCC Quality: {ncc_quality} ({metrics['ncc']:.3f})")
    
    # Overall assessment
    if metrics['ssim'] > 0.6 and metrics['ncc'] > 0.5:
        overall = "✅ Registration appears successful"
    elif metrics['ssim'] > 0.4 or metrics['ncc'] > 0.3:
        overall = "⚠️ Registration is moderate - check visually"
    else:
        overall = "❌ Registration may have failed - manual inspection needed"
    
    print(f"\n{overall}")
    
    print("\n💡 RECOMMENDATIONS:")
    if metrics['ssim'] < 0.4:
        print("- SSIM is low - consider adjusting VALIS parameters")
        print("- Check if images have sufficient overlapping features")
    if metrics['ncc'] < 0.3:
        print("- Low correlation - images may not be properly aligned") 
        print("- Consider using different reference image or preprocessing")
    if metrics['ssim'] > 0.6:
        print("- Registration quality looks good!")
        print("- Proceed with confidence to core extraction")


def main():
    parser = argparse.ArgumentParser(description="Inspect registration quality")
    
    parser.add_argument("--registered_dir", required=True, 
                       help="Directory containing registered images")
    parser.add_argument("--output", default="./registration_inspection", 
                       help="Output directory for inspection results")
    parser.add_argument("--he_name", default="he_slide.ome.tiff",
                       help="Name of registered H&E file")
    parser.add_argument("--orion_name", default="orion_slide.ome.tiff", 
                       help="Name of registered Orion file")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find registered images
    registered_dir = Path(args.registered_dir)
    he_path = registered_dir / args.he_name
    orion_path = registered_dir / args.orion_name
    
    print("🔍 REGISTRATION QUALITY INSPECTION")
    print("=" * 50)
    print(f"Registered Directory: {registered_dir}")
    print(f"H&E File: {he_path}")
    print(f"Orion File: {orion_path}")
    print(f"Output: {output_path}")
    
    # Check if files exist
    if not he_path.exists():
        print(f"❌ H&E file not found: {he_path}")
        return
    
    if not orion_path.exists():
        print(f"❌ Orion file not found: {orion_path}")
        return
    
    print("\n📂 Loading registered images...")
    he_img = load_registered_image(he_path)
    orion_img = load_registered_image(orion_path)
    
    print("\n📏 Calculating registration metrics...")
    metrics = calculate_registration_metrics(he_img, orion_img)
    
    # Print assessment
    print_registration_assessment(metrics)
    
    print("\n🎨 Creating visualizations...")
    
    # Create overlay visualization
    fig_overlay = create_overlay_visualization(he_img, orion_img, 
                                             "Registration Quality Inspection")
    fig_overlay.savefig(output_path / "registration_overlay.png", 
                       dpi=150, bbox_inches='tight')
    
    # Create metrics plot
    fig_metrics = create_metrics_plot(metrics)
    fig_metrics.savefig(output_path / "registration_metrics.png", 
                       dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Inspection complete! Results saved to: {output_path}")
    print("\n📄 Generated Files:")
    print(f"  📊 registration_overlay.png - Visual comparison of registered images")
    print(f"  📈 registration_metrics.png - Quantitative quality metrics")
    
    print("\n💡 Next Steps:")
    if metrics['ssim'] > 0.6 and metrics['ncc'] > 0.5:
        print("  🎯 Registration looks good - proceed to core detection tuning")
        print("  🔧 Use tune_core_detection.py to optimize core detection parameters")
    else:
        print("  ⚠️ Consider re-running registration with different parameters")
        print("  🔧 Check VALIS parameters or preprocessing settings")


if __name__ == "__main__":
    main() 