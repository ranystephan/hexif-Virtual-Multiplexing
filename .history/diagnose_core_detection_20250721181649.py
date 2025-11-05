#!/usr/bin/env python3
"""
Diagnostic script for core detection parameters.
This helps visualize what the core detection algorithm sees and tune parameters.
"""

import numpy as np
import cv2
from skimage import filters, morphology, measure
from skimage import io
import matplotlib.pyplot as plt
import pathlib
import argparse
from tifffile import imread

def visualize_core_detection_process(image, image_name, config_params):
    """Visualize each step of the core detection process."""
    
    print(f"\n=== Core Detection Diagnostics for {image_name} ===")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: {image.min()} - {image.max()}")
    print(f"Image mean: {image.mean():.2f}")
    print(f"Image std: {image.std():.2f}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:  # RGB (H, W, C)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            print("Converted RGB to grayscale")
        elif image.shape[0] <= 50 and image.shape[0] < min(image.shape[1], image.shape[2]):
            # Multi-channel format (C, H, W) - channels first
            print(f"Detected (C, H, W) format with {image.shape[0]} channels, using channel 0")
            gray = image[0, :, :]  # Take first channel (e.g., DAPI)
        else:
            # Assume (H, W, C) format - channels last
            print(f"Detected (H, W, C) format, using first channel")
            gray = image[:, :, 0]
    elif len(image.shape) == 2:
        gray = image
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")
    
    print(f"Grayscale shape: {gray.shape}")
    print(f"Grayscale range: {gray.min()} - {gray.max()}")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Core Detection Diagnostics: {image_name}', fontsize=16)
    
    # 1. Original grayscale
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Grayscale')
    axes[0, 0].axis('off')
    
    # 2. Gaussian smoothing
    gray_smooth = filters.gaussian(gray, sigma=config_params['gaussian_sigma'])
    axes[0, 1].imshow(gray_smooth, cmap='gray')
    axes[0, 1].set_title(f'Gaussian Smooth (σ={config_params["gaussian_sigma"]})')
    axes[0, 1].axis('off')
    
    # 3. Otsu thresholding
    threshold = filters.threshold_otsu(gray_smooth)
    binary = gray_smooth < threshold  # Tissue is typically darker
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title(f'Otsu Threshold ({threshold:.1f})')
    axes[0, 2].axis('off')
    
    # 4. Remove small objects
    binary_cleaned = morphology.remove_small_objects(binary, min_size=1000)
    axes[1, 0].imshow(binary_cleaned, cmap='gray')
    axes[1, 0].set_title('Remove Small Objects')
    axes[1, 0].axis('off')
    
    # 5. Fill holes
    binary_filled = morphology.binary_fill_holes(binary_cleaned)
    axes[1, 1].imshow(binary_filled, cmap='gray')
    axes[1, 1].set_title('Fill Holes')
    axes[1, 1].axis('off')
    
    # 6. Find connected components and analyze
    labeled = measure.label(binary_filled)
    regions = measure.regionprops(labeled, intensity_image=gray)
    
    # Filter regions by current parameters
    filtered_regions = []
    for i, region in enumerate(regions):
        # Filter by area
        if not (config_params['core_min_area'] <= region.area <= config_params['core_max_area']):
            continue
        
        # Filter by circularity
        if region.perimeter > 0:
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
            if circularity < config_params['core_circularity_threshold']:
                continue
        else:
            continue
        
        # Filter by diameter
        equiv_diameter = np.sqrt(4 * region.area / np.pi)
        if equiv_diameter < config_params['min_core_diameter']:
            continue
            
        filtered_regions.append(region)
    
    # Show final result with detected cores
    axes[1, 2].imshow(gray, cmap='gray')
    for region in filtered_regions:
        # Draw bounding box
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, 
                           fill=False, edgecolor='red', linewidth=2)
        axes[1, 2].add_patch(rect)
        
        # Draw centroid
        axes[1, 2].plot(region.centroid[1], region.centroid[0], 'ro', markersize=4)
    
    axes[1, 2].set_title(f'Detected Cores: {len(filtered_regions)}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n--- Core Detection Statistics ---")
    print(f"Total regions found: {len(regions)}")
    print(f"Regions after area filter ({config_params['core_min_area']}-{config_params['core_max_area']}): {len([r for r in regions if config_params['core_min_area'] <= r.area <= config_params['core_max_area']])}")
    print(f"Regions after circularity filter (>{config_params['core_circularity_threshold']}): {len([r for r in regions if r.perimeter > 0 and 4 * np.pi * r.area / (r.perimeter ** 2) >= config_params['core_circularity_threshold']])}")
    print(f"Regions after diameter filter (>{config_params['min_core_diameter']}): {len([r for r in regions if np.sqrt(4 * r.area / np.pi) >= config_params['min_core_diameter']])}")
    print(f"Final detected cores: {len(filtered_regions)}")
    
    if len(regions) > 0:
        areas = [r.area for r in regions]
        circularities = [4 * np.pi * r.area / (r.perimeter ** 2) if r.perimeter > 0 else 0 for r in regions]
        diameters = [np.sqrt(4 * r.area / np.pi) for r in regions]
        
        print(f"\n--- Region Statistics ---")
        print(f"Area range: {min(areas)} - {max(areas)}")
        print(f"Circularity range: {min(circularities):.3f} - {max(circularities):.3f}")
        print(f"Diameter range: {min(diameters):.1f} - {max(diameters):.1f}")
        
        # Show histogram of areas
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(areas, bins=50)
        plt.axvline(config_params['core_min_area'], color='red', linestyle='--', label=f'Min ({config_params["core_min_area"]})')
        plt.axvline(config_params['core_max_area'], color='red', linestyle='--', label=f'Max ({config_params["core_max_area"]})')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Count')
        plt.title('Region Areas')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.hist(circularities, bins=50)
        plt.axvline(config_params['core_circularity_threshold'], color='red', linestyle='--', label=f'Threshold ({config_params["core_circularity_threshold"]})')
        plt.xlabel('Circularity')
        plt.ylabel('Count')
        plt.title('Region Circularity')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.hist(diameters, bins=50)
        plt.axvline(config_params['min_core_diameter'], color='red', linestyle='--', label=f'Min ({config_params["min_core_diameter"]})')
        plt.xlabel('Diameter (pixels)')
        plt.ylabel('Count')
        plt.title('Region Diameters')
        plt.legend()
        
        plt.tight_layout()
    
    return fig, len(filtered_regions)

def suggest_parameters(image, image_name):
    """Suggest parameters based on image analysis."""
    print(f"\n=== Parameter Suggestions for {image_name} ===")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 3:  # RGB
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[0] <= 50:  # (C, H, W)
            gray = image[0, :, :]
        else:  # (H, W, C)
            gray = image[:, :, 0]
    else:
        gray = image
    
    # Analyze image characteristics
    print(f"Image size: {gray.shape}")
    print(f"Image range: {gray.min()} - {gray.max()}")
    print(f"Image mean: {gray.mean():.2f}")
    print(f"Image std: {gray.std():.2f}")
    
    # Apply basic thresholding to see what we get
    gray_smooth = filters.gaussian(gray, sigma=2.0)
    threshold = filters.threshold_otsu(gray_smooth)
    binary = gray_smooth < threshold
    binary_cleaned = morphology.remove_small_objects(binary, min_size=1000)
    binary_filled = morphology.binary_fill_holes(binary_cleaned)
    
    labeled = measure.label(binary_filled)
    regions = measure.regionprops(labeled, intensity_image=gray)
    
    if len(regions) > 0:
        areas = [r.area for r in regions]
        circularities = [4 * np.pi * r.area / (r.perimeter ** 2) if r.perimeter > 0 else 0 for r in regions]
        diameters = [np.sqrt(4 * r.area / np.pi) for r in regions]
        
        print(f"\nFound {len(regions)} potential regions")
        print(f"Area range: {min(areas)} - {max(areas)}")
        print(f"Circularity range: {min(circularities):.3f} - {max(circularities):.3f}")
        print(f"Diameter range: {min(diameters):.1f} - {max(diameters):.1f}")
        
        # Suggest parameters
        suggested_min_area = max(1000, int(np.percentile(areas, 10)))
        suggested_max_area = int(np.percentile(areas, 90))
        suggested_min_circularity = max(0.1, np.percentile(circularities, 25))
        suggested_min_diameter = max(50, int(np.percentile(diameters, 10)))
        
        print(f"\n--- Suggested Parameters ---")
        print(f"core_min_area: {suggested_min_area}")
        print(f"core_max_area: {suggested_max_area}")
        print(f"core_circularity_threshold: {suggested_min_circularity:.3f}")
        print(f"min_core_diameter: {suggested_min_diameter}")
        
        return {
            'core_min_area': suggested_min_area,
            'core_max_area': suggested_max_area,
            'core_circularity_threshold': suggested_min_circularity,
            'min_core_diameter': suggested_min_diameter,
            'gaussian_sigma': 2.0
        }
    else:
        print("No regions found with basic thresholding!")
        return None

def main():
    parser = argparse.ArgumentParser(description='Diagnose core detection parameters')
    parser.add_argument('--he_image', required=True, help='Path to H&E image')
    parser.add_argument('--orion_image', required=True, help='Path to Orion image')
    parser.add_argument('--output_dir', default='./core_detection_diagnostics', help='Output directory for diagnostics')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    print("Loading images...")
    he_image = imread(args.he_image)
    orion_image = imread(args.orion_image)
    
    # Current parameters (from the pipeline)
    current_params = {
        'core_min_area': 50000,
        'core_max_area': 500000,
        'core_circularity_threshold': 0.4,
        'min_core_diameter': 200,
        'gaussian_sigma': 2.0
    }
    
    # Analyze and suggest parameters
    print("\n" + "="*60)
    he_suggestions = suggest_parameters(he_image, "H&E")
    print("\n" + "="*60)
    orion_suggestions = suggest_parameters(orion_image, "Orion")
    
    # Visualize with current parameters
    print("\n" + "="*60)
    print("Visualizing with CURRENT parameters...")
    
    fig1, he_count = visualize_core_detection_process(he_image, "H&E", current_params)
    fig1.savefig(output_dir / 'he_current_params.png', dpi=150, bbox_inches='tight')
    
    fig2, orion_count = visualize_core_detection_process(orion_image, "Orion", current_params)
    fig2.savefig(output_dir / 'orion_current_params.png', dpi=150, bbox_inches='tight')
    
    # Visualize with suggested parameters if available
    if he_suggestions and orion_suggestions:
        print("\n" + "="*60)
        print("Visualizing with SUGGESTED parameters...")
        
        # Use Orion suggestions for both (since that's where the problem is)
        suggested_params = orion_suggestions
        
        fig3, he_count_suggested = visualize_core_detection_process(he_image, "H&E", suggested_params)
        fig3.savefig(output_dir / 'he_suggested_params.png', dpi=150, bbox_inches='tight')
        
        fig4, orion_count_suggested = visualize_core_detection_process(orion_image, "Orion", suggested_params)
        fig4.savefig(output_dir / 'orion_suggested_params.png', dpi=150, bbox_inches='tight')
        
        print(f"\n=== Results Summary ===")
        print(f"H&E cores (current): {he_count}")
        print(f"Orion cores (current): {orion_count}")
        print(f"H&E cores (suggested): {he_count_suggested}")
        print(f"Orion cores (suggested): {orion_count_suggested}")
        
        print(f"\n=== Suggested Parameters ===")
        for key, value in suggested_params.items():
            print(f"{key}: {value}")
    
    plt.show()

if __name__ == "__main__":
    main() 