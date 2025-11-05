#!/usr/bin/env python3
"""
Quick parameter tuning for core detection on fluorescence data.
Tests different parameter combinations to find what works for Orion data.
"""

import numpy as np
import cv2
from skimage import filters, morphology, measure
import matplotlib.pyplot as plt
from tifffile import imread
import pathlib

def test_core_detection(image, params, image_name):
    """Test core detection with given parameters."""
    
    # Convert to grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 3:  # RGB
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[0] <= 50:  # (C, H, W)
            gray = image[0, :, :]  # DAPI channel
        else:  # (H, W, C)
            gray = image[:, :, 0]
    else:
        gray = image
    
    # Core detection pipeline
    gray_smooth = filters.gaussian(gray, sigma=params['gaussian_sigma'])
    
    # Try different thresholding methods
    if params.get('use_adaptive', False):
        # Adaptive thresholding for fluorescence
        threshold = cv2.adaptiveThreshold(
            (gray_smooth * 255).astype(np.uint8), 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        binary = threshold > 0
    else:
        # Otsu thresholding
        threshold = filters.threshold_otsu(gray_smooth)
        binary = gray_smooth < threshold
    
    # Clean up
    binary_cleaned = morphology.remove_small_objects(binary, min_size=1000)
    binary_filled = morphology.binary_fill_holes(binary_cleaned)
    
    # Find regions
    labeled = measure.label(binary_filled)
    regions = measure.regionprops(labeled, intensity_image=gray)
    
    # Filter by parameters
    filtered_regions = []
    for region in regions:
        # Area filter
        if not (params['core_min_area'] <= region.area <= params['core_max_area']):
            continue
        
        # Circularity filter
        if region.perimeter > 0:
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
            if circularity < params['core_circularity_threshold']:
                continue
        else:
            continue
        
        # Diameter filter
        equiv_diameter = np.sqrt(4 * region.area / np.pi)
        if equiv_diameter < params['min_core_diameter']:
            continue
            
        filtered_regions.append(region)
    
    return len(filtered_regions), len(regions), filtered_regions

def main():
    # Load the preprocessed images
    he_path = "/tmp/wsi_preprocess_xe4ggh3p/he_processed_rgb.tif"
    orion_path = "/tmp/wsi_preprocess_xe4ggh3p/orion_processed_multichannel.tif"
    
    if not pathlib.Path(he_path).exists() or not pathlib.Path(orion_path).exists():
        print("❌ Preprocessed files not found!")
        print("Please run the WSI workflow first to generate the preprocessed files.")
        return
    
    print("Loading images...")
    he_image = imread(he_path)
    orion_image = imread(orion_path)
    
    print(f"H&E shape: {he_image.shape}")
    print(f"Orion shape: {orion_image.shape}")
    
    # Test different parameter combinations
    parameter_sets = [
        # Original parameters (too strict for fluorescence)
        {
            'name': 'Original (Current)',
            'core_min_area': 50000,
            'core_max_area': 500000,
            'core_circularity_threshold': 0.4,
            'min_core_diameter': 200,
            'gaussian_sigma': 2.0,
            'use_adaptive': False
        },
        # More permissive for fluorescence
        {
            'name': 'Fluorescence-Friendly 1',
            'core_min_area': 10000,
            'core_max_area': 200000,
            'core_circularity_threshold': 0.2,
            'min_core_diameter': 100,
            'gaussian_sigma': 1.5,
            'use_adaptive': False
        },
        # Even more permissive
        {
            'name': 'Fluorescence-Friendly 2',
            'core_min_area': 5000,
            'core_max_area': 100000,
            'core_circularity_threshold': 0.15,
            'min_core_diameter': 80,
            'gaussian_sigma': 1.0,
            'use_adaptive': False
        },
        # With adaptive thresholding
        {
            'name': 'Adaptive Threshold',
            'core_min_area': 10000,
            'core_max_area': 200000,
            'core_circularity_threshold': 0.2,
            'min_core_diameter': 100,
            'gaussian_sigma': 1.5,
            'use_adaptive': True
        },
        # Very permissive for testing
        {
            'name': 'Very Permissive',
            'core_min_area': 2000,
            'core_max_area': 50000,
            'core_circularity_threshold': 0.1,
            'min_core_diameter': 50,
            'gaussian_sigma': 0.5,
            'use_adaptive': False
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING DIFFERENT PARAMETER COMBINATIONS")
    print("="*80)
    
    results = []
    
    for params in parameter_sets:
        print(f"\n--- Testing: {params['name']} ---")
        
        # Test H&E
        he_cores, he_total, he_regions = test_core_detection(he_image, params, "H&E")
        
        # Test Orion
        orion_cores, orion_total, orion_regions = test_core_detection(orion_image, params, "Orion")
        
        results.append({
            'params': params,
            'he_cores': he_cores,
            'he_total': he_total,
            'orion_cores': orion_cores,
            'orion_total': orion_total
        })
        
        print(f"H&E: {he_cores} cores (from {he_total} regions)")
        print(f"Orion: {orion_cores} cores (from {orion_total} regions)")
        
        # Check if this looks promising
        if orion_cores > 0 and he_cores > 0:
            print(f"✅ PROMISING: Both images have cores!")
            if abs(he_cores - orion_cores) <= 20:  # Within 20 cores
                print(f"🎯 EXCELLENT: Core counts are similar!")
    
    # Find best parameters
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find parameters that work for both
    working_params = [r for r in results if r['orion_cores'] > 0 and r['he_cores'] > 0]
    
    if working_params:
        # Sort by how close the core counts are
        working_params.sort(key=lambda x: abs(x['he_cores'] - x['orion_cores']))
        best = working_params[0]
        
        print(f"🎯 BEST PARAMETERS: {best['params']['name']}")
        print(f"H&E cores: {best['he_cores']}")
        print(f"Orion cores: {best['orion_cores']}")
        print(f"Difference: {abs(best['he_cores'] - best['orion_cores'])}")
        
        print(f"\n📋 RECOMMENDED PARAMETERS:")
        for key, value in best['params'].items():
            if key != 'name':
                print(f"  {key}: {value}")
        
        # Create a config snippet
        print(f"\n🔧 CONFIG SNIPPET FOR PIPELINE:")
        print(f"config = WSIRegistrationConfig(")
        print(f"    he_wsi_path='your_he_path',")
        print(f"    orion_wsi_path='your_orion_path',")
        print(f"    output_dir='your_output_dir',")
        print(f"    core_min_area={best['params']['core_min_area']},")
        print(f"    core_max_area={best['params']['core_max_area']},")
        print(f"    core_circularity_threshold={best['params']['core_circularity_threshold']},")
        print(f"    min_core_diameter={best['params']['min_core_diameter']},")
        print(f"    gaussian_sigma={best['params']['gaussian_sigma']}")
        print(f")")
        
    else:
        print("❌ No parameter combination worked for both images!")
        print("This suggests the fluorescence data may need different preprocessing.")
        
        # Show what we found
        print(f"\n📊 SUMMARY:")
        for r in results:
            print(f"{r['params']['name']}: H&E={r['he_cores']}, Orion={r['orion_cores']}")

if __name__ == "__main__":
    main() 