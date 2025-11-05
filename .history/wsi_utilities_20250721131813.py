"""
Utility functions for WSI Registration Pipeline

This module provides helpful utilities for working with the WSI registration pipeline,
including parameter estimation, batch processing, and result analysis.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tifffile import imread
import cv2

from registration_pipeline_wsi import WSIRegistrationConfig, WSIRegistrationPipeline


def estimate_core_parameters(image_path: str, 
                           sample_cores: int = 5,
                           visualize: bool = True) -> Dict:
    """
    Estimate core detection parameters from a sample image.
    
    Args:
        image_path: Path to a sample TMA slide
        sample_cores: Number of cores to sample for parameter estimation
        visualize: Whether to show visualization
        
    Returns:
        Dictionary with recommended parameters
    """
    print(f"Analyzing {image_path} to estimate core parameters...")
    
    # Load image (use a lower resolution for analysis)
    img = imread(image_path)
    if len(img.shape) > 2:
        if img.shape[0] < img.shape[2]:  # Multi-channel (C, H, W)
            img = img[0]  # Use first channel
        else:  # RGB (H, W, C)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Downsample if very large
    max_dim = 2048
    if max(img.shape) > max_dim:
        scale = max_dim / max(img.shape)
        new_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_shape)
        scale_factor = 1 / scale
    else:
        scale_factor = 1
    
    # Simple thresholding to find tissue
    from skimage import filters, measure, morphology
    from scipy import ndimage
    
    # Preprocessing
    img_smooth = filters.gaussian(img, sigma=2.0)
    threshold = filters.threshold_otsu(img_smooth)
    binary = img_smooth < threshold
    
    # Clean up
    binary_cleaned = morphology.remove_small_objects(binary, min_size=1000)
    binary_filled = ndimage.binary_fill_holes(binary_cleaned)
    
    # Find regions
    labeled = measure.label(binary_filled)
    regions = measure.regionprops(labeled)
    
    # Calculate stats
    areas = [r.area * (scale_factor ** 2) for r in regions]  # Scale back to original resolution
    circularities = []
    diameters = []
    
    for region in regions:
        if region.perimeter > 0:
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
            circularities.append(circularity)
            
            equiv_diameter = np.sqrt(4 * region.area / np.pi) * scale_factor
            diameters.append(equiv_diameter)
    
    if not areas:
        print("⚠️ No tissue regions found. Check image quality and try manual parameters.")
        return {
            'core_min_area': 50000,
            'core_max_area': 500000,
            'core_circularity_threshold': 0.4,
            'expected_core_diameter': 400,
            'core_padding': 50
        }
    
    # Calculate statistics
    areas = np.array(areas)
    circularities = np.array(circularities) if circularities else np.array([0.5])
    diameters = np.array(diameters) if diameters else np.array([300])
    
    # Filter reasonable core sizes (remove very small and very large regions)
    reasonable_indices = (areas > np.percentile(areas, 10)) & (areas < np.percentile(areas, 90))
    if reasonable_indices.sum() > 0:
        filtered_areas = areas[reasonable_indices]
        filtered_circularities = circularities[reasonable_indices] if len(circularities) > 0 else circularities
        filtered_diameters = diameters[reasonable_indices] if len(diameters) > 0 else diameters
    else:
        filtered_areas = areas
        filtered_circularities = circularities
        filtered_diameters = diameters
    
    # Estimate parameters
    recommended = {
        'core_min_area': max(10000, int(np.percentile(filtered_areas, 25) * 0.7)),
        'core_max_area': min(1000000, int(np.percentile(filtered_areas, 75) * 1.5)),
        'core_circularity_threshold': max(0.2, np.percentile(filtered_circularities, 10)),
        'expected_core_diameter': int(np.median(filtered_diameters)),
        'core_padding': max(25, int(np.median(filtered_diameters) * 0.15))
    }
    
    print(f"📊 Analysis Results:")
    print(f"  Found {len(regions)} tissue regions")
    print(f"  Area range: {int(areas.min()):,} - {int(areas.max()):,} pixels")
    print(f"  Median area: {int(np.median(areas)):,} pixels") 
    print(f"  Circularity range: {circularities.min():.2f} - {circularities.max():.2f}")
    print(f"  Diameter range: {int(diameters.min())} - {int(diameters.max())} pixels")
    print(f"")
    print(f"🎯 Recommended Parameters:")
    for key, value in recommended.items():
        print(f"  --{key.replace('_', '-')}: {value}")
    
    if visualize:
        _visualize_parameter_estimation(img, binary_filled, regions, recommended)
    
    return recommended


def _visualize_parameter_estimation(img: np.ndarray, binary: np.ndarray, 
                                  regions: List, params: Dict):
    """Visualize parameter estimation results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image (Downsampled)')
    axes[0].axis('off')
    
    # Binary segmentation
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title('Tissue Segmentation')
    axes[1].axis('off')
    
    # Detected regions with filtering
    axes[2].imshow(img, cmap='gray')
    
    # Show regions that would pass the filters
    for region in regions:
        area = region.area
        if region.perimeter > 0:
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
        else:
            circularity = 0
        
        # Check if region passes filters
        passes_area = params['core_min_area'] <= area <= params['core_max_area']
        passes_circularity = circularity >= params['core_circularity_threshold']
        
        if passes_area and passes_circularity:
            # Draw accepted regions in green
            minr, minc, maxr, maxc = region.bbox
            rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                               linewidth=2, edgecolor='green', facecolor='none')
            axes[2].add_patch(rect)
            axes[2].plot(region.centroid[1], region.centroid[0], 'go', markersize=8)
        else:
            # Draw rejected regions in red
            minr, minc, maxr, maxc = region.bbox
            rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                               linewidth=1, edgecolor='red', facecolor='none', alpha=0.5)
            axes[2].add_patch(rect)
    
    axes[2].set_title('Filtered Regions (Green=Accept, Red=Reject)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def batch_process_wsi_pairs(pairs_list: List[Tuple[str, str]], 
                          output_base_dir: str,
                          config_template: Optional[WSIRegistrationConfig] = None) -> List[Dict]:
    """
    Process multiple WSI pairs in batch.
    
    Args:
        pairs_list: List of (he_wsi_path, orion_wsi_path) tuples
        output_base_dir: Base directory for all outputs
        config_template: Template configuration to use for all pairs
        
    Returns:
        List of results for each pair
    """
    results = []
    
    for i, (he_path, orion_path) in enumerate(pairs_list):
        pair_name = f"pair_{i+1:03d}"
        pair_output_dir = os.path.join(output_base_dir, pair_name)
        
        print(f"\n{'='*60}")
        print(f"Processing Pair {i+1}/{len(pairs_list)}: {pair_name}")
        print(f"H&E: {he_path}")
        print(f"Orion: {orion_path}")
        print(f"Output: {pair_output_dir}")
        print(f"{'='*60}")
        
        try:
            # Create config for this pair
            if config_template:
                config = WSIRegistrationConfig(
                    he_wsi_path=he_path,
                    orion_wsi_path=orion_path,
                    output_dir=pair_output_dir,
                    max_processed_image_dim_px=config_template.max_processed_image_dim_px,
                    max_non_rigid_registration_dim_px=config_template.max_non_rigid_registration_dim_px,
                    core_min_area=config_template.core_min_area,
                    core_max_area=config_template.core_max_area,
                    core_circularity_threshold=config_template.core_circularity_threshold,
                    expected_core_diameter=config_template.expected_core_diameter,
                    core_padding=config_template.core_padding,
                    compression=config_template.compression,
                    save_core_detection_plots=config_template.save_core_detection_plots,
                    save_quality_plots=config_template.save_quality_plots
                )
            else:
                config = WSIRegistrationConfig(
                    he_wsi_path=he_path,
                    orion_wsi_path=orion_path,
                    output_dir=pair_output_dir
                )
            
            # Run pipeline
            pipeline = WSIRegistrationPipeline(config)
            result = pipeline.run()
            
            # Add pair info to result
            result['pair_name'] = pair_name
            result['he_wsi_path'] = he_path
            result['orion_wsi_path'] = orion_path
            result['output_dir'] = pair_output_dir
            
            results.append(result)
            
            if result['success']:
                print(f"✅ {pair_name} completed successfully!")
                print(f"   Cores extracted: {result.get('cores_extracted', 0)}")
            else:
                print(f"❌ {pair_name} failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"💥 {pair_name} crashed: {e}")
            results.append({
                'pair_name': pair_name,
                'he_wsi_path': he_path,
                'orion_wsi_path': orion_path,
                'output_dir': pair_output_dir,
                'success': False,
                'error': f"Crashed: {e}"
            })
    
    # Save batch summary
    batch_summary_path = os.path.join(output_base_dir, "batch_summary.json")
    with open(batch_summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"Total Pairs Processed: {len(pairs_list)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_cores = sum(r.get('cores_extracted', 0) for r in successful)
        print(f"Total Cores Extracted: {total_cores}")
    
    if failed:
        print("\nFailed Pairs:")
        for r in failed:
            print(f"  {r['pair_name']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nBatch summary saved to: {batch_summary_path}")
    
    return results


def analyze_pipeline_results(output_dir: str) -> Dict:
    """
    Analyze results from a completed WSI pipeline run.
    
    Args:
        output_dir: Directory containing pipeline results
        
    Returns:
        Analysis summary dictionary
    """
    output_path = Path(output_dir)
    
    # Load pipeline summary
    summary_file = output_path / "wsi_pipeline_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Pipeline summary not found: {summary_file}")
    
    with open(summary_file) as f:
        pipeline_results = json.load(f)
    
    # Count extracted cores
    cores_dir = output_path / "extracted_cores"
    core_folders = [d for d in cores_dir.glob("core_*") if d.is_dir()] if cores_dir.exists() else []
    
    # Analyze core sizes
    core_sizes = []
    he_shapes = []
    orion_shapes = []
    
    for core_folder in core_folders:
        metadata_file = core_folder / f"{core_folder.name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            he_shape = metadata.get('he_extracted_shape', [])
            orion_shape = metadata.get('orion_extracted_shape', [])
            
            if he_shape:
                he_shapes.append(he_shape)
                core_sizes.append(he_shape[0] * he_shape[1])  # Area in pixels
            if orion_shape:
                orion_shapes.append(orion_shape)
    
    analysis = {
        'pipeline_success': pipeline_results.get('success', False),
        'registration_success': pipeline_results.get('registration_success', False),
        'cores_detected_he': pipeline_results.get('he_cores_detected', 0),
        'cores_detected_orion': pipeline_results.get('orion_cores_detected', 0),
        'cores_matched': pipeline_results.get('matched_pairs', 0),
        'cores_extracted': pipeline_results.get('cores_extracted', 0),
        'core_folders_found': len(core_folders),
        'avg_core_size_pixels': np.mean(core_sizes) if core_sizes else 0,
        'core_size_std': np.std(core_sizes) if core_sizes else 0,
        'avg_he_shape': np.mean(he_shapes, axis=0).tolist() if he_shapes else [],
        'avg_orion_shape': np.mean(orion_shapes, axis=0).tolist() if orion_shapes else [],
        'output_directory': str(output_path),
        'has_quality_plots': (output_path / "quality_plots").exists(),
        'has_registered_wsi': (output_path / "registered_wsi").exists()
    }
    
    return analysis


def create_batch_config(base_config: Dict, 
                      pairs_list: List[Tuple[str, str]],
                      output_file: str):
    """
    Create a batch configuration file for processing multiple WSI pairs.
    
    Args:
        base_config: Base configuration dictionary
        pairs_list: List of (he_wsi_path, orion_wsi_path) tuples  
        output_file: Path to save batch configuration
    """
    batch_config = {
        'base_config': base_config,
        'pairs': [
            {
                'he_wsi_path': he_path,
                'orion_wsi_path': orion_path,
                'pair_id': f"pair_{i+1:03d}"
            }
            for i, (he_path, orion_path) in enumerate(pairs_list)
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(batch_config, f, indent=2)
    
    print(f"Batch configuration saved to: {output_file}")
    print(f"Configuration includes {len(pairs_list)} WSI pairs")


# Example usage functions
def example_single_wsi():
    """Example of processing a single WSI pair."""
    config = WSIRegistrationConfig(
        he_wsi_path="/path/to/he_tma.tif",
        orion_wsi_path="/path/to/orion_tma.tif",
        output_dir="./single_wsi_output",
        core_min_area=75000,
        core_max_area=400000,
        expected_core_diameter=350
    )
    
    pipeline = WSIRegistrationPipeline(config)
    results = pipeline.run()
    
    if results['success']:
        analysis = analyze_pipeline_results(config.output_dir)
        print("Analysis Results:", analysis)
    
    return results


def example_batch_processing():
    """Example of batch processing multiple WSI pairs."""
    pairs = [
        ("/path/to/slide1_he.tif", "/path/to/slide1_orion.tif"),
        ("/path/to/slide2_he.tif", "/path/to/slide2_orion.tif"),
        ("/path/to/slide3_he.tif", "/path/to/slide3_orion.tif")
    ]
    
    # Create template config
    template_config = WSIRegistrationConfig(
        he_wsi_path="",  # Will be overridden
        orion_wsi_path="",  # Will be overridden
        output_dir="",  # Will be overridden
        core_min_area=60000,
        core_max_area=450000,
        expected_core_diameter=400
    )
    
    results = batch_process_wsi_pairs(
        pairs_list=pairs,
        output_base_dir="./batch_wsi_output",
        config_template=template_config
    )
    
    return results


if __name__ == "__main__":
    # Example: Estimate parameters from a sample image
    # recommended_params = estimate_core_parameters("/path/to/sample_tma.tif")
    print("WSI Utilities loaded. Use the functions above for parameter estimation and batch processing.") 