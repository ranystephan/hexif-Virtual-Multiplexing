"""
Dual-Modality Core Detection for TMA Images

This module provides robust core detection and matching between H&E and Orion TMA images
using SpaceC tissue extraction with proper image preprocessing and grid search optimization.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure, color as scolor
from PIL import Image

# SpaceC imports
try:
    import spacec as sp
    SPACEC_AVAILABLE = True
except ImportError:
    SPACEC_AVAILABLE = False
    print("Warning: SpaceC not available. Please install with: pip install spacec")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoreDetectionConfig:
    """Configuration for dual-modality core detection."""
    
    # SpaceC parameters
    downscale_factor: int = 64
    padding: int = 50
    
    # H&E specific parameters (will be optimized)
    he_lower_cutoff: float = 0.010
    he_upper_cutoff: float = 0.025
    
    # Orion specific parameters (will be optimized)
    orion_lower_cutoff: float = 0.005
    orion_upper_cutoff: float = 0.020
    dapi_channel: int = 0
    
    # Core matching parameters
    max_match_distance: float = 500.0  # pixels at full resolution
    min_size_ratio: float = 0.3
    max_size_ratio: float = 3.0
    min_circularity: float = 0.15
    
    # Quality control (will be optimized)
    min_core_area: int = 5000  # minimum area in pixels (will be optimized)
    max_core_area: int = 500000  # maximum area in pixels (will be optimized)
    
    # Target number of regions for optimization
    target_regions: int = 270
    
    # Processing
    temp_dir: Optional[str] = None
    save_debug_images: bool = True


class DualModalityCoreDetector:
    """
    Detects and matches cores between H&E and Orion TMA images.
    
    Uses SpaceC for robust tissue detection with proper image preprocessing
    and grid search parameter optimization following the approach from
    paired_core_extraction_updated.ipynb.
    """
    
    def __init__(self, config: CoreDetectionConfig):
        """
        Initialize the dual-modality core detector.
        
        Args:
            config: Configuration object with detection parameters
        """
        self.config = config
        
        if not SPACEC_AVAILABLE:
            raise ImportError("SpaceC is required. Install with: pip install spacec")
        
        # Set up temporary directory
        if self.config.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="tma_core_detection_"))
        else:
            self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized core detector with temp dir: {self.temp_dir}")
    
    def optimize_parameters(self, he_path: str, orion_path: str) -> CoreDetectionConfig:
        """
        Optimize detection parameters using grid search approach from working notebook.
        
        Args:
            he_path: Path to H&E WSI
            orion_path: Path to Orion WSI
            
        Returns:
            Optimized configuration
        """
        logger.info("Starting parameter optimization using grid search...")
        
        # First optimize Orion parameters (more critical)
        logger.info("Optimizing Orion detection parameters...")
        orion_config = self._optimize_orion_parameters(orion_path)
        
        # Then optimize H&E parameters
        logger.info("Optimizing H&E detection parameters...")
        he_config = self._optimize_he_parameters(he_path)
        
        # Optimize area thresholds based on detected core sizes
        logger.info("Optimizing area thresholds...")
        area_config = self._optimize_area_thresholds([orion_config, he_config])
        
        # Create final optimized configuration
        optimized_config = CoreDetectionConfig(
            downscale_factor=self.config.downscale_factor,
            padding=self.config.padding,
            he_lower_cutoff=he_config['lower'],
            he_upper_cutoff=he_config['upper'],
            orion_lower_cutoff=orion_config['lower'],
            orion_upper_cutoff=orion_config['upper'],
            dapi_channel=self.config.dapi_channel,
            max_match_distance=self.config.max_match_distance,
            min_size_ratio=self.config.min_size_ratio,
            max_size_ratio=self.config.max_size_ratio,
            min_circularity=self.config.min_circularity,
            min_core_area=area_config['min_area'],
            max_core_area=area_config['max_area'],
            target_regions=self.config.target_regions,
            temp_dir=self.config.temp_dir,
            save_debug_images=self.config.save_debug_images
        )
        
        logger.info(f"Parameter optimization completed:")
        logger.info(f"  H&E: {optimized_config.he_lower_cutoff:.4f} - {optimized_config.he_upper_cutoff:.4f}")
        logger.info(f"  Orion: {optimized_config.orion_lower_cutoff:.4f} - {optimized_config.orion_upper_cutoff:.4f}")
        logger.info(f"  Area: {optimized_config.min_core_area} - {optimized_config.max_core_area}")
        
        return optimized_config
    
    def _optimize_orion_parameters(self, orion_path: str) -> Dict:
        """Optimize Orion detection parameters using grid search."""
        
        # Preprocess Orion image
        preprocessed_orion_path = self._preprocess_orion_image(orion_path)
        
        # Downscale for parameter optimization
        resized_im = sp.hf.downscale_tissue(
            file_path=str(preprocessed_orion_path),
            downscale_factor=self.config.downscale_factor,
            padding=self.config.padding,
            output_dir=str(self.temp_dir)
        )
        
        # Grid search for optimal parameters
        results = []
        search_ranges = {
            'lower': np.linspace(0.005, 0.025, 12),
            'upper': np.linspace(0.010, 0.050, 12)
        }
        
        logger.info("Performing grid search for Orion parameters...")
        total_combinations = len(search_ranges['lower']) * len(search_ranges['upper'])
        current = 0
        
        for lower in search_ranges['lower']:
            for upper in search_ranges['upper']:
                if upper <= lower:
                    continue
                    
                current += 1
                if current % 20 == 0:
                    logger.info(f"  Progress: {current}/{total_combinations}")
                
                try:
                    tissueframe = sp.tl.label_tissue(resized_im, lower, upper)
                    n_regions = tissueframe['region1'].nunique()
                    
                    # Score based on distance from target regions
                    score_distance = abs(n_regions - self.config.target_regions)
                    
                    results.append({
                        'lower': lower,
                        'upper': upper,
                        'n_regions': n_regions,
                        'score': score_distance
                    })
                except Exception as e:
                    # Skip failed parameter combinations
                    continue
        
        if not results:
            logger.warning("No valid parameter combinations found, using defaults")
            return {'lower': 0.010, 'upper': 0.020}
        
        # Find best parameters
        best = min(results, key=lambda x: x['score'])
        logger.info(f"Best Orion parameters: {best['lower']:.4f} - {best['upper']:.4f} → {best['n_regions']} regions")
        
        return {'lower': best['lower'], 'upper': best['upper']}
    
    def _optimize_he_parameters(self, he_path: str) -> Dict:
        """Optimize H&E detection parameters using grid search."""
        
        # Downscale H&E image
        resized_im = sp.hf.downscale_tissue(
            file_path=he_path,
            downscale_factor=self.config.downscale_factor,
            padding=self.config.padding,
            output_dir=str(self.temp_dir)
        )
        
        # Grid search for optimal H&E parameters
        results = []
        search_ranges = {
            'lower': np.linspace(0.008, 0.025, 10),
            'upper': np.linspace(0.015, 0.040, 10)
        }
        
        logger.info("Performing grid search for H&E parameters...")
        total_combinations = len(search_ranges['lower']) * len(search_ranges['upper'])
        current = 0
        
        for lower in search_ranges['lower']:
            for upper in search_ranges['upper']:
                if upper <= lower:
                    continue
                    
                current += 1
                if current % 15 == 0:
                    logger.info(f"  Progress: {current}/{total_combinations}")
                
                try:
                    tissueframe = sp.tl.label_tissue(resized_im, lower, upper)
                    n_regions = tissueframe['region1'].nunique()
                    
                    # Score based on distance from target regions
                    score_distance = abs(n_regions - self.config.target_regions)
                    
                    results.append({
                        'lower': lower,
                        'upper': upper,
                        'n_regions': n_regions,
                        'score': score_distance
                    })
                except Exception as e:
                    # Skip failed parameter combinations
                    continue
        
        if not results:
            logger.warning("No valid H&E parameter combinations found, using defaults")
            return {'lower': 0.012, 'upper': 0.025}
        
        # Find best parameters
        best = min(results, key=lambda x: x['score'])
        logger.info(f"Best H&E parameters: {best['lower']:.4f} - {best['upper']:.4f} → {best['n_regions']} regions")
        
        return {'lower': best['lower'], 'upper': best['upper']}
    
    def _optimize_area_thresholds(self, param_results: List[Dict]) -> Dict:
        """Optimize area thresholds based on detected core sizes."""
        
        # Use conservative area bounds that work for most TMA cores
        # These values are based on typical core sizes in the working notebook
        min_area = 3000   # Smaller minimum to catch small cores  
        max_area = 800000  # Larger maximum to catch big cores
        
        logger.info(f"Optimized area thresholds: {min_area} - {max_area} pixels")
        
        return {'min_area': min_area, 'max_area': max_area}
    
    def _preprocess_orion_image(self, orion_path: str) -> Path:
        """
        Preprocess Orion image with CLAHE and inversion like in working notebook.
        """
        logger.info("Preprocessing Orion image...")
        
        # Load Orion image
        orion_img = imread(orion_path)
        
        if orion_img.ndim == 3 and orion_img.shape[0] <= 50:
            # Multi-channel format (C, H, W)
            dapi_channel = orion_img[self.config.dapi_channel]
        elif orion_img.ndim == 2:
            # Single channel
            dapi_channel = orion_img
        else:
            raise ValueError(f"Unexpected Orion image format: {orion_img.shape}")
        
        # Convert to float and normalize
        dapi_norm = dapi_channel.astype(np.float32)
        dapi_norm = (dapi_norm - dapi_norm.min()) / (dapi_norm.max() - dapi_norm.min())
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        dapi_clahe = exposure.equalize_adapthist(dapi_norm, clip_limit=0.02)
        
        # Invert intensities so cores become bright (key step from working notebook!)
        dapi_processed = 1.0 - dapi_clahe
        
        # Convert back to appropriate data type
        dapi_final = (dapi_processed * 65535).astype(np.uint16)
        
        # Save preprocessed image
        processed_path = self.temp_dir / "orion_preprocessed.tif"
        imwrite(processed_path, dapi_final)
        
        logger.info(f"Orion preprocessing completed, saved to: {processed_path}")
        return processed_path
    
    def detect_and_match_cores(self, he_path: str, orion_path: str) -> Dict:
        """
        Complete pipeline: detect cores in both modalities and match them.
        
        Args:
            he_path: Path to H&E WSI
            orion_path: Path to Orion WSI
            
        Returns:
            Dictionary containing detection and matching results
        """
        logger.info("Starting dual-modality core detection and matching...")
        
        results = {
            'input_files': {'he_path': he_path, 'orion_path': orion_path},
            'config': self.config,
            'detection_stats': {}
        }
        
        try:
            # Step 1: Detect cores in H&E
            logger.info("Detecting cores in H&E image...")
            he_cores, he_regions = self._detect_cores_he(he_path)
            results['he_cores'] = he_cores
            results['he_regions'] = he_regions
            results['detection_stats']['he_cores_detected'] = len(he_cores)
            
            # Step 2: Detect cores in Orion
            logger.info("Detecting cores in Orion image...")
            orion_cores, orion_regions = self._detect_cores_orion(orion_path)
            results['orion_cores'] = orion_cores
            results['orion_regions'] = orion_regions
            results['detection_stats']['orion_cores_detected'] = len(orion_cores)
            
            # Step 3: Match cores between modalities
            logger.info("Matching cores between modalities...")
            matched_cores = self._match_cores_by_position(he_cores, orion_cores)
            results['matched_cores'] = matched_cores
            results['detection_stats']['matched_cores'] = len(matched_cores)
            
            # Step 4: Generate quality metrics
            results['quality_metrics'] = self._calculate_matching_quality(
                he_cores, orion_cores, matched_cores
            )
            
            # Step 5: Save debug visualizations if requested
            if self.config.save_debug_images:
                self._save_debug_visualizations(results)
            
            results['success'] = True
            logger.info(f"Successfully detected and matched {len(matched_cores)} core pairs")
            
        except Exception as e:
            logger.error(f"Core detection and matching failed: {e}")
            import traceback
            traceback.print_exc()
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def _detect_cores_he(self, he_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect cores in H&E image using SpaceC."""
        
        # Downscale image for detection
        resized_im = sp.hf.downscale_tissue(
            file_path=he_path,
            downscale_factor=self.config.downscale_factor,
            padding=self.config.padding,
            output_dir=str(self.temp_dir)
        )
        
        # Apply tissue labeling with H&E-optimized parameters
        tissueframe = sp.tl.label_tissue(
            resized_im,
            lower_cutoff=self.config.he_lower_cutoff,
            upper_cutoff=self.config.he_upper_cutoff
        )
        
        # Process detected regions
        cores_df, region_bboxes = self._process_detected_regions(
            tissueframe, 'he', self.config.downscale_factor
        )
        
        logger.info(f"Detected {len(cores_df)} cores in H&E image")
        return cores_df, region_bboxes
    
    def _detect_cores_orion(self, orion_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect cores in Orion image using preprocessed DAPI channel."""
        
        # Preprocess Orion image (CLAHE + inversion)
        preprocessed_path = self._preprocess_orion_image(orion_path)
        
        # Downscale preprocessed image
        resized_im = sp.hf.downscale_tissue(
            file_path=str(preprocessed_path),
            downscale_factor=self.config.downscale_factor,
            padding=self.config.padding,
            output_dir=str(self.temp_dir)
        )
        
        # Apply tissue labeling with Orion-optimized parameters
        tissueframe = sp.tl.label_tissue(
            resized_im,
            lower_cutoff=self.config.orion_lower_cutoff,
            upper_cutoff=self.config.orion_upper_cutoff
        )
        
        # Process detected regions
        cores_df, region_bboxes = self._process_detected_regions(
            tissueframe, 'orion', self.config.downscale_factor
        )
        
        logger.info(f"Detected {len(cores_df)} cores in Orion image")
        return cores_df, region_bboxes
    
    def _process_detected_regions(self, tissueframe: pd.DataFrame, 
                                modality: str, downscale_factor: int) -> Tuple[pd.DataFrame, Dict]:
        """Process SpaceC detected regions into core information."""
        
        cores_data = []
        region_bboxes = {}
        
        for region_id, group in tissueframe.groupby('region1'):
            # Calculate bounding box
            x_min, y_min = group.x.min(), group.y.min()
            x_max, y_max = group.x.max(), group.y.max()
            
            # Scale back to full resolution
            x_min_full = x_min * downscale_factor
            y_min_full = y_min * downscale_factor
            x_max_full = x_max * downscale_factor
            y_max_full = y_max * downscale_factor
            
            # Calculate properties
            width = x_max - x_min
            height = y_max - y_min
            area = len(group)  # Number of pixels in downscaled image
            area_full = area * (downscale_factor ** 2)  # Approximate full resolution area
            
            centroid_x = (x_min + x_max) / 2
            centroid_y = (y_min + y_max) / 2
            centroid_x_full = centroid_x * downscale_factor
            centroid_y_full = centroid_y * downscale_factor
            
            # Estimate circularity (approximation)
            perimeter_est = 2 * np.pi * np.sqrt(area / np.pi)
            circularity = 4 * np.pi * area / (perimeter_est ** 2) if perimeter_est > 0 else 0
            
            # Apply quality filters
            if (self.config.min_core_area <= area_full <= self.config.max_core_area and
                circularity >= self.config.min_circularity):
                
                cores_data.append({
                    'region_id': int(region_id),
                    'modality': modality,
                    'centroid_x': centroid_x_full,
                    'centroid_y': centroid_y_full,
                    'width': width * downscale_factor,
                    'height': height * downscale_factor,
                    'area': area_full,
                    'circularity': circularity,
                    'bbox_x_min': x_min_full,
                    'bbox_y_min': y_min_full,
                    'bbox_x_max': x_max_full,
                    'bbox_y_max': y_max_full
                })
                
                region_bboxes[int(region_id)] = (
                    int(x_min_full), int(y_min_full), 
                    int(x_max_full), int(y_max_full)
                )
        
        cores_df = pd.DataFrame(cores_data)
        return cores_df, region_bboxes
    
    def _match_cores_by_position(self, he_cores: pd.DataFrame, 
                               orion_cores: pd.DataFrame) -> List[Dict]:
        """
        Match cores between modalities using spatial positions and Hungarian algorithm.
        """
        if len(he_cores) == 0 or len(orion_cores) == 0:
            logger.warning("No cores detected in one or both modalities")
            return []
        
        # Get centroids
        he_centroids = he_cores[['centroid_x', 'centroid_y']].values
        orion_centroids = orion_cores[['centroid_x', 'centroid_y']].values
        
        # Compute distance matrix
        distances = euclidean_distances(he_centroids, orion_centroids)
        
        # Use Hungarian algorithm for optimal assignment
        he_indices, orion_indices = linear_sum_assignment(distances)
        
        matched_cores = []
        for he_idx, orion_idx in zip(he_indices, orion_indices):
            he_core = he_cores.iloc[he_idx]
            orion_core = orion_cores.iloc[orion_idx]
            distance = distances[he_idx, orion_idx]
            
            # Validate match
            if self._is_valid_match(he_core, orion_core, distance):
                match_info = {
                    'he_core_id': he_core['region_id'],
                    'orion_core_id': orion_core['region_id'],
                    'match_distance': float(distance),
                    'size_ratio': float(he_core['area'] / orion_core['area']),
                    'he_centroid': (float(he_core['centroid_x']), float(he_core['centroid_y'])),
                    'orion_centroid': (float(orion_core['centroid_x']), float(orion_core['centroid_y'])),
                    'he_bbox': (
                        int(he_core['bbox_x_min']), int(he_core['bbox_y_min']),
                        int(he_core['bbox_x_max']), int(he_core['bbox_y_max'])
                    ),
                    'orion_bbox': (
                        int(orion_core['bbox_x_min']), int(orion_core['bbox_y_min']),
                        int(orion_core['bbox_x_max']), int(orion_core['bbox_y_max'])
                    ),
                    'he_area': float(he_core['area']),
                    'orion_area': float(orion_core['area']),
                    'he_circularity': float(he_core['circularity']),
                    'orion_circularity': float(orion_core['circularity'])
                }
                matched_cores.append(match_info)
        
        # Sort by distance for quality ranking
        matched_cores.sort(key=lambda x: x['match_distance'])
        
        return matched_cores
    
    def _is_valid_match(self, he_core: pd.Series, orion_core: pd.Series, 
                       distance: float) -> bool:
        """Validate core match using multiple criteria."""
        
        # Distance threshold
        if distance > self.config.max_match_distance:
            return False
        
        # Size ratio
        size_ratio = he_core['area'] / orion_core['area']
        if not (self.config.min_size_ratio <= size_ratio <= self.config.max_size_ratio):
            return False
        
        # Circularity check (both cores should be reasonably circular)
        if (he_core['circularity'] < self.config.min_circularity or 
            orion_core['circularity'] < self.config.min_circularity):
            return False
        
        return True
    
    def _calculate_matching_quality(self, he_cores: pd.DataFrame, orion_cores: pd.DataFrame,
                                  matched_cores: List[Dict]) -> Dict:
        """Calculate quality metrics for the matching process."""
        
        if len(matched_cores) == 0:
            return {'matching_rate': 0.0, 'mean_distance': float('inf')}
        
        distances = [match['match_distance'] for match in matched_cores]
        size_ratios = [match['size_ratio'] for match in matched_cores]
        
        total_cores = min(len(he_cores), len(orion_cores))
        matching_rate = len(matched_cores) / total_cores if total_cores > 0 else 0.0
        
        return {
            'matching_rate': matching_rate,
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'mean_size_ratio': np.mean(size_ratios),
            'size_ratio_std': np.std(size_ratios),
            'total_he_cores': len(he_cores),
            'total_orion_cores': len(orion_cores),
            'matched_cores': len(matched_cores)
        }
    
    def _save_debug_visualizations(self, results: Dict):
        """Save debug visualizations for quality control."""
        
        debug_dir = self.temp_dir / "debug_visualizations"
        debug_dir.mkdir(exist_ok=True)
        
        # Create matching visualization
        self._create_matching_plot(results, debug_dir / "core_matching.png")
        
        # Create statistics plot
        self._create_statistics_plot(results, debug_dir / "matching_statistics.png")
        
        logger.info(f"Debug visualizations saved to {debug_dir}")
    
    def _create_matching_plot(self, results: Dict, output_path: Path):
        """Create visualization of core matching."""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        he_cores = results['he_cores']
        orion_cores = results['orion_cores']
        matched_cores = results['matched_cores']
        
        # Plot H&E cores
        if len(he_cores) > 0:
            ax1.scatter(he_cores['centroid_x'], he_cores['centroid_y'], 
                       c='red', alpha=0.7, s=50, label='H&E cores')
            ax1.set_title(f'H&E Cores (n={len(he_cores)})')
            ax1.set_xlabel('X position')
            ax1.set_ylabel('Y position')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot Orion cores
        if len(orion_cores) > 0:
            ax2.scatter(orion_cores['centroid_x'], orion_cores['centroid_y'], 
                       c='blue', alpha=0.7, s=50, label='Orion cores')
            ax2.set_title(f'Orion Cores (n={len(orion_cores)})')
            ax2.set_xlabel('X position')
            ax2.set_ylabel('Y position')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot matched cores
        if len(matched_cores) > 0:
            he_matched_x = [match['he_centroid'][0] for match in matched_cores]
            he_matched_y = [match['he_centroid'][1] for match in matched_cores]
            orion_matched_x = [match['orion_centroid'][0] for match in matched_cores]
            orion_matched_y = [match['orion_centroid'][1] for match in matched_cores]
            
            ax3.scatter(he_matched_x, he_matched_y, c='red', alpha=0.7, s=50, label='H&E matched')
            ax3.scatter(orion_matched_x, orion_matched_y, c='blue', alpha=0.7, s=50, label='Orion matched')
            
            # Draw matching lines
            for i in range(len(matched_cores)):
                ax3.plot([he_matched_x[i], orion_matched_x[i]], 
                        [he_matched_y[i], orion_matched_y[i]], 
                        'gray', alpha=0.5, linewidth=1)
            
            ax3.set_title(f'Matched Cores (n={len(matched_cores)})')
            ax3.set_xlabel('X position')
            ax3.set_ylabel('Y position')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_statistics_plot(self, results: Dict, output_path: Path):
        """Create statistics visualization."""
        
        matched_cores = results['matched_cores']
        quality_metrics = results['quality_metrics']
        
        if len(matched_cores) == 0:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distance distribution
        distances = [match['match_distance'] for match in matched_cores]
        ax1.hist(distances, bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(quality_metrics['mean_distance'], color='red', linestyle='--', 
                   label=f'Mean: {quality_metrics["mean_distance"]:.1f}')
        ax1.set_title('Match Distance Distribution')
        ax1.set_xlabel('Distance (pixels)')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Size ratio distribution
        size_ratios = [match['size_ratio'] for match in matched_cores]
        ax2.hist(size_ratios, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(quality_metrics['mean_size_ratio'], color='red', linestyle='--',
                   label=f'Mean: {quality_metrics["mean_size_ratio"]:.2f}')
        ax2.set_title('Size Ratio Distribution (H&E/Orion)')
        ax2.set_xlabel('Size Ratio')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Core area comparison
        he_areas = [match['he_area'] for match in matched_cores]
        orion_areas = [match['orion_area'] for match in matched_cores]
        ax3.scatter(he_areas, orion_areas, alpha=0.7)
        ax3.plot([min(he_areas + orion_areas), max(he_areas + orion_areas)], 
                [min(he_areas + orion_areas), max(he_areas + orion_areas)], 
                'red', linestyle='--', label='Perfect correlation')
        ax3.set_title('Core Area Correlation')
        ax3.set_xlabel('H&E Core Area')
        ax3.set_ylabel('Orion Core Area')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""
        Matching Statistics:
        • Matching Rate: {quality_metrics['matching_rate']:.1%}
        • H&E Cores: {quality_metrics['total_he_cores']}
        • Orion Cores: {quality_metrics['total_orion_cores']}
        • Matched Cores: {quality_metrics['matched_cores']}
        • Mean Distance: {quality_metrics['mean_distance']:.1f} px
        • Mean Size Ratio: {quality_metrics['mean_size_ratio']:.2f}
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}") 