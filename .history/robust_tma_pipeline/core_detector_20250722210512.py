"""
Dual-Modality Core Detection for TMA Images

This module provides robust core detection for both H&E and Orion images using
SpaceC with optimized parameters for each modality.
"""

import spacec as sp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import logging
from tifffile import imread
import cv2
from skimage import filters, morphology, measure

logger = logging.getLogger(__name__)


class DualModalityCoreDetector:
    """Detects cores in both H&E and Orion images with modality-specific optimization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.temp_dir = Path(config.get('temp_dir', tempfile.gettempdir()))
        self.temp_dir.mkdir(exist_ok=True)
    
    def detect_cores_both_modalities(self, he_path: str, orion_path: str) -> Dict:
        """
        Detect cores in both H&E and Orion images.
        
        Args:
            he_path: Path to H&E WSI
            orion_path: Path to Orion WSI
            
        Returns:
            Dictionary with detection results for both modalities
        """
        logger.info(f"Detecting cores in H&E: {Path(he_path).name}")
        logger.info(f"Detecting cores in Orion: {Path(orion_path).name}")
        
        results = {}
        
        try:
            # Detect cores in H&E
            he_cores = self._detect_cores_he(he_path)
            results['he_cores'] = he_cores
            logger.info(f"Detected {len(he_cores)} cores in H&E")
            
            # Detect cores in Orion
            orion_cores = self._detect_cores_orion(orion_path)
            results['orion_cores'] = orion_cores
            logger.info(f"Detected {len(orion_cores)} cores in Orion")
            
            # Store paths for later use
            results['he_path'] = he_path
            results['orion_path'] = orion_path
            
        except Exception as e:
            logger.error(f"Core detection failed: {e}")
            raise
        
        return results
    
    def _detect_cores_he(self, he_path: str) -> pd.DataFrame:
        """Detect cores in H&E image using optimized SpaceC parameters."""
        
        # Downscale for efficient detection
        resized_im = sp.hf.downscale_tissue(
            file_path=he_path,
            downscale_factor=self.config['downscale_factor'],
            padding=self.config['padding'],
            output_dir=str(self.temp_dir)
        )
        
        # Use H&E-optimized parameters
        tissueframe = sp.tl.label_tissue(
            resized_im,
            lower_cutoff=self.config['he_lower_cutoff'],
            upper_cutoff=self.config['he_upper_cutoff']
        )
        
        return self._process_detected_regions(tissueframe, 'he')
    
    def _detect_cores_orion(self, orion_path: str) -> pd.DataFrame:
        """Detect cores in Orion image using DAPI channel."""
        
        # First, extract DAPI channel and prepare for SpaceC
        dapi_path = self._prepare_orion_for_spacec(orion_path)
        
        # Downscale DAPI image
        resized_im = sp.hf.downscale_tissue(
            file_path=dapi_path,
            downscale_factor=self.config['downscale_factor'],
            padding=self.config['padding'],
            output_dir=str(self.temp_dir)
        )
        
        # Use Orion-optimized parameters
        tissueframe = sp.tl.label_tissue(
            resized_im,
            lower_cutoff=self.config['orion_lower_cutoff'],
            upper_cutoff=self.config['orion_upper_cutoff']
        )
        
        return self._process_detected_regions(tissueframe, 'orion')
    
    def _prepare_orion_for_spacec(self, orion_path: str) -> str:
        """
        Extract DAPI channel from Orion image and save as TIFF for SpaceC.
        
        Args:
            orion_path: Path to multi-channel Orion image
            
        Returns:
            Path to extracted DAPI TIFF
        """
        # Load Orion image
        orion_img = imread(orion_path)
        
        # Extract DAPI channel (typically channel 0)
        dapi_channel = self.config.get('dapi_channel', 0)
        
        if orion_img.ndim == 3 and orion_img.shape[0] <= 50:
            # Multi-channel format (C, H, W)
            dapi = orion_img[dapi_channel]
        elif orion_img.ndim == 2:
            # Already single channel
            dapi = orion_img
        else:
            raise ValueError(f"Unsupported Orion image format: {orion_img.shape}")
        
        # Normalize to uint16 for SpaceC compatibility
        if dapi.dtype != np.uint16:
            dapi = cv2.normalize(dapi, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
        
        # Save as temporary TIFF
        dapi_path = self.temp_dir / f"orion_dapi_{Path(orion_path).stem}.tif"
        
        # Create a 3D array for SpaceC (expects C, H, W format)
        dapi_3d = np.expand_dims(dapi, axis=0)
        
        from tifffile import imwrite
        imwrite(dapi_path, dapi_3d)
        
        return str(dapi_path)
    
    def _process_detected_regions(self, tissueframe: pd.DataFrame, modality: str) -> pd.DataFrame:
        """
        Process SpaceC detection results and add quality metrics.
        
        Args:
            tissueframe: SpaceC output dataframe
            modality: 'he' or 'orion'
            
        Returns:
            Processed dataframe with core information
        """
        if tissueframe.empty:
            logger.warning(f"No regions detected in {modality} image")
            return pd.DataFrame()
        
        # Group by region to get core-level information
        cores_data = []
        
        for region_id, group in tissueframe.groupby('region1'):
            # Calculate bounding box
            x_min, x_max = group.x.min(), group.x.max()
            y_min, y_max = group.y.min(), group.y.max()
            
            # Calculate core properties
            area = len(group)
            centroid_x = group.x.mean()
            centroid_y = group.y.mean()
            
            # Estimate circularity (approximate)
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            circularity = 1.0 / aspect_ratio  # Simple circularity estimate
            
            core_info = {
                'region_id': f"{modality}_reg{region_id:03d}",
                'modality': modality,
                'area': area,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'bbox': (x_min, y_min, x_max, y_max),
                'width': width,
                'height': height,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio
            }
            
            cores_data.append(core_info)
        
        cores_df = pd.DataFrame(cores_data)
        
        # Filter cores by quality criteria
        if not cores_df.empty:
            cores_df = self._filter_cores_by_quality(cores_df, modality)
        
        logger.info(f"Processed {len(cores_df)} quality cores in {modality}")
        return cores_df
    
    def _filter_cores_by_quality(self, cores_df: pd.DataFrame, modality: str) -> pd.DataFrame:
        """Filter cores based on size, shape, and quality criteria."""
        
        initial_count = len(cores_df)
        
        # Get modality-specific thresholds
        if modality == 'he':
            min_area = self.config.get('he_min_core_area', 10000)
            max_area = self.config.get('he_max_core_area', 500000)
            min_circularity = self.config.get('he_min_circularity', 0.3)
        else:  # orion
            min_area = self.config.get('orion_min_core_area', 8000)
            max_area = self.config.get('orion_max_core_area', 600000)
            min_circularity = self.config.get('orion_min_circularity', 0.2)
        
        # Apply filters
        filtered_df = cores_df[
            (cores_df['area'] >= min_area) &
            (cores_df['area'] <= max_area) &
            (cores_df['circularity'] >= min_circularity) &
            (cores_df['aspect_ratio'] <= 3.0)  # Not too elongated
        ].copy()
        
        filtered_count = len(filtered_df)
        logger.info(f"Quality filtering: {initial_count} → {filtered_count} cores ({modality})")
        
        return filtered_df
    
    def optimize_detection_parameters(self, he_sample: str, orion_sample: str) -> Dict:
        """
        Automatically optimize SpaceC parameters for core detection.
        
        Args:
            he_sample: Path to sample H&E image
            orion_sample: Path to sample Orion image
            
        Returns:
            Optimized parameters dictionary
        """
        logger.info("Optimizing detection parameters...")
        
        # Parameter search ranges
        he_lower_range = np.linspace(0.10, 0.25, 8)
        he_upper_range = np.linspace(0.15, 0.30, 8)
        orion_lower_range = np.linspace(0.05, 0.20, 8)
        orion_upper_range = np.linspace(0.10, 0.25, 8)
        
        best_score = 0
        best_params = {}
        
        # Test H&E parameters
        for he_lo in he_lower_range:
            for he_hi in he_upper_range:
                if he_hi <= he_lo:
                    continue
                
                try:
                    # Test these parameters
                    test_config = self.config.copy()
                    test_config.update({
                        'he_lower_cutoff': he_lo,
                        'he_upper_cutoff': he_hi
                    })
                    
                    # Quick detection test
                    score = self._evaluate_detection_quality(he_sample, 'he', test_config)
                    
                    if score > best_score:
                        best_score = score
                        best_params.update({
                            'he_lower_cutoff': he_lo,
                            'he_upper_cutoff': he_hi
                        })
                        
                except Exception as e:
                    logger.debug(f"Parameter test failed: {e}")
                    continue
        
        logger.info(f"Optimized parameters found (score: {best_score:.3f})")
        return best_params
    
    def _evaluate_detection_quality(self, image_path: str, modality: str, test_config: Dict) -> float:
        """
        Evaluate detection quality for parameter optimization.
        
        Returns:
            Quality score (higher is better)
        """
        try:
            # Create temporary detector with test config
            temp_detector = DualModalityCoreDetector(test_config)
            
            if modality == 'he':
                cores = temp_detector._detect_cores_he(image_path)
            else:
                cores = temp_detector._detect_cores_orion(image_path)
            
            if cores.empty:
                return 0.0
            
            # Score based on number of reasonable cores and their quality
            num_cores = len(cores)
            avg_circularity = cores['circularity'].mean()
            size_variance = cores['area'].std() / cores['area'].mean()
            
            # Target: ~100-300 cores, high circularity, consistent sizes
            core_count_score = min(num_cores / 200.0, 1.0)  # Optimal around 200
            circularity_score = avg_circularity
            consistency_score = max(0, 1.0 - size_variance)
            
            total_score = (core_count_score * 0.5 + 
                          circularity_score * 0.3 + 
                          consistency_score * 0.2)
            
            return total_score
            
        except Exception as e:
            logger.debug(f"Quality evaluation failed: {e}")
            return 0.0 