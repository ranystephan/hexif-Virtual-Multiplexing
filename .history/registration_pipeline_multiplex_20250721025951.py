"""
Registration Pipeline for H&E to Multiplex Protein Prediction

This module provides a comprehensive registration pipeline using VALIS to align
H&E images with multiplex protein images (Orion, CODEX, etc.) for training
deep learning models for in-silico staining.

Key Features:
- Per-core registration using VALIS
- Multi-channel warping and alignment
- Training dataset preparation with tiling
- Quality control and validation metrics
- Integration with existing ROSIE baseline
"""

import os
import sys
import pathlib
import tempfile
import gc
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
import skimage.io as skio
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from tifffile import imread, imwrite
import pandas as pd
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

# VALIS registration
try:
    from valis import registration
    VALIS_AVAILABLE = True
except ImportError:
    VALIS_AVAILABLE = False
    warnings.warn("VALIS not available. Please install with: pip install valis")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RegistrationConfig:
    """Configuration for registration pipeline."""
    
    # Input/Output paths
    input_dir: str
    output_dir: str
    he_suffix: str = "_HE.tif"
    orion_suffix: str = "_Orion.tif"
    
    # Registration parameters
    max_processed_image_dim_px: int = 1024
    max_non_rigid_registration_dim_px: int = 1500
    reference_img: str = "he"  # Use H&E as reference
    
    # Tiling parameters
    patch_size: int = 256
    stride: int = 256
    min_background_threshold: int = 10
    
    # Quality control
    min_ssim_threshold: float = 0.3
    min_ncc_threshold: float = 0.2
    min_mi_threshold: float = 0.5
    
    # Processing
    num_workers: int = 4
    compression: str = "jp2k"
    compression_quality: int = 95
    
    # Error handling
    skip_failed_registrations: bool = True
    max_failures_before_stop: int = 50  # Stop if too many failures
    
    # Output formats
    save_ome_tiff: bool = True
    save_npy_pairs: bool = True
    save_quality_plots: bool = True


class ImagePreprocessor:
    """Handles image preprocessing for registration."""
    
    @staticmethod
    def load_and_preprocess_image(image_path: str, target_channels: Optional[List[int]] = None) -> np.ndarray:
        """
        Load and preprocess image for registration.
        
        Args:
            image_path: Path to image file
            target_channels: Specific channels to extract (None for all)
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            img = imread(image_path)
            
            # Validate image
            if img is None or img.size == 0:
                raise ValueError(f"Empty or corrupted image: {image_path}")
            
            # Check for problematic dimensions
            if any(dim <= 0 for dim in img.shape):
                raise ValueError(f"Invalid image dimensions: {img.shape}")
            
            logger.info(f"Loaded H&E image with shape: {img.shape}")
            
            # Handle different image formats
            if img.ndim == 3:
                if img.shape[2] <= 4:  # Likely RGB/RGBA format (H, W, C)
                    if img.shape[2] == 4:  # RGBA
                        img = img[:, :, :3]  # Remove alpha channel
                    # Keep as RGB
                elif img.shape[0] <= 10:  # Likely multi-channel format (C, H, W) 
                    if target_channels is not None:
                        img = img[target_channels]
                        if img.ndim == 3:
                            img = np.transpose(img, (1, 2, 0))  # Convert to H, W, C
                    elif img.shape[0] == 1:  # Single channel stack
                        img = img[0]
                        img = np.stack([img] * 3, axis=-1)  # Convert to RGB
                    else:  # Use first 3 channels for RGB
                        if img.shape[0] >= 3:
                            img = np.transpose(img[:3], (1, 2, 0))  # Convert to H, W, 3
                        else:
                            # Use first channel and replicate for RGB
                            img = img[0]
                            img = np.stack([img] * 3, axis=-1)
                else:
                    raise ValueError(f"Unsupported image format: {img.shape}")
            elif img.ndim == 2:
                # Convert grayscale to RGB
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim > 3:
                raise ValueError(f"Unsupported image dimensions: {img.shape}")
            
            # Get spatial dimensions for validation
            if img.ndim == 3:
                height, width = img.shape[0], img.shape[1]
            else:
                height, width = img.shape[0], img.shape[1]
            
            # Ensure minimum size for VALIS
            min_size = 50  # Minimum dimension for VALIS to work
            if height < min_size or width < min_size:
                raise ValueError(f"Image too small for registration: {height}x{width}, minimum size: {min_size}x{min_size}")
            
            # Check for very unusual aspect ratios that might cause OpenCV errors
            aspect_ratio = max(height/width, width/height)
            if aspect_ratio > 10:  # More reasonable threshold
                logger.warning(f"High aspect ratio detected: {height}x{width} (ratio: {aspect_ratio:.1f})")
            
            # Normalize to uint8 for VALIS
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            return img
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    @staticmethod
    def extract_dapi_channel(image_path: str, dapi_channel: int = 0) -> np.ndarray:
        """
        Extract DAPI channel from multi-channel image.
        
        Args:
            image_path: Path to multi-channel image
            dapi_channel: Channel index for DAPI
            
        Returns:
            DAPI channel as 2D array
        """
        img = imread(image_path)
        
        # Validate image
        if img is None or img.size == 0:
            raise ValueError(f"Empty or corrupted image: {image_path}")
        
        # Check for problematic dimensions
        if any(dim <= 0 for dim in img.shape):
            raise ValueError(f"Invalid image dimensions: {img.shape}")
        
        logger.info(f"Loaded Orion image with shape: {img.shape}")
        
        if img.ndim == 3:
            # For multi-channel images, assume channel-first format (C, H, W)
            # This handles your 20-channel Orion images: (20, height, width)
            if dapi_channel >= img.shape[0]:
                logger.warning(f"DAPI channel {dapi_channel} not available (image has {img.shape[0]} channels), using channel 0")
                dapi_channel = 0
            
            dapi = img[dapi_channel]  # Extract the specified channel
            logger.info(f"Extracted channel {dapi_channel} with shape: {dapi.shape}")
            
        elif img.ndim == 2:
            # Already 2D image
            dapi = img
        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape}")
        
        # Ensure minimum size for the spatial dimensions
        min_size = 50
        if dapi.shape[0] < min_size or dapi.shape[1] < min_size:
            raise ValueError(f"DAPI channel too small: {dapi.shape}, minimum: {min_size}x{min_size}")
        
        # Check for very extreme aspect ratios
        height, width = dapi.shape[:2]
        aspect_ratio = max(height/width, width/height)
        if aspect_ratio > 10:  # More reasonable threshold for tissue cores
            logger.warning(f"High aspect ratio in DAPI channel: {height}x{width} (ratio: {aspect_ratio:.1f})")
        
        # Normalize to uint8
        dapi_u8 = cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return dapi_u8


class VALISRegistrar:
    """Handles VALIS-based image registration."""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
        if not VALIS_AVAILABLE:
            raise ImportError("VALIS is required for registration. Install with: pip install valis")
    
    def register_core_pair(self, he_path: str, orion_path: str, core_id: str) -> Dict:
        """
        Register a single core pair using VALIS.
        
        Args:
            he_path: Path to H&E image
            orion_path: Path to Orion/multiplex image
            core_id: Identifier for the core
            
        Returns:
            Dictionary with registration results and metrics
        """
        logger.info(f"Registering core {core_id}")
        
        try:
            # Pre-validate images before processing
            try:
                he_img = ImagePreprocessor.load_and_preprocess_image(he_path)
                orion_img = ImagePreprocessor.extract_dapi_channel(orion_path)
            except Exception as e:
                raise ValueError(f"Image preprocessing failed: {e}")
            
            # Additional validation for VALIS compatibility
            if he_img.shape[0] < 100 or he_img.shape[1] < 100:
                raise ValueError(f"H&E image too small for reliable registration: {he_img.shape}")
            
            if orion_img.shape[0] < 100 or orion_img.shape[1] < 100:
                raise ValueError(f"Orion image too small for reliable registration: {orion_img.shape}")
            
            # Check if images are completely black or white (likely problematic)
            he_mean = np.mean(he_img)
            orion_mean = np.mean(orion_img)
            
            if he_mean < 5 or he_mean > 250:
                logger.warning(f"H&E image may be problematic (mean intensity: {he_mean:.1f})")
            
            if orion_mean < 5 or orion_mean > 250:
                logger.warning(f"Orion image may be problematic (mean intensity: {orion_mean:.1f})")
            
            # Create temporary directory for VALIS
            with tempfile.TemporaryDirectory() as tmpdir:
                src_dir = pathlib.Path(tmpdir)
                dst_dir = src_dir / "valis_logs"
                out_dir = src_dir / "warped"
                
                # Create directories
                src_dir.mkdir(exist_ok=True)
                dst_dir.mkdir(exist_ok=True)
                out_dir.mkdir(exist_ok=True)
                
                # Save images for VALIS
                he_file = src_dir / "he.png"
                orion_file = src_dir / "orion.png"
                
                try:
                    skio.imsave(he_file, he_img, check_contrast=False)
                    skio.imsave(orion_file, orion_img, check_contrast=False)
                except Exception as e:
                    raise ValueError(f"Failed to save images for VALIS: {e}")
                
                # Run VALIS registration with error handling
                try:
                    registrar = registration.Valis(
                        str(src_dir),
                        str(dst_dir),
                        reference_img_f=he_file.name,
                        image_type="multi",
                        imgs_ordered=True,
                        align_to_reference=True,
                        max_processed_image_dim_px=self.config.max_processed_image_dim_px,
                        max_non_rigid_registration_dim_px=self.config.max_non_rigid_registration_dim_px,
                    )
                    
                    # Register images
                    registrar.register()
                    
                except Exception as e:
                    # Clean up JVM on error
                    try:
                        registration.kill_jvm()
                    except:
                        pass
                    
                    # Check for specific VALIS errors
                    error_msg = str(e).lower()
                    if "'nonetype' object has no attribute 'shape'" in error_msg:
                        raise ValueError("VALIS feature detection failed - images may lack sufficient features for registration")
                    elif "opencv" in error_msg and "resize" in error_msg:
                        raise ValueError("OpenCV resize error - image dimensions may be problematic")
                    else:
                        raise ValueError(f"VALIS registration failed: {e}")
                
                try:
                    # Warp and save
                    registrar.warp_and_save_slides(
                        str(out_dir),
                        crop="reference",
                        non_rigid=True,
                        compression=self.config.compression,
                        Q=self.config.compression_quality
                    )
                    
                    # Clean up JVM
                    registration.kill_jvm()
                    
                    # Load warped results
                    warped_orion_path = out_dir / orion_file.with_suffix(".ome.tiff").name
                    
                    if not warped_orion_path.exists():
                        raise ValueError("VALIS failed to generate warped output")
                    
                    warped_orion = imread(warped_orion_path)
                    
                    if warped_orion is None or warped_orion.size == 0:
                        raise ValueError("Warped output is empty or corrupted")
                    
                except Exception as e:
                    # Clean up JVM on error
                    try:
                        registration.kill_jvm()
                    except:
                        pass
                    raise ValueError(f"VALIS warping failed: {e}")
                
                # Calculate quality metrics
                try:
                    metrics = self._calculate_registration_metrics(he_img, warped_orion)
                except Exception as e:
                    logger.warning(f"Quality metrics calculation failed for {core_id}: {e}")
                    metrics = {'ssim': 0.0, 'ncc': 0.0, 'mutual_info': 0.0}
                
                # Save results
                results = {
                    'core_id': core_id,
                    'he_path': he_path,
                    'orion_path': orion_path,
                    'warped_orion': warped_orion,
                    'he_img': he_img,
                    'metrics': metrics,
                    'success': True
                }
                
                return results
                
        except Exception as e:
            # Ensure JVM is cleaned up on any error
            try:
                registration.kill_jvm()
            except:
                pass
            
            logger.error(f"Registration failed for core {core_id}: {e}")
            return {
                'core_id': core_id,
                'he_path': he_path,
                'orion_path': orion_path,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_registration_metrics(self, he_img: np.ndarray, warped_orion: np.ndarray) -> Dict:
        """Calculate registration quality metrics."""
        # Convert H&E to grayscale for comparison
        if he_img.ndim == 3:
            he_gray = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY)
        else:
            he_gray = he_img
        
        # Resize H&E to match warped Orion
        he_gray_rs = cv2.resize(he_gray, warped_orion.shape[::-1], interpolation=cv2.INTER_AREA)
        
        # Calculate metrics
        ssim_val = ssim(he_gray_rs, warped_orion, data_range=warped_orion.max() - warped_orion.min())
        ncc_val = np.corrcoef(he_gray_rs.flatten(), warped_orion.flatten())[0, 1]
        
        # Mutual information
        hist_2d = np.histogram2d(he_gray_rs.flatten(), warped_orion.flatten(), bins=64)[0]
        mi_val = mutual_info_score(None, None, contingency=hist_2d)
        
        return {
            'ssim': ssim_val,
            'ncc': ncc_val,
            'mutual_info': mi_val
        }

class DatasetPreparatorMultiplex(DatasetPreparator):
    """Prepares training datasets from registered images (multi-channel version)."""
    def create_training_pairs(self, registration_results: List[Dict]) -> str:
        pairs_dir = pathlib.Path(self.config.output_dir) / "training_pairs_multiplex"
        pairs_dir.mkdir(exist_ok=True)
        successful_results = [r for r in registration_results if r['success']]
        logger.info(f"Creating multiplex training pairs from {len(successful_results)} successful registrations")
        pair_count = 0
        for result in successful_results:
            he_img = result['he_img']
            warped_orion = result['warped_orion']
            core_id = result['core_id']
            # Create patches
            patches = self._extract_patches(he_img, warped_orion, core_id)
            for i, (he_patch, orion_patch) in enumerate(patches):
                if self._is_valid_patch(orion_patch):
                    patch_id = f"{core_id}_patch_{i:04d}"
                    if self.config.save_npy_pairs:
                        np.save(pairs_dir / f"{patch_id}_HE.npy", he_patch)
                        np.save(pairs_dir / f"{patch_id}_ORION.npy", orion_patch)
                    pair_count += 1
        logger.info(f"Created {pair_count} multiplex training pairs")
        return str(pairs_dir)
    def _extract_patches(self, he_img: np.ndarray, orion_img: np.ndarray, core_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        patches = []
        for y in range(0, he_img.shape[0] - self.config.patch_size + 1, self.config.stride):
            for x in range(0, he_img.shape[1] - self.config.patch_size + 1, self.config.stride):
                he_patch = he_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                # If orion_img is multi-channel (C, H, W) or (H, W, C), handle both
                if orion_img.ndim == 3 and (orion_img.shape[0] == he_img.shape[0] and orion_img.shape[1] == he_img.shape[1]):
                    # (H, W, C) format
                    orion_patch = orion_img[y:y + self.config.patch_size, x:x + self.config.patch_size, :]
                elif orion_img.ndim == 3 and (orion_img.shape[1] == he_img.shape[0] and orion_img.shape[2] == he_img.shape[1]):
                    # (C, H, W) format
                    orion_patch = orion_img[:, y:y + self.config.patch_size, x:x + self.config.patch_size]
                    orion_patch = np.transpose(orion_patch, (1, 2, 0))
                else:
                    # Fallback: treat as single channel
                    orion_patch = orion_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                patches.append((he_patch, orion_patch))
        return patches

# In the RegistrationPipeline class, use DatasetPreparatorMultiplex
class RegistrationPipelineMultiplex(RegistrationPipeline):
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.registrar = VALISRegistrar(config)
        self.preparator = DatasetPreparatorMultiplex(config)
        self.qc = QualityController(config)
        self.output_path = pathlib.Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        (self.output_path / "registered_images").mkdir(exist_ok=True)
        (self.output_path / "quality_plots").mkdir(exist_ok=True)
        (self.output_path / "training_pairs_multiplex").mkdir(exist_ok=True)

# In main(), instantiate RegistrationPipelineMultiplex
if __name__ == "__main__":
    config = RegistrationConfig(
        input_dir="/path/to/your/image/pairs",
        output_dir="./registration_output_multiplex",
        he_suffix="_HE.tif",
        orion_suffix="_Orion.tif",
        patch_size=256,
        stride=256,
        num_workers=4
    )
    pipeline = RegistrationPipelineMultiplex(config)
    results = pipeline.run()
    print("Multiplex Registration Pipeline Results:")
    print(f"Total pairs: {results['total_image_pairs']}")
    print(f"Successful registrations: {results['successful_registrations']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Multiplex training pairs directory: {results['training_pairs_directory']}") 