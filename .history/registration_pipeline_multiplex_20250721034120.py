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

# Import base classes from the original registration pipeline
try:
    from registration_pipeline import DatasetPreparator, RegistrationPipeline, QualityController
    BASE_CLASSES_AVAILABLE = True
except ImportError:
    BASE_CLASSES_AVAILABLE = False
    logger.warning("Base classes not available, will define them locally")

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
                # Load the FULL multi-channel Orion image - VALIS will handle channels internally
                full_orion_img = imread(orion_path)
                # Extract DAPI channel only for quality metrics later
                dapi_img = ImagePreprocessor.extract_dapi_channel(orion_path)
            except Exception as e:
                raise ValueError(f"Image preprocessing failed: {e}")
            
            # Validate full Orion image
            if full_orion_img is None or full_orion_img.size == 0:
                raise ValueError(f"Empty or corrupted Orion image: {orion_path}")
                
            logger.info(f"Loaded full Orion image with shape: {full_orion_img.shape}")
            
            # Additional validation for VALIS compatibility
            if he_img.shape[0] < 100 or he_img.shape[1] < 100:
                raise ValueError(f"H&E image too small for reliable registration: {he_img.shape}")
            
            if dapi_img.shape[0] < 100 or dapi_img.shape[1] < 100:
                raise ValueError(f"Orion image too small for reliable registration: {dapi_img.shape}")
            
            # Check if images are completely black or white (likely problematic)
            he_mean = np.mean(he_img)
            orion_mean = np.mean(dapi_img)
            
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
                
                # Save images for VALIS - use original filenames to preserve multi-channel info
                he_file = src_dir / f"{core_id}_he.tif"
                orion_file = src_dir / f"{core_id}_orion.tif"
                
                try:
                    # Save H&E as regular RGB
                    imwrite(he_file, he_img)
                    
                    # Save the FULL multi-channel Orion image as TIFF with proper metadata
                    # Ensure it's in (C, H, W) format for VALIS
                    if full_orion_img.ndim == 3:
                        if full_orion_img.shape[0] > full_orion_img.shape[2]:
                            # Already in (C, H, W) format
                            orion_for_valis = full_orion_img
                        else:
                            # Convert from (H, W, C) to (C, H, W)
                            orion_for_valis = np.transpose(full_orion_img, (2, 0, 1))
                    else:
                        orion_for_valis = full_orion_img
                    
                    # Save with proper metadata to indicate it's multi-channel, not RGB
                    # Use OME-TIFF format with proper channel information
                    from tifffile import imwrite
                    imwrite(
                        orion_file, 
                        orion_for_valis,
                        photometric='minisblack',  # Prevent RGB interpretation
                        metadata={'axes': 'CYX'},   # Specify channel-first format
                        imagej=False,               # Disable ImageJ format which might convert to RGB
                        ome=True                    # Use OME-TIFF format
                    )
                    
                    logger.info(f"Saved H&E image: {he_file} (shape: {he_img.shape})")
                    logger.info(f"Saved Orion image: {orion_file} (shape: {orion_for_valis.shape}) as multi-channel OME-TIFF")
                except Exception as e:
                    raise ValueError(f"Failed to save images for VALIS: {e}")
                
                # Run VALIS registration with error handling
                try:
                    # Create VALIS registrar with H&E as reference
                    # Specify that this contains fluorescence images to prevent RGB conversion
                    registrar = registration.Valis(
                        str(src_dir),
                        str(dst_dir),
                        reference_img_f=he_file.name,  # Use H&E as reference
                        image_type=None,               # Let VALIS auto-detect each image type
                        imgs_ordered=True,             # Preserve order (H&E first, then Orion)
                        align_to_reference=True,       # Align Orion to H&E
                        max_processed_image_dim_px=self.config.max_processed_image_dim_px,
                        max_non_rigid_registration_dim_px=self.config.max_non_rigid_registration_dim_px,
                    )
                    
                    # Register images - VALIS will handle multi-channel automatically
                    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
                    
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
                    # Warp and save - VALIS will preserve all channels in the Orion image
                    registrar.warp_and_save_slides(
                        str(out_dir),
                        crop="reference",  # Crop to H&E reference
                        non_rigid=True,
                        compression=self.config.compression,
                        Q=self.config.compression_quality
                    )
                    
                    # Clean up JVM
                    registration.kill_jvm()
                    
                    # Load warped multi-channel Orion result
                    warped_orion_path = out_dir / orion_file.with_suffix(".ome.tiff").name
                    
                    if not warped_orion_path.exists():
                        raise ValueError(f"VALIS failed to generate warped Orion output: {warped_orion_path}")
                    
                    warped_orion = imread(warped_orion_path)
                    logger.info(f"Loaded warped Orion with shape: {warped_orion.shape}")
                    
                    if warped_orion is None or warped_orion.size == 0:
                        raise ValueError("VALIS warped Orion output is empty or corrupted")
                    
                    # Verify we have multi-channel data
                    if warped_orion.ndim == 3:
                        if warped_orion.shape[0] < warped_orion.shape[2]:
                            # Likely (C, H, W) format
                            n_channels = warped_orion.shape[0]
                        else:
                            # Likely (H, W, C) format
                            n_channels = warped_orion.shape[2]
                        logger.info(f"✅ Successfully warped {n_channels}-channel Orion image!")
                    else:
                        logger.warning(f"⚠️ Warped Orion is not multi-channel: {warped_orion.shape}")
                    
                except Exception as e:
                    # Clean up JVM on error
                    try:
                        registration.kill_jvm()
                    except:
                        pass
                    raise ValueError(f"VALIS warping failed: {e}")
                
                # Calculate quality metrics using DAPI channel for comparison
                try:
                    # For quality metrics, use the DAPI channel regardless of what we return
                    if warped_orion.ndim == 3:
                        if warped_orion.shape[0] < warped_orion.shape[2]:
                            # (C, H, W) format - use first channel (should be DAPI)
                            dapi_for_metrics = warped_orion[0]
                        else:
                            # (H, W, C) format - use first channel
                            dapi_for_metrics = warped_orion[:, :, 0]
                    else:
                        dapi_for_metrics = warped_orion
                    
                    metrics = self._calculate_registration_metrics(he_img, dapi_for_metrics)
                except Exception as e:
                    logger.warning(f"Quality metrics calculation failed for {core_id}: {e}")
                    metrics = {'ssim': 0.0, 'ncc': 0.0, 'mutual_info': 0.0}
                
                # Save results - now warped_orion contains all channels!
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


# Fallback base class definitions if not available from registration_pipeline
if not BASE_CLASSES_AVAILABLE:
    
    class DatasetPreparator:
        """Base class for dataset preparation."""
        
        def __init__(self, config: RegistrationConfig):
            self.config = config
        
        def _is_valid_patch(self, orion_patch: np.ndarray) -> bool:
            """Check if patch contains sufficient signal."""
            if orion_patch.ndim == 3:
                # For multi-channel, check if any channel has sufficient signal
                return np.max(orion_patch) >= self.config.min_background_threshold
            else:
                return orion_patch.max() >= self.config.min_background_threshold
    
    class RegistrationPipeline:
        """Base pipeline class."""
        
        def __init__(self, config: RegistrationConfig):
            self.config = config
        
        def _find_image_pairs(self) -> List[Tuple[str, str, str]]:
            """Find H&E and Orion image pairs."""
            input_path = pathlib.Path(self.config.input_dir)
            
            if not input_path.exists():
                raise ValueError(f"Input directory does not exist: {input_path}")
            
            # Find H&E files
            he_files = list(input_path.glob(f"*{self.config.he_suffix}"))
            
            if not he_files:
                raise ValueError(f"No H&E files found with suffix '{self.config.he_suffix}' in {input_path}")
            
            image_pairs = []
            
            for he_file in he_files:
                # Extract core identifier
                core_id = he_file.stem.replace(self.config.he_suffix.replace('.tif', ''), '')
                
                # Find corresponding Orion file
                orion_file = input_path / f"{core_id}{self.config.orion_suffix}"
                
                if orion_file.exists():
                    image_pairs.append((str(he_file), str(orion_file), core_id))
                else:
                    logger.warning(f"No Orion file found for {he_file}")
            
            return image_pairs
    
    class QualityController:
        """Quality control for registrations."""
        
        def __init__(self, config: RegistrationConfig):
            self.config = config
        
        def assess_registration_quality(self, registration_results: List[Dict]) -> pd.DataFrame:
            """Assess registration quality and return metrics DataFrame."""
            quality_data = []
            
            for result in registration_results:
                if result['success']:
                    metrics = result['metrics']
                    quality_data.append({
                        'core_id': result['core_id'],
                        'ssim': metrics.get('ssim', 0.0),
                        'ncc': metrics.get('ncc', 0.0),
                        'mutual_info': metrics.get('mutual_info', 0.0),
                        'success': True
                    })
                else:
                    quality_data.append({
                        'core_id': result['core_id'],
                        'ssim': 0.0,
                        'ncc': 0.0,
                        'mutual_info': 0.0,
                        'success': False
                    })
            
            return pd.DataFrame(quality_data)
        
        def create_quality_plots(self, registration_results: List[Dict], output_dir: str):
            """Create quality control plots."""
            # Simple implementation - just log that plots would be created
            logger.info(f"Quality plots would be saved to: {output_dir}")


class DatasetPreparatorMultiplex(DatasetPreparator):
    """Prepares training datasets from registered images (multi-channel version)."""
    
    def create_training_pairs(self, registration_results: List[Dict]) -> str:
        """Create training pairs from multi-channel registered images."""
        pairs_dir = pathlib.Path(self.config.output_dir) / "training_pairs_multiplex"
        pairs_dir.mkdir(exist_ok=True)
        
        successful_results = [r for r in registration_results if r['success']]
        logger.info(f"Creating multiplex training pairs from {len(successful_results)} successful registrations")
        
        pair_count = 0
        
        for result in successful_results:
            he_img = result['he_img']
            warped_orion = result['warped_orion']
            core_id = result['core_id']
            
            logger.info(f"Processing {core_id}: H&E shape {he_img.shape}, Orion shape {warped_orion.shape}")
            
            # Create patches
            patches = self._extract_patches(he_img, warped_orion, core_id)
            
            # Save patches
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
        """Extract patches from registered images, handling multi-channel Orion properly."""
        patches = []
        
        logger.info(f"Extracting patches from {core_id}: H&E {he_img.shape}, Orion {orion_img.shape}")
        
        # Determine the spatial dimensions for H&E (should be H, W, C)
        he_height, he_width = he_img.shape[:2]
        
        # Handle different Orion image formats
        if orion_img.ndim == 3:
            # Check if it's (C, H, W) or (H, W, C)
            if orion_img.shape[0] <= 50 and orion_img.shape[1] > 50 and orion_img.shape[2] > 50:
                # Likely (C, H, W) format - channels first
                orion_channels, orion_height, orion_width = orion_img.shape
                orion_format = "CHW"
                logger.info(f"Detected Orion format: (C, H, W) = ({orion_channels}, {orion_height}, {orion_width})")
            else:
                # Likely (H, W, C) format - channels last
                orion_height, orion_width, orion_channels = orion_img.shape
                orion_format = "HWC"
                logger.info(f"Detected Orion format: (H, W, C) = ({orion_height}, {orion_width}, {orion_channels})")
        elif orion_img.ndim == 2:
            # Single channel (H, W)
            orion_height, orion_width = orion_img.shape
            orion_channels = 1
            orion_format = "HW"
            logger.info(f"Detected Orion format: (H, W) = ({orion_height}, {orion_width})")
        else:
            raise ValueError(f"Unsupported Orion image dimensions: {orion_img.shape}")
        
        # Ensure spatial dimensions match between H&E and Orion
        if he_height != orion_height or he_width != orion_width:
            logger.warning(f"Size mismatch: H&E ({he_height}x{he_width}) vs Orion ({orion_height}x{orion_width})")
            # Resize Orion to match H&E
            if orion_format == "CHW":
                # Resize each channel separately
                resized_channels = []
                for c in range(orion_channels):
                    channel = cv2.resize(orion_img[c], (he_width, he_height), interpolation=cv2.INTER_AREA)
                    resized_channels.append(channel)
                orion_img = np.stack(resized_channels, axis=0)
            elif orion_format == "HWC":
                orion_img = cv2.resize(orion_img, (he_width, he_height), interpolation=cv2.INTER_AREA)
            else:  # HW
                orion_img = cv2.resize(orion_img, (he_width, he_height), interpolation=cv2.INTER_AREA)
            
            orion_height, orion_width = he_height, he_width
            logger.info(f"Resized Orion to match H&E: {orion_img.shape}")
        
        # Extract patches
        for y in range(0, min(he_height, orion_height) - self.config.patch_size + 1, self.config.stride):
            for x in range(0, min(he_width, orion_width) - self.config.patch_size + 1, self.config.stride):
                
                # Extract H&E patch
                he_patch = he_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                
                # Extract Orion patch based on format
                if orion_format == "CHW":
                    # (C, H, W) format
                    orion_patch = orion_img[:, y:y + self.config.patch_size, x:x + self.config.patch_size]
                    # Convert to (H, W, C) for consistency
                    orion_patch = np.transpose(orion_patch, (1, 2, 0))
                elif orion_format == "HWC":
                    # (H, W, C) format - already correct
                    orion_patch = orion_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                else:  # HW
                    # (H, W) format - single channel
                    orion_patch = orion_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                
                patches.append((he_patch, orion_patch))
        
        logger.info(f"Extracted {len(patches)} patches from {core_id}")
        return patches

# In the RegistrationPipeline class, use DatasetPreparatorMultiplex
class RegistrationPipelineMultiplex(RegistrationPipeline):
    """Main registration pipeline orchestrator for multi-channel images."""
    
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
    
    def run(self) -> Dict:
        """
        Run the complete multiplex registration pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting multiplex registration pipeline")
        
        # Find image pairs
        image_pairs = self._find_image_pairs()
        logger.info(f"Found {len(image_pairs)} image pairs")
        
        if not image_pairs:
            raise ValueError("No image pairs found")
        
        # Register images
        registration_results = self._register_images(image_pairs)
        
        # Assess quality
        quality_df = self.qc.assess_registration_quality(registration_results)
        quality_df.to_csv(self.output_path / "registration_quality_multiplex.csv", index=False)
        
        # Create quality plots
        self.qc.create_quality_plots(registration_results, str(self.output_path / "quality_plots"))
        
        # Create training dataset
        successful_results = [r for r in registration_results if r['success']]
        if successful_results:
            pairs_dir = self.preparator.create_training_pairs(successful_results)
        else:
            pairs_dir = None
            logger.warning("No successful registrations to create training pairs")
        
        # Calculate summary statistics
        total_pairs = len(image_pairs)
        successful_count = len(successful_results)
        success_rate = successful_count / total_pairs if total_pairs > 0 else 0.0
        
        # Save summary
        summary = {
            'total_image_pairs': total_pairs,
            'successful_registrations': successful_count,
            'failed_registrations': total_pairs - successful_count,
            'success_rate': success_rate,
            'training_pairs_directory': pairs_dir,
            'quality_csv': str(self.output_path / "registration_quality_multiplex.csv")
        }
        
        # Save summary to JSON
        import json
        with open(self.output_path / "pipeline_summary_multiplex.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Multiplex registration pipeline completed")
        logger.info(f"Success rate: {success_rate:.2%}")
        
        return summary
    
    def _register_images(self, image_pairs: List[Tuple[str, str, str]]) -> List[Dict]:
        """Register all image pairs."""
        registration_results = []
        failure_count = 0
        
        if self.config.num_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []
                for he_path, orion_path, core_id in image_pairs:
                    future = executor.submit(self.registrar.register_core_pair, he_path, orion_path, core_id)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        registration_results.append(result)
                        
                        # Track failures
                        if not result['success']:
                            failure_count += 1
                            if failure_count >= self.config.max_failures_before_stop:
                                logger.error(f"Too many failures ({failure_count}), stopping registration")
                                # Cancel remaining futures
                                for remaining_future in futures:
                                    remaining_future.cancel()
                                break
                    except Exception as e:
                        logger.error(f"Registration task failed with exception: {e}")
                        failure_count += 1
        else:
            # Sequential processing
            for he_path, orion_path, core_id in image_pairs:
                try:
                    result = self.registrar.register_core_pair(he_path, orion_path, core_id)
                    registration_results.append(result)
                    
                    # Track failures
                    if not result['success']:
                        failure_count += 1
                        if failure_count >= self.config.max_failures_before_stop:
                            logger.error(f"Too many failures ({failure_count}), stopping registration")
                            break
                except Exception as e:
                    logger.error(f"Registration failed for {core_id}: {e}")
                    failure_count += 1
                    registration_results.append({
                        'core_id': core_id,
                        'he_path': he_path,
                        'orion_path': orion_path,
                        'success': False,
                        'error': str(e)
                    })
        
        logger.info(f"Registration completed: {len([r for r in registration_results if r['success']])} successful, {failure_count} failed")
        return registration_results


def main():
    """Main function to run the multiplex registration pipeline."""
    # Example configuration
    config = RegistrationConfig(
        input_dir="/path/to/your/image/pairs",
        output_dir="./registration_output_multiplex",
        he_suffix="_HE.tif",
        orion_suffix="_Orion.tif",
        patch_size=256,
        stride=256,
        num_workers=4
    )
    
    # Create and run pipeline
    pipeline = RegistrationPipelineMultiplex(config)
    results = pipeline.run()
    
    print("Multiplex Registration Pipeline Results:")
    print(f"Total pairs: {results['total_image_pairs']}")
    print(f"Successful registrations: {results['successful_registrations']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Multiplex training pairs directory: {results['training_pairs_directory']}")


if __name__ == "__main__":
    main() 