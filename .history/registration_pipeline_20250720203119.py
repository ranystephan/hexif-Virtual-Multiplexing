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
            
            # Handle different image formats
            if img.ndim == 3 and img.shape[0] <= 10:  # Multi-channel
                if target_channels is not None:
                    img = img[target_channels]
                elif img.shape[0] == 1:  # Single channel stack
                    img = img[0]
                else:  # Use first channel or max projection
                    img = np.max(img, axis=0)
            elif img.ndim == 3 and img.shape[2] <= 4:  # RGB/RGBA
                if img.shape[2] == 4:  # RGBA
                    img = img[:, :, :3]  # Remove alpha channel
            elif img.ndim > 3:
                raise ValueError(f"Unsupported image dimensions: {img.shape}")
            
            # Ensure minimum size for VALIS
            min_size = 50  # Minimum dimension for VALIS to work
            if img.shape[0] < min_size or img.shape[1] < min_size:
                raise ValueError(f"Image too small for registration: {img.shape}, minimum size: {min_size}x{min_size}")
            
            # Check for very unusual aspect ratios that might cause OpenCV errors
            height, width = img.shape[:2]
            aspect_ratio = max(height/width, width/height)
            if aspect_ratio > 100:  # Very extreme aspect ratio
                logger.warning(f"Extreme aspect ratio detected: {height}x{width} (ratio: {aspect_ratio:.1f})")
            
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
        
        if img.ndim == 3:
            if img.shape[0] <= 10:  # Channel-first format
                if dapi_channel >= img.shape[0]:
                    logger.warning(f"DAPI channel {dapi_channel} not available, using channel 0")
                    dapi_channel = 0
                dapi = img[dapi_channel]
            else:  # Spatial-first format
                if dapi_channel >= img.shape[2]:
                    logger.warning(f"DAPI channel {dapi_channel} not available, using channel 0")
                    dapi_channel = 0
                dapi = img[:, :, dapi_channel]
        else:
            dapi = img
        
        # Ensure minimum size
        min_size = 50
        if dapi.shape[0] < min_size or dapi.shape[1] < min_size:
            raise ValueError(f"DAPI channel too small: {dapi.shape}, minimum: {min_size}x{min_size}")
        
        # Check for very extreme aspect ratios
        height, width = dapi.shape[:2]
        aspect_ratio = max(height/width, width/height)
        if aspect_ratio > 100:
            logger.warning(f"Extreme aspect ratio in DAPI channel: {height}x{width} (ratio: {aspect_ratio:.1f})")
        
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


class DatasetPreparator:
    """Prepares training datasets from registered images."""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
    
    def create_training_pairs(self, registration_results: List[Dict]) -> str:
        """
        Create training pairs from registered images.
        
        Args:
            registration_results: List of registration results
            
        Returns:
            Path to training pairs directory
        """
        pairs_dir = pathlib.Path(self.config.output_dir) / "training_pairs"
        pairs_dir.mkdir(exist_ok=True)
        
        successful_results = [r for r in registration_results if r['success']]
        logger.info(f"Creating training pairs from {len(successful_results)} successful registrations")
        
        pair_count = 0
        
        for result in successful_results:
            he_img = result['he_img']
            warped_orion = result['warped_orion']
            core_id = result['core_id']
            
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
        
        logger.info(f"Created {pair_count} training pairs")
        return str(pairs_dir)
    
    def _extract_patches(self, he_img: np.ndarray, orion_img: np.ndarray, core_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract patches from registered images."""
        patches = []
        
        for y in range(0, he_img.shape[0] - self.config.patch_size + 1, self.config.stride):
            for x in range(0, he_img.shape[1] - self.config.patch_size + 1, self.config.stride):
                he_patch = he_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                orion_patch = orion_img[y:y + self.config.patch_size, x:x + self.config.patch_size]
                
                patches.append((he_patch, orion_patch))
        
        return patches
    
    def _is_valid_patch(self, orion_patch: np.ndarray) -> bool:
        """Check if patch contains sufficient signal."""
        return orion_patch.max() >= self.config.min_background_threshold


class QualityController:
    """Handles quality control for registration results."""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
    
    def assess_registration_quality(self, registration_results: List[Dict]) -> pd.DataFrame:
        """
        Assess quality of registration results.
        
        Args:
            registration_results: List of registration results
            
        Returns:
            DataFrame with quality metrics
        """
        quality_data = []
        
        for result in registration_results:
            if result['success']:
                metrics = result['metrics']
                quality_data.append({
                    'core_id': result['core_id'],
                    'ssim': metrics['ssim'],
                    'ncc': metrics['ncc'],
                    'mutual_info': metrics['mutual_info'],
                    'passed_ssim': metrics['ssim'] >= self.config.min_ssim_threshold,
                    'passed_ncc': metrics['ncc'] >= self.config.min_ncc_threshold,
                    'passed_mi': metrics['mutual_info'] >= self.config.min_mi_threshold
                })
            else:
                quality_data.append({
                    'core_id': result['core_id'],
                    'ssim': np.nan,
                    'ncc': np.nan,
                    'mutual_info': np.nan,
                    'passed_ssim': False,
                    'passed_ncc': False,
                    'passed_mi': False,
                    'error': result.get('error', 'Unknown error')
                })
        
        return pd.DataFrame(quality_data)
    
    def create_quality_plots(self, registration_results: List[Dict], output_dir: str):
        """Create quality control plots."""
        if not self.config.save_quality_plots:
            return
        
        output_path = pathlib.Path(output_dir)
        
        # Create overview plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        successful_results = [r for r in registration_results if r['success']]
        
        if successful_results:
            # Plot metrics distributions
            ssim_vals = [r['metrics']['ssim'] for r in successful_results]
            ncc_vals = [r['metrics']['ncc'] for r in successful_results]
            mi_vals = [r['metrics']['mutual_info'] for r in successful_results]
            
            axes[0, 0].hist(ssim_vals, bins=20, alpha=0.7)
            axes[0, 0].axvline(self.config.min_ssim_threshold, color='red', linestyle='--')
            axes[0, 0].set_title('SSIM Distribution')
            axes[0, 0].set_xlabel('SSIM')
            axes[0, 0].set_ylabel('Count')
            
            axes[0, 1].hist(ncc_vals, bins=20, alpha=0.7)
            axes[0, 1].axvline(self.config.min_ncc_threshold, color='red', linestyle='--')
            axes[0, 1].set_title('NCC Distribution')
            axes[0, 1].set_xlabel('NCC')
            axes[0, 1].set_ylabel('Count')
            
            axes[1, 0].hist(mi_vals, bins=20, alpha=0.7)
            axes[1, 0].axvline(self.config.min_mi_threshold, color='red', linestyle='--')
            axes[1, 0].set_title('Mutual Information Distribution')
            axes[1, 0].set_xlabel('Mutual Information')
            axes[1, 0].set_ylabel('Count')
            
            # Plot example overlay
            if successful_results:
                example = successful_results[0]
                he_img = example['he_img']
                warped_orion = example['warped_orion']
                
                overlay = cv2.merge([
                    cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY),
                    warped_orion,
                    np.zeros_like(warped_orion)
                ])
                
                axes[1, 1].imshow(overlay)
                axes[1, 1].set_title(f'Example Overlay (Core: {example["core_id"]})')
                axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'registration_quality.png', dpi=300, bbox_inches='tight')
        plt.close()


class RegistrationPipeline:
    """Main registration pipeline orchestrator."""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.registrar = VALISRegistrar(config)
        self.preparator = DatasetPreparator(config)
        self.qc = QualityController(config)
        
        # Create output directories
        self.output_path = pathlib.Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        (self.output_path / "registered_images").mkdir(exist_ok=True)
        (self.output_path / "quality_plots").mkdir(exist_ok=True)
        (self.output_path / "training_pairs").mkdir(exist_ok=True)
    
    def run(self) -> Dict:
        """
        Run the complete registration pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting registration pipeline")
        
        # Find image pairs
        image_pairs = self._find_image_pairs()
        logger.info(f"Found {len(image_pairs)} image pairs")
        
        if not image_pairs:
            raise ValueError("No image pairs found")
        
        # Register images
        registration_results = self._register_images(image_pairs)
        
        # Assess quality
        quality_df = self.qc.assess_registration_quality(registration_results)
        quality_df.to_csv(self.output_path / "registration_quality.csv", index=False)
        
        # Create quality plots
        self.qc.create_quality_plots(registration_results, str(self.output_path / "quality_plots"))
        
        # Create training dataset
        successful_results = [r for r in registration_results if r['success']]
        if successful_results:
            pairs_dir = self.preparator.create_training_pairs(successful_results)
        else:
            pairs_dir = None
            logger.warning("No successful registrations to create training pairs")
        
        # Generate summary
        summary = self._generate_summary(registration_results, quality_df, pairs_dir)
        
        logger.info("Registration pipeline completed")
        return summary
    
    def _find_image_pairs(self) -> List[Tuple[str, str, str]]:
        """Find matching H&E and Orion image pairs."""
        input_path = pathlib.Path(self.config.input_dir)
        pairs = []
        
        # Find H&E images
        he_files = list(input_path.glob(f"*{self.config.he_suffix}"))
        
        for he_file in he_files:
            # Extract core ID by removing the HE suffix from the end
            # Handle cases where core ID contains "HE" in the name
            filename = he_file.stem
            if filename.endswith("_HE"):
                core_id = filename[:-3]  # Remove "_HE" suffix
            else:
                # Fallback: try to replace the full suffix
                core_id = filename.replace(self.config.he_suffix.replace(".tif", ""), "")
            
            # Look for corresponding Orion file
            orion_file = input_path / f"{core_id}{self.config.orion_suffix}"
            
            if orion_file.exists():
                pairs.append((str(he_file), str(orion_file), core_id))
            else:
                logger.warning(f"No Orion file found for core {core_id}")
        
        return pairs
    
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
        else:
            # Sequential processing
            for he_path, orion_path, core_id in image_pairs:
                result = self.registrar.register_core_pair(he_path, orion_path, core_id)
                registration_results.append(result)
                
                # Track failures
                if not result['success']:
                    failure_count += 1
                    if failure_count >= self.config.max_failures_before_stop:
                        logger.error(f"Too many failures ({failure_count}), stopping registration")
                        break
        
        logger.info(f"Registration completed: {len([r for r in registration_results if r['success']])} successful, {failure_count} failed")
        return registration_results
    
    def _generate_summary(self, registration_results: List[Dict], quality_df: pd.DataFrame, pairs_dir: Optional[str]) -> Dict:
        """Generate pipeline summary."""
        total_pairs = len(registration_results)
        successful_registrations = len([r for r in registration_results if r['success']])
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif pd.isna(obj):
                return None
            return obj
        
        summary = {
            'total_image_pairs': int(total_pairs),
            'successful_registrations': int(successful_registrations),
            'success_rate': float(successful_registrations / total_pairs if total_pairs > 0 else 0),
            'quality_metrics': {
                'mean_ssim': convert_numpy_types(quality_df['ssim'].mean()),
                'mean_ncc': convert_numpy_types(quality_df['ncc'].mean()),
                'mean_mutual_info': convert_numpy_types(quality_df['mutual_info'].mean()),
                'passed_quality_thresholds': convert_numpy_types(quality_df[['passed_ssim', 'passed_ncc', 'passed_mi']].all(axis=1).sum())
            },
            'training_pairs_directory': pairs_dir,
            'output_directory': str(self.output_path)
        }
        
        # Save summary
        import json
        with open(self.output_path / "pipeline_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    """Main function to run the registration pipeline."""
    # Example configuration
    config = RegistrationConfig(
        input_dir="/path/to/your/image/pairs",
        output_dir="./registration_output",
        he_suffix="_HE.tif",
        orion_suffix="_Orion.tif",
        patch_size=256,
        stride=256,
        num_workers=4
    )
    
    # Create and run pipeline
    pipeline = RegistrationPipeline(config)
    results = pipeline.run()
    
    print("Registration Pipeline Results:")
    print(f"Total pairs: {results['total_image_pairs']}")
    print(f"Successful registrations: {results['successful_registrations']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Training pairs directory: {results['training_pairs_directory']}")


if __name__ == "__main__":
    main() 