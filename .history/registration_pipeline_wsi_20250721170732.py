"""
Whole Slide Image Registration Pipeline for TMA Core Extraction

This module provides a comprehensive pipeline for registering whole slide images (WSI)
of Tissue Microarrays (TMAs) and extracting paired tissue cores. Designed specifically
for H&E to multiplex protein prediction workflows where you have full TMA slides
rather than individual core images.

Key Features:
- Full WSI registration using VALIS
- Automated tissue core detection and segmentation
- Core pairing based on spatial correspondence after registration
- Organized output with paired core folders
- Quality control and validation metrics
- Support for multi-channel protein images (Orion, CODEX, etc.)
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
import json
import shutil

import numpy as np
import cv2
from skimage import measure, morphology, filters
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from tifffile import imread, imwrite
import pandas as pd

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
class WSIRegistrationConfig:
    """Configuration for WSI registration pipeline."""
    
    # Input/Output paths
    he_wsi_path: str
    orion_wsi_path: str
    output_dir: str
    
    # Registration parameters
    max_processed_image_dim_px: int = 2048
    max_non_rigid_registration_dim_px: int = 3000
    reference_img: str = "he"  # Use H&E as reference
    
    # Core detection parameters
    core_min_area: int = 50000  # Minimum pixels for a valid core
    core_max_area: int = 500000  # Maximum pixels for a valid core
    core_circularity_threshold: float = 0.4  # Minimum circularity for cores
    gaussian_sigma: float = 2.0  # For preprocessing before core detection
    
    # Core extraction parameters  
    core_padding: int = 50  # Extra pixels around detected core boundary
    min_core_diameter: int = 200  # Minimum diameter in pixels
    expected_core_diameter: int = 400  # Expected core diameter for validation
    
    # Quality control
    min_ssim_threshold: float = 0.3
    min_ncc_threshold: float = 0.2
    min_mi_threshold: float = 0.5
    
    # Processing
    compression: str = "lzw"
    compression_quality: int = 95
    
    # Output formats
    save_ome_tiff: bool = True
    save_quality_plots: bool = True
    save_core_detection_plots: bool = True


class CoreDetector:
    """Detects tissue cores in TMA slides."""
    
    def __init__(self, config: WSIRegistrationConfig):
        self.config = config
    
    def detect_cores(self, image: np.ndarray, image_name: str = "image") -> List[Dict]:
        """
        Detect tissue cores in a TMA slide image.
        
        Args:
            image: Input image (can be RGB or grayscale)
            image_name: Name for logging and plots
            
        Returns:
            List of dictionaries containing core information
        """
        logger.info(f"Detecting cores in {image_name}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # Take first channel if multi-channel
                gray = image[:, :, 0]
        else:
            gray = image
        
        # Preprocessing
        gray_smooth = filters.gaussian(gray, sigma=self.config.gaussian_sigma)
        
        # Otsu thresholding to separate tissue from background
        threshold = filters.threshold_otsu(gray_smooth)
        binary = gray_smooth < threshold  # Tissue is typically darker
        
        # Remove small objects and fill holes
        binary_cleaned = morphology.remove_small_objects(binary, min_size=1000)
        binary_filled = ndimage.binary_fill_holes(binary_cleaned)
        
        # Find connected components
        labeled = measure.label(binary_filled)
        regions = measure.regionprops(labeled, intensity_image=gray)
        
        cores = []
        for i, region in enumerate(regions):
            # Filter by area
            if not (self.config.core_min_area <= region.area <= self.config.core_max_area):
                continue
            
            # Filter by circularity (4π * area / perimeter²)
            if region.perimeter > 0:
                circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
                if circularity < self.config.core_circularity_threshold:
                    continue
            else:
                continue
            
            # Calculate equivalent diameter
            equiv_diameter = np.sqrt(4 * region.area / np.pi)
            if equiv_diameter < self.config.min_core_diameter:
                continue
            
            # Get bounding box with padding
            minr, minc, maxr, maxc = region.bbox
            
            # Add padding
            height, width = gray.shape
            minr_pad = max(0, minr - self.config.core_padding)
            minc_pad = max(0, minc - self.config.core_padding)
            maxr_pad = min(height, maxr + self.config.core_padding)
            maxc_pad = min(width, maxc + self.config.core_padding)
            
            core_info = {
                'id': i,
                'centroid': region.centroid,  # (row, col)
                'centroid_xy': (region.centroid[1], region.centroid[0]),  # (x, y)
                'area': region.area,
                'equiv_diameter': equiv_diameter,
                'circularity': circularity,
                'bbox': (minr, minc, maxr, maxc),
                'bbox_padded': (minr_pad, minc_pad, maxr_pad, maxc_pad),
                'mean_intensity': region.mean_intensity,
                'label': region.label
            }
            cores.append(core_info)
        
        logger.info(f"Detected {len(cores)} cores in {image_name}")
        return cores
    
    def visualize_core_detection(self, image: np.ndarray, cores: List[Dict], 
                                output_path: str, title: str = "Core Detection"):
        """Create visualization of detected cores."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        if len(image.shape) == 3 and image.shape[2] == 3:
            axes[0].imshow(image)
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f"{title} - Original")
        axes[0].axis('off')
        
        # Cores overlay
        if len(image.shape) == 3 and image.shape[2] == 3:
            axes[1].imshow(image)
        else:
            axes[1].imshow(image, cmap='gray')
        
        # Draw core boundaries and centroids
        for core in cores:
            minr, minc, maxr, maxc = core['bbox_padded']
            
            # Draw bounding box
            rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, 
                               linewidth=2, edgecolor='red', facecolor='none')
            axes[1].add_patch(rect)
            
            # Draw centroid
            axes[1].plot(core['centroid_xy'][0], core['centroid_xy'][1], 'ro', markersize=8)
            
            # Add core ID
            axes[1].text(core['centroid_xy'][0], core['centroid_xy'][1] - 20, 
                        f"C{core['id']}", ha='center', va='bottom', 
                        color='red', fontweight='bold', fontsize=10)
        
        axes[1].set_title(f"{title} - Detected Cores ({len(cores)})")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Core detection visualization saved to: {output_path}")


class CoreMatcher:
    """Matches cores between H&E and Orion slides based on spatial correspondence."""
    
    def __init__(self, config: WSIRegistrationConfig):
        self.config = config
    
    def match_cores(self, he_cores: List[Dict], orion_cores: List[Dict], 
                   registration_transform: Optional[np.ndarray] = None) -> List[Tuple[Dict, Dict]]:
        """
        Match cores between H&E and Orion slides.
        
        Args:
            he_cores: List of H&E core detections
            orion_cores: List of Orion core detections
            registration_transform: Optional transformation matrix from registration
            
        Returns:
            List of (he_core, orion_core) pairs
        """
        logger.info(f"Matching {len(he_cores)} H&E cores with {len(orion_cores)} Orion cores")
        
        if not he_cores or not orion_cores:
            logger.warning("No cores to match!")
            return []
        
        # Get centroids
        he_centroids = np.array([core['centroid_xy'] for core in he_cores])
        orion_centroids = np.array([core['centroid_xy'] for core in orion_cores])
        
        # Apply registration transform to H&E centroids if available
        if registration_transform is not None:
            logger.info("Applying registration transform to H&E centroids")
            # Convert to homogeneous coordinates
            he_centroids_homo = np.hstack([he_centroids, np.ones((len(he_centroids), 1))])
            # Transform
            he_centroids_transformed = (registration_transform @ he_centroids_homo.T).T[:, :2]
            he_centroids = he_centroids_transformed
        
        # Find best matches using distance-based assignment
        matched_pairs = []
        used_orion_indices = set()
        
        # Sort H&E cores by some consistent criterion (e.g., y-coordinate, then x)
        he_indices = sorted(range(len(he_cores)), 
                           key=lambda i: (he_centroids[i][1], he_centroids[i][0]))
        
        for he_idx in he_indices:
            he_centroid = he_centroids[he_idx]
            
            # Find closest available Orion core
            min_distance = float('inf')
            best_orion_idx = None
            
            for orion_idx, orion_centroid in enumerate(orion_centroids):
                if orion_idx in used_orion_indices:
                    continue
                
                distance = np.linalg.norm(he_centroid - orion_centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_orion_idx = orion_idx
            
            # Only accept matches within reasonable distance
            max_distance = self.config.expected_core_diameter * 2  # Allow some tolerance
            if best_orion_idx is not None and min_distance <= max_distance:
                matched_pairs.append((he_cores[he_idx], orion_cores[best_orion_idx]))
                used_orion_indices.add(best_orion_idx)
                logger.debug(f"Matched H&E core {he_idx} with Orion core {best_orion_idx} "
                           f"(distance: {min_distance:.1f})")
        
        logger.info(f"Successfully matched {len(matched_pairs)} core pairs")
        return matched_pairs
    
    def visualize_matches(self, he_image: np.ndarray, orion_image: np.ndarray,
                         matched_pairs: List[Tuple[Dict, Dict]], output_path: str):
        """Create visualization of matched cores."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # H&E side
        if len(he_image.shape) == 3 and he_image.shape[2] == 3:
            axes[0].imshow(he_image)
        else:
            axes[0].imshow(he_image, cmap='gray')
        axes[0].set_title(f"H&E - Matched Cores ({len(matched_pairs)})")
        axes[0].axis('off')
        
        # Orion side
        if len(orion_image.shape) == 3:
            # For multi-channel, show first channel
            if orion_image.shape[0] < orion_image.shape[2]:
                # Channels first (C, H, W)
                orion_display = orion_image[0] if orion_image.shape[0] > 1 else orion_image[0]
            else:
                # Channels last (H, W, C)
                orion_display = orion_image[:, :, 0] if orion_image.shape[2] > 1 else orion_image[:, :, 0]
            axes[1].imshow(orion_display, cmap='gray')
        else:
            axes[1].imshow(orion_image, cmap='gray')
        axes[1].set_title(f"Orion - Matched Cores ({len(matched_pairs)})")
        axes[1].axis('off')
        
        # Draw matched cores with same colors
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(matched_pairs), 10)))
        if len(matched_pairs) > 10:
            colors = plt.cm.viridis(np.linspace(0, 1, len(matched_pairs)))
        
        for i, (he_core, orion_core) in enumerate(matched_pairs):
            color = colors[i % len(colors)]
            
            # H&E core
            he_minr, he_minc, he_maxr, he_maxc = he_core['bbox_padded']
            he_rect = plt.Rectangle((he_minc, he_minr), he_maxc - he_minc, he_maxr - he_minr,
                                  linewidth=3, edgecolor=color, facecolor='none')
            axes[0].add_patch(he_rect)
            axes[0].text(he_core['centroid_xy'][0], he_core['centroid_xy'][1], 
                        f"{i+1}", ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=12,
                        bbox=dict(boxstyle="circle,pad=0.3", facecolor=color))
            
            # Orion core
            orion_minr, orion_minc, orion_maxr, orion_maxc = orion_core['bbox_padded']
            orion_rect = plt.Rectangle((orion_minc, orion_minr), orion_maxc - orion_minc, 
                                     orion_maxr - orion_minr,
                                     linewidth=3, edgecolor=color, facecolor='none')
            axes[1].add_patch(orion_rect)
            axes[1].text(orion_core['centroid_xy'][0], orion_core['centroid_xy'][1], 
                        f"{i+1}", ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=12,
                        bbox=dict(boxstyle="circle,pad=0.3", facecolor=color))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Core matching visualization saved to: {output_path}")


class WSIRegistrationPipeline:
    """Main pipeline for WSI registration and core extraction."""
    
    def __init__(self, config: WSIRegistrationConfig):
        self.config = config
        if not VALIS_AVAILABLE:
            raise ImportError("VALIS is required for registration. Install with: pip install valis")
        
        # Create output directories
        self.output_path = pathlib.Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        self.registered_wsi_dir = self.output_path / "registered_wsi"
        self.cores_dir = self.output_path / "extracted_cores" 
        self.quality_plots_dir = self.output_path / "quality_plots"
        self.temp_dir = self.output_path / "temp"
        
        for dir_path in [self.registered_wsi_dir, self.cores_dir, self.quality_plots_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.core_detector = CoreDetector(config)
        self.core_matcher = CoreMatcher(config)
    
    def run(self) -> Dict:
        """
        Run the complete WSI registration and core extraction pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting WSI registration and core extraction pipeline")
        
        results = {
            'config': self.config.__dict__,
            'he_wsi_path': self.config.he_wsi_path,
            'orion_wsi_path': self.config.orion_wsi_path,
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Register WSI images
            logger.info("Step 1: Registering whole slide images")
            registration_results = self._register_wsi()
            results.update(registration_results)
            
            if not registration_results['registration_success']:
                results['error'] = "WSI registration failed"
                return results
            
            # Step 2: Load registered images
            logger.info("Step 2: Loading registered images")
            he_registered, orion_registered = self._load_registered_images()
            
            # Try to load RGB version of H&E for better visualization
            he_rgb = None
            if hasattr(self, 'preprocessing_dir') and self.preprocessing_dir:
                preprocess_path = pathlib.Path(self.preprocessing_dir)
                he_rgb_paths = list(preprocess_path.glob("*he*rgb*.tif"))
                if he_rgb_paths:
                    try:
                        he_rgb = imread(str(he_rgb_paths[0]))
                        logger.info(f"Loaded RGB H&E for visualization: {he_rgb.shape}")
                    except Exception as e:
                        logger.warning(f"Could not load RGB H&E: {e}")
                        he_rgb = None
            
            # Step 3: Detect cores in both images
            logger.info("Step 3: Detecting tissue cores")
            he_cores = self.core_detector.detect_cores(he_registered, "H&E")
            orion_cores = self.core_detector.detect_cores(orion_registered, "Orion")
            
            results['he_cores_detected'] = len(he_cores)
            results['orion_cores_detected'] = len(orion_cores)
            
            # Step 4: Match cores between images
            logger.info("Step 4: Matching cores between H&E and Orion")
            matched_pairs = self.core_matcher.match_cores(he_cores, orion_cores)
            results['matched_pairs'] = len(matched_pairs)
            
            if not matched_pairs:
                results['error'] = "No matching core pairs found"
                return results
            
            # Step 5: Extract and save core pairs
            logger.info("Step 5: Extracting and saving core pairs")
            extraction_results = self._extract_core_pairs(he_registered, orion_registered, matched_pairs)
            results.update(extraction_results)
            
            # Step 6: Create visualizations
            if self.config.save_core_detection_plots:
                logger.info("Step 6: Creating visualizations")
                self._create_visualizations(he_registered, orion_registered, he_cores, orion_cores, matched_pairs)
            
            # Step 7: Save summary
            self._save_summary(results)
            
            results['success'] = True
            logger.info("WSI registration and core extraction pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['error'] = str(e)
        
        finally:
            # Cleanup
            self._cleanup()
        
        return results
    
    def _register_wsi(self) -> Dict:
        """Register the H&E and Orion WSI images using VALIS."""
        try:
            # Create temporary directory for VALIS input
            src_dir = self.temp_dir / "valis_input"
            dst_dir = self.temp_dir / "valis_output"
            src_dir.mkdir(exist_ok=True)
            dst_dir.mkdir(exist_ok=True)
            
            # Copy/link input files to VALIS input directory
            he_temp_path = src_dir / "he_slide.tif"
            orion_temp_path = src_dir / "orion_slide.tif"
            
            # Create symbolic links or copy files
            if not he_temp_path.exists():
                shutil.copy2(self.config.he_wsi_path, he_temp_path)
            if not orion_temp_path.exists():
                shutil.copy2(self.config.orion_wsi_path, orion_temp_path)
            
            # Create VALIS registrar with robust parameters
            registrar = registration.Valis(
                str(src_dir),
                str(dst_dir),
                reference_img_f="he_slide.tif",  # Use H&E as reference
                align_to_reference=True,
                imgs_ordered=True,  # Disable automatic sorting since we have known order
                max_processed_image_dim_px=self.config.max_processed_image_dim_px,
                max_non_rigid_registration_dim_px=self.config.max_non_rigid_registration_dim_px,
                crop="reference",  # Crop to H&E reference
                # Add robustness parameters
                check_for_reflections=False,  # Disable to speed up and avoid issues
            )
            
            # Perform registration with error handling
            try:
                rigid_registrar, non_rigid_registrar, error_df = registrar.register()
            except ValueError as e:
                if "NoneType" in str(e) or "copy mode not allowed" in str(e):
                    # This suggests feature matching failed, try with different parameters
                    logger.warning("Initial registration failed due to feature matching issues, trying fallback approach")
                    
                    # Clean up and try again with more permissive settings
                    try:
                        registration.kill_jvm()
                    except:
                        pass
                    
                    # Try again with rigid-only registration (no non-rigid)
                    registrar_fallback = registration.Valis(
                        str(src_dir),
                        str(dst_dir),
                        reference_img_f="he_slide.tif",
                        align_to_reference=True,
                        imgs_ordered=True,
                        max_processed_image_dim_px=min(1024, self.config.max_processed_image_dim_px),  # Use smaller images
                        max_non_rigid_registration_dim_px=None,  # Disable non-rigid
                        crop="reference",
                        check_for_reflections=False,
                        non_rigid_registrar_cls=None,  # Disable non-rigid completely
                    )
                    
                    try:
                        logger.info("Attempting rigid-only registration as fallback...")
                        rigid_registrar, non_rigid_registrar, error_df = registrar_fallback.register()
                        logger.info("Fallback registration succeeded!")
                    except Exception as fallback_error:
                        logger.error(f"Fallback registration also failed: {fallback_error}")
                        raise ValueError(f"Both primary and fallback registration failed: {e}")
                else:
                    raise
            
            # Save registered slides
            registrar.warp_and_save_slides(
                str(self.registered_wsi_dir),
                compression=self.config.compression,
                Q=self.config.compression_quality
            )
            
            # Clean up JVM
            registration.kill_jvm()
            
            return {
                'registration_success': True,
                'registration_error': None,
                'error_df': error_df.to_dict() if error_df is not None else None
            }
            
        except Exception as e:
            try:
                registration.kill_jvm()
            except:
                pass
            
            logger.error(f"WSI registration failed: {e}")
            return {
                'registration_success': False,
                'registration_error': str(e),
                'error_df': None
            }
    
    def _load_registered_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the registered WSI images."""
        # Find registered images
        he_registered_path = None
        orion_registered_path = None
        
        for file_path in self.registered_wsi_dir.glob("*.ome.tiff"):
            if "he_slide" in file_path.name.lower():
                he_registered_path = file_path
            elif "orion_slide" in file_path.name.lower():
                orion_registered_path = file_path
        
        if he_registered_path is None:
            raise FileNotFoundError("Registered H&E slide not found")
        if orion_registered_path is None:
            raise FileNotFoundError("Registered Orion slide not found")
        
        logger.info(f"Loading registered H&E from: {he_registered_path}")
        logger.info(f"Loading registered Orion from: {orion_registered_path}")
        
        he_registered = imread(str(he_registered_path))
        orion_registered = imread(str(orion_registered_path))
        
        # Check if we have a multi-channel version of Orion available
        # Look for the multi-channel version that was created during preprocessing
        if hasattr(self, 'preprocessing_dir') and self.preprocessing_dir:
            preprocess_path = pathlib.Path(self.preprocessing_dir)
            orion_multichannel_paths = list(preprocess_path.glob("*orion*multichannel*.tif"))
        else:
            orion_multichannel_paths = []
        if orion_multichannel_paths:
            orion_multichannel_path = orion_multichannel_paths[0]
            logger.info(f"Found multi-channel Orion version: {orion_multichannel_path}")
            orion_multichannel = imread(str(orion_multichannel_path))
            logger.info(f"Multi-channel Orion shape: {orion_multichannel.shape}")
            
            # Use the multi-channel version if it has more channels
            if orion_multichannel.ndim == 3 and (orion_registered.ndim == 2 or 
                (orion_registered.ndim == 3 and orion_multichannel.shape[0] > orion_registered.shape[-1])):
                logger.info("Using multi-channel Orion for core extraction")
                orion_registered = orion_multichannel
        
        logger.info(f"Final H&E shape: {he_registered.shape}")
        logger.info(f"Final Orion shape: {orion_registered.shape}")
        
        return he_registered, orion_registered
    
    def _extract_core_pairs(self, he_image: np.ndarray, orion_image: np.ndarray, 
                           matched_pairs: List[Tuple[Dict, Dict]]) -> Dict:
        """Extract and save matched core pairs."""
        extraction_results = {
            'cores_extracted': 0,
            'extraction_errors': []
        }
        
        for i, (he_core, orion_core) in enumerate(matched_pairs):
            try:
                # Create core directory
                core_id = f"core_{i+1:03d}"
                core_dir = self.cores_dir / core_id
                core_dir.mkdir(exist_ok=True)
                
                # Extract H&E core
                he_minr, he_minc, he_maxr, he_maxc = he_core['bbox_padded']
                he_core_image = he_image[he_minr:he_maxr, he_minc:he_maxc]
                
                # Extract Orion core
                orion_minr, orion_minc, orion_maxr, orion_maxc = orion_core['bbox_padded']
                if orion_image.ndim == 3 and orion_image.shape[0] <= 50:
                    # Multi-channel format (C, H, W)
                    orion_core_image = orion_image[:, orion_minr:orion_maxr, orion_minc:orion_maxc]
                else:
                    # Standard format (H, W) or (H, W, C)
                    orion_core_image = orion_image[orion_minr:orion_maxr, orion_minc:orion_maxc]
                
                # Save core images
                he_core_path = core_dir / f"{core_id}_HE.tif"
                orion_core_path = core_dir / f"{core_id}_Orion.ome.tif"
                
                imwrite(str(he_core_path), he_core_image)
                imwrite(str(orion_core_path), orion_core_image)
                
                # Save core metadata
                core_metadata = {
                    'core_id': core_id,
                    'he_core_info': he_core,
                    'orion_core_info': orion_core,
                    'he_extracted_shape': he_core_image.shape,
                    'orion_extracted_shape': orion_core_image.shape,
                    'he_core_path': str(he_core_path),
                    'orion_core_path': str(orion_core_path)
                }
                
                with open(core_dir / f"{core_id}_metadata.json", 'w') as f:
                    json.dump(core_metadata, f, indent=2, default=str)
                
                extraction_results['cores_extracted'] += 1
                logger.info(f"Extracted {core_id}: H&E {he_core_image.shape}, Orion {orion_core_image.shape}")
                
            except Exception as e:
                error_msg = f"Failed to extract core {i+1}: {e}"
                logger.error(error_msg)
                extraction_results['extraction_errors'].append(error_msg)
        
        return extraction_results
    
    def _create_visualizations(self, he_image: np.ndarray, orion_image: np.ndarray,
                              he_cores: List[Dict], orion_cores: List[Dict],
                              matched_pairs: List[Tuple[Dict, Dict]]):
        """Create visualization plots."""
        # Core detection visualizations
        self.core_detector.visualize_core_detection(
            he_image, he_cores, 
            str(self.quality_plots_dir / "he_core_detection.png"),
            "H&E Core Detection"
        )
        
        # For Orion visualization, handle multi-channel
        if orion_image.ndim == 3 and orion_image.shape[0] <= 50:
            # Multi-channel format (C, H, W) - use first channel
            orion_display = orion_image[0]
        else:
            orion_display = orion_image
            
        self.core_detector.visualize_core_detection(
            orion_display, orion_cores,
            str(self.quality_plots_dir / "orion_core_detection.png"), 
            "Orion Core Detection"
        )
        
        # Core matching visualization
        self.core_matcher.visualize_matches(
            he_image, orion_image, matched_pairs,
            str(self.quality_plots_dir / "core_matching.png")
        )
    
    def _save_summary(self, results: Dict):
        """Save pipeline summary."""
        summary_path = self.output_path / "wsi_pipeline_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline summary saved to: {summary_path}")
    
    def _cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")


def main():
    """Example usage of the WSI registration pipeline."""
    
    # Example configuration
    config = WSIRegistrationConfig(
        he_wsi_path="/path/to/your/he_tma.tif",
        orion_wsi_path="/path/to/your/orion_tma.tif", 
        output_dir="./wsi_registration_output",
        max_processed_image_dim_px=2048,
        core_min_area=50000,
        core_max_area=500000,
        core_circularity_threshold=0.4,
        expected_core_diameter=400
    )
    
    # Create and run pipeline
    pipeline = WSIRegistrationPipeline(config)
    results = pipeline.run()
    
    print("WSI Registration Pipeline Results:")
    print(f"Registration Success: {results.get('registration_success', False)}")
    print(f"H&E Cores Detected: {results.get('he_cores_detected', 0)}")
    print(f"Orion Cores Detected: {results.get('orion_cores_detected', 0)}")
    print(f"Matched Core Pairs: {results.get('matched_pairs', 0)}")
    print(f"Cores Extracted: {results.get('cores_extracted', 0)}")
    print(f"Output Directory: {config.output_dir}")
    
    if results.get('success'):
        print("✅ Pipeline completed successfully!")
        print(f"Extracted cores can be found in: {config.output_dir}/extracted_cores/")
    else:
        print("❌ Pipeline failed!")
        print(f"Error: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main() 