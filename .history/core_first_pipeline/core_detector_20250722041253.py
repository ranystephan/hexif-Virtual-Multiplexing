"""
Core Detection Module for TMA Processing

This module provides advanced core detection algorithms specifically designed for
Tissue Microarray (TMA) analysis. It can handle both H&E and multi-channel
fluorescence images (Orion, CODEX, etc.) and extract individual tissue cores
while preserving all channel information.

Key Features:
- Robust core detection using multiple methods (morphology, Hough circles, contours)
- Multi-channel image support with channel preservation
- Adaptive parameter selection based on image characteristics
- Quality filtering and validation
- Visualization and diagnostic tools
"""

import numpy as np
import cv2
from skimage import filters, morphology, measure, feature, segmentation
from skimage.restoration import rolling_ball
from scipy import ndimage, spatial
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoreDetectionConfig:
    """Configuration for core detection."""
    
    # Core size parameters (in pixels at full resolution)
    min_core_area: int = 30000      # Minimum area for a valid core
    max_core_area: int = 800000     # Maximum area for a valid core  
    min_core_diameter: int = 150    # Minimum diameter in pixels
    max_core_diameter: int = 1000   # Maximum diameter in pixels
    
    # Image downsampling for detection (NEW)
    max_detection_dimension: int = 4000  # Maximum dimension for detection processing
    detection_downsample_method: str = "area"  # "area", "linear", "cubic"
    
    # Shape filtering
    min_circularity: float = 0.3    # Minimum circularity (4π*area/perimeter²)
    min_solidity: float = 0.6       # Minimum solidity (area/convex_hull_area)
    aspect_ratio_threshold: float = 2.5  # Maximum width/height ratio
    
    # Image preprocessing
    gaussian_sigma: float = 1.5     # Gaussian smoothing sigma
    rolling_ball_radius: int = 50   # Background subtraction radius
    remove_small_objects_size: int = 1000  # Remove objects smaller than this
    
    # Detection method
    detection_method: str = "hybrid"  # "morphology", "hough", "contours", "hybrid"
    
    # Hough circle parameters (for method="hough")
    hough_dp: float = 1            # Inverse ratio of accumulator resolution
    hough_min_dist: int = 200      # Minimum distance between circle centers
    hough_param1: int = 50         # Upper threshold for edge detection
    hough_param2: int = 30         # Accumulator threshold for center detection
    
    # Morphology parameters
    morphology_iterations: int = 2  # Morphological operation iterations
    watershed_threshold: float = 0.6  # Threshold for watershed segmentation
    
    # Quality control
    min_mean_intensity: float = 5.0   # Minimum mean intensity for valid cores
    max_background_ratio: float = 0.8 # Maximum ratio of background pixels
    
    # Output
    core_padding: int = 20          # Padding around extracted cores
    save_intermediate_images: bool = False
    create_visualizations: bool = True


class CoreDetector:
    """Advanced core detector for TMA images."""
    
    def __init__(self, config: CoreDetectionConfig):
        self.config = config
    
    def detect_cores(self, image_path: str, image_type: str = "auto") -> Dict:
        """
        Detect cores in a TMA image.
        
        Args:
            image_path: Path to the image file
            image_type: Type of image ("he", "orion", "auto")
            
        Returns:
            Dictionary containing detected cores and metadata
        """
        logger.info(f"Detecting cores in {image_path}")
        
        # Load and preprocess image
        image = imread(image_path)
        original_shape = image.shape
        
        # Determine image type if auto
        if image_type == "auto":
            image_type = self._determine_image_type(image, image_path)
        
        logger.info(f"Image type: {image_type}, shape: {original_shape}, dtype: {image.dtype}")
        
        # Get detection channel (grayscale representation for detection)
        detection_image = self._prepare_detection_image(image, image_type)
        
        # Apply intelligent downsampling if image is too large
        downsample_factor = 1.0
        if max(detection_image.shape) > self.config.max_detection_dimension:
            downsample_factor = self.config.max_detection_dimension / max(detection_image.shape)
            logger.info(f"Image is large ({max(detection_image.shape)}px), downsampling by factor {downsample_factor:.3f} for detection")
            
            # Downsample image for detection
            new_height = int(detection_image.shape[0] * downsample_factor)
            new_width = int(detection_image.shape[1] * downsample_factor)
            
            if self.config.detection_downsample_method == "area":
                detection_image_small = cv2.resize(detection_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            elif self.config.detection_downsample_method == "linear":
                detection_image_small = cv2.resize(detection_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            elif self.config.detection_downsample_method == "cubic":
                detection_image_small = cv2.resize(detection_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            else:
                detection_image_small = cv2.resize(detection_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            logger.info(f"Downsampled detection image to {detection_image_small.shape}")
        else:
            detection_image_small = detection_image
        
        # Apply detection algorithm on potentially downsampled image
        cores = self._detect_cores_with_method(detection_image_small, self.config.detection_method)
        
        # Scale core coordinates and sizes back to full resolution
        if downsample_factor != 1.0:
            cores = self._scale_cores_to_full_resolution(cores, downsample_factor)
        
        # Filter and validate cores using full-resolution parameters
        filtered_cores = self._filter_cores(cores, detection_image)
        
        # Add metadata
        results = {
            'image_path': str(image_path),
            'image_type': image_type,
            'original_shape': original_shape,
            'detection_image_shape': detection_image.shape,
            'downsample_factor': downsample_factor,
            'detection_processed_shape': detection_image_small.shape,
            'total_cores_detected': len(cores),
            'filtered_cores_count': len(filtered_cores),
            'cores': filtered_cores,
            'detection_method': self.config.detection_method,
            'config': self.config.__dict__
        }
        
        logger.info(f"Detected {len(filtered_cores)} valid cores from {len(cores)} candidates")
        return results
    
    def _determine_image_type(self, image: np.ndarray, image_path: str) -> str:
        """Automatically determine image type based on characteristics."""
        path_lower = str(image_path).lower()
        
        if 'he' in path_lower or 'h&e' in path_lower:
            return 'he'
        elif 'orion' in path_lower or 'codex' in path_lower:
            return 'orion'
        
        # Analyze image characteristics
        if image.ndim == 3:
            if image.shape[0] > 10:  # Many channels, likely multi-channel fluorescence
                return 'orion'
            elif image.shape[2] == 3:  # RGB, likely H&E
                return 'he'
        
        # Default to H&E for 2D images
        return 'he'
    
    def _prepare_detection_image(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """Prepare image for core detection by converting to optimal grayscale representation."""
        
        if image_type == 'he':
            if image.ndim == 3 and image.shape[2] == 3:
                # RGB H&E - convert to grayscale using weighted average
                # Use custom weights that work well for H&E
                detection_img = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            elif image.ndim == 2:
                detection_img = image.copy()
            else:
                # Fallback: use first channel
                detection_img = image[0] if image.ndim == 3 and image.shape[0] < image.shape[2] else image[:, :, 0]
        
        elif image_type == 'orion':
            if image.ndim == 3 and image.shape[0] > image.shape[2]:
                # Multi-channel format (C, H, W) - use DAPI channel (typically channel 0)
                detection_img = image[0]
                logger.info("Using channel 0 (likely DAPI) for Orion core detection")
            elif image.ndim == 3:
                # (H, W, C) format
                detection_img = image[:, :, 0]
            else:
                detection_img = image
        
        else:
            # Default: use first channel or convert to grayscale
            if image.ndim == 3:
                if image.shape[0] < image.shape[2]:
                    detection_img = image[0]
                else:
                    detection_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
            else:
                detection_img = image.copy()
        
        # Ensure proper data type and range
        if detection_img.dtype != np.uint8:
            detection_img = cv2.normalize(detection_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return detection_img
    
    def _scale_cores_to_full_resolution(self, cores: List[Dict], downsample_factor: float) -> List[Dict]:
        """Scale core coordinates and sizes from detection resolution to full resolution."""
        
        scale_factor = 1.0 / downsample_factor  # Scale back up
        
        scaled_cores = []
        for core in cores:
            scaled_core = core.copy()
            
            # Scale coordinates
            if 'centroid' in scaled_core:
                cy, cx = scaled_core['centroid']
                scaled_core['centroid'] = (cy * scale_factor, cx * scale_factor)
            
            if 'centroid_xy' in scaled_core:
                cx, cy = scaled_core['centroid_xy']
                scaled_core['centroid_xy'] = (cx * scale_factor, cy * scale_factor)
            
            # Scale bounding box
            if 'bbox' in scaled_core:
                minr, minc, maxr, maxc = scaled_core['bbox']
                scaled_core['bbox'] = (
                    int(minr * scale_factor), 
                    int(minc * scale_factor),
                    int(maxr * scale_factor), 
                    int(maxc * scale_factor)
                )
            
            # Scale sizes
            if 'area' in scaled_core:
                scaled_core['area'] = scaled_core['area'] * (scale_factor ** 2)
            
            if 'equiv_diameter' in scaled_core:
                scaled_core['equiv_diameter'] = scaled_core['equiv_diameter'] * scale_factor
            
            if 'perimeter' in scaled_core:
                scaled_core['perimeter'] = scaled_core['perimeter'] * scale_factor
            
            if 'radius' in scaled_core:  # For Hough circles
                scaled_core['radius'] = scaled_core['radius'] * scale_factor
            
            # Circularity and solidity remain unchanged (they're ratios)
            
            scaled_cores.append(scaled_core)
        
        logger.debug(f"Scaled {len(cores)} cores from detection resolution to full resolution (scale factor: {scale_factor:.3f})")
        return scaled_cores
    
    def _detect_cores_with_method(self, image: np.ndarray, method: str) -> List[Dict]:
        """Apply the specified detection method."""
        
        if method == "morphology":
            return self._detect_cores_morphology(image)
        elif method == "hough":
            return self._detect_cores_hough(image)
        elif method == "contours":
            return self._detect_cores_contours(image)
        elif method == "hybrid":
            return self._detect_cores_hybrid(image)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _detect_cores_morphology(self, image: np.ndarray) -> List[Dict]:
        """Detect cores using morphological operations and watershed segmentation."""
        
        # Preprocessing
        smoothed = filters.gaussian(image, sigma=self.config.gaussian_sigma)
        
        # Background subtraction using rolling ball
        if self.config.rolling_ball_radius > 0:
            background = rolling_ball(smoothed, radius=self.config.rolling_ball_radius)
            smoothed = smoothed - background
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
        
        # Threshold using multiple methods and combine
        otsu_thresh = filters.threshold_otsu(smoothed)
        local_thresh = filters.threshold_local(smoothed, block_size=201)
        
        # For tissue detection, use inverted threshold (tissue is darker)
        binary_otsu = smoothed > otsu_thresh
        binary_local = smoothed > local_thresh
        binary = binary_otsu & binary_local
        
        # Morphological cleaning
        binary = morphology.remove_small_objects(binary, min_size=self.config.remove_small_objects_size)
        binary = morphology.binary_fill_holes(binary)
        
        # Apply morphological operations
        selem = morphology.disk(self.config.morphology_iterations * 3)
        binary = morphology.opening(binary, selem)
        binary = morphology.closing(binary, selem)
        
        # Watershed segmentation to separate touching cores
        distance = ndimage.distance_transform_edt(binary)
        local_maxima = feature.peak_local_maxima(
            distance, 
            min_distance=self.config.min_core_diameter // 2,
            threshold_abs=self.config.watershed_threshold * distance.max()
        )
        
        markers = np.zeros_like(binary, dtype=int)
        markers[tuple(local_maxima)] = np.arange(1, len(local_maxima[0]) + 1)
        
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
        # Extract core properties
        regions = measure.regionprops(labels, intensity_image=image)
        cores = []
        
        for region in regions:
            core_info = self._extract_core_properties(region, image)
            cores.append(core_info)
        
        return cores
    
    def _detect_cores_hough(self, image: np.ndarray) -> List[Dict]:
        """Detect cores using Hough circle transform."""
        
        # Preprocessing for circle detection
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.config.hough_dp,
            minDist=self.config.hough_min_dist,
            param1=self.config.hough_param1,
            param2=self.config.hough_param2,
            minRadius=self.config.min_core_diameter // 2,
            maxRadius=self.config.max_core_diameter // 2
        )
        
        cores = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Create circular mask and extract properties
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # Calculate properties
                area = np.pi * r * r
                perimeter = 2 * np.pi * r
                circularity = 1.0  # Perfect circle by definition
                
                # Extract intensity properties
                masked_region = np.ma.masked_where(mask == 0, image)
                mean_intensity = np.mean(masked_region)
                
                core_info = {
                    'centroid': (y, x),  # (row, col)
                    'centroid_xy': (x, y),  # (x, y)
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'equiv_diameter': 2 * r,
                    'bbox': (max(0, y-r), max(0, x-r), min(image.shape[0], y+r), min(image.shape[1], x+r)),
                    'mean_intensity': mean_intensity,
                    'radius': r,
                    'solidity': 1.0,  # Approximation for circle
                    'aspect_ratio': 1.0  # Circle has aspect ratio of 1
                }
                cores.append(core_info)
        
        return cores
    
    def _detect_cores_contours(self, image: np.ndarray) -> List[Dict]:
        """Detect cores using contour detection."""
        
        # Preprocessing
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Multiple thresholding approaches
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
        
        # Combine thresholds
        combined_thresh = cv2.bitwise_and(thresh_otsu, thresh_adaptive)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cores = []
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            if area < self.config.min_core_area or area > self.config.max_core_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Shape analysis
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            
            # Convex hull for solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Moments for centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Mean intensity
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            mean_intensity = cv2.mean(image, mask=mask)[0]
            
            core_info = {
                'centroid': (cy, cx),  # (row, col)
                'centroid_xy': (cx, cy),  # (x, y)
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'equiv_diameter': np.sqrt(4 * area / np.pi),
                'bbox': (y, x, y + h, x + w),
                'mean_intensity': mean_intensity,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'contour': contour
            }
            cores.append(core_info)
        
        return cores
    
    def _detect_cores_hybrid(self, image: np.ndarray) -> List[Dict]:
        """Detect cores using a hybrid approach combining multiple methods."""
        
        # Get detections from multiple methods
        morphology_cores = self._detect_cores_morphology(image)
        hough_cores = self._detect_cores_hough(image)
        contour_cores = self._detect_cores_contours(image)
        
        logger.info(f"Detection results - Morphology: {len(morphology_cores)}, Hough: {len(hough_cores)}, Contours: {len(contour_cores)}")
        
        # Combine all detections
        all_cores = morphology_cores + hough_cores + contour_cores
        
        if not all_cores:
            return []
        
        # Remove duplicates by clustering nearby detections
        positions = np.array([core['centroid_xy'] for core in all_cores])
        
        # Use DBSCAN clustering to group nearby detections
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=self.config.min_core_diameter * 0.3, min_samples=1)
        clusters = clustering.fit_predict(positions)
        
        # For each cluster, select the best detection
        unique_cores = []
        for cluster_id in np.unique(clusters):
            cluster_cores = [all_cores[i] for i in range(len(all_cores)) if clusters[i] == cluster_id]
            
            # Select core with best circularity
            best_core = max(cluster_cores, key=lambda x: x.get('circularity', 0))
            unique_cores.append(best_core)
        
        logger.info(f"After deduplication: {len(unique_cores)} unique cores")
        return unique_cores
    
    def _extract_core_properties(self, region, image: np.ndarray) -> Dict:
        """Extract comprehensive properties from a detected region."""
        
        # Basic properties
        area = region.area
        perimeter = region.perimeter if region.perimeter > 0 else 1
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        equiv_diameter = region.equivalent_diameter
        
        # Shape properties
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        height = maxr - minr
        aspect_ratio = max(width, height) / min(width, height)
        
        return {
            'centroid': region.centroid,  # (row, col)
            'centroid_xy': (region.centroid[1], region.centroid[0]),  # (x, y)
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'equiv_diameter': equiv_diameter,
            'bbox': region.bbox,
            'mean_intensity': region.mean_intensity,
            'solidity': region.solidity,
            'aspect_ratio': aspect_ratio,
            'label': region.label
        }
    
    def _filter_cores(self, cores: List[Dict], image: np.ndarray) -> List[Dict]:
        """Filter detected cores based on quality criteria."""
        
        filtered_cores = []
        
        for core in cores:
            # Size filters
            if not (self.config.min_core_area <= core['area'] <= self.config.max_core_area):
                continue
            
            if core['equiv_diameter'] < self.config.min_core_diameter:
                continue
            
            # Shape filters
            if core['circularity'] < self.config.min_circularity:
                continue
            
            if core.get('solidity', 1.0) < self.config.min_solidity:
                continue
            
            if core.get('aspect_ratio', 1.0) > self.config.aspect_ratio_threshold:
                continue
            
            # Intensity filters
            if core['mean_intensity'] < self.config.min_mean_intensity:
                continue
            
            # Add padding to bounding box
            minr, minc, maxr, maxc = core['bbox']
            height, width = image.shape[:2]
            
            padded_bbox = (
                max(0, minr - self.config.core_padding),
                max(0, minc - self.config.core_padding),
                min(height, maxr + self.config.core_padding),
                min(width, maxc + self.config.core_padding)
            )
            
            core['bbox_padded'] = padded_bbox
            filtered_cores.append(core)
        
        # Sort by area (largest first) for consistent ordering
        filtered_cores.sort(key=lambda x: x['area'], reverse=True)
        
        # Add sequential IDs
        for i, core in enumerate(filtered_cores):
            core['id'] = i
        
        return filtered_cores
    
    def visualize_detection(self, results: Dict, output_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of core detection results."""
        
        image_path = results['image_path']
        cores = results['cores']
        
        # Load original image for visualization
        original_image = imread(image_path)
        
        # Prepare display image
        if original_image.ndim == 3:
            if original_image.shape[0] <= 10:  # Multi-channel (C, H, W)
                # Use first 3 channels or repeat first channel
                if original_image.shape[0] >= 3:
                    display_img = np.transpose(original_image[:3], (1, 2, 0))
                else:
                    display_img = np.stack([original_image[0]] * 3, axis=2)
            else:  # RGB (H, W, C)
                display_img = original_image
        else:
            # Grayscale - convert to RGB
            display_img = np.stack([original_image] * 3, axis=2)
        
        # Normalize for display
        if display_img.dtype != np.uint8:
            display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        axes[0].imshow(display_img)
        axes[0].set_title(f'Original Image\nShape: {original_image.shape}')
        axes[0].axis('off')
        
        # Detected cores overlay
        overlay = display_img.copy()
        
        for core in cores:
            minr, minc, maxr, maxc = core['bbox_padded']
            
            # Draw bounding box
            cv2.rectangle(overlay, (minc, minr), (maxc, maxr), (0, 255, 0), 3)
            
            # Draw centroid
            cx, cy = int(core['centroid_xy'][0]), int(core['centroid_xy'][1])
            cv2.circle(overlay, (cx, cy), 8, (255, 0, 0), -1)
            
            # Add core ID and metrics
            text = f"C{core['id']}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(overlay, (cx - text_size[0]//2 - 5, cy - 25), 
                         (cx + text_size[0]//2 + 5, cy - 5), (255, 255, 255), -1)
            cv2.putText(overlay, text, (cx - text_size[0]//2, cy - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        axes[1].imshow(overlay)
        axes[1].set_title(f'Detected Cores: {len(cores)}\nMethod: {results["detection_method"]}')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        return fig


def main():
    """Example usage of the CoreDetector."""
    
    # Configuration
    config = CoreDetectionConfig(
        detection_method="hybrid",
        min_core_area=30000,
        max_core_area=800000,
        min_circularity=0.3,
        create_visualizations=True
    )
    
    # Create detector
    detector = CoreDetector(config)
    
    # Example paths (update these to your actual paths)
    he_path = "data/raw/TA118-HEraw.ome.tiff"
    orion_path = "data/raw/TA118-Orionraw.ome.tiff"
    
    # Detect cores in H&E
    if Path(he_path).exists():
        he_results = detector.detect_cores(he_path, image_type="he")
        print(f"H&E: Detected {he_results['filtered_cores_count']} cores")
        
        if config.create_visualizations:
            detector.visualize_detection(he_results, "he_cores_detected.png")
    
    # Detect cores in Orion
    if Path(orion_path).exists():
        orion_results = detector.detect_cores(orion_path, image_type="orion")
        print(f"Orion: Detected {orion_results['filtered_cores_count']} cores")
        
        if config.create_visualizations:
            detector.visualize_detection(orion_results, "orion_cores_detected.png")


if __name__ == "__main__":
    main() 