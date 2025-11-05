"""
TMA Grid-Based Core Detection

This module implements a grid-aware approach to TMA core detection that leverages
the known regular structure of tissue microarrays. Instead of trying to detect
individual cores using traditional computer vision, this approach:

1. Identifies the overall grid structure and spacing
2. Uses template matching and autocorrelation to find the grid pattern
3. Extracts cores based on predicted grid positions
4. Handles both H&E and multi-channel fluorescence images
5. Maintains grid correspondence between different stains

This approach is fundamentally more reliable for TMAs because it uses the
inherent structure rather than fighting against it.

Key Features:
- Grid pattern detection using autocorrelation and FFT
- Template-based core identification
- Automatic spacing and size estimation
- Multi-scale analysis for robust detection
- Grid coordinate system for cross-stain matching
"""

import numpy as np
import cv2
from skimage import filters, morphology, measure, feature
from skimage.feature import match_template, peak_local_maxima
from scipy import ndimage, fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TMAGridConfig:
    """Configuration for TMA grid detection."""
    
    # Grid detection parameters
    expected_core_diameter_range: Tuple[int, int] = (800, 1200)  # Expected core diameter in pixels
    expected_spacing_range: Tuple[int, int] = (1000, 1600)      # Expected center-to-center spacing
    grid_detection_downsample: int = 8                          # Downsample factor for grid detection
    
    # Template matching
    template_size_range: Tuple[int, int] = (100, 200)          # Template size range (at detection resolution)
    template_match_threshold: float = 0.4                      # Minimum template matching score
    
    # Autocorrelation parameters
    autocorr_peak_threshold: float = 0.3                       # Minimum autocorrelation peak height
    autocorr_peak_prominence: float = 0.1                      # Minimum peak prominence
    
    # Grid validation
    min_grid_regularity: float = 0.7                           # Minimum grid regularity score
    max_angle_deviation: float = 10.0                          # Maximum angle deviation (degrees)
    
    # Core extraction
    core_extraction_padding: int = 50                          # Padding around each core
    min_tissue_coverage: float = 0.3                           # Minimum tissue coverage in core
    
    # Quality control
    expected_core_count_range: Tuple[int, int] = (200, 400)    # Expected number of cores
    enable_visualizations: bool = True
    save_debug_images: bool = False


class TMAGridDetector:
    """Grid-aware TMA core detector."""
    
    def __init__(self, config: TMAGridConfig):
        self.config = config
        
    def detect_tma_grid(self, image_path: str, image_type: str = "auto") -> Dict:
        """
        Detect TMA grid and extract all cores.
        
        Args:
            image_path: Path to the TMA image
            image_type: Type of image ("he", "orion", "auto")
            
        Returns:
            Dictionary with grid information and detected cores
        """
        logger.info(f"Detecting TMA grid in {image_path}")
        
        # Load and prepare image
        image = self._load_image(image_path, image_type)
        detection_image = self._prepare_detection_image(image, image_type)
        
        logger.info(f"Image loaded: {image.shape}, detection image: {detection_image.shape}")
        
        # Downsample for grid detection
        downsample_factor = self.config.grid_detection_downsample
        detection_small = self._downsample_image(detection_image, downsample_factor)
        
        logger.info(f"Detection image downsampled to: {detection_small.shape} (factor: {downsample_factor})")
        
        # Step 1: Estimate grid parameters using autocorrelation
        grid_params = self._estimate_grid_parameters(detection_small)
        
        if grid_params is None:
            logger.error("Failed to estimate grid parameters")
            return self._create_empty_result(image_path, image_type, image.shape)
        
        logger.info(f"Estimated grid parameters: spacing=({grid_params['spacing_x']:.1f}, {grid_params['spacing_y']:.1f}), "
                   f"angle={grid_params['angle']:.1f}°")
        
        # Step 2: Create template from detected cores
        template = self._create_core_template(detection_small, grid_params)
        
        if template is None:
            logger.error("Failed to create core template")
            return self._create_empty_result(image_path, image_type, image.shape)
        
        # Step 3: Find all grid positions using template matching
        grid_positions = self._find_grid_positions(detection_small, template, grid_params)
        
        logger.info(f"Found {len(grid_positions)} potential grid positions")
        
        # Step 4: Validate and refine grid
        validated_positions = self._validate_grid_positions(detection_image, grid_positions, downsample_factor)
        
        logger.info(f"Validated {len(validated_positions)} grid positions")
        
        # Step 5: Extract core information
        cores = self._extract_cores_from_grid(image, validated_positions, image_type)
        
        # Create results
        results = {
            'image_path': str(image_path),
            'image_type': image_type,
            'original_shape': image.shape,
            'detection_method': 'tma_grid',
            'grid_parameters': grid_params,
            'downsample_factor': downsample_factor,
            'total_positions_found': len(grid_positions),
            'validated_positions': len(validated_positions),
            'cores_extracted': len(cores),
            'cores': cores,
            'success': len(cores) > 0
        }
        
        logger.info(f"TMA grid detection complete: {len(cores)} cores extracted")
        return results
    
    def _load_image(self, image_path: str, image_type: str) -> np.ndarray:
        """Load image with proper error handling."""
        try:
            image = imread(image_path)
            if image is None or image.size == 0:
                raise ValueError(f"Empty or corrupted image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _prepare_detection_image(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """Prepare image for grid detection."""
        
        if image_type == "he":
            if image.ndim == 3 and image.shape[2] == 3:
                # RGB H&E - convert to grayscale
                detection_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.ndim == 3 and image.shape[0] == 3:
                # Channel-first RGB
                detection_img = cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
            else:
                detection_img = image.squeeze()
        
        elif image_type == "orion":
            if image.ndim == 3 and image.shape[0] > 10:
                # Multi-channel Orion - use DAPI (channel 0)
                detection_img = image[0]
                logger.info("Using DAPI channel (0) for Orion grid detection")
            else:
                detection_img = image.squeeze()
        
        else:
            # Auto-detect or fallback
            if image.ndim == 3:
                if image.shape[0] > 10:  # Multi-channel
                    detection_img = image[0]
                elif image.shape[2] == 3:  # RGB
                    detection_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    detection_img = image[:, :, 0]
            else:
                detection_img = image
        
        # Ensure proper data type
        if detection_img.dtype != np.uint8:
            detection_img = cv2.normalize(detection_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return detection_img
    
    def _downsample_image(self, image: np.ndarray, factor: int) -> np.ndarray:
        """Downsample image for efficient processing."""
        new_height = image.shape[0] // factor
        new_width = image.shape[1] // factor
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _estimate_grid_parameters(self, image: np.ndarray) -> Optional[Dict]:
        """
        Estimate grid parameters using autocorrelation and FFT analysis.
        
        This method uses the regular structure of TMAs to automatically
        determine the grid spacing and orientation.
        """
        logger.info("Estimating grid parameters using autocorrelation...")
        
        # Preprocess image for better autocorrelation
        # Apply slight Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(blurred)
        
        # Compute 2D autocorrelation using FFT
        try:
            autocorr = self._compute_2d_autocorrelation(enhanced)
        except Exception as e:
            logger.error(f"Failed to compute autocorrelation: {e}")
            return None
        
        # Find peaks in autocorrelation
        peaks = self._find_autocorrelation_peaks(autocorr)
        
        if len(peaks) < 2:
            logger.warning("Insufficient peaks found in autocorrelation")
            return None
        
        # Estimate spacing from peaks
        spacings = self._estimate_spacing_from_peaks(peaks, autocorr.shape)
        
        if not spacings:
            logger.warning("Could not estimate spacing from peaks")
            return None
        
        # Estimate angle from peak positions
        angle = self._estimate_grid_angle(peaks)
        
        # Validate parameters against expected ranges
        spacing_x, spacing_y = spacings
        expected_min, expected_max = self.config.expected_spacing_range
        
        # Scale back to full resolution
        factor = self.config.grid_detection_downsample
        spacing_x *= factor
        spacing_y *= factor
        
        # Sanity check - if spacing is way too large, it's probably wrong
        if spacing_x > 5000 or spacing_y > 5000:
            logger.warning(f"Detected spacing seems too large ({spacing_x}, {spacing_y}), using default estimate")
            # Use a reasonable default based on typical TMA spacing
            spacing_x = 1200  # pixels
            spacing_y = 1200  # pixels
        
        if not (expected_min <= spacing_x <= expected_max and expected_min <= spacing_y <= expected_max):
            logger.warning(f"Estimated spacing ({spacing_x:.1f}, {spacing_y:.1f}) outside expected range {self.config.expected_spacing_range}")
            # Continue anyway - ranges might be conservative
        
        return {
            'spacing_x': spacing_x,
            'spacing_y': spacing_y,
            'angle': angle,
            'autocorr_peaks': len(peaks),
            'confidence': min(len(peaks) / 10.0, 1.0)  # Rough confidence based on peak count
        }
    
    def _compute_2d_autocorrelation(self, image: np.ndarray) -> np.ndarray:
        """Compute 2D autocorrelation using FFT."""
        # Normalize image
        image_norm = image.astype(np.float32) - np.mean(image)
        
        # Compute FFT
        f_image = fft.fft2(image_norm)
        
        # Compute power spectrum (magnitude squared)
        power_spectrum = np.abs(f_image) ** 2
        
        # Inverse FFT to get autocorrelation
        autocorr = np.real(fft.ifft2(power_spectrum))
        
        # Shift zero frequency to center
        autocorr = fft.fftshift(autocorr)
        
        # Normalize
        autocorr = autocorr / np.max(autocorr)
        
        return autocorr
    
    def _find_autocorrelation_peaks(self, autocorr: np.ndarray) -> List[Tuple[int, int]]:
        """Find peaks in 2D autocorrelation function."""
        
        # Suppress the central peak (zero lag)
        center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        mask_radius = 20  # Suppress peaks within this radius of center
        
        y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
        center_mask = (x - center_x)**2 + (y - center_y)**2 < mask_radius**2
        autocorr_masked = autocorr.copy()
        autocorr_masked[center_mask] = 0
        
        # Find local maxima
        peaks = peak_local_maxima(
            autocorr_masked,
            min_distance=10,
            threshold_abs=self.config.autocorr_peak_threshold,
            exclude_border=True
        )
        
        # Convert to list of tuples and sort by peak value
        peak_coords = [(peaks[0][i], peaks[1][i]) for i in range(len(peaks[0]))]
        peak_values = [autocorr_masked[y, x] for y, x in peak_coords]
        
        # Sort by peak strength and take strongest peaks
        sorted_peaks = sorted(zip(peak_coords, peak_values), key=lambda x: x[1], reverse=True)
        
        # Return top peaks (up to 20)
        return [coord for coord, _ in sorted_peaks[:20]]
    
    def _estimate_spacing_from_peaks(self, peaks: List[Tuple[int, int]], shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Estimate grid spacing from autocorrelation peaks."""
        
        center_y, center_x = shape[0] // 2, shape[1] // 2
        
        # Calculate distances from center to peaks
        distances = []
        for py, px in peaks:
            dx = px - center_x
            dy = py - center_y
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 5:  # Ignore very close peaks
                distances.append((distance, dx, dy))
        
        if not distances:
            return None
        
        # Find the most common distance (fundamental frequency)
        distances.sort(key=lambda x: x[0])
        
        # Look for the first significant peak distance
        min_expected = self.config.expected_spacing_range[0] // self.config.grid_detection_downsample
        max_expected = self.config.expected_spacing_range[1] // self.config.grid_detection_downsample
        
        valid_distances = [d for d in distances if min_expected <= d[0] <= max_expected]
        
        if not valid_distances:
            # Fallback: use smallest non-trivial distance
            valid_distances = [d for d in distances if d[0] > 10]
        
        if not valid_distances:
            return None
        
        # Use the first valid distance as fundamental spacing
        fundamental_dist, dx, dy = valid_distances[0]
        
        # Estimate x and y spacing
        spacing_x = abs(dx) if abs(dx) > 5 else fundamental_dist
        spacing_y = abs(dy) if abs(dy) > 5 else fundamental_dist
        
        return (spacing_x, spacing_y)
    
    def _estimate_grid_angle(self, peaks: List[Tuple[int, int]]) -> float:
        """Estimate grid orientation angle from peak positions."""
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate angles between center and peaks
        angles = []
        for py, px in peaks[:4]:  # Use top 4 peaks
            angle = np.arctan2(py, px) * 180 / np.pi
            angles.append(angle)
        
        # Find the most common angle direction
        angles = np.array(angles)
        
        # Normalize angles to [0, 90] degrees (grid symmetry)
        normalized_angles = np.abs(angles) % 90
        
        # Return median angle
        return np.median(normalized_angles)
    
    def _create_core_template(self, image: np.ndarray, grid_params: Dict) -> Optional[np.ndarray]:
        """
        Create a template by extracting a typical core from the image.
        
        This method identifies high-quality cores in the image and creates
        a template for template matching.
        """
        logger.info("Creating core template from image...")
        
        spacing_x = grid_params['spacing_x'] / self.config.grid_detection_downsample
        spacing_y = grid_params['spacing_y'] / self.config.grid_detection_downsample
        
        # Estimate template size based on spacing
        template_size = int(min(spacing_x, spacing_y) * 0.6)  # 60% of spacing
        template_size = max(self.config.template_size_range[0], 
                           min(template_size, self.config.template_size_range[1]))
        
        logger.info(f"Using template size: {template_size}x{template_size}")
        
        # Find regions with high local contrast (likely cores)
        contrast_img = self._compute_local_contrast(image, template_size // 4)
        
        # Find candidate template locations
        candidates = self._find_template_candidates(contrast_img, template_size)
        
        if not candidates:
            logger.warning("No template candidates found")
            return None
        
        # Extract templates from top candidates
        templates = []
        for y, x in candidates[:5]:  # Top 5 candidates
            if (y + template_size < image.shape[0] and 
                x + template_size < image.shape[1]):
                template = image[y:y+template_size, x:x+template_size]
                templates.append(template)
        
        if not templates:
            return None
        
        # Average templates to create final template
        final_template = np.mean(templates, axis=0).astype(np.uint8)
        
        logger.info(f"Created template from {len(templates)} candidates")
        return final_template
    
    def _compute_local_contrast(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """Compute local contrast to identify interesting regions."""
        
        # Use local standard deviation as contrast measure
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        
        # Mean
        mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        
        # Variance = E[X^2] - E[X]^2
        sqr_mean = cv2.filter2D((image.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        # Standard deviation (contrast)
        contrast = np.sqrt(np.maximum(variance, 0))
        
        return contrast
    
    def _find_template_candidates(self, contrast_img: np.ndarray, template_size: int) -> List[Tuple[int, int]]:
        """Find locations with high contrast suitable for template creation."""
        
        # Find local maxima in contrast image
        peaks = peak_local_maxima(
            contrast_img,
            min_distance=template_size,
            threshold_abs=np.percentile(contrast_img, 80),  # Top 20% contrast
            exclude_border=template_size//2
        )
        
        # Convert to list and sort by contrast value
        candidates = [(peaks[0][i], peaks[1][i]) for i in range(len(peaks[0]))]
        
        # Validate candidates are within image bounds
        valid_candidates = []
        for y, x in candidates:
            if (0 <= y < contrast_img.shape[0] and 
                0 <= x < contrast_img.shape[1]):
                valid_candidates.append((y, x))
        
        # Get contrast values for valid candidates
        contrast_values = [contrast_img[y, x] for y, x in valid_candidates]
        
        # Sort by contrast (highest first)
        sorted_candidates = sorted(zip(valid_candidates, contrast_values), 
                                 key=lambda x: x[1], reverse=True)
        
        return [coord for coord, _ in sorted_candidates]
    
    def _find_grid_positions(self, image: np.ndarray, template: np.ndarray, 
                           grid_params: Dict) -> List[Dict]:
        """Find all grid positions using template matching."""
        
        logger.info("Finding grid positions using template matching...")
        
        # Perform template matching
        result = match_template(image, template, pad_input=True)
        
        # Find peaks in template matching result
        peaks = peak_local_maxima(
            result,
            min_distance=int(min(grid_params['spacing_x'], grid_params['spacing_y']) 
                           / self.config.grid_detection_downsample * 0.7),
            threshold_abs=self.config.template_match_threshold,
            exclude_border=template.shape[0]//2
        )
        
        # Convert peaks to grid positions
        positions = []
        for i in range(len(peaks[0])):
            y, x = peaks[0][i], peaks[1][i]
            score = result[y, x]
            
            positions.append({
                'grid_x': x,
                'grid_y': y,
                'match_score': score,
                'template_size': template.shape
            })
        
        # Sort by match score
        positions.sort(key=lambda p: p['match_score'], reverse=True)
        
        logger.info(f"Found {len(positions)} template matches above threshold")
        return positions
    
    def _validate_grid_positions(self, full_image: np.ndarray, grid_positions: List[Dict], 
                                downsample_factor: int) -> List[Dict]:
        """Validate grid positions on full-resolution image."""
        
        logger.info("Validating grid positions on full-resolution image...")
        
        validated_positions = []
        
        for pos in grid_positions:
            # Scale position to full resolution
            full_x = int(pos['grid_x'] * downsample_factor)
            full_y = int(pos['grid_y'] * downsample_factor)
            
            # Estimate core size at full resolution
            core_size = int(np.mean(self.config.expected_core_diameter_range))
            half_size = core_size // 2
            
            # Check bounds
            if (half_size <= full_x < full_image.shape[1] - half_size and
                half_size <= full_y < full_image.shape[0] - half_size):
                
                # Extract region around position
                region = full_image[full_y-half_size:full_y+half_size,
                                  full_x-half_size:full_x+half_size]
                
                # Validate region has sufficient tissue content
                if self._validate_core_region(region):
                    validated_positions.append({
                        'grid_x': pos['grid_x'],
                        'grid_y': pos['grid_y'],
                        'full_x': full_x,
                        'full_y': full_y,
                        'match_score': pos['match_score'],
                        'core_size': core_size
                    })
        
        logger.info(f"Validated {len(validated_positions)} positions")
        return validated_positions
    
    def _validate_core_region(self, region: np.ndarray) -> bool:
        """Validate that a region contains sufficient tissue content."""
        
        if region.size == 0:
            return False
        
        # Simple validation: check if region has sufficient non-background content
        # Background is typically very bright in H&E or very dark in fluorescence
        mean_intensity = np.mean(region)
        std_intensity = np.std(region)
        
        # Require some variation in intensity (not just background)
        if std_intensity < 10:  # Too uniform
            return False
        
        # Check for reasonable intensity range
        if mean_intensity < 5 or mean_intensity > 250:  # Too dark or too bright
            return False
        
        return True
    
    def _extract_cores_from_grid(self, image: np.ndarray, positions: List[Dict], 
                                image_type: str) -> List[Dict]:
        """Extract core information from validated grid positions."""
        
        logger.info(f"Extracting {len(positions)} cores from grid positions...")
        
        cores = []
        
        for i, pos in enumerate(positions):
            full_x = pos['full_x']
            full_y = pos['full_y']
            core_size = pos['core_size']
            
            # Calculate bounding box with padding
            padding = self.config.core_extraction_padding
            half_size = core_size // 2
            
            minr = max(0, full_y - half_size - padding)
            maxr = min(image.shape[0], full_y + half_size + padding)
            minc = max(0, full_x - half_size - padding)
            maxc = min(image.shape[1], full_x + half_size + padding)
            
            # Calculate core properties
            area = np.pi * (half_size ** 2)
            diameter = core_size
            
            core_info = {
                'id': i,
                'centroid': (full_y, full_x),
                'centroid_xy': (full_x, full_y),
                'grid_position': (pos['grid_x'], pos['grid_y']),
                'area': area,
                'equiv_diameter': diameter,
                'bbox': (minr, minc, maxr, maxc),
                'bbox_padded': (minr, minc, maxr, maxc),
                'match_score': pos['match_score'],
                'detection_method': 'tma_grid',
                'grid_based': True,
                # Standard properties for compatibility
                'circularity': 1.0,  # Assume circular
                'solidity': 1.0,
                'aspect_ratio': 1.0,
                'perimeter': 2 * np.pi * half_size,
                'mean_intensity': 128  # Placeholder
            }
            
            cores.append(core_info)
        
        logger.info(f"Extracted {len(cores)} core information records")
        return cores
    
    def _create_empty_result(self, image_path: str, image_type: str, image_shape: Tuple) -> Dict:
        """Create empty result dictionary for failed detection."""
        return {
            'image_path': str(image_path),
            'image_type': image_type,
            'original_shape': image_shape,
            'detection_method': 'tma_grid',
            'grid_parameters': None,
            'cores': [],
            'success': False,
            'error': 'Grid detection failed'
        }
    
    def visualize_grid_detection(self, results: Dict, output_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of grid detection results."""
        
        if not results['success']:
            logger.warning("Cannot visualize failed grid detection")
            return None
        
        # Load original image for visualization
        image = self._load_image(results['image_path'], results['image_type'])
        display_img = self._prepare_display_image(image, results['image_type'])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        axes[0].imshow(display_img, cmap='gray' if len(display_img.shape) == 2 else None)
        axes[0].set_title(f'Original Image\nShape: {results["original_shape"]}')
        axes[0].axis('off')
        
        # Grid overlay
        overlay = display_img.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        # Draw detected cores
        for core in results['cores']:
            center_x, center_y = core['centroid_xy']
            size = core['equiv_diameter'] // 2
            
            # Draw circle
            cv2.circle(overlay, (int(center_x), int(center_y)), int(size), (0, 255, 0), 3)
            
            # Draw grid position
            grid_x, grid_y = core['grid_position']
            cv2.putText(overlay, f"({grid_x},{grid_y})", 
                       (int(center_x-size), int(center_y-size-10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        axes[1].imshow(overlay)
        axes[1].set_title(f'Grid Detection Results\n{len(results["cores"])} cores detected')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Grid visualization saved to {output_path}")
        
        return fig
    
    def _prepare_display_image(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """Prepare image for display."""
        
        if image_type == "orion" and image.ndim == 3 and image.shape[0] > 10:
            # Multi-channel - create RGB composite
            if image.shape[0] >= 3:
                display_img = np.transpose(image[:3], (1, 2, 0))
            else:
                display_img = np.stack([image[0]] * 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 3:
            display_img = image
        elif image.ndim == 3:
            display_img = image[:, :, 0]
        else:
            display_img = image
        
        # Normalize for display
        if display_img.dtype != np.uint8:
            display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return display_img


def main():
    """Example usage of TMA grid detector."""
    
    config = TMAGridConfig(
        expected_core_diameter_range=(800, 1200),
        expected_spacing_range=(1000, 1600),
        enable_visualizations=True
    )
    
    detector = TMAGridDetector(config)
    
    # Test on example images
    he_path = "data/raw/TA118-HEraw.ome.tiff"
    orion_path = "data/raw/TA118-Orionraw.ome.tiff"
    
    if Path(he_path).exists():
        results = detector.detect_tma_grid(he_path, image_type="he")
        print(f"H&E: Detected {len(results['cores'])} cores")
        
        if results['success']:
            detector.visualize_grid_detection(results, "he_grid_detection.png")
    
    if Path(orion_path).exists():
        results = detector.detect_tma_grid(orion_path, image_type="orion")
        print(f"Orion: Detected {len(results['cores'])} cores")
        
        if results['success']:
            detector.visualize_grid_detection(results, "orion_grid_detection.png")


if __name__ == "__main__":
    main() 