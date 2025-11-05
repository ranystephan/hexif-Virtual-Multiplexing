#!/usr/bin/env python3
"""
TMA Detection Diagnostic Tool

This tool helps you visualize your TMA images at different scales and 
tune detection parameters visually. It shows what's actually in your images
and helps you choose appropriate parameters for core detection.

Usage:
    python diagnose_tma_detection.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import filters, morphology, measure
from tifffile import imread
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TMADiagnostic:
    """Interactive TMA diagnostic tool."""
    
    def __init__(self):
        self.image = None
        self.image_name = None
        self.detection_image = None
        self.scale_factor = 1.0
        
    def analyze_tma_structure(self, image_path: str, image_type: str = "auto"):
        """
        Comprehensive analysis of TMA structure with multiple visualizations.
        """
        print(f"🔬 ANALYZING TMA STRUCTURE: {image_path}")
        print("=" * 60)
        
        # Load image
        self.image = imread(image_path)
        self.image_name = Path(image_path).stem
        
        print(f"Image shape: {self.image.shape}")
        print(f"Image dtype: {self.image.dtype}")
        
        # Prepare detection image
        self.detection_image = self._prepare_detection_image(self.image, image_type)
        print(f"Detection image shape: {self.detection_image.shape}")
        
        # Calculate scale factor for display
        max_display_size = 2000
        if max(self.detection_image.shape) > max_display_size:
            self.scale_factor = max_display_size / max(self.detection_image.shape)
            display_height = int(self.detection_image.shape[0] * self.scale_factor)
            display_width = int(self.detection_image.shape[1] * self.scale_factor)
            display_image = cv2.resize(self.detection_image, (display_width, display_height), 
                                     interpolation=cv2.INTER_AREA)
        else:
            display_image = self.detection_image
            self.scale_factor = 1.0
        
        print(f"Display scale factor: {self.scale_factor:.3f}")
        print(f"Display image shape: {display_image.shape}")
        
        # Create comprehensive visualization
        self._create_overview_plot(display_image)
        
        # Analyze core characteristics
        self._analyze_core_characteristics(display_image)
        
        # Create interactive parameter tuning
        self._create_interactive_tuning(display_image)
        
        # Estimate optimal parameters
        optimal_params = self._estimate_optimal_parameters(display_image)
        self._print_parameter_recommendations(optimal_params)
        
        return optimal_params
    
    def _prepare_detection_image(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """Prepare image for detection analysis."""
        
        if image_type == "auto":
            if image.ndim == 3 and image.shape[0] > 10:
                image_type = "orion"
            else:
                image_type = "he"
        
        if image_type == "he":
            if image.ndim == 3 and image.shape[2] == 3:
                # RGB H&E
                detection_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.ndim == 3 and image.shape[0] == 3:
                # Channel-first RGB
                detection_img = cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
            else:
                detection_img = image.squeeze()
        else:  # Orion
            if image.ndim == 3 and image.shape[0] > 10:
                # Multi-channel - use DAPI (channel 0)
                detection_img = image[0]
            else:
                detection_img = image.squeeze()
        
        # Normalize to uint8
        if detection_img.dtype != np.uint8:
            detection_img = cv2.normalize(detection_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return detection_img
    
    def _create_overview_plot(self, image: np.ndarray):
        """Create overview visualization showing the image at different processing stages."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'TMA Overview Analysis - {self.image_name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title(f'Original Image\n{image.shape} pixels')
        axes[0, 0].axis('off')
        
        # Add scale bars
        self._add_scale_bar(axes[0, 0], image.shape, "1mm", 1000)
        
        # Histogram
        axes[0, 1].hist(image.ravel(), bins=100, alpha=0.7)
        axes[0, 1].set_title('Intensity Histogram')
        axes[0, 1].set_xlabel('Pixel Intensity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Otsu threshold
        otsu_thresh = filters.threshold_otsu(image)
        binary_otsu = image > otsu_thresh
        axes[0, 2].imshow(binary_otsu, cmap='gray')
        axes[0, 2].set_title(f'Otsu Threshold\nThreshold: {otsu_thresh}')
        axes[0, 2].axis('off')
        
        # Gaussian smoothed
        smoothed = cv2.GaussianBlur(image, (5, 5), 1.0)
        axes[1, 0].imshow(smoothed, cmap='gray')
        axes[1, 0].set_title('Gaussian Smoothed')
        axes[1, 0].axis('off')
        
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title('Edge Detection (Canny)')
        axes[1, 1].axis('off')
        
        # Morphological operations
        binary_cleaned = morphology.remove_small_objects(binary_otsu, min_size=1000)
        binary_cleaned = morphology.binary_fill_holes(binary_cleaned)
        axes[1, 2].imshow(binary_cleaned, cmap='gray')
        axes[1, 2].set_title('Morphological Cleaning')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.image_name}_overview.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _add_scale_bar(self, ax, image_shape, label, pixels_per_mm):
        """Add scale bar to image."""
        
        # Assume 1 pixel = 0.5 microns (typical for high-res TMA)
        pixel_size_um = 0.5
        scale_bar_mm = 1.0  # 1mm scale bar
        scale_bar_pixels = int(scale_bar_mm * 1000 / pixel_size_um * self.scale_factor)
        
        # Position scale bar
        height, width = image_shape
        x_pos = width - scale_bar_pixels - 50
        y_pos = height - 30
        
        # Draw scale bar
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_pos, y_pos), scale_bar_pixels, 10, 
                        color='white', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x_pos + scale_bar_pixels/2, y_pos - 20, label, 
               ha='center', va='top', color='white', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def _analyze_core_characteristics(self, image: np.ndarray):
        """Analyze potential core characteristics in the image."""
        
        print("\n🔍 ANALYZING POTENTIAL CORE CHARACTERISTICS")
        print("-" * 50)
        
        # Apply basic thresholding and morphology
        otsu_thresh = filters.threshold_otsu(image)
        binary = image > otsu_thresh
        binary_cleaned = morphology.remove_small_objects(binary, min_size=1000)
        binary_filled = morphology.binary_fill_holes(binary_cleaned)
        
        # Find connected components
        labeled = measure.label(binary_filled)
        regions = measure.regionprops(labeled, intensity_image=image)
        
        if not regions:
            print("❌ No regions found! Image may need different preprocessing.")
            return
        
        # Calculate statistics (scale back to original resolution)
        scale_back = 1.0 / self.scale_factor
        
        areas = [(r.area * scale_back**2) for r in regions]
        diameters = [np.sqrt(4 * area / np.pi) for area in areas]
        circularities = []
        
        for region in regions:
            if region.perimeter > 0:
                circ = 4 * np.pi * region.area / (region.perimeter ** 2)
                circularities.append(circ)
        
        areas = np.array(areas)
        diameters = np.array(diameters)
        circularities = np.array(circularities) if circularities else np.array([0])
        
        print(f"Found {len(regions)} potential objects")
        print(f"Area range: {areas.min():.0f} - {areas.max():.0f} pixels²")
        print(f"Diameter range: {diameters.min():.0f} - {diameters.max():.0f} pixels")
        print(f"Circularity range: {circularities.min():.3f} - {circularities.max():.3f}")
        
        # Show size distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Core Characteristics Analysis')
        
        axes[0, 0].hist(areas, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.median(areas), color='red', linestyle='--', label='Median')
        axes[0, 0].set_xlabel('Area (pixels²)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Area Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(diameters, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.median(diameters), color='red', linestyle='--', label='Median')
        axes[0, 1].set_xlabel('Equivalent Diameter (pixels)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Diameter Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist(circularities, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.median(circularities), color='red', linestyle='--', label='Median')
        axes[1, 0].set_xlabel('Circularity')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Circularity Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot: area vs circularity
        axes[1, 1].scatter(areas, circularities, alpha=0.6)
        axes[1, 1].set_xlabel('Area (pixels²)')
        axes[1, 1].set_ylabel('Circularity')
        axes[1, 1].set_title('Area vs Circularity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.image_name}_characteristics.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return areas, diameters, circularities
    
    def _create_interactive_tuning(self, image: np.ndarray):
        """Create interactive parameter tuning interface."""
        
        print("\n🎛️  INTERACTIVE PARAMETER TUNING")
        print("-" * 50)
        print("Adjust the sliders to see how parameters affect detection:")
        
        # Create interactive plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(bottom=0.3)
        
        # Initial parameters
        min_area = 30000
        max_area = 300000
        min_circularity = 0.3
        hough_param2 = 30
        
        def update_detection(min_a, max_a, min_circ, h_param2):
            """Update detection visualization."""
            
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            # Hough circles detection
            blurred = cv2.GaussianBlur(image, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1,
                minDist=200, param1=50, param2=int(h_param2),
                minRadius=100, maxRadius=500
            )
            
            # Show original with Hough circles
            axes[0, 0].imshow(image, cmap='gray')
            hough_count = 0
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
                        axes[0, 0].add_patch(circle)
                        hough_count += 1
            
            axes[0, 0].set_title(f'Hough Circles: {hough_count} found')
            axes[0, 0].axis('off')
            
            # Morphological detection
            otsu_thresh = filters.threshold_otsu(image)
            binary = image > otsu_thresh
            binary_cleaned = morphology.remove_small_objects(binary, min_size=1000)
            binary_filled = morphology.binary_fill_holes(binary_cleaned)
            
            axes[0, 1].imshow(binary_filled, cmap='gray')
            axes[0, 1].set_title('Binary After Morphology')
            axes[0, 1].axis('off')
            
            # Find regions and filter
            labeled = measure.label(binary_filled)
            regions = measure.regionprops(labeled, intensity_image=image)
            
            filtered_regions = []
            scale_back = 1.0 / self.scale_factor
            
            for region in regions:
                area = region.area * scale_back**2
                if not (min_a <= area <= max_a):
                    continue
                
                if region.perimeter > 0:
                    circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
                    if circularity < min_circ:
                        continue
                
                filtered_regions.append(region)
            
            # Show filtered results
            axes[1, 0].imshow(image, cmap='gray')
            for region in filtered_regions:
                minr, minc, maxr, maxc = region.bbox
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                   fill=False, edgecolor='green', linewidth=2)
                axes[1, 0].add_patch(rect)
                axes[1, 0].plot(region.centroid[1], region.centroid[0], 'go', markersize=4)
            
            axes[1, 0].set_title(f'Morphology + Filter: {len(filtered_regions)} cores')
            axes[1, 0].axis('off')
            
            # Combined results
            axes[1, 1].imshow(image, cmap='gray')
            
            # Show both Hough and morphology results
            if circles is not None:
                for (x, y, r) in circles:
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=1, alpha=0.7)
                        axes[1, 1].add_patch(circle)
            
            for region in filtered_regions:
                minr, minc, maxr, maxc = region.bbox
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                   fill=False, edgecolor='green', linewidth=1, alpha=0.7)
                axes[1, 1].add_patch(rect)
            
            axes[1, 1].set_title(f'Combined: H={hough_count}, M={len(filtered_regions)}')
            axes[1, 1].axis('off')
            
            plt.draw()
        
        # Create sliders
        ax_min_area = plt.axes([0.1, 0.2, 0.35, 0.03])
        ax_max_area = plt.axes([0.55, 0.2, 0.35, 0.03])
        ax_circularity = plt.axes([0.1, 0.15, 0.35, 0.03])
        ax_hough_param = plt.axes([0.55, 0.15, 0.35, 0.03])
        
        slider_min_area = Slider(ax_min_area, 'Min Area', 10000, 100000, valinit=min_area, valfmt='%d')
        slider_max_area = Slider(ax_max_area, 'Max Area', 100000, 800000, valinit=max_area, valfmt='%d')
        slider_circularity = Slider(ax_circularity, 'Min Circularity', 0.1, 0.9, valinit=min_circularity, valfmt='%.2f')
        slider_hough_param = Slider(ax_hough_param, 'Hough Param2', 10, 80, valinit=hough_param2, valfmt='%d')
        
        def on_slider_change(val):
            update_detection(
                slider_min_area.val,
                slider_max_area.val, 
                slider_circularity.val,
                slider_hough_param.val
            )
        
        slider_min_area.on_changed(on_slider_change)
        slider_max_area.on_changed(on_slider_change)
        slider_circularity.on_changed(on_slider_change)
        slider_hough_param.on_changed(on_slider_change)
        
        # Initial update
        update_detection(min_area, max_area, min_circularity, hough_param2)
        
        plt.savefig(f'{self.image_name}_interactive_tuning.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _estimate_optimal_parameters(self, image: np.ndarray) -> dict:
        """Estimate optimal detection parameters based on image analysis."""
        
        print("\n🎯 ESTIMATING OPTIMAL PARAMETERS")
        print("-" * 50)
        
        # Basic morphological analysis
        otsu_thresh = filters.threshold_otsu(image)
        binary = image > otsu_thresh
        binary_cleaned = morphology.remove_small_objects(binary, min_size=1000)
        binary_filled = morphology.binary_fill_holes(binary_cleaned)
        
        # Find regions
        labeled = measure.label(binary_filled)
        regions = measure.regionprops(labeled, intensity_image=image)
        
        if not regions:
            print("⚠️ No regions found - using default parameters")
            return self._get_default_parameters()
        
        # Calculate statistics (scale back to original resolution)
        scale_back = 1.0 / self.scale_factor
        areas = np.array([r.area * scale_back**2 for r in regions])
        diameters = np.array([np.sqrt(4 * area / np.pi) for area in areas])
        circularities = []
        
        for region in regions:
            if region.perimeter > 0:
                circ = 4 * np.pi * region.area / (region.perimeter ** 2)
                circularities.append(circ)
        
        circularities = np.array(circularities) if circularities else np.array([0.5])
        
        # Filter out obvious outliers (keep middle 80%)
        area_p10, area_p90 = np.percentile(areas, [10, 90])
        diameter_p10, diameter_p90 = np.percentile(diameters, [10, 90])
        
        reasonable_areas = areas[(areas >= area_p10) & (areas <= area_p90)]
        reasonable_diameters = diameters[(diameters >= diameter_p10) & (diameters <= diameter_p90)]
        
        if len(reasonable_areas) == 0:
            reasonable_areas = areas
            reasonable_diameters = diameters
        
        # Estimate core parameters
        median_area = np.median(reasonable_areas)
        median_diameter = np.median(reasonable_diameters)
        median_circularity = np.median(circularities)
        
        # Conservative parameter estimation
        min_area = max(10000, int(median_area * 0.3))  # 30% of median
        max_area = min(1000000, int(median_area * 3.0))  # 3x median
        min_diameter = max(100, int(median_diameter * 0.5))  # 50% of median
        max_diameter = min(2000, int(median_diameter * 2.0))  # 2x median
        min_circularity = max(0.2, median_circularity * 0.6)  # 60% of median
        
        # Hough circle parameters
        hough_min_radius = min_diameter // 2
        hough_max_radius = max_diameter // 2
        hough_min_dist = int(median_diameter * 0.8)  # 80% of typical diameter
        
        optimal_params = {
            # Size parameters
            'min_core_area': min_area,
            'max_core_area': max_area,
            'min_core_diameter': min_diameter,
            'max_core_diameter': max_diameter,
            
            # Shape parameters
            'min_circularity': min_circularity,
            'min_solidity': 0.6,
            'aspect_ratio_threshold': 2.0,
            
            # Hough parameters
            'hough_min_radius': hough_min_radius,
            'hough_max_radius': hough_max_radius,
            'hough_min_dist': hough_min_dist,
            'hough_param1': 50,
            'hough_param2': 30,
            
            # Statistics for reference
            'estimated_core_count': len(reasonable_areas),
            'median_area': median_area,
            'median_diameter': median_diameter,
            'median_circularity': median_circularity,
        }
        
        return optimal_params
    
    def _get_default_parameters(self) -> dict:
        """Get default parameters when analysis fails."""
        return {
            'min_core_area': 30000,
            'max_core_area': 300000,
            'min_core_diameter': 150,
            'max_core_diameter': 800,
            'min_circularity': 0.3,
            'min_solidity': 0.6,
            'aspect_ratio_threshold': 2.0,
            'hough_min_radius': 75,
            'hough_max_radius': 400,
            'hough_min_dist': 200,
            'hough_param1': 50,
            'hough_param2': 30,
            'estimated_core_count': 0,
            'median_area': 100000,
            'median_diameter': 300,
            'median_circularity': 0.5,
        }
    
    def _print_parameter_recommendations(self, params: dict):
        """Print parameter recommendations."""
        
        print("\n📋 PARAMETER RECOMMENDATIONS")
        print("=" * 60)
        print(f"Based on analysis of {params['estimated_core_count']} potential cores:")
        print()
        print("🔸 SIZE PARAMETERS:")
        print(f"   min_core_area: {params['min_core_area']:,}")
        print(f"   max_core_area: {params['max_core_area']:,}")  
        print(f"   min_core_diameter: {params['min_core_diameter']}")
        print(f"   max_core_diameter: {params['max_core_diameter']}")
        print()
        print("🔸 SHAPE PARAMETERS:")
        print(f"   min_circularity: {params['min_circularity']:.3f}")
        print(f"   min_solidity: {params['min_solidity']:.3f}")
        print(f"   aspect_ratio_threshold: {params['aspect_ratio_threshold']:.1f}")
        print()
        print("🔸 HOUGH CIRCLE PARAMETERS:")
        print(f"   hough_min_radius: {params['hough_min_radius']}")
        print(f"   hough_max_radius: {params['hough_max_radius']}")
        print(f"   hough_min_dist: {params['hough_min_dist']}")
        print(f"   hough_param2: {params['hough_param2']}")
        print()
        print("🔸 ESTIMATED STATISTICS:")
        print(f"   Median core area: {params['median_area']:,.0f} pixels²")
        print(f"   Median core diameter: {params['median_diameter']:.0f} pixels")
        print(f"   Median circularity: {params['median_circularity']:.3f}")
        print()
        print("💡 USAGE:")
        print("Copy these parameters into your detection configuration:")
        print("```python")
        print("config = CoreDetectionConfig(")
        print(f"    min_core_area={params['min_core_area']},")
        print(f"    max_core_area={params['max_core_area']},")
        print(f"    min_core_diameter={params['min_core_diameter']},")
        print(f"    max_core_diameter={params['max_core_diameter']},")
        print(f"    min_circularity={params['min_circularity']:.3f},")
        print(f"    hough_min_dist={params['hough_min_dist']},")
        print(f"    hough_param2={params['hough_param2']},")
        print("    detection_method='hough'  # or 'hybrid'")
        print(")")
        print("```")


def main():
    """Run TMA diagnostic analysis."""
    
    print("🔬 TMA DETECTION DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Test paths
    he_path = "data/raw/TA118-HEraw.ome.tiff"
    orion_path = "data/raw/TA118-Orionraw.ome.tiff"
    
    diagnostic = TMADiagnostic()
    
    # Analyze H&E
    if Path(he_path).exists():
        print(f"\n🎯 ANALYZING H&E IMAGE")
        he_params = diagnostic.analyze_tma_structure(he_path, "he")
        
        # Save parameters to file
        import json
        with open('he_optimal_parameters.json', 'w') as f:
            json.dump(he_params, f, indent=2)
        print(f"\n💾 H&E parameters saved to: he_optimal_parameters.json")
    else:
        print(f"❌ H&E image not found: {he_path}")
    
    print("\n" + "=" * 60)
    
    # Analyze Orion  
    if Path(orion_path).exists():
        print(f"\n🎯 ANALYZING ORION IMAGE")
        orion_params = diagnostic.analyze_tma_structure(orion_path, "orion")
        
        # Save parameters to file
        import json
        with open('orion_optimal_parameters.json', 'w') as f:
            json.dump(orion_params, f, indent=2)
        print(f"\n💾 Orion parameters saved to: orion_optimal_parameters.json")
    else:
        print(f"❌ Orion image not found: {orion_path}")
    
    print("\n🎉 DIAGNOSTIC ANALYSIS COMPLETE!")
    print("Check the generated plots and parameter files.")
    print("Use the recommended parameters in your detection configuration.")


if __name__ == "__main__":
    main() 