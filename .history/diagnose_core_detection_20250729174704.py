#!/usr/bin/env python3
"""
diagnose_core_detection.py
--------------------------
Debug and visualize the core detection process to understand why so few cores are being detected.

Usage:
python diagnose_core_detection.py \
    --he data/raw/TA118-HEraw.ome.tiff \
    --orion data/raw/TA118-Orionraw.ome.tiff \
    --min_area 500 \
    --circularity 0.1
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tifffile import TiffFile
from typing import List, Dict, Tuple

class DiagnosticCoreDetector:
    def __init__(self, thumb_size: int = 4096, min_core_area: int = 2000):
        self.thumb_size = thumb_size
        self.min_core_area = min_core_area

    def load_thumbnail(self, slide_path: str, use_channel0: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load thumbnail using the same method as the main script"""
        with TiffFile(slide_path) as tif:
            series = tif.series[0]
            full_shape = series.shape[-2:]
            full_shape = (int(full_shape[0]), int(full_shape[1]))
            level = series.levels[-1]
            arr = level.asarray()
            
            if arr.ndim == 3:
                if arr.shape[0] < arr.shape[-1]:
                    arr = arr[0] if use_channel0 else arr[0]
                else:
                    arr = arr[..., 0] if use_channel0 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            
            thumb = arr.astype(np.uint8)
        
        h, w = thumb.shape
        if max(h, w) < self.thumb_size:
            scale = self.thumb_size / max(h, w)
            thumb = cv2.resize(thumb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
        return thumb, full_shape

    def detect_cores_with_debug(
        self, 
        image: np.ndarray, 
        brightfield: bool = True, 
        min_area: int = None,
        circularity_thresh: float = 0.3,
        show_steps: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """Detect cores with detailed debugging information"""
        min_area = min_area or self.min_core_area
        
        # Step 1: Normalize
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Step 2: Blur
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Step 3: Optional CLAHE for non-brightfield
        if not brightfield:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            blur = clahe.apply(blur)
        
        # Step 4: Adaptive threshold
        bs = max(31, int(min(blur.shape) // 50) * 2 + 1)
        thresh_type = cv2.THRESH_BINARY_INV if brightfield else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            bs, 2
        )
        
        # Step 5: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        morph_final = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)
        
        # Step 6: Find contours
        contours, _ = cv2.findContours(morph_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 7: Filter contours
        all_contours = []
        filtered_cores = []
        filter_stats = {
            'total_contours': len(contours),
            'area_filtered': 0,
            'circularity_filtered': 0,
            'moment_filtered': 0,
            'final_cores': 0
        }
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            all_contours.append({'area': area, 'contour': cnt})
            
            if area < min_area:
                filter_stats['area_filtered'] += 1
                continue
                
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                filter_stats['moment_filtered'] += 1
                continue
                
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            
            if circularity < circularity_thresh:
                filter_stats['circularity_filtered'] += 1
                continue
                
            filtered_cores.append({
                'center': (cx, cy), 
                'area': area, 
                'circularity': circularity,
                'contour': cnt
            })
        
        filter_stats['final_cores'] = len(filtered_cores)
        
        # Debug info
        debug_info = {
            'original': img,
            'blurred': blur,
            'binary': binary,
            'morph_open': morph_open,
            'morph_final': morph_final,
            'filter_stats': filter_stats,
            'all_contours': all_contours,
            'block_size': bs,
            'min_area': min_area,
            'circularity_thresh': circularity_thresh
        }
        
        if show_steps:
            self.visualize_detection_steps(debug_info, filtered_cores, image.shape)
        
        return filtered_cores, debug_info

    def visualize_detection_steps(self, debug_info: Dict, cores: List[Dict], original_shape: Tuple):
        """Visualize the detection process step by step"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Core Detection Steps - Found {len(cores)} cores', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(debug_info['original'], cmap='gray')
        axes[0, 0].set_title('1. Original/Normalized')
        axes[0, 0].axis('off')
        
        # Blurred
        axes[0, 1].imshow(debug_info['blurred'], cmap='gray')
        axes[0, 1].set_title('2. Gaussian Blur')
        axes[0, 1].axis('off')
        
        # Binary threshold
        axes[0, 2].imshow(debug_info['binary'], cmap='gray')
        axes[0, 2].set_title(f'3. Adaptive Threshold (bs={debug_info["block_size"]})')
        axes[0, 2].axis('off')
        
        # Morphological opening
        axes[1, 0].imshow(debug_info['morph_open'], cmap='gray')
        axes[1, 0].set_title('4. Morphological Opening')
        axes[1, 0].axis('off')
        
        # Final morphology
        axes[1, 1].imshow(debug_info['morph_final'], cmap='gray')
        axes[1, 1].set_title('5. Morphological Closing')
        axes[1, 1].axis('off')
        
        # Final result with detected cores
        result_img = cv2.cvtColor(debug_info['original'], cv2.COLOR_GRAY2RGB)
        for core in cores:
            cx, cy = core['center']
            cv2.circle(result_img, (cx, cy), 10, (255, 0, 0), 2)
            # Also draw the contour
            cv2.drawContours(result_img, [core['contour']], -1, (0, 255, 0), 2)
        
        axes[1, 2].imshow(result_img)
        axes[1, 2].set_title(f'6. Final Result ({len(cores)} cores)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print filtering statistics
        stats = debug_info['filter_stats']
        print(f"\n=== FILTERING STATISTICS ===")
        print(f"Total contours found: {stats['total_contours']}")
        print(f"Filtered by area (< {debug_info['min_area']}): {stats['area_filtered']}")
        print(f"Filtered by invalid moments: {stats['moment_filtered']}")
        print(f"Filtered by circularity (< {debug_info['circularity_thresh']:.2f}): {stats['circularity_filtered']}")
        print(f"Final cores: {stats['final_cores']}")
        
        # Show area distribution
        if debug_info['all_contours']:
            areas = [c['area'] for c in debug_info['all_contours']]
            plt.figure(figsize=(10, 6))
            plt.hist(areas, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(debug_info['min_area'], color='red', linestyle='--', 
                       label=f'Min area threshold: {debug_info["min_area"]}')
            plt.xlabel('Contour Area')
            plt.ylabel('Count')
            plt.title('Distribution of Contour Areas')
            plt.legend()
            plt.yscale('log')
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Diagnose core detection issues")
    parser.add_argument("--he", required=True, help="Path to H&E OME-TIFF")
    parser.add_argument("--orion", required=True, help="Path to Orion OME-TIFF")
    parser.add_argument("--min_area", type=int, default=500, help="Minimum core area (try lower values)")
    parser.add_argument("--circularity", type=float, default=0.1, help="Minimum circularity (try lower values)")
    parser.add_argument("--thumb_size", type=int, default=4096, help="Thumbnail size")
    args = parser.parse_args()
    
    detector = DiagnosticCoreDetector(args.thumb_size)
    
    print("=== ANALYZING H&E IMAGE ===")
    he_thumb, he_full = detector.load_thumbnail(args.he)
    print(f"H&E thumbnail shape: {he_thumb.shape}, full shape: {he_full}")
    
    he_cores, he_debug = detector.detect_cores_with_debug(
        he_thumb, 
        brightfield=True, 
        min_area=args.min_area,
        circularity_thresh=args.circularity,
        show_steps=True
    )
    
    print(f"\nH&E cores detected: {len(he_cores)}")
    if he_cores:
        areas = [c['area'] for c in he_cores]
        circularities = [c['circularity'] for c in he_cores]
        print(f"Area range: {min(areas):.0f} - {max(areas):.0f}")
        print(f"Circularity range: {min(circularities):.3f} - {max(circularities):.3f}")
    
    print("\n" + "="*50)
    print("=== ANALYZING ORION IMAGE ===")
    or_thumb, or_full = detector.load_thumbnail(args.orion, use_channel0=True)
    print(f"Orion thumbnail shape: {or_thumb.shape}, full shape: {or_full}")
    
    or_cores, or_debug = detector.detect_cores_with_debug(
        or_thumb, 
        brightfield=False, 
        min_area=args.min_area,
        circularity_thresh=args.circularity,
        show_steps=True
    )
    
    print(f"\nOrion cores detected: {len(or_cores)}")
    if or_cores:
        areas = [c['area'] for c in or_cores]
        circularities = [c['circularity'] for c in or_cores]
        print(f"Area range: {min(areas):.0f} - {max(areas):.0f}")
        print(f"Circularity range: {min(circularities):.3f} - {max(circularities):.3f}")
    
    print(f"\n=== RECOMMENDATIONS ===")
    print(f"Current min_area: {args.min_area}")
    print(f"Current circularity: {args.circularity}")
    
    if len(he_cores) < 50 or len(or_cores) < 50:
        print("\nToo few cores detected! Try these parameters:")
        print(f"  --min_area 100")
        print(f"  --circularity 0.05")
        print("\nOr even more aggressive:")
        print(f"  --min_area 50")
        print(f"  --circularity 0.01")

if __name__ == "__main__":
    main() 