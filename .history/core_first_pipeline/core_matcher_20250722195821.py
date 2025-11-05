"""
Core Matching Module for TMA Processing

This module handles the spatial matching of tissue cores detected in paired 
H&E and Orion TMA images. It uses various strategies to find corresponding
cores between the two stains based on spatial proximity, size similarity,
and other features.

Key Features:
- Spatial proximity-based matching
- Multiple matching algorithms (Hungarian, nearest neighbor, etc.)
- Quality assessment of matches
- Visualization of matching results
- Handling of missing or extra cores
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add import for the new grid matcher
from .tma_grid_matcher import TMAGridMatcher, TMAGridMatchingConfig


@dataclass
class CoreMatchingConfig:
    """Configuration for core matching."""
    
    # Matching parameters
    max_distance_threshold: float = 300.0  # Maximum distance between matched cores (pixels)
    size_similarity_weight: float = 0.3    # Weight for size similarity in matching
    distance_weight: float = 0.7           # Weight for distance in matching
    
    # Matching algorithms
    matching_method: str = "hungarian"     # "hungarian", "nearest_neighbor", "greedy"
    
    # Quality control
    min_match_confidence: float = 0.5      # Minimum confidence for accepting a match
    max_size_ratio_difference: float = 2.0 # Maximum ratio difference in core sizes
    
    # Handling unmatched cores  
    allow_unmatched: bool = True           # Whether to allow unmatched cores
    max_unmatched_fraction: float = 0.3    # Maximum fraction of unmatched cores allowed
    
    # Output
    save_visualizations: bool = True
    create_paired_output: bool = True


class CoreMatcher:
    """Matches cores between H&E and Orion images based on spatial proximity."""
    
    def __init__(self, config: CoreMatchingConfig):
        self.config = config
    
    def match_cores(self, he_detection_results: Dict, orion_detection_results: Dict) -> Dict:
        """
        Match cores between H&E and Orion detection results.
        
        Args:
            he_detection_results: Detection results from H&E image
            orion_detection_results: Detection results from Orion image
            
        Returns:
            Dictionary containing matching results
        """
        logger.info(f"Matching {len(he_detection_results['cores'])} H&E cores with {len(orion_detection_results['cores'])} Orion cores")
        
        he_cores = he_detection_results['cores']
        orion_cores = orion_detection_results['cores']
        
        if not he_cores or not orion_cores:
            logger.warning("No cores to match in one or both images")
            return self._create_empty_matching_results(he_detection_results, orion_detection_results)
        
        # Check if both sets of cores have grid position information
        he_has_grid = all('grid_position' in core for core in he_cores[:3])  # Check first 3
        orion_has_grid = all('grid_position' in core for core in orion_cores[:3])
        
        if he_has_grid and orion_has_grid:
            logger.info("Using grid-based matching (cores have grid position information)")
            return self._match_cores_grid_based(he_detection_results, orion_detection_results)
        else:
            logger.info("Using traditional spatial matching (no grid information available)")
            return self._match_cores_traditional(he_detection_results, orion_detection_results)
    
    def _match_cores_grid_based(self, he_detection_results: Dict, orion_detection_results: Dict) -> Dict:
        """Match cores using grid-based approach."""
        
        # Create grid matcher configuration
        grid_config = TMAGridMatchingConfig(
            max_grid_shift=3,
            min_match_confidence=self.config.min_match_confidence,
            save_visualizations=self.config.save_visualizations
        )
        
        # Create grid matcher and perform matching
        grid_matcher = TMAGridMatcher(grid_config)
        grid_results = grid_matcher.match_grid_cores(he_detection_results, orion_detection_results)
        
        # Convert to our expected format
        if grid_results['success']:
            # Convert grid matches to our standard format
            matches = []
            for match in grid_results['matches']:
                standard_match = {
                    'he_core_id': match['he_core_id'],
                    'orion_core_id': match['orion_core_id'],
                    'distance': match['distance'],
                    'confidence': match['confidence'],
                    'size_ratio': match['size_ratio'],
                    'matching_method': 'grid_based'
                }
                matches.append(standard_match)
            
            # Create our standard results format
            matching_results = {
                'matches': matches,
                'high_quality_matches': len(matches),
                'matching_method': 'grid_based',
                'matching_statistics': grid_results['matching_statistics'],
                'grid_alignment': grid_results.get('grid_alignment'),
                'success': True
            }
            
            return matching_results
        else:
            logger.warning("Grid-based matching failed, falling back to traditional matching")
            return self._match_cores_traditional(he_detection_results, orion_detection_results)
    
    def _match_cores_traditional(self, he_detection_results: Dict, orion_detection_results: Dict) -> Dict:
        """Match cores using traditional spatial approach."""
        
        he_cores = he_detection_results['cores']
        orion_cores = orion_detection_results['cores']
        
        # Extract features for matching
        he_features = self._extract_core_features(he_cores)
        orion_features = self._extract_core_features(orion_cores)
        
        # Perform matching
        if self.config.matching_method == "hungarian":
            matches = self._match_hungarian(he_features, orion_features, he_cores, orion_cores)
        elif self.config.matching_method == "nearest_neighbor":
            matches = self._match_nearest_neighbor(he_features, orion_features, he_cores, orion_cores)
        elif self.config.matching_method == "greedy":
            matches = self._match_greedy(he_features, orion_features, he_cores, orion_cores)
        else:
            raise ValueError(f"Unknown matching method: {self.config.matching_method}")
        
        # Assess match quality
        matches_with_quality = self._assess_match_quality(matches, he_cores, orion_cores)
        
        # Filter matches based on quality criteria
        high_quality_matches = self._filter_high_quality_matches(matches_with_quality)
        
        # Compute matching statistics
        statistics = self._compute_matching_statistics(matches_with_quality, he_cores, orion_cores)
        
        # Create results dictionary
        matching_results = {
            'matches': high_quality_matches,
            'high_quality_matches': len(high_quality_matches),
            'all_matches': matches_with_quality,
            'matching_method': self.config.matching_method,
            'matching_statistics': statistics,
            'success': len(high_quality_matches) > 0
        }
        
        return matching_results
    
    def _extract_core_features(self, cores: List[Dict]) -> np.ndarray:
        """Extract features from cores for matching."""
        
        features = []
        for core in cores:
            # Spatial features (centroid coordinates)
            cx, cy = core['centroid_xy']
            
            # Size features
            area = core['area']
            diameter = core['equiv_diameter']
            
            # Shape features
            circularity = core.get('circularity', 0.5)
            solidity = core.get('solidity', 0.8)
            
            # Combine features
            feature_vector = [cx, cy, area, diameter, circularity, solidity]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _match_hungarian(self, he_features: np.ndarray, orion_features: np.ndarray, 
                        he_cores: List[Dict], orion_cores: List[Dict]) -> List[Dict]:
        """Match cores using the Hungarian algorithm (optimal assignment)."""
        
        # Calculate cost matrix based on distance and size similarity
        he_positions = he_features[:, :2]  # x, y coordinates
        orion_positions = orion_features[:, :2]
        
        # Distance matrix
        distance_matrix = cdist(he_positions, orion_positions, metric='euclidean')
        
        # Size similarity matrix
        he_sizes = he_features[:, 2]  # area
        orion_sizes = orion_features[:, 2]
        
        size_ratio_matrix = np.abs(np.log(he_sizes.reshape(-1, 1) / orion_sizes.reshape(1, -1)))
        
        # Combined cost matrix
        # Normalize both matrices to [0, 1]
        distance_norm = distance_matrix / np.max(distance_matrix) if np.max(distance_matrix) > 0 else distance_matrix
        size_norm = size_ratio_matrix / np.max(size_ratio_matrix) if np.max(size_ratio_matrix) > 0 else size_ratio_matrix
        
        cost_matrix = (self.config.distance_weight * distance_norm + 
                      self.config.size_similarity_weight * size_norm)
        
        # Apply distance threshold by setting high costs for distant pairs
        cost_matrix[distance_matrix > self.config.max_distance_threshold] = np.inf
        
        # Solve assignment problem
        he_indices, orion_indices = linear_sum_assignment(cost_matrix)
        
        # Create matches
        matches = []
        for he_idx, orion_idx in zip(he_indices, orion_indices):
            if cost_matrix[he_idx, orion_idx] != np.inf:  # Valid assignment
                match = {
                    'he_core_id': he_cores[he_idx]['id'],
                    'orion_core_id': orion_cores[orion_idx]['id'],
                    'he_core_idx': int(he_idx),
                    'orion_core_idx': int(orion_idx),
                    'distance': distance_matrix[he_idx, orion_idx],
                    'size_ratio': he_sizes[he_idx] / orion_sizes[orion_idx],
                    'cost': cost_matrix[he_idx, orion_idx]
                }
                matches.append(match)
        
        return matches
    
    def _match_nearest_neighbor(self, he_features: np.ndarray, orion_features: np.ndarray,
                               he_cores: List[Dict], orion_cores: List[Dict]) -> List[Dict]:
        """Match cores using nearest neighbor approach."""
        
        he_positions = he_features[:, :2]
        orion_positions = orion_features[:, :2]
        he_sizes = he_features[:, 2]
        orion_sizes = orion_features[:, 2]
        
        matches = []
        used_orion_indices = set()
        
        # Sort H&E cores by some criterion (e.g., size, position)
        he_order = sorted(range(len(he_cores)), key=lambda i: -he_sizes[i])  # Largest first
        
        for he_idx in he_order:
            he_pos = he_positions[he_idx]
            he_size = he_sizes[he_idx]
            
            # Find closest available Orion core
            best_orion_idx = None
            best_distance = float('inf')
            
            for orion_idx in range(len(orion_cores)):
                if orion_idx in used_orion_indices:
                    continue
                
                orion_pos = orion_positions[orion_idx]
                orion_size = orion_sizes[orion_idx]
                
                distance = np.linalg.norm(he_pos - orion_pos)
                
                # Check distance threshold
                if distance > self.config.max_distance_threshold:
                    continue
                
                # Check size similarity
                size_ratio = he_size / orion_size
                if (size_ratio > self.config.max_size_ratio_difference or 
                    size_ratio < 1/self.config.max_size_ratio_difference):
                    continue
                
                if distance < best_distance:
                    best_distance = distance
                    best_orion_idx = orion_idx
            
            # Create match if found
            if best_orion_idx is not None:
                match = {
                    'he_core_id': he_cores[he_idx]['id'],
                    'orion_core_id': orion_cores[best_orion_idx]['id'],
                    'he_core_idx': int(he_idx),
                    'orion_core_idx': int(best_orion_idx),
                    'distance': best_distance,
                    'size_ratio': he_size / orion_sizes[best_orion_idx],
                }
                matches.append(match)
                used_orion_indices.add(best_orion_idx)
        
        return matches
    
    def _match_greedy(self, he_features: np.ndarray, orion_features: np.ndarray,
                     he_cores: List[Dict], orion_cores: List[Dict]) -> List[Dict]:
        """Match cores using a greedy approach based on combined distance and size similarity."""
        
        he_positions = he_features[:, :2]
        orion_positions = orion_features[:, :2]
        he_sizes = he_features[:, 2]
        orion_sizes = orion_features[:, 2]
        
        # Calculate all pairwise distances and similarities
        distance_matrix = cdist(he_positions, orion_positions, metric='euclidean')
        
        # Create list of all possible matches with scores
        potential_matches = []
        for he_idx in range(len(he_cores)):
            for orion_idx in range(len(orion_cores)):
                distance = distance_matrix[he_idx, orion_idx]
                
                if distance > self.config.max_distance_threshold:
                    continue
                
                size_ratio = he_sizes[he_idx] / orion_sizes[orion_idx]
                if (size_ratio > self.config.max_size_ratio_difference or 
                    size_ratio < 1/self.config.max_size_ratio_difference):
                    continue
                
                # Calculate combined score (lower is better)
                distance_score = distance / self.config.max_distance_threshold
                size_score = abs(np.log(size_ratio))
                
                combined_score = (self.config.distance_weight * distance_score + 
                                self.config.size_similarity_weight * size_score)
                
                potential_matches.append({
                    'he_idx': he_idx,
                    'orion_idx': orion_idx,
                    'distance': distance,
                    'size_ratio': size_ratio,
                    'score': combined_score
                })
        
        # Sort by score (best matches first)
        potential_matches.sort(key=lambda x: x['score'])
        
        # Greedily select matches
        matches = []
        used_he = set()
        used_orion = set()
        
        for match in potential_matches:
            he_idx = match['he_idx']
            orion_idx = match['orion_idx']
            
            if he_idx not in used_he and orion_idx not in used_orion:
                final_match = {
                    'he_core_id': he_cores[he_idx]['id'],
                    'orion_core_id': orion_cores[orion_idx]['id'],
                    'he_core_idx': int(he_idx),
                    'orion_core_idx': int(orion_idx),
                    'distance': match['distance'],
                    'size_ratio': match['size_ratio'],
                    'score': match['score']
                }
                matches.append(final_match)
                used_he.add(he_idx)
                used_orion.add(orion_idx)
        
        return matches
    
    def _assess_match_quality(self, matches: List[Dict], he_cores: List[Dict], orion_cores: List[Dict]) -> List[Dict]:
        """Assess the quality of each match."""
        
        for match in matches:
            he_core = he_cores[match['he_core_idx']]
            orion_core = orion_cores[match['orion_core_idx']]
            
            # Distance-based confidence (closer is better)
            distance_confidence = max(0, 1 - match['distance'] / self.config.max_distance_threshold)
            
            # Size similarity confidence
            size_ratio = match['size_ratio']
            size_confidence = max(0, 1 - abs(np.log(size_ratio)) / np.log(self.config.max_size_ratio_difference))
            
            # Shape similarity (if available)
            he_circularity = he_core.get('circularity', 0.5)
            orion_circularity = orion_core.get('circularity', 0.5)
            shape_confidence = 1 - abs(he_circularity - orion_circularity)
            
            # Combined confidence
            overall_confidence = (0.5 * distance_confidence + 
                                0.3 * size_confidence + 
                                0.2 * shape_confidence)
            
            match['quality_metrics'] = {
                'distance_confidence': distance_confidence,
                'size_confidence': size_confidence,
                'shape_confidence': shape_confidence,
                'overall_confidence': overall_confidence
            }
            
            match['confidence'] = overall_confidence
        
        return matches
    
    def _filter_matches(self, matches: List[Dict]) -> List[Dict]:
        """Filter matches based on quality criteria."""
        
        filtered_matches = []
        
        for match in matches:
            # Check confidence threshold
            if match['confidence'] < self.config.min_match_confidence:
                logger.debug(f"Filtering out match with low confidence: {match['confidence']:.3f}")
                continue
            
            # Check distance threshold (should already be satisfied, but double-check)
            if match['distance'] > self.config.max_distance_threshold:
                continue
            
            # Check size ratio threshold
            size_ratio = match['size_ratio']
            if (size_ratio > self.config.max_size_ratio_difference or 
                size_ratio < 1/self.config.max_size_ratio_difference):
                continue
            
            filtered_matches.append(match)
        
        return filtered_matches
    
    def _find_unmatched_cores(self, cores: List[Dict], matches: List[Dict], core_type: str) -> List[Dict]:
        """Find cores that were not matched."""
        
        if core_type == 'he':
            matched_indices = {match['he_core_idx'] for match in matches}
        else:  # orion
            matched_indices = {match['orion_core_idx'] for match in matches}
        
        unmatched_cores = []
        for i, core in enumerate(cores):
            if i not in matched_indices:
                unmatched_cores.append({
                    'core_id': core['id'],
                    'index': i,
                    'centroid_xy': core['centroid_xy'],
                    'area': core['area'],
                    'reason': 'no_suitable_match_found'
                })
        
        return unmatched_cores
    
    def _calculate_matching_statistics(self, matches: List[Dict], he_cores: List[Dict], orion_cores: List[Dict]) -> Dict:
        """Calculate statistics about the matching results."""
        
        if not matches:
            return {
                'match_rate_he': 0.0,
                'match_rate_orion': 0.0,
                'mean_distance': None,
                'median_distance': None,
                'mean_confidence': None,
                'mean_size_ratio': None
            }
        
        distances = [match['distance'] for match in matches]
        confidences = [match['confidence'] for match in matches]
        size_ratios = [match['size_ratio'] for match in matches]
        
        return {
            'match_rate_he': len(matches) / len(he_cores),
            'match_rate_orion': len(matches) / len(orion_cores),
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'mean_size_ratio': np.mean(size_ratios),
            'median_size_ratio': np.median(size_ratios)
        }
    
    def _create_empty_matching_results(self, he_detection_results: Dict, orion_detection_results: Dict) -> Dict:
        """Create empty matching results when no cores are available."""
        
        return {
            'he_detection_results': he_detection_results,
            'orion_detection_results': orion_detection_results,
            'matching_method': self.config.matching_method,
            'total_he_cores': len(he_detection_results.get('cores', [])),
            'total_orion_cores': len(orion_detection_results.get('cores', [])),
            'total_matches_found': 0,
            'high_quality_matches': 0,
            'matches': [],
            'unmatched_he_cores': he_detection_results.get('cores', []),
            'unmatched_orion_cores': orion_detection_results.get('cores', []),
            'matching_statistics': self._calculate_matching_statistics([], [], []),
            'config': self.config.__dict__
        }
    
    def visualize_matches(self, matching_results: Dict, he_image_path: str, orion_image_path: str, 
                         output_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of matching results."""
        
        from tifffile import imread
        
        # Load images for visualization
        he_image = imread(he_image_path)
        orion_image = imread(orion_image_path)
        
        # Prepare display images
        he_display = self._prepare_display_image(he_image)
        orion_display = self._prepare_display_image(orion_image)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # H&E with cores
        axes[0].imshow(he_display)
        axes[0].set_title(f'H&E Cores ({matching_results["total_he_cores"]})')
        axes[0].axis('off')
        
        # Draw H&E cores
        for core in matching_results['he_detection_results']['cores']:
            cx, cy = core['centroid_xy']
            axes[0].plot(cx, cy, 'ro', markersize=8)
            axes[0].text(cx, cy-30, f"H{core['id']}", ha='center', color='red', fontweight='bold')
        
        # Orion with cores
        axes[1].imshow(orion_display)
        axes[1].set_title(f'Orion Cores ({matching_results["total_orion_cores"]})')
        axes[1].axis('off')
        
        # Draw Orion cores
        for core in matching_results['orion_detection_results']['cores']:
            cx, cy = core['centroid_xy']
            axes[1].plot(cx, cy, 'bo', markersize=8)
            axes[1].text(cx, cy-30, f"O{core['id']}", ha='center', color='blue', fontweight='bold')
        
        # Combined view with matches
        axes[2].imshow(he_display, alpha=0.7)
        axes[2].set_title(f'Matches ({matching_results["high_quality_matches"]})')
        axes[2].axis('off')
        
        # Draw matches
        he_cores = {core['id']: core for core in matching_results['he_detection_results']['cores']}
        orion_cores = {core['id']: core for core in matching_results['orion_detection_results']['cores']}
        
        for match in matching_results['matches']:
            he_core = he_cores[match['he_core_id']]
            orion_core = orion_cores[match['orion_core_id']]
            
            he_cx, he_cy = he_core['centroid_xy']
            orion_cx, orion_cy = orion_core['centroid_xy']
            
            # Draw cores
            axes[2].plot(he_cx, he_cy, 'ro', markersize=8)
            axes[2].plot(orion_cx, orion_cy, 'bo', markersize=8)
            
            # Draw connection line
            axes[2].plot([he_cx, orion_cx], [he_cy, orion_cy], 'g-', linewidth=2, alpha=0.7)
            
            # Add match info
            mid_x, mid_y = (he_cx + orion_cx) / 2, (he_cy + orion_cy) / 2
            confidence = match['confidence']
            axes[2].text(mid_x, mid_y, f"{confidence:.2f}", ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                        fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Matching visualization saved to {output_path}")
        
        return fig
    
    def _prepare_display_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for display."""
        
        if image.ndim == 3:
            if image.shape[0] <= 10:  # Multi-channel (C, H, W)
                if image.shape[0] >= 3:
                    display_img = np.transpose(image[:3], (1, 2, 0))
                else:
                    display_img = np.stack([image[0]] * 3, axis=2)
            else:  # RGB (H, W, C)
                display_img = image
        else:
            display_img = np.stack([image] * 3, axis=2)
        
        # Normalize for display
        if display_img.dtype != np.uint8:
            display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return display_img


def main():
    """Example usage of the CoreMatcher."""
    
    from core_detector import CoreDetector, CoreDetectionConfig
    
    # Configuration
    detection_config = CoreDetectionConfig()
    matching_config = CoreMatchingConfig(
        matching_method="hungarian",
        max_distance_threshold=250.0,
        save_visualizations=True
    )
    
    # Create detector and matcher
    detector = CoreDetector(detection_config)
    matcher = CoreMatcher(matching_config)
    
    # Example paths
    he_path = "data/raw/TA118-HEraw.ome.tiff"
    orion_path = "data/raw/TA118-Orionraw.ome.tiff"
    
    # Detect cores in both images
    if Path(he_path).exists() and Path(orion_path).exists():
        he_results = detector.detect_cores(he_path, image_type="he")
        orion_results = detector.detect_cores(orion_path, image_type="orion")
        
        # Match cores
        matching_results = matcher.match_cores(he_results, orion_results)
        
        print(f"Matched {matching_results['high_quality_matches']} core pairs")
        print(f"Match rate: H&E {matching_results['matching_statistics']['match_rate_he']:.2%}, "
              f"Orion {matching_results['matching_statistics']['match_rate_orion']:.2%}")
        
        # Create visualization
        if matching_config.save_visualizations:
            matcher.visualize_matches(matching_results, he_path, orion_path, "core_matching_results.png")


if __name__ == "__main__":
    main() 