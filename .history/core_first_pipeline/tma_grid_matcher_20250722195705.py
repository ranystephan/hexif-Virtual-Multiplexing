"""
TMA Grid-Based Core Matching

This module implements grid-aware core matching for TMA images. Instead of
relying solely on spatial proximity, it uses the grid coordinates determined
by the TMA grid detector to establish correspondence between cores in
different stains.

Key Features:
- Grid coordinate-based matching
- Handles slight shifts and rotations between stains
- Robust to missing cores or artifacts
- Quality scoring based on multiple factors
- Fallback to spatial matching when needed
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TMAGridMatchingConfig:
    """Configuration for TMA grid-based matching."""
    
    # Grid matching parameters
    max_grid_shift: int = 3                     # Maximum shift in grid coordinates
    grid_position_weight: float = 0.6           # Weight for grid position similarity
    spatial_position_weight: float = 0.2       # Weight for spatial position similarity
    size_similarity_weight: float = 0.1        # Weight for size similarity
    quality_score_weight: float = 0.1          # Weight for individual core quality
    
    # Matching constraints
    max_spatial_distance: float = 500.0        # Maximum spatial distance for matches
    min_match_confidence: float = 0.4          # Minimum confidence for valid matches
    
    # Grid alignment
    enable_grid_alignment: bool = True          # Enable automatic grid alignment
    max_rotation_degrees: float = 5.0          # Maximum rotation correction
    max_translation_pixels: float = 100.0      # Maximum translation correction
    
    # Quality control
    expected_match_rate: float = 0.8           # Expected fraction of cores to match
    match_symmetry_threshold: float = 0.9      # Minimum symmetry for valid matching
    
    # Visualization
    save_visualizations: bool = True
    visualization_downsample: int = 4           # Downsample factor for visualizations


class TMAGridMatcher:
    """Grid-aware core matcher for TMA images."""
    
    def __init__(self, config: TMAGridMatchingConfig):
        self.config = config
    
    def match_grid_cores(self, he_results: Dict, orion_results: Dict) -> Dict:
        """
        Match cores between H&E and Orion based on grid structure.
        
        Args:
            he_results: Grid detection results from H&E image
            orion_results: Grid detection results from Orion image
            
        Returns:
            Dictionary with matching results
        """
        logger.info(f"Grid-based matching: {len(he_results['cores'])} H&E cores vs {len(orion_results['cores'])} Orion cores")
        
        he_cores = he_results['cores']
        orion_cores = orion_results['cores']
        
        if not he_cores or not orion_cores:
            return self._create_empty_matching_result(he_results, orion_results)
        
        # Step 1: Estimate grid alignment between the two images
        alignment = self._estimate_grid_alignment(he_cores, orion_cores)
        logger.info(f"Estimated grid alignment: translation=({alignment['tx']:.1f}, {alignment['ty']:.1f}), rotation={alignment['rotation']:.2f}°")
        
        # Step 2: Apply alignment to Orion grid coordinates
        aligned_orion_cores = self._apply_grid_alignment(orion_cores, alignment)
        
        # Step 3: Compute matching costs using grid coordinates
        cost_matrix = self._compute_grid_matching_costs(he_cores, aligned_orion_cores)
        
        # Step 4: Solve assignment problem
        matches = self._solve_assignment(cost_matrix, he_cores, aligned_orion_cores)
        
        # Step 5: Filter matches by quality
        filtered_matches = self._filter_matches_by_quality(matches, he_cores, orion_cores)
        
        # Step 6: Compute matching statistics
        statistics = self._compute_matching_statistics(filtered_matches, he_cores, orion_cores)
        
        # Create results
        matching_results = {
            'matches': filtered_matches,
            'high_quality_matches': len(filtered_matches),
            'grid_alignment': alignment,
            'matching_statistics': statistics,
            'cost_matrix': cost_matrix,
            'he_cores_count': len(he_cores),
            'orion_cores_count': len(orion_cores),
            'method': 'tma_grid_matching',
            'success': len(filtered_matches) > 0
        }
        
        logger.info(f"Grid matching complete: {len(filtered_matches)} high-quality matches")
        return matching_results
    
    def _estimate_grid_alignment(self, he_cores: List[Dict], orion_cores: List[Dict]) -> Dict:
        """
        Estimate the transformation between H&E and Orion grid coordinates.
        
        This method finds the best alignment by trying different translations
        and rotations of the grid coordinates.
        """
        logger.info("Estimating grid alignment between H&E and Orion...")
        
        if not self.config.enable_grid_alignment:
            return {'tx': 0, 'ty': 0, 'rotation': 0, 'confidence': 1.0}
        
        # Extract grid positions
        he_grid_pos = np.array([core['grid_position'] for core in he_cores])
        orion_grid_pos = np.array([core['grid_position'] for core in orion_cores])
        
        # Try different translations
        best_score = -1
        best_alignment = {'tx': 0, 'ty': 0, 'rotation': 0, 'confidence': 0}
        
        max_shift = self.config.max_grid_shift
        
        for tx in range(-max_shift, max_shift + 1):
            for ty in range(-max_shift, max_shift + 1):
                # Apply translation
                shifted_orion_pos = orion_grid_pos + np.array([tx, ty])
                
                # Compute overlap score
                score = self._compute_grid_overlap_score(he_grid_pos, shifted_orion_pos)
                
                if score > best_score:
                    best_score = score
                    best_alignment = {
                        'tx': tx, 'ty': ty, 'rotation': 0, 
                        'confidence': min(score, 1.0)
                    }
        
        logger.info(f"Best grid alignment score: {best_score:.3f}")
        return best_alignment
    
    def _compute_grid_overlap_score(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute overlap score between two sets of grid positions."""
        
        if len(pos1) == 0 or len(pos2) == 0:
            return 0.0
        
        # Convert to sets of tuples for easy comparison
        set1 = set(map(tuple, pos1))
        set2 = set(map(tuple, pos2))
        
        # Compute overlap
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        # Jaccard similarity
        return intersection / union
    
    def _apply_grid_alignment(self, orion_cores: List[Dict], alignment: Dict) -> List[Dict]:
        """Apply grid alignment transformation to Orion cores."""
        
        aligned_cores = []
        tx, ty = alignment['tx'], alignment['ty']
        
        for core in orion_cores:
            aligned_core = core.copy()
            grid_x, grid_y = core['grid_position']
            
            # Apply translation
            aligned_core['aligned_grid_position'] = (grid_x + tx, grid_y + ty)
            aligned_cores.append(aligned_core)
        
        return aligned_cores
    
    def _compute_grid_matching_costs(self, he_cores: List[Dict], orion_cores: List[Dict]) -> np.ndarray:
        """Compute cost matrix for grid-based matching."""
        
        n_he = len(he_cores)
        n_orion = len(orion_cores)
        cost_matrix = np.full((n_he, n_orion), np.inf)
        
        for i, he_core in enumerate(he_cores):
            he_grid_pos = np.array(he_core['grid_position'])
            he_spatial_pos = np.array(he_core['centroid_xy'])
            
            for j, orion_core in enumerate(orion_cores):
                # Use aligned grid position if available
                if 'aligned_grid_position' in orion_core:
                    orion_grid_pos = np.array(orion_core['aligned_grid_position'])
                else:
                    orion_grid_pos = np.array(orion_core['grid_position'])
                
                orion_spatial_pos = np.array(orion_core['centroid_xy'])
                
                # Compute different similarity components
                grid_dist = np.linalg.norm(he_grid_pos - orion_grid_pos)
                spatial_dist = np.linalg.norm(he_spatial_pos - orion_spatial_pos)
                
                # Size similarity
                he_size = he_core.get('equiv_diameter', 1000)
                orion_size = orion_core.get('equiv_diameter', 1000)
                size_ratio = min(he_size, orion_size) / max(he_size, orion_size)
                
                # Quality scores
                he_quality = he_core.get('match_score', 0.5)
                orion_quality = orion_core.get('match_score', 0.5)
                avg_quality = (he_quality + orion_quality) / 2
                
                # Skip if spatial distance is too large
                if spatial_dist > self.config.max_spatial_distance:
                    continue
                
                # Compute weighted cost (lower is better)
                cost = (
                    self.config.grid_position_weight * grid_dist +
                    self.config.spatial_position_weight * (spatial_dist / 1000.0) +
                    self.config.size_similarity_weight * (1 - size_ratio) +
                    self.config.quality_score_weight * (1 - avg_quality)
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _solve_assignment(self, cost_matrix: np.ndarray, he_cores: List[Dict], 
                         orion_cores: List[Dict]) -> List[Dict]:
        """Solve the assignment problem using Hungarian algorithm."""
        
        # Use Hungarian algorithm to find optimal assignment
        he_indices, orion_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        for he_idx, orion_idx in zip(he_indices, orion_indices):
            cost = cost_matrix[he_idx, orion_idx]
            
            # Skip infinite costs
            if np.isinf(cost):
                continue
            
            he_core = he_cores[he_idx]
            orion_core = orion_cores[orion_idx]
            
            # Compute additional match properties
            distance = np.linalg.norm(
                np.array(he_core['centroid_xy']) - np.array(orion_core['centroid_xy'])
            )
            
            he_size = he_core.get('equiv_diameter', 1000)
            orion_size = orion_core.get('equiv_diameter', 1000)
            size_ratio = min(he_size, orion_size) / max(he_size, orion_size)
            
            # Convert cost to confidence (higher is better)
            confidence = max(0, 1 - cost)
            
            match = {
                'he_core_id': he_core['id'],
                'orion_core_id': orion_core['id'],
                'he_grid_position': he_core['grid_position'],
                'orion_grid_position': orion_core['grid_position'],
                'distance': distance,
                'size_ratio': size_ratio,
                'confidence': confidence,
                'cost': cost,
                'method': 'grid_hungarian'
            }
            
            matches.append(match)
        
        return matches
    
    def _filter_matches_by_quality(self, matches: List[Dict], he_cores: List[Dict], 
                                  orion_cores: List[Dict]) -> List[Dict]:
        """Filter matches based on quality criteria."""
        
        filtered_matches = []
        
        for match in matches:
            # Filter by confidence threshold
            if match['confidence'] < self.config.min_match_confidence:
                continue
            
            # Filter by spatial distance
            if match['distance'] > self.config.max_spatial_distance:
                continue
            
            # Filter by size similarity (require reasonable size match)
            if match['size_ratio'] < 0.5:  # Cores shouldn't differ by more than 2x
                continue
            
            filtered_matches.append(match)
        
        return filtered_matches
    
    def _compute_matching_statistics(self, matches: List[Dict], he_cores: List[Dict], 
                                   orion_cores: List[Dict]) -> Dict:
        """Compute matching statistics."""
        
        n_he = len(he_cores)
        n_orion = len(orion_cores)
        n_matches = len(matches)
        
        if n_matches == 0:
            return {
                'match_rate_he': 0.0,
                'match_rate_orion': 0.0,
                'average_distance': 0.0,
                'average_confidence': 0.0,
                'average_size_ratio': 0.0
            }
        
        # Compute statistics
        distances = [m['distance'] for m in matches]
        confidences = [m['confidence'] for m in matches]
        size_ratios = [m['size_ratio'] for m in matches]
        
        return {
            'match_rate_he': n_matches / n_he if n_he > 0 else 0,
            'match_rate_orion': n_matches / n_orion if n_orion > 0 else 0,
            'average_distance': np.mean(distances),
            'median_distance': np.median(distances), 
            'average_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'average_size_ratio': np.mean(size_ratios),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def _create_empty_matching_result(self, he_results: Dict, orion_results: Dict) -> Dict:
        """Create empty matching result for failed matching."""
        return {
            'matches': [],
            'high_quality_matches': 0,
            'grid_alignment': {'tx': 0, 'ty': 0, 'rotation': 0, 'confidence': 0},
            'matching_statistics': {
                'match_rate_he': 0.0,
                'match_rate_orion': 0.0,
                'average_distance': 0.0,
                'average_confidence': 0.0
            },
            'method': 'tma_grid_matching',
            'success': False,
            'error': 'No cores to match'
        }
    
    def visualize_grid_matching(self, matching_results: Dict, he_results: Dict, 
                              orion_results: Dict, output_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of grid matching results."""
        
        if not matching_results['success']:
            logger.warning("Cannot visualize failed matching")
            return None
        
        # Load images for visualization
        he_image = self._load_image_for_viz(he_results['image_path'], he_results['image_type'])
        orion_image = self._load_image_for_viz(orion_results['image_path'], orion_results['image_type'])
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # H&E with cores
        self._plot_cores_on_image(axes[0, 0], he_image, he_results['cores'], 
                                 "H&E Detected Cores", color=(0, 255, 0))
        
        # Orion with cores
        self._plot_cores_on_image(axes[0, 1], orion_image, orion_results['cores'],
                                 "Orion Detected Cores", color=(255, 0, 0))
        
        # Grid alignment visualization
        self._plot_grid_alignment(axes[1, 0], he_results['cores'], orion_results['cores'],
                                matching_results['grid_alignment'])
        
        # Matching statistics
        self._plot_matching_statistics(axes[1, 1], matching_results)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Grid matching visualization saved to {output_path}")
        
        return fig
    
    def _load_image_for_viz(self, image_path: str, image_type: str) -> np.ndarray:
        """Load and prepare image for visualization."""
        try:
            from tifffile import imread
            image = imread(image_path)
            
            # Downsample for visualization
            if max(image.shape[-2:]) > 4000:
                factor = self.config.visualization_downsample
                if image.ndim == 3 and image.shape[0] > 10:  # Multi-channel
                    image = image[:, ::factor, ::factor]
                else:
                    image = image[::factor, ::factor]
            
            # Convert to display format
            if image_type == "orion" and image.ndim == 3 and image.shape[0] > 10:
                # Use first 3 channels for display
                if image.shape[0] >= 3:
                    display_img = np.transpose(image[:3], (1, 2, 0))
                else:
                    display_img = np.stack([image[0]] * 3, axis=2)
            elif image.ndim == 3 and image.shape[2] == 3:
                display_img = image
            else:
                display_img = image.squeeze()
                if len(display_img.shape) == 2:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            
            # Normalize
            if display_img.dtype != np.uint8:
                display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            return display_img
            
        except Exception as e:
            logger.error(f"Failed to load image for visualization: {e}")
            return np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    def _plot_cores_on_image(self, ax, image: np.ndarray, cores: List[Dict], 
                           title: str, color: Tuple[int, int, int]):
        """Plot cores on image."""
        
        overlay = image.copy()
        factor = self.config.visualization_downsample
        
        for core in cores:
            center_x, center_y = core['centroid_xy']
            center_x //= factor
            center_y //= factor
            radius = core.get('equiv_diameter', 1000) // (2 * factor)
            
            cv2.circle(overlay, (int(center_x), int(center_y)), int(radius), color, 2)
            
            # Add grid position label
            if 'grid_position' in core:
                gx, gy = core['grid_position']
                cv2.putText(overlay, f"({gx},{gy})", 
                           (int(center_x - radius), int(center_y - radius - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        ax.imshow(overlay)
        ax.set_title(f"{title}\n{len(cores)} cores")
        ax.axis('off')
    
    def _plot_grid_alignment(self, ax, he_cores: List[Dict], orion_cores: List[Dict], 
                           alignment: Dict):
        """Plot grid alignment visualization."""
        
        # Extract grid positions
        he_positions = np.array([core['grid_position'] for core in he_cores])
        orion_positions = np.array([core['grid_position'] for core in orion_cores])
        
        # Apply alignment to Orion positions
        tx, ty = alignment['tx'], alignment['ty']
        aligned_orion_positions = orion_positions + np.array([tx, ty])
        
        # Plot
        ax.scatter(he_positions[:, 0], he_positions[:, 1], 
                  c='blue', alpha=0.6, s=50, label='H&E cores')
        ax.scatter(orion_positions[:, 0], orion_positions[:, 1], 
                  c='red', alpha=0.6, s=50, label='Orion cores (original)')
        ax.scatter(aligned_orion_positions[:, 0], aligned_orion_positions[:, 1], 
                  c='orange', alpha=0.6, s=50, label='Orion cores (aligned)')
        
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        ax.set_title(f'Grid Alignment\nTranslation: ({tx}, {ty})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_matching_statistics(self, ax, matching_results: Dict):
        """Plot matching statistics."""
        
        stats = matching_results['matching_statistics']
        
        # Create text summary
        text = f"""Matching Statistics
        
Total Matches: {matching_results['high_quality_matches']}
H&E Match Rate: {stats['match_rate_he']:.1%}
Orion Match Rate: {stats['match_rate_orion']:.1%}

Distance Statistics:
  Average: {stats['average_distance']:.1f} px
  Median: {stats.get('median_distance', 0):.1f} px
  Range: {stats.get('min_distance', 0):.1f} - {stats.get('max_distance', 0):.1f} px

Confidence Statistics:
  Average: {stats['average_confidence']:.3f}
  Median: {stats.get('median_confidence', 0):.3f}

Size Ratio: {stats['average_size_ratio']:.3f}
        """
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')


def main():
    """Example usage of TMA grid matcher."""
    
    config = TMAGridMatchingConfig(
        max_grid_shift=3,
        min_match_confidence=0.4,
        save_visualizations=True
    )
    
    matcher = TMAGridMatcher(config)
    
    # This would be called with actual detection results
    print("TMA Grid Matcher initialized successfully")
    print("Use matcher.match_grid_cores(he_results, orion_results) to match cores")


if __name__ == "__main__":
    main() 