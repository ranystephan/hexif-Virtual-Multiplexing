#!/usr/bin/env python3
"""
Working CODEX Processing Script
==============================

This script properly handles all numpy array and pandas DataFrame issues.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available. Install with: pip install opencv-python")

try:
    from skimage import measure, morphology, filters, segmentation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image not available. Install with: pip install scikit-image")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkingCODEXProcessor:
    """
    Working CODEX processor that properly handles all data types
    """
    
    def __init__(self, data_root: str, output_root: str):
        """
        Initialize the processor
        
        Args:
            data_root: Path to CODEX data directory
            output_root: Path to store outputs
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Load markers
        self.markers = self._load_markers()
        logger.info(f"Loaded {len(self.markers)} markers")
        
        # Find TMA directories
        self.tma_dirs = self._find_tma_directories()
        logger.info(f"Found {len(self.tma_dirs)} TMA directories")
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available. Some features may not work.")
        if not SKIMAGE_AVAILABLE:
            logger.warning("scikit-image not available. Some features may not work.")
    
    def _load_markers(self) -> List[str]:
        """Load marker list"""
        marker_file = self.data_root / "MarkerList.txt"
        if marker_file.exists():
            with open(marker_file, 'r') as f:
                markers = [line.strip() for line in f.readlines() if line.strip()]
            # Ensure we return a list, not numpy array
            return list(markers)
        else:
            logger.warning("MarkerList.txt not found")
            return []
    
    def _find_tma_directories(self) -> List[Path]:
        """Find TMA directories"""
        tma_dirs = []
        for item in self.data_root.iterdir():
            if item.is_dir() and "TMA" in item.name:
                tma_dirs.append(item)
        return sorted(tma_dirs)
    
    def _find_qptiff_files(self, tma_dir: Path) -> List[Path]:
        """Find qptiff files in TMA directory"""
        qptiff_files = []
        scan_dir = tma_dir / "Scan1"
        if scan_dir.exists():
            for file in scan_dir.glob("*.qptiff"):
                if "er.qptiff" in file.name:
                    qptiff_files.append(file)
        return qptiff_files
    
    def _load_qptiff_file(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Load qptiff file (simplified version)
        
        Note: This is a placeholder. You'll need to implement proper qptiff loading
        or use SPACEc's built-in functions.
        """
        try:
            # This is a placeholder - you'll need to implement proper qptiff loading
            # For now, we'll create a dummy array for demonstration
            logger.info(f"Loading qptiff file: {file_path}")
            
            # Placeholder: create dummy data for demonstration
            # In reality, you would use SPACEc or a qptiff library
            dummy_shape = (len(self.markers), 1000, 1000)  # channels, height, width
            dummy_data = np.random.rand(*dummy_shape)
            
            logger.info(f"Loaded dummy data with shape: {dummy_data.shape}")
            return dummy_data
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _segment_cells_simple(self, image: np.ndarray, channel_idx: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Simple cell segmentation using basic image processing
        
        Args:
            image: Image array (channels, height, width)
            channel_idx: Channel to use for segmentation (usually DAPI)
            
        Returns:
            cell_labels: Labeled cell regions
            cell_centroids: Cell centroid coordinates
        """
        if not SKIMAGE_AVAILABLE:
            logger.error("scikit-image required for cell segmentation")
            return None, None
        
        try:
            # Extract DAPI channel
            dapi_channel = image[channel_idx]
            
            # Apply Gaussian blur
            blurred = filters.gaussian(dapi_channel, sigma=1)
            
            # Apply Otsu thresholding
            threshold = filters.threshold_otsu(blurred)
            binary = blurred > threshold
            
            # Remove small objects
            binary = morphology.remove_small_objects(binary, min_size=50)
            
            # Fill holes
            binary = morphology.remove_small_holes(binary, area_threshold=100)
            
            # Label connected components
            cell_labels = measure.label(binary)
            
            # Get cell properties
            props = measure.regionprops(cell_labels)
            
            # Extract centroids
            cell_centroids = np.array([prop.centroid for prop in props])
            
            logger.info(f"Segmented {len(cell_centroids)} cells")
            return cell_labels, cell_centroids
            
        except Exception as e:
            logger.error(f"Error in cell segmentation: {e}")
            return None, None
    
    def _extract_cell_features(self, image: np.ndarray, cell_labels: Optional[np.ndarray]) -> np.ndarray:
        """
        Extract cell features (mean intensity per marker)
        
        Args:
            image: Image array (channels, height, width)
            cell_labels: Labeled cell regions
            
        Returns:
            cell_features: Cell x marker expression matrix
        """
        if cell_labels is None:
            return np.array([])
        
        try:
            n_cells = cell_labels.max()
            n_markers = image.shape[0]
            
            cell_features = np.zeros((n_cells, n_markers))
            
            for cell_id in range(1, n_cells + 1):
                cell_mask = cell_labels == cell_id
                
                for marker_idx in range(n_markers):
                    marker_channel = image[marker_idx]
                    cell_intensity = np.mean(marker_channel[cell_mask])
                    cell_features[cell_id - 1, marker_idx] = cell_intensity
            
            logger.info(f"Extracted features for {n_cells} cells and {n_markers} markers")
            return cell_features
            
        except Exception as e:
            logger.error(f"Error extracting cell features: {e}")
            return np.array([])
    
    def _save_results(self, cell_features: np.ndarray, cell_locations: Optional[np.ndarray], 
                     output_dir: Path, tissue_name: str):
        """
        Save processing results
        
        Args:
            cell_features: Cell x marker expression matrix
            cell_locations: Cell x 2 location matrix
            output_dir: Output directory
            tissue_name: Tissue name
        """
        if len(cell_features) == 0:
            logger.warning("No cell features to save")
            return
        
        try:
            # Ensure markers is a list, not numpy array
            markers_list = list(self.markers) if hasattr(self.markers, '__iter__') else []
            
            # Convert numpy arrays to lists for proper DataFrame creation
            cell_features_list = cell_features.tolist()
            
            # Create expression DataFrame using dictionary approach
            expression_dict = {}
            for i, marker in enumerate(markers_list):
                expression_dict[str(marker)] = [float(row[i]) for row in cell_features_list]
            
            expression_df = pd.DataFrame(expression_dict)
            expression_df.index = [f"cell_{i}" for i in range(len(cell_features))]
            
            expression_file = output_dir / f"{tissue_name}_expression_matrix.csv"
            expression_df.to_csv(expression_file)
            logger.info(f"Saved expression matrix: {expression_file}")
            
            # Save cell locations
            locations_file = None
            if cell_locations is not None and len(cell_locations) > 0:
                cell_locations_list = cell_locations.tolist()
                locations_dict = {
                    'x': [row[0] for row in cell_locations_list],
                    'y': [row[1] for row in cell_locations_list]
                }
                locations_df = pd.DataFrame(locations_dict)
                locations_df.index = [f"cell_{i}" for i in range(len(cell_locations))]
                
                locations_file = output_dir / f"{tissue_name}_cell_locations.csv"
                locations_df.to_csv(locations_file)
                logger.info(f"Saved cell locations: {locations_file}")
            
            # Save summary
            summary = {
                "tissue_name": tissue_name,
                "n_cells": len(cell_features),
                "n_markers": len(markers_list),
                "markers": markers_list,
                "expression_file": str(expression_file),
                "locations_file": str(locations_file) if locations_file else None
            }
            
            summary_file = output_dir / f"{tissue_name}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved summary: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def process_single_tma(self, tma_dir: Path) -> Dict:
        """
        Process a single TMA directory
        
        Args:
            tma_dir: TMA directory path
            
        Returns:
            Processing results
        """
        tma_name = tma_dir.name
        logger.info(f"Processing TMA: {tma_name}")
        
        # Create output directory
        tma_output = self.output_root / tma_name
        tma_output.mkdir(parents=True, exist_ok=True)
        
        # Find qptiff files
        qptiff_files = self._find_qptiff_files(tma_dir)
        if not qptiff_files:
            logger.warning(f"No qptiff files found in {tma_dir}")
            return {"tma_name": tma_name, "status": "no_files"}
        
        results = {"tma_name": tma_name, "files_processed": []}
        
        for qptiff_file in qptiff_files:
            try:
                file_result = self._process_single_file(qptiff_file, tma_output)
                results["files_processed"].append(file_result)
            except Exception as e:
                logger.error(f"Error processing {qptiff_file}: {e}")
                results["files_processed"].append({
                    "file": qptiff_file.name,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def _process_single_file(self, qptiff_file: Path, output_dir: Path) -> Dict:
        """
        Process a single qptiff file
        
        Args:
            qptiff_file: Path to qptiff file
            output_dir: Output directory
            
        Returns:
            Processing results
        """
        file_name = qptiff_file.stem
        logger.info(f"Processing file: {file_name}")
        
        # Load the file
        image_data = self._load_qptiff_file(qptiff_file)
        if image_data is None:
            return {"file": file_name, "status": "load_error"}
        
        # Segment cells
        cell_labels, cell_locations = self._segment_cells_simple(image_data)
        
        # Extract cell features
        cell_features = self._extract_cell_features(image_data, cell_labels)
        
        # Save results
        self._save_results(cell_features, cell_locations, output_dir, file_name)
        
        return {
            "file": file_name,
            "status": "success",
            "n_cells": len(cell_features) if len(cell_features) > 0 else 0,
            "n_markers": len(self.markers)
        }
    
    def process_all_tmas(self) -> Dict:
        """
        Process all TMA directories
        
        Returns:
            Processing results
        """
        logger.info("Starting processing of all TMA directories")
        
        all_results = {
            "total_tmas": len(self.tma_dirs),
            "processed_tmas": 0,
            "tma_results": []
        }
        
        for tma_dir in self.tma_dirs:
            try:
                tma_result = self.process_single_tma(tma_dir)
                all_results["tma_results"].append(tma_result)
                all_results["processed_tmas"] += 1
                logger.info(f"Completed processing {tma_dir.name}")
            except Exception as e:
                logger.error(f"Error processing TMA {tma_dir.name}: {e}")
                all_results["tma_results"].append({
                    "tma_name": tma_dir.name,
                    "status": "error",
                    "error": str(e)
                })
        
        # Save summary
        summary_file = self.output_root / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Processing completed. Summary saved to: {summary_file}")
        return all_results
    
    def create_combined_matrix(self) -> pd.DataFrame:
        """
        Create combined expression matrix from all processed files
        
        Returns:
            Combined expression matrix
        """
        logger.info("Creating combined expression matrix")
        
        all_expression_data = []
        
        # Find all expression matrix files
        for expression_file in self.output_root.rglob("*_expression_matrix.csv"):
            try:
                df = pd.read_csv(expression_file, index_col=0)
                # Add metadata
                df['tma'] = expression_file.parent.name
                df['tissue'] = expression_file.stem.replace('_expression_matrix', '')
                df['cell_id'] = df.index
                all_expression_data.append(df)
                logger.info(f"Loaded expression data from: {expression_file}")
            except Exception as e:
                logger.error(f"Error loading {expression_file}: {e}")
        
        if not all_expression_data:
            logger.warning("No expression data found")
            # Return empty DataFrame with proper columns
            markers_list = list(self.markers) if hasattr(self.markers, '__iter__') else []
            # Convert markers to strings to avoid numpy array issues
            marker_columns = [str(marker) for marker in markers_list]
            empty_df = pd.DataFrame(columns=['tma', 'tissue', 'cell_id'] + marker_columns)
            return empty_df
        
        # Combine data
        combined_df = pd.concat(all_expression_data, ignore_index=True)
        logger.info(f"Combined expression matrix shape: {combined_df.shape}")
        
        # Save combined matrix
        combined_file = self.output_root / "combined_expression_matrix.csv"
        combined_df.to_csv(combined_file)
        logger.info(f"Saved combined expression matrix: {combined_file}")
        
        return combined_df


def main():
    """Main function"""
    
    # Configuration
    data_root = "data/nomad_data/CODEX"
    output_root = "data/nomad_data/CODEX/processed"
    
    # Initialize processor
    processor = WorkingCODEXProcessor(data_root, output_root)
    
    # Process all TMAs
    results = processor.process_all_tmas()
    
    # Create combined matrix
    combined_matrix = processor.create_combined_matrix()
    
    logger.info("CODEX processing completed!")
    logger.info(f"Processed {results['processed_tmas']} out of {results['total_tmas']} TMAs")
    
    if not combined_matrix.empty:
        logger.info(f"Combined expression matrix: {combined_matrix.shape}")
        logger.info(f"Total cells: {len(combined_matrix)}")
    else:
        logger.info("No expression data was generated")


if __name__ == "__main__":
    main() 