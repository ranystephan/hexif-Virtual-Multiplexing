#!/usr/bin/env python3
"""
CODEX Processing Pipeline using SPACEc
=====================================

This script processes CODEX data from multiple TMA directories to extract:
1. Protein expression matrices (protein x cell)
2. Cell locations (x, y coordinates)
3. Cell segmentation and annotation

Author: AI Assistant
Date: 2024
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import SPACEc
try:
    import spacec as sp
    logger.info("SPACEc imported successfully")
except ImportError:
    logger.error("SPACEc not found. Please install it from: https://github.com/yuqiyuqitan/SPACEc")
    sys.exit(1)

class CODEXProcessor:
    """
    Main class for processing CODEX data using SPACEc
    """
    
    def __init__(self, data_root: str, output_root: str):
        """
        Initialize the CODEX processor
        
        Args:
            data_root: Path to the CODEX data directory
            output_root: Path to store processed outputs
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Load marker list
        self.markers = self._load_markers()
        logger.info(f"Loaded {len(self.markers)} markers")
        
        # Find TMA directories
        self.tma_dirs = self._find_tma_directories()
        logger.info(f"Found {len(self.tma_dirs)} TMA directories")
        
        # Set scanpy settings
        sc.settings.set_figure_params(dpi=80, facecolor='white')
        plt.rcParams["image.cmap"] = 'viridis'
    
    def _load_markers(self) -> List[str]:
        """Load marker list from MarkerList.txt"""
        marker_file = self.data_root / "MarkerList.txt"
        if marker_file.exists():
            with open(marker_file, 'r') as f:
                markers = [line.strip() for line in f.readlines() if line.strip()]
            return markers
        else:
            logger.warning("MarkerList.txt not found, using default markers")
            return []
    
    def _find_tma_directories(self) -> List[Path]:
        """Find all TMA directories"""
        tma_dirs = []
        for item in self.data_root.iterdir():
            if item.is_dir() and "TMA" in item.name:
                tma_dirs.append(item)
        return sorted(tma_dirs)
    
    def _find_qptiff_files(self, tma_dir: Path) -> List[Path]:
        """Find .qptiff files in TMA directory"""
        qptiff_files = []
        scan_dir = tma_dir / "Scan1"
        if scan_dir.exists():
            for file in scan_dir.glob("*.qptiff"):
                if "er.qptiff" in file.name:  # Look for processed files
                    qptiff_files.append(file)
        return qptiff_files
    
    def process_single_tma(self, tma_dir: Path, output_dir: Path) -> Dict:
        """
        Process a single TMA directory
        
        Args:
            tma_dir: Path to TMA directory
            output_dir: Output directory for this TMA
            
        Returns:
            Dictionary with processing results
        """
        tma_name = tma_dir.name
        logger.info(f"Processing TMA: {tma_name}")
        
        # Create output directory
        tma_output = output_dir / tma_name
        tma_output.mkdir(parents=True, exist_ok=True)
        
        # Find qptiff files
        qptiff_files = self._find_qptiff_files(tma_dir)
        if not qptiff_files:
            logger.warning(f"No qptiff files found in {tma_dir}")
            return {"tma_name": tma_name, "status": "no_files"}
        
        results = {"tma_name": tma_name, "files_processed": []}
        
        for qptiff_file in qptiff_files:
            try:
                file_result = self._process_qptiff_file(qptiff_file, tma_output)
                results["files_processed"].append(file_result)
            except Exception as e:
                logger.error(f"Error processing {qptiff_file}: {e}")
                results["files_processed"].append({
                    "file": qptiff_file.name,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def _process_qptiff_file(self, qptiff_file: Path, output_dir: Path) -> Dict:
        """
        Process a single qptiff file
        
        Args:
            qptiff_file: Path to qptiff file
            output_dir: Output directory
            
        Returns:
            Dictionary with processing results
        """
        file_name = qptiff_file.stem
        logger.info(f"Processing file: {file_name}")
        
        # Step 1: Downscale the image for tissue extraction
        downscale_output = output_dir / "downscaled"
        downscale_output.mkdir(exist_ok=True)
        
        try:
            resized_im = sp.hf.downscale_tissue(
                file_path=str(qptiff_file),
                downscale_factor=64,
                padding=50,
                output_dir=str(downscale_output)
            )
            logger.info(f"Downscaled image created for {file_name}")
        except Exception as e:
            logger.error(f"Error downscaling {file_name}: {e}")
            return {"file": file_name, "status": "downscale_error", "error": str(e)}
        
        # Step 2: Segment tissues
        try:
            tissueframe = sp.tl.label_tissue(
                resized_im,
                lower_cutoff=0.2,
                upper_cutoff=0.21
            )
            logger.info(f"Tissue segmentation completed for {file_name}")
        except Exception as e:
            logger.error(f"Error segmenting tissues for {file_name}: {e}")
            return {"file": file_name, "status": "segmentation_error", "error": str(e)}
        
        # Step 3: Extract individual tissues
        tissue_output = output_dir / "tissues"
        tissue_output.mkdir(exist_ok=True)
        
        try:
            sp.tl.save_labelled_tissue(
                filepath=str(qptiff_file),
                tissueframe=tissueframe,
                output_dir=str(tissue_output),
                downscale_factor=64,
                region='region1',
                padding=50
            )
            logger.info(f"Tissue extraction completed for {file_name}")
        except Exception as e:
            logger.error(f"Error extracting tissues for {file_name}: {e}")
            return {"file": file_name, "status": "extraction_error", "error": str(e)}
        
        # Step 4: Process each extracted tissue for cell segmentation
        tissue_files = list(tissue_output.glob(f"*{file_name}*.tif"))
        cell_results = []
        
        for tissue_file in tissue_files:
            try:
                cell_result = self._process_tissue_for_cells(tissue_file, output_dir)
                cell_results.append(cell_result)
            except Exception as e:
                logger.error(f"Error processing tissue {tissue_file.name}: {e}")
                cell_results.append({
                    "tissue": tissue_file.name,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "file": file_name,
            "status": "success",
            "tissues_processed": len(tissue_files),
            "cell_results": cell_results
        }
    
    def _process_tissue_for_cells(self, tissue_file: Path, output_dir: Path) -> Dict:
        """
        Process a single tissue file for cell segmentation and expression
        
        Args:
            tissue_file: Path to tissue file
            output_dir: Output directory
            
        Returns:
            Dictionary with cell processing results
        """
        tissue_name = tissue_file.stem
        logger.info(f"Processing tissue for cells: {tissue_name}")
        
        # Create output directory for this tissue
        tissue_output = output_dir / "cells" / tissue_name
        tissue_output.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load the tissue image
            tissue_im = sp.hf.load_tiff(str(tissue_file))
            logger.info(f"Loaded tissue image: {tissue_im.shape}")
            
            # Perform cell segmentation
            # Note: This is a simplified version. You may need to adjust parameters
            cell_labels = sp.tl.segment_cells(
                tissue_im,
                channel_idx=0,  # DAPI channel
                min_area=50,
                max_area=1000
            )
            
            # Extract cell features
            cell_features = sp.tl.extract_cell_features(
                tissue_im,
                cell_labels,
                markers=self.markers
            )
            
            # Get cell locations
            cell_locations = sp.tl.get_cell_locations(cell_labels)
            
            # Save results
            self._save_cell_results(
                cell_features, 
                cell_locations, 
                tissue_output,
                tissue_name
            )
            
            return {
                "tissue": tissue_name,
                "status": "success",
                "n_cells": len(cell_features),
                "n_markers": len(self.markers)
            }
            
        except Exception as e:
            logger.error(f"Error processing tissue {tissue_name}: {e}")
            return {
                "tissue": tissue_name,
                "status": "error",
                "error": str(e)
            }
    
    def _save_cell_results(self, cell_features: np.ndarray, cell_locations: np.ndarray, 
                          output_dir: Path, tissue_name: str):
        """
        Save cell features and locations
        
        Args:
            cell_features: Cell x marker expression matrix
            cell_locations: Cell x 2 location matrix (x, y)
            output_dir: Output directory
            tissue_name: Name of the tissue
        """
        # Save expression matrix
        expression_df = pd.DataFrame(
            cell_features,
            columns=self.markers,
            index=[f"cell_{i}" for i in range(len(cell_features))]
        )
        expression_file = output_dir / f"{tissue_name}_expression_matrix.csv"
        expression_df.to_csv(expression_file)
        logger.info(f"Saved expression matrix: {expression_file}")
        
        # Save cell locations
        locations_df = pd.DataFrame(
            cell_locations,
            columns=['x', 'y'],
            index=[f"cell_{i}" for i in range(len(cell_locations))]
        )
        locations_file = output_dir / f"{tissue_name}_cell_locations.csv"
        locations_df.to_csv(locations_file)
        logger.info(f"Saved cell locations: {locations_file}")
        
        # Create AnnData object for further analysis
        adata = sc.AnnData(X=cell_features)
        adata.var_names = self.markers
        adata.obs_names = [f"cell_{i}" for i in range(len(cell_features))]
        adata.obsm['spatial'] = cell_locations
        
        # Save AnnData object
        adata_file = output_dir / f"{tissue_name}_adata.h5ad"
        adata.write(adata_file)
        logger.info(f"Saved AnnData object: {adata_file}")
    
    def process_all_tmas(self) -> Dict:
        """
        Process all TMA directories
        
        Returns:
            Dictionary with processing results for all TMAs
        """
        logger.info("Starting processing of all TMA directories")
        
        all_results = {
            "total_tmas": len(self.tma_dirs),
            "processed_tmas": 0,
            "tma_results": []
        }
        
        for tma_dir in self.tma_dirs:
            try:
                tma_result = self.process_single_tma(tma_dir, self.output_root)
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
        
        # Save summary results
        summary_file = self.output_root / "processing_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Processing completed. Summary saved to: {summary_file}")
        return all_results
    
    def create_combined_expression_matrix(self) -> pd.DataFrame:
        """
        Create a combined expression matrix from all processed TMAs
        
        Returns:
            Combined expression matrix
        """
        logger.info("Creating combined expression matrix")
        
        all_expression_data = []
        
        # Find all expression matrix files
        for expression_file in self.output_root.rglob("*_expression_matrix.csv"):
            try:
                df = pd.read_csv(expression_file, index_col=0)
                # Add metadata columns
                df['tma'] = expression_file.parent.parent.name
                df['tissue'] = expression_file.stem.replace('_expression_matrix', '')
                df['cell_id'] = df.index
                all_expression_data.append(df)
                logger.info(f"Loaded expression data from: {expression_file}")
            except Exception as e:
                logger.error(f"Error loading {expression_file}: {e}")
        
        if not all_expression_data:
            logger.warning("No expression data found")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_expression_data, ignore_index=True)
        logger.info(f"Combined expression matrix shape: {combined_df.shape}")
        
        # Save combined matrix
        combined_file = self.output_root / "combined_expression_matrix.csv"
        combined_df.to_csv(combined_file)
        logger.info(f"Saved combined expression matrix: {combined_file}")
        
        return combined_df


def main():
    """Main function to run the CODEX processing pipeline"""
    
    # Configuration
    data_root = "data/nomad_data/CODEX"
    output_root = "data/nomad_data/CODEX/processed"
    
    # Initialize processor
    processor = CODEXProcessor(data_root, output_root)
    
    # Process all TMAs
    results = processor.process_all_tmas()
    
    # Create combined expression matrix
    combined_matrix = processor.create_combined_expression_matrix()
    
    logger.info("CODEX processing pipeline completed!")
    logger.info(f"Processed {results['processed_tmas']} out of {results['total_tmas']} TMAs")
    
    if not combined_matrix.empty:
        logger.info(f"Combined expression matrix: {combined_matrix.shape}")
        logger.info(f"Markers: {list(combined_matrix.columns[:-3])}")  # Exclude metadata columns
        logger.info(f"Total cells: {len(combined_matrix)}")


if __name__ == "__main__":
    main() 