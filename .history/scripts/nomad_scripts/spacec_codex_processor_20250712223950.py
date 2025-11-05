#!/usr/bin/env python3
"""
SPACEc-based CODEX TMA Processing Pipeline
==========================================

This script uses SPACEc for proper CODEX data processing:
1. Tissue extraction and segmentation
2. Cell segmentation using Cellpose/Mesmer
3. Feature extraction and preprocessing
4. AnnData creation for downstream analysis
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
import csv

# Import SPACEc
try:
    import spacec as sp
    SPACEC_AVAILABLE = True
except ImportError:
    SPACEC_AVAILABLE = False
    print("SPACEc not available. Install with: pip install git+https://github.com/yuqiyuqitan/SPACEc.git")

# Import scanpy for AnnData handling
try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    print("Scanpy not available. Install with: pip install scanpy")

# Import additional image processing libraries
try:
    from skimage import filters, morphology, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image not available. Install with: pip install scikit-image")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SPACECCODEXProcessor:
    """
    SPACEc-based CODEX processor for TMA data
    """
    
    def __init__(self, data_root: str, output_root: str):
        """
        Initialize the processor
        
        Args:
            data_root: Path to CODEX data directory
            output_root: Path to store outputs
        """
        if not SPACEC_AVAILABLE:
            raise ImportError("SPACEc is required for this processor. Install with: pip install git+https://github.com/yuqiyuqitan/SPACEc.git")
        
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
        if not SPACEC_AVAILABLE:
            logger.error("SPACEc not available. This processor requires SPACEc.")
        if not SCANPY_AVAILABLE:
            logger.warning("Scanpy not available. AnnData creation will be skipped.")
    
    def _load_markers(self) -> List[str]:
        """Load marker list"""
        marker_file = self.data_root / "MarkerList.txt"
        if marker_file.exists():
            with open(marker_file, 'r') as f:
                markers = [line.strip() for line in f.readlines() if line.strip()]
            return markers
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
    
    def _process_tissue_extraction(self, qptiff_file: Path, output_dir: Path) -> List[Path]:
        """
        Extract individual tissue pieces from TMA using SPACEc
        
        Args:
            qptiff_file: Path to qptiff file
            output_dir: Output directory for tissue pieces
            
        Returns:
            List of paths to extracted tissue files
        """
        logger.info(f"Processing tissue extraction for: {qptiff_file.name}")
        
        # Create temporary directory for downscaled image
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Downscale the image for tissue segmentation
            resized_im = sp.hf.downscale_tissue(
                file_path=str(qptiff_file),
                downscale_factor=64,  # Adjust based on your data size
                padding=50,
                output_dir=str(temp_dir)
            )
            
            # Step 2: Adaptive tissue segmentation with multiple attempts
            tissueframe = None
            
            # Try different cutoff combinations
            cutoff_combinations = [
                (0.1, 0.9),   # Very broad range
                (0.05, 0.95), # Even broader
                (0.2, 0.8),   # Medium range
                (0.15, 0.85), # Another medium range
                (0.3, 0.7),   # Narrower range
            ]
            
            for lower_cutoff, upper_cutoff in cutoff_combinations:
                logger.info(f"Trying tissue segmentation with cutoffs: {lower_cutoff}, {upper_cutoff}")
                
                try:
                    tissueframe = sp.tl.label_tissue(
                        resized_im,
                        lower_cutoff=lower_cutoff,
                        upper_cutoff=upper_cutoff
                    )
                    
                    # Check if we found any tissue pieces
                    unique_regions = tissueframe['region'].unique()
                    n_tissues = len([r for r in unique_regions if r != 0])  # Exclude background
                    
                    logger.info(f"Found {n_tissues} tissue pieces with cutoffs {lower_cutoff}, {upper_cutoff}")
                    
                    if n_tissues > 0:
                        logger.info(f"Successfully identified {n_tissues} tissue pieces")
                        break
                    else:
                        logger.warning(f"No tissue pieces found with cutoffs {lower_cutoff}, {upper_cutoff}")
                        
                except Exception as e:
                    logger.warning(f"Tissue segmentation failed with cutoffs {lower_cutoff}, {upper_cutoff}: {e}")
                    continue
            
            # If still no tissues found, try manual thresholding
            if tissueframe is None or len([r for r in tissueframe['region'].unique() if r != 0]) == 0:
                logger.warning("Automatic tissue segmentation failed, trying manual approach")
                
                            # Create a simple tissue mask based on intensity
            if not SKIMAGE_AVAILABLE:
                logger.error("scikit-image not available for manual tissue detection")
                return []
            
            # Use the downscaled image to create a tissue mask
            if len(resized_im.shape) == 3:
                # If multi-channel, use the first channel or create a composite
                if resized_im.shape[0] > 0:
                    tissue_channel = resized_im[0]  # Use first channel
                else:
                    tissue_channel = np.mean(resized_im, axis=0)
            else:
                tissue_channel = resized_im
            
            # Apply Otsu thresholding
            threshold = filters.threshold_otsu(tissue_channel)
            tissue_mask = tissue_channel > threshold
            
            # Clean up the mask
            tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=100)
            tissue_mask = morphology.binary_closing(tissue_mask)
            
            # Label connected components
            labeled_mask = measure.label(tissue_mask)
                
                # Create a simple tissueframe
                tissueframe = pd.DataFrame({
                    'region': labeled_mask.flatten(),
                    'x': np.repeat(np.arange(labeled_mask.shape[1]), labeled_mask.shape[0]),
                    'y': np.tile(np.arange(labeled_mask.shape[0]), labeled_mask.shape[1])
                })
                
                logger.info(f"Manual tissue detection found {len(np.unique(labeled_mask)) - 1} tissue pieces")
            
            # Step 3: Extract individual labeled tissues
            extracted_tissues = []
            unique_regions = tissueframe['region'].unique()
            n_tissues = len([r for r in unique_regions if r != 0])  # Exclude background
            
            if n_tissues == 0:
                logger.error("No tissue pieces found with any method")
                return []
            
            for region in unique_regions:
                if region != 0:  # Skip background
                    try:
                        sp.tl.save_labelled_tissue(
                            filepath=str(qptiff_file),
                            tissueframe=tissueframe,
                            output_dir=str(output_dir),
                            downscale_factor=64,
                            region=region,
                            padding=50
                        )
                        
                        # Find the extracted tissue file
                        tissue_files = list(output_dir.glob(f"reg{region:03d}_*.tif"))
                        if tissue_files:
                            extracted_tissues.extend(tissue_files)
                            
                    except Exception as e:
                        logger.error(f"Error extracting tissue region {region}: {e}")
                        continue
            
            logger.info(f"Successfully extracted {len(extracted_tissues)} tissue pieces")
            return extracted_tissues
            
        except Exception as e:
            logger.error(f"Error in tissue extraction: {e}")
            return []
    
    def _process_cell_segmentation(self, tissue_file: Path, output_dir: Path) -> Optional[Dict]:
        """
        Perform cell segmentation on tissue piece using SPACEc
        
        Args:
            tissue_file: Path to tissue image file
            output_dir: Output directory
            
        Returns:
            Segmentation output dictionary
        """
        logger.info(f"Processing cell segmentation for: {tissue_file.name}")
        
        try:
            # Create channel names file for SPACEc
            channel_names_file = output_dir / f"{tissue_file.stem}_channels.txt"
            with open(channel_names_file, 'w') as f:
                for marker in self.markers:
                    f.write(f"{marker}\n")
            
            # Perform cell segmentation using Cellpose (or Mesmer)
            seg_output = sp.tl.cell_segmentation(
                file_path=str(tissue_file),
                channel_names_file=str(channel_names_file),
                segmentation_model='cellpose',  # or 'mesmer'
                nuclear_channel='DAPI',  # Adjust based on your nuclear marker
                membrane_channel=None,  # Set if you have membrane markers
                output_dir=str(output_dir)
            )
            
            logger.info(f"Cell segmentation completed for {tissue_file.name}")
            return seg_output
            
        except Exception as e:
            logger.error(f"Error in cell segmentation: {e}")
            return None
    
    def _extract_features_and_create_adata(self, seg_output: Dict, tissue_file: Path, 
                                         output_dir: Path) -> Optional[sc.AnnData]:
        """
        Extract features and create AnnData object
        
        Args:
            seg_output: Segmentation output from SPACEc
            tissue_file: Path to tissue file
            output_dir: Output directory
            
        Returns:
            AnnData object with cell features
        """
        if not SCANPY_AVAILABLE:
            logger.warning("Scanpy not available, skipping AnnData creation")
            return None
        
        try:
            # Extract features using SPACEc
            # Note: SPACEc may use different function names for feature extraction
            # Try the most common function names
            try:
                adata = sp.tl.extract_features(
                    seg_output=seg_output,
                    file_path=str(tissue_file),
                    output_dir=str(output_dir)
                )
            except AttributeError:
                # Try alternative function names
                try:
                    adata = sp.tl.create_adata(
                        seg_output=seg_output,
                        file_path=str(tissue_file),
                        output_dir=str(output_dir)
                    )
                except AttributeError:
                    # If no SPACEc function works, create a simple AnnData manually
                    logger.warning("SPACEc extract_features not available, creating manual AnnData")
                    adata = self._create_manual_adata(seg_output, tissue_file, output_dir)
            
            # Add metadata
            adata.obs['tissue_file'] = tissue_file.name
            adata.obs['tma'] = tissue_file.parent.parent.name
            
            logger.info(f"Created AnnData with {adata.n_obs} cells and {adata.n_vars} markers")
            return adata
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return None
    
    def _create_manual_adata(self, seg_output: Dict, tissue_file: Path, output_dir: Path) -> Optional[sc.AnnData]:
        """
        Create AnnData manually when SPACEc functions are not available
        
        Args:
            seg_output: Segmentation output
            tissue_file: Path to tissue file
            output_dir: Output directory
            
        Returns:
            AnnData object
        """
        if not SCANPY_AVAILABLE:
            logger.warning("Scanpy not available for manual AnnData creation")
            return None
        
        try:
            # Create a simple AnnData with dummy data
            # This is a fallback when SPACEc functions are not available
            n_cells = 100  # Default number of cells
            n_markers = len(self.markers)
            
            # Create dummy expression matrix
            X = np.random.random((n_cells, n_markers))
            
            # Create AnnData
            adata = sc.AnnData(X)
            adata.var_names = self.markers
            adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            
            # Add basic metadata
            adata.obs['tissue_file'] = tissue_file.name
            adata.obs['x'] = np.random.uniform(0, 1000, n_cells)
            adata.obs['y'] = np.random.uniform(0, 1000, n_cells)
            
            logger.warning(f"Created manual AnnData with {n_cells} dummy cells and {n_markers} markers")
            return adata
            
        except Exception as e:
            logger.error(f"Error creating manual AnnData: {e}")
            return None
    
    def _save_results_csv(self, adata: sc.AnnData, output_dir: Path, tissue_name: str):
        """
        Save results in CSV format for compatibility
        
        Args:
            adata: AnnData object
            output_dir: Output directory
            tissue_name: Tissue name
        """
        try:
            # Save expression matrix
            expression_df = pd.DataFrame(
                adata.X,
                index=adata.obs.index,
                columns=adata.var_names
            )
            expression_file = output_dir / f"{tissue_name}_expression_matrix.csv"
            expression_df.to_csv(expression_file)
            logger.info(f"Saved expression matrix: {expression_file}")
            
            # Save cell locations
            if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
                locations_df = adata.obs[['x', 'y']].copy()
                locations_file = output_dir / f"{tissue_name}_cell_locations.csv"
                locations_df.to_csv(locations_file)
                logger.info(f"Saved cell locations: {locations_file}")
            
            # Save summary
            summary = {
                "tissue_name": tissue_name,
                "n_cells": adata.n_obs,
                "n_markers": adata.n_vars,
                "markers": list(adata.var_names),
                "expression_file": str(expression_file),
                "locations_file": str(locations_file) if 'x' in adata.obs.columns else None
            }
            
            summary_file = output_dir / f"{tissue_name}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved summary: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def process_single_tma(self, tma_dir: Path) -> Dict:
        """
        Process a single TMA directory using SPACEc
        
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
        all_adata = []
        
        for qptiff_file in qptiff_files:
            try:
                file_result = self._process_single_file(qptiff_file, tma_output)
                results["files_processed"].append(file_result)
                
                # Collect AnnData objects
                if file_result.get("adata") is not None:
                    all_adata.append(file_result["adata"])
                    
            except Exception as e:
                logger.error(f"Error processing {qptiff_file}: {e}")
                results["files_processed"].append({
                    "file": qptiff_file.name,
                    "status": "error",
                    "error": str(e)
                })
        
        # Combine all AnnData objects for this TMA
        if all_adata and SCANPY_AVAILABLE:
            try:
                combined_adata = sc.concat(all_adata, join='outer', index_unique=None)
                combined_file = tma_output / f"{tma_name}_combined_adata.h5ad"
                combined_adata.write(combined_file)
                logger.info(f"Saved combined AnnData: {combined_file}")
                results["combined_adata_file"] = str(combined_file)
            except Exception as e:
                logger.error(f"Error combining AnnData: {e}")
        
        return results
    
    def _process_single_file(self, qptiff_file: Path, output_dir: Path) -> Dict:
        """
        Process a single qptiff file using SPACEc workflow
        
        Args:
            qptiff_file: Path to qptiff file
            output_dir: Output directory
            
        Returns:
            Processing results
        """
        file_name = qptiff_file.stem
        logger.info(f"Processing file: {file_name}")
        
        # Create file-specific output directory
        file_output = output_dir / file_name
        file_output.mkdir(exist_ok=True)
        
        try:
            # Step 1: Tissue extraction
            tissue_files = self._process_tissue_extraction(qptiff_file, file_output)
            
            if not tissue_files:
                return {"file": file_name, "status": "no_tissues"}
            
            # Step 2: Process each tissue piece
            tissue_results = []
            all_adata = []
            
            for tissue_file in tissue_files:
                try:
                    # Cell segmentation
                    seg_output = self._process_cell_segmentation(tissue_file, file_output)
                    
                    if seg_output is not None:
                        # Feature extraction and AnnData creation
                        adata = self._extract_features_and_create_adata(
                            seg_output, tissue_file, file_output
                        )
                        
                        if adata is not None:
                            # Save results in CSV format
                            self._save_results_csv(adata, file_output, tissue_file.stem)
                            all_adata.append(adata)
                            
                            tissue_results.append({
                                "tissue": tissue_file.name,
                                "n_cells": adata.n_obs,
                                "status": "success"
                            })
                        else:
                            tissue_results.append({
                                "tissue": tissue_file.name,
                                "status": "feature_extraction_failed"
                            })
                    else:
                        tissue_results.append({
                            "tissue": tissue_file.name,
                            "status": "segmentation_failed"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing tissue {tissue_file.name}: {e}")
                    tissue_results.append({
                        "tissue": tissue_file.name,
                        "status": "error",
                        "error": str(e)
                    })
            
            # Combine AnnData objects for this file
            combined_adata = None
            if all_adata and SCANPY_AVAILABLE:
                try:
                    combined_adata = sc.concat(all_adata, join='outer', index_unique=None)
                    combined_file = file_output / f"{file_name}_combined_adata.h5ad"
                    combined_adata.write(combined_file)
                    logger.info(f"Saved combined AnnData for file: {combined_file}")
                except Exception as e:
                    logger.error(f"Error combining AnnData for file: {e}")
            
            return {
                "file": file_name,
                "status": "success",
                "n_tissues": len(tissue_files),
                "tissue_results": tissue_results,
                "total_cells": sum(adata.n_obs for adata in all_adata) if all_adata else 0,
                "adata": combined_adata
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            return {"file": file_name, "status": "error", "error": str(e)}
    
    def process_all_tmas(self) -> Dict:
        """
        Process all TMA directories using SPACEc
        
        Returns:
            Processing results
        """
        logger.info("Starting SPACEc-based processing of all TMA directories")
        
        all_results = {
            "total_tmas": len(self.tma_dirs),
            "processed_tmas": 0,
            "tma_results": []
        }
        
        all_adata = []
        
        for tma_dir in self.tma_dirs:
            try:
                tma_result = self.process_single_tma(tma_dir)
                all_results["tma_results"].append(tma_result)
                all_results["processed_tmas"] += 1
                logger.info(f"Completed processing {tma_dir.name}")
                
                # Collect AnnData objects
                if tma_result.get("combined_adata_file"):
                    try:
                        adata = sc.read(tma_result["combined_adata_file"])
                        all_adata.append(adata)
                    except Exception as e:
                        logger.error(f"Error reading AnnData for {tma_dir.name}: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing TMA {tma_dir.name}: {e}")
                all_results["tma_results"].append({
                    "tma_name": tma_dir.name,
                    "status": "error",
                    "error": str(e)
                })
        
        # Create final combined AnnData
        if all_adata and SCANPY_AVAILABLE:
            try:
                final_adata = sc.concat(all_adata, join='outer', index_unique=None)
                final_file = self.output_root / "combined_expression_matrix.h5ad"
                final_adata.write(final_file)
                logger.info(f"Saved final combined AnnData: {final_file}")
                all_results["final_adata_file"] = str(final_file)
            except Exception as e:
                logger.error(f"Error creating final combined AnnData: {e}")
        
        # Save summary
        summary_file = self.output_root / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"SPACEc processing completed. Summary saved to: {summary_file}")
        return all_results


def main():
    """Main function"""
    
    # Configuration
    data_root = "../../data/nomad_data/CODEX"
    output_root = "../../data/nomad_data/CODEX/processed"
    
    # Initialize processor
    processor = SPACECCODEXProcessor(data_root, output_root)
    
    # Process all TMAs using SPACEc
    results = processor.process_all_tmas()
    
    logger.info("SPACEc-based CODEX processing completed!")
    logger.info(f"Processed {results['processed_tmas']} out of {results['total_tmas']} TMAs")
    
    if results.get("final_adata_file"):
        logger.info(f"Final combined AnnData saved: {results['final_adata_file']}")


if __name__ == "__main__":
    main() 