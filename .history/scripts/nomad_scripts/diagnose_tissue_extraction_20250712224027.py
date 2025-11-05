#!/usr/bin/env python3
"""
Diagnostic script to understand tissue extraction issues
"""

import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import spacec as sp
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import logging
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_tissue_extraction(qptiff_file: Path, output_dir: Path):
    """
    Diagnose tissue extraction issues
    """
    logger.info(f"Diagnosing tissue extraction for: {qptiff_file.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Downscale the image
        logger.info("Step 1: Downscaling image...")
        resized_im = sp.hf.downscale_tissue(
            file_path=str(qptiff_file),
            downscale_factor=64,
            padding=50,
            output_dir=str(output_dir)
        )
        
        logger.info(f"Downscaled image shape: {resized_im.shape}")
        logger.info(f"Downscaled image dtype: {resized_im.dtype}")
        logger.info(f"Downscaled image min/max: {resized_im.min():.3f}/{resized_im.max():.3f}")
        
        # Step 2: Analyze image intensity distribution
        logger.info("Step 2: Analyzing intensity distribution...")
        
        if len(resized_im.shape) == 3:
            # Multi-channel image
            logger.info("Multi-channel image detected")
            for i in range(min(3, resized_im.shape[0])):  # Look at first 3 channels
                channel = resized_im[i]
                logger.info(f"Channel {i}: min={channel.min():.3f}, max={channel.max():.3f}, mean={channel.mean():.3f}")
        else:
            # Single channel image
            logger.info("Single-channel image detected")
            logger.info(f"Intensity: min={resized_im.min():.3f}, max={resized_im.max():.3f}, mean={resized_im.mean():.3f}")
        
        # Step 3: Try different cutoff values
        logger.info("Step 3: Testing different cutoff values...")
        
        cutoff_tests = [
            (0.05, 0.95),
            (0.1, 0.9),
            (0.15, 0.85),
            (0.2, 0.8),
            (0.25, 0.75),
            (0.3, 0.7),
            (0.35, 0.65),
            (0.4, 0.6),
        ]
        
        for lower_cutoff, upper_cutoff in cutoff_tests:
            logger.info(f"Testing cutoffs: {lower_cutoff}, {upper_cutoff}")
            
            try:
                tissueframe = sp.tl.label_tissue(
                    resized_im,
                    lower_cutoff=lower_cutoff,
                    upper_cutoff=upper_cutoff
                )
                
                unique_regions = tissueframe['region'].unique()
                n_tissues = len([r for r in unique_regions if r != 0])
                
                logger.info(f"  Found {n_tissues} tissue pieces")
                
                if n_tissues > 0:
                    logger.info(f"  SUCCESS! Found {n_tissues} tissues with cutoffs {lower_cutoff}, {upper_cutoff}")
                    return True, tissueframe, (lower_cutoff, upper_cutoff)
                    
            except Exception as e:
                logger.warning(f"  Failed with cutoffs {lower_cutoff}, {upper_cutoff}: {e}")
        
        # Step 4: Try manual thresholding
        logger.info("Step 4: Trying manual thresholding...")
        
        try:
            from skimage import filters, morphology, measure
            
            # Use the downscaled image
            if len(resized_im.shape) == 3:
                tissue_channel = resized_im[0]  # Use first channel
            else:
                tissue_channel = resized_im
            
            # Apply Otsu thresholding
            threshold = filters.threshold_otsu(tissue_channel)
            logger.info(f"Otsu threshold: {threshold:.3f}")
            
            tissue_mask = tissue_channel > threshold
            
            # Clean up the mask
            tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=100)
            tissue_mask = morphology.binary_closing(tissue_mask)
            
            # Label connected components
            labeled_mask = measure.label(tissue_mask)
            n_regions = len(np.unique(labeled_mask)) - 1  # Exclude background
            
            logger.info(f"Manual thresholding found {n_regions} regions")
            
            if n_regions > 0:
                logger.info("SUCCESS! Manual thresholding worked")
                return True, labeled_mask, "manual"
                
        except Exception as e:
            logger.error(f"Manual thresholding failed: {e}")
        
        logger.error("All tissue extraction methods failed")
        return False, None, None
        
    except Exception as e:
        logger.error(f"Error in diagnosis: {e}")
        return False, None, None

def main():
    """Main function"""
    
    # Find a qptiff file to test
    data_root = Path("../../data/nomad_data/CODEX")
    
    # Find TMA directories
    tma_dirs = [d for d in data_root.iterdir() if d.is_dir() and "TMA" in d.name]
    
    if not tma_dirs:
        logger.error("No TMA directories found")
        return
    
    # Use the first TMA
    tma_dir = tma_dirs[0]
    logger.info(f"Using TMA: {tma_dir.name}")
    
    # Find qptiff files
    scan_dir = tma_dir / "Scan1"
    if not scan_dir.exists():
        logger.error(f"Scan1 directory not found in {tma_dir}")
        return
    
    qptiff_files = list(scan_dir.glob("*.qptiff"))
    if not qptiff_files:
        logger.error("No qptiff files found")
        return
    
    # Use the first qptiff file
    qptiff_file = qptiff_files[0]
    logger.info(f"Using file: {qptiff_file.name}")
    
    # Create output directory
    output_dir = Path("../../data/nomad_data/CODEX/diagnostic_output")
    
    # Run diagnosis
    success, result, method = diagnose_tissue_extraction(qptiff_file, output_dir)
    
    if success:
        logger.info(f"Tissue extraction successful using method: {method}")
        if isinstance(result, np.ndarray):
            logger.info(f"Found {len(np.unique(result)) - 1} tissue regions")
        else:
            logger.info(f"Found {len(result['region'].unique()) - 1} tissue regions")
    else:
        logger.error("Tissue extraction failed with all methods")

if __name__ == "__main__":
    main() 