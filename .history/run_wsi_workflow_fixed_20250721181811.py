#!/usr/bin/env python3
"""
WSI Workflow with fluorescence-optimized core detection parameters.
This version uses more permissive parameters that work better for fluorescence data.
"""

import sys
import pathlib
import tempfile
import shutil
import logging
from preprocess_ome_tiff import convert_ome_tiff
from registration_pipeline_wsi import WSIRegistrationConfig, WSIRegistrationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='WSI Workflow with fluorescence-optimized parameters')
    parser.add_argument('--he_wsi', required=True, help='Path to H&E WSI file')
    parser.add_argument('--orion_wsi', required=True, help='Path to Orion WSI file')
    parser.add_argument('--output', required=True, help='Output directory for extracted cores')
    parser.add_argument('--keep_preprocessed', action='store_true', help='Keep temporary preprocessing files')
    
    args = parser.parse_args()
    
    print("="*70)
    print("WSI WORKFLOW: Preprocessing + Registration + Core Extraction")
    print("="*70)
    print(f"H&E WSI (OME-TIFF): {args.he_wsi}")
    print(f"Orion WSI (OME-TIFF): {args.orion_wsi}")
    print(f"Output Directory: {args.output}")
    print("="*70)
    print()
    
    # Create temporary directory for preprocessing
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="wsi_preprocess_")
        
        print("🔄 STEP 1: Preprocessing OME-TIFF files to VALIS-compatible format")
        print("-" * 50)
        
        # Preprocess H&E
        he_output = pathlib.Path(temp_dir) / "he_processed.tif"
        print(f"Converting H&E: {args.he_wsi} -> {he_output}")
        convert_ome_tiff(args.he_wsi, str(he_output))
        
        # Preprocess Orion
        orion_output = pathlib.Path(temp_dir) / "orion_processed.tif"
        print(f"Converting Orion: {args.orion_wsi} -> {orion_output}")
        convert_ome_tiff(args.orion_wsi, str(orion_output))
        
        print("✅ Preprocessing completed successfully!")
        print()
        
        print("🔄 STEP 2: Running WSI registration and core extraction")
        print("-" * 50)
        
        # Create configuration with FLUORESCENCE-OPTIMIZED parameters
        config = WSIRegistrationConfig(
            he_wsi_path=str(he_output),
            orion_wsi_path=str(orion_output),
            output_dir=args.output,
            # FLUORESCENCE-OPTIMIZED PARAMETERS
            core_min_area=10000,        # Reduced from 50000
            core_max_area=200000,       # Reduced from 500000
            core_circularity_threshold=0.2,  # Reduced from 0.4
            min_core_diameter=100,      # Reduced from 200
            gaussian_sigma=1.5,         # Reduced from 2.0
            max_processed_image_dim_px=2048,
            max_non_rigid_registration_dim_px=3000
        )
        
        print("Registration Parameters:")
        print(f"  Max Processed Dimension: {config.max_processed_image_dim_px}")
        print(f"  Max Non-rigid Dimension: {config.max_non_rigid_registration_dim_px}")
        print()
        print("Core Detection Parameters (FLUORESCENCE-OPTIMIZED):")
        print(f"  Min Core Area: {config.core_min_area}")
        print(f"  Max Core Area: {config.core_max_area}")
        print(f"  Circularity Threshold: {config.core_circularity_threshold}")
        print(f"  Expected Diameter: {config.min_core_diameter}")
        print(f"  Core Padding: {config.core_padding}")
        print()
        
        # Create and run pipeline
        pipeline = WSIRegistrationPipeline(config)
        pipeline.preprocessing_dir = temp_dir  # Pass preprocessing directory
        
        success = pipeline.run()
        
        if success:
            print("="*70)
            print("WORKFLOW RESULTS")
            print("="*70)
            print("✅ Workflow completed successfully!")
            print(f"📁 Extracted cores saved to: {args.output}")
        else:
            print("="*70)
            print("WORKFLOW RESULTS")
            print("="*70)
            print("❌ Workflow failed!")
            sys.exit(1)
            
    except Exception as e:
        print("="*70)
        print("WORKFLOW RESULTS")
        print("="*70)
        print(f"❌ Workflow failed!")
        print(f"Error: {e}")
        logger.exception("Workflow error")
        sys.exit(1)
        
    finally:
        # Clean up temporary files if not keeping them
        if temp_dir and not args.keep_preprocessed:
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary preprocessing files")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")

if __name__ == "__main__":
    main() 