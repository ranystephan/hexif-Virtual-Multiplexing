#!/usr/bin/env python3
"""
Complete WSI Workflow: Preprocessing + Registration + Core Extraction

This script combines OME-TIFF preprocessing with WSI registration to handle
problematic OME-TIFF formats that cause pyvips errors.

Usage:
    python run_wsi_workflow.py --he_wsi input_he.ome.tiff --orion_wsi input_orion.ome.tiff --output output_dir
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path
import logging

from preprocess_ome_tiff import convert_ome_tiff
from run_wsi_registration import parse_arguments, validate_inputs, main as run_registration
from registration_pipeline_wsi import WSIRegistrationConfig, WSIRegistrationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_workflow_arguments():
    """Parse command line arguments for the workflow."""
    parser = argparse.ArgumentParser(
        description="Complete WSI workflow: preprocess OME-TIFF and run registration pipeline"
    )
    
    # Required arguments
    parser.add_argument(
        "--he_wsi", 
        required=True,
        help="Path to H&E whole slide image (OME-TIFF)"
    )
    
    parser.add_argument(
        "--orion_wsi",
        required=True, 
        help="Path to Orion/multiplex whole slide image (OME-TIFF)"
    )
    
    parser.add_argument(
        "--output",
        default="./wsi_workflow_output",
        help="Output directory (default: ./wsi_workflow_output)"
    )
    
    # Preprocessing options
    parser.add_argument(
        "--max_preprocess_resolution",
        type=int,
        help="Maximum resolution for preprocessing (optional, for very large images)"
    )
    
    parser.add_argument(
        "--keep_preprocessed",
        action="store_true",
        help="Keep preprocessed TIFF files after completion"
    )
    
    # Registration parameters (same as run_wsi_registration.py)
    parser.add_argument(
        "--max_processed_dim",
        type=int,
        default=2048,
        help="Maximum dimension for processed images (default: 2048)"
    )
    
    parser.add_argument(
        "--max_nonrigid_dim", 
        type=int,
        default=3000,
        help="Maximum dimension for non-rigid registration (default: 3000)"
    )
    
    # Core detection parameters
    parser.add_argument(
        "--core_min_area",
        type=int,
        default=50000,
        help="Minimum area for valid cores in pixels (default: 50000)"
    )
    
    parser.add_argument(
        "--core_max_area",
        type=int, 
        default=500000,
        help="Maximum area for valid cores in pixels (default: 500000)"
    )
    
    parser.add_argument(
        "--core_circularity",
        type=float,
        default=0.4,
        help="Minimum circularity threshold for cores (default: 0.4)"
    )
    
    parser.add_argument(
        "--expected_core_diameter",
        type=int,
        default=400,
        help="Expected core diameter in pixels (default: 400)"
    )
    
    parser.add_argument(
        "--core_padding",
        type=int,
        default=50,
        help="Padding around cores in pixels (default: 50)"
    )
    
    # Processing options
    parser.add_argument(
        "--compression",
        default="lzw",
        choices=["lzw", "jpeg", "jp2k"],
        help="Compression method for output images (default: lzw)"
    )
    
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable generation of quality control plots"
    )
    
    return parser.parse_args()


def validate_workflow_inputs(args):
    """Validate input arguments for the workflow."""
    errors = []
    
    # Check input files exist
    he_path = Path(args.he_wsi)
    if not he_path.exists():
        errors.append(f"H&E WSI file not found: {he_path}")
    
    orion_path = Path(args.orion_wsi)
    if not orion_path.exists():
        errors.append(f"Orion WSI file not found: {orion_path}")
    
    # Check output directory is writable
    output_path = Path(args.output)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory {output_path}: {e}")
    
    # Validate numeric parameters
    if args.core_min_area <= 0:
        errors.append("Core minimum area must be positive")
    
    if args.core_max_area <= args.core_min_area:
        errors.append("Core maximum area must be greater than minimum area")
    
    if not (0.0 <= args.core_circularity <= 1.0):
        errors.append("Core circularity must be between 0.0 and 1.0")
    
    if args.expected_core_diameter <= 0:
        errors.append("Expected core diameter must be positive")
    
    return errors


def main():
    """Main workflow function."""
    args = parse_workflow_arguments()
    
    # Validate inputs
    errors = validate_workflow_inputs(args)
    if errors:
        print("Input validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print("=" * 70)
    print("WSI WORKFLOW: Preprocessing + Registration + Core Extraction")
    print("=" * 70)
    print(f"H&E WSI (OME-TIFF): {args.he_wsi}")
    print(f"Orion WSI (OME-TIFF): {args.orion_wsi}")
    print(f"Output Directory: {args.output}")
    print("=" * 70)
    
    temp_dir = None
    
    try:
        # Step 1: Preprocess OME-TIFF files
        print("\n🔄 STEP 1: Preprocessing OME-TIFF files to VALIS-compatible format")
        print("-" * 50)
        
        # Create temporary directory for preprocessed files
        if args.keep_preprocessed:
            preprocess_dir = Path(args.output) / "preprocessed_tiff"
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = str(preprocess_dir)
        else:
            temp_dir = tempfile.mkdtemp(prefix="wsi_preprocess_")
        
        he_processed = Path(temp_dir) / "he_processed.tif"
        orion_processed = Path(temp_dir) / "orion_processed.tif"
        
        # Convert H&E
        print(f"Converting H&E: {args.he_wsi} -> {he_processed}")
        he_success = convert_ome_tiff(
            args.he_wsi, 
            str(he_processed), 
            args.max_preprocess_resolution
        )
        
        if not he_success:
            print("❌ Failed to preprocess H&E image")
            sys.exit(1)
        
        # Convert Orion (will create both single-channel for registration and multi-channel for extraction)
        print(f"Converting Orion: {args.orion_wsi} -> {orion_processed}")
        orion_success = convert_ome_tiff(
            args.orion_wsi, 
            str(orion_processed), 
            args.max_preprocess_resolution,
            registration_channel=0  # Use channel 0 (typically DAPI) for registration
        )
        
        if not orion_success:
            print("❌ Failed to preprocess Orion image")
            sys.exit(1)
        
        print("✅ Preprocessing completed successfully!")
        
        # Step 2: Run WSI registration pipeline
        print("\n🔄 STEP 2: Running WSI registration and core extraction")
        print("-" * 50)
        
        # Create configuration for the registration pipeline
        config = WSIRegistrationConfig(
            he_wsi_path=str(he_processed),
            orion_wsi_path=str(orion_processed),
            output_dir=args.output,
            max_processed_image_dim_px=args.max_processed_dim,
            max_non_rigid_registration_dim_px=args.max_nonrigid_dim,
            core_min_area=args.core_min_area,
            core_max_area=args.core_max_area,
            core_circularity_threshold=args.core_circularity,
            expected_core_diameter=args.expected_core_diameter,
            core_padding=args.core_padding,
            compression=args.compression,
            save_core_detection_plots=not args.no_plots,
            save_quality_plots=not args.no_plots
        )
        
        # Pass the preprocessing directory to the pipeline
        pipeline = WSIRegistrationPipeline(config)
        pipeline.preprocessing_dir = temp_dir  # Add reference to preprocessing files
        
        # Print registration configuration
        print("Registration Parameters:")
        print(f"  Max Processed Dimension: {config.max_processed_image_dim_px}")
        print(f"  Max Non-rigid Dimension: {config.max_non_rigid_registration_dim_px}")
        print()
        print("Core Detection Parameters:")
        print(f"  Min Core Area: {config.core_min_area}")
        print(f"  Max Core Area: {config.core_max_area}")
        print(f"  Circularity Threshold: {config.core_circularity_threshold}")
        print(f"  Expected Diameter: {config.expected_core_diameter}")
        print(f"  Core Padding: {config.core_padding}")
        print()
        
        # Run the registration pipeline
        pipeline = WSIRegistrationPipeline(config)
        results = pipeline.run()
        
        # Step 3: Report results
        print("\n" + "=" * 70)
        print("WORKFLOW RESULTS")
        print("=" * 70)
        
        if results.get('success', False):
            print("✅ Workflow completed successfully!")
            print()
            print("Preprocessing Results:")
            print(f"  H&E Converted: ✅ {args.he_wsi} -> {he_processed}")
            print(f"  Orion Converted: ✅ {args.orion_wsi} -> {orion_processed}")
            if args.keep_preprocessed:
                print(f"  Preprocessed files saved in: {temp_dir}")
            print()
            print("Registration Results:")
            print(f"  Registration Success: {results.get('registration_success', False)}")
            print()
            print("Core Detection Results:")
            print(f"  H&E Cores Detected: {results.get('he_cores_detected', 0)}")
            print(f"  Orion Cores Detected: {results.get('orion_cores_detected', 0)}")
            print(f"  Matched Core Pairs: {results.get('matched_pairs', 0)}")
            print()
            print("Core Extraction Results:")
            print(f"  Cores Successfully Extracted: {results.get('cores_extracted', 0)}")
            if results.get('extraction_errors'):
                print(f"  Extraction Errors: {len(results.get('extraction_errors', []))}")
            print()
            print("Output Structure:")
            print(f"  📁 {args.output}/")
            print(f"    ├── 📁 registered_wsi/          # Registered whole slide images")
            print(f"    ├── 📁 extracted_cores/         # Individual core pairs")
            if args.keep_preprocessed:
                print(f"    ├── 📁 preprocessed_tiff/       # Converted TIFF files")
            if config.save_core_detection_plots:
                print(f"    ├── 📁 quality_plots/           # Visualization plots")
            print(f"    └── 📄 wsi_pipeline_summary.json # Pipeline summary")
            print()
            print(f"🎯 Found and extracted {results.get('matched_pairs', 0)} matched core pairs!")
            print(f"📂 Cores saved to: {args.output}/extracted_cores/")
            
        else:
            print("❌ Workflow failed!")
            error_msg = results.get('error', 'Unknown error')
            print(f"Error: {error_msg}")
            
            # Additional error details
            if not results.get('registration_success', True):
                reg_error = results.get('registration_error')
                if reg_error:
                    print(f"Registration Error: {reg_error}")
            
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error in workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up temporary files if not keeping them
        if temp_dir and not args.keep_preprocessed:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary preprocessing files")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")


if __name__ == "__main__":
    main() 