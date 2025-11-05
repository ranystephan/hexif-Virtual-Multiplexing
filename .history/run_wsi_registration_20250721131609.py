#!/usr/bin/env python3
"""
Example script for running WSI registration and core extraction pipeline.

This script demonstrates how to use the WSI registration pipeline to:
1. Register full TMA slides (H&E and Orion)
2. Detect tissue cores automatically 
3. Extract matched core pairs
4. Organize output for downstream analysis

Usage:
    python run_wsi_registration.py --he_wsi /path/to/he_tma.tif --orion_wsi /path/to/orion_tma.tif --output ./output
"""

import argparse
import sys
from pathlib import Path

from registration_pipeline_wsi import WSIRegistrationConfig, WSIRegistrationPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run WSI registration and core extraction pipeline"
    )
    
    # Required arguments
    parser.add_argument(
        "--he_wsi", 
        required=True,
        help="Path to H&E whole slide image"
    )
    
    parser.add_argument(
        "--orion_wsi",
        required=True, 
        help="Path to Orion/multiplex whole slide image"
    )
    
    parser.add_argument(
        "--output",
        default="./wsi_registration_output",
        help="Output directory (default: ./wsi_registration_output)"
    )
    
    # Registration parameters
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


def validate_inputs(args):
    """Validate input arguments."""
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
    """Main function."""
    args = parse_arguments()
    
    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        print("Input validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Create configuration
    config = WSIRegistrationConfig(
        he_wsi_path=args.he_wsi,
        orion_wsi_path=args.orion_wsi,
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
    
    # Print configuration
    print("=" * 60)
    print("WSI Registration and Core Extraction Pipeline")
    print("=" * 60)
    print(f"H&E WSI: {config.he_wsi_path}")
    print(f"Orion WSI: {config.orion_wsi_path}")
    print(f"Output Directory: {config.output_dir}")
    print()
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
    print("Processing Options:")
    print(f"  Compression: {config.compression}")
    print(f"  Generate Plots: {config.save_core_detection_plots}")
    print("=" * 60)
    
    try:
        # Create and run pipeline
        pipeline = WSIRegistrationPipeline(config)
        results = pipeline.run()
        
        # Print results
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS")
        print("=" * 60)
        
        if results.get('success', False):
            print("✅ Pipeline completed successfully!")
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
            print(f"  📁 {config.output_dir}/")
            print(f"    ├── 📁 registered_wsi/          # Registered whole slide images")
            print(f"    ├── 📁 extracted_cores/         # Individual core pairs")
            if config.save_core_detection_plots:
                print(f"    ├── 📁 quality_plots/           # Visualization plots")
            print(f"    └── 📄 wsi_pipeline_summary.json # Pipeline summary")
            print()
            print(f"🎯 Found and extracted {results.get('matched_pairs', 0)} matched core pairs!")
            print(f"📂 Cores saved to: {config.output_dir}/extracted_cores/")
            
        else:
            print("❌ Pipeline failed!")
            error_msg = results.get('error', 'Unknown error')
            print(f"Error: {error_msg}")
            
            # Additional error details
            if not results.get('registration_success', True):
                reg_error = results.get('registration_error')
                if reg_error:
                    print(f"Registration Error: {reg_error}")
            
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 