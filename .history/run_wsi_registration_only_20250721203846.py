#!/usr/bin/env python3
"""
Registration-Only WSI Workflow: Preprocessing + Registration (No Core Detection)

This script separates the registration step from core detection, allowing you to:
1. Preprocess OME-TIFF files to VALIS-compatible format
2. Register the whole slide images using VALIS
3. Save registered images without attempting core detection

This is useful for debugging registration issues separately from core detection problems.

Usage:
    python run_wsi_registration_only.py --he_wsi input_he.ome.tiff --orion_wsi input_orion.ome.tiff --output output_dir
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path
import logging
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from preprocess_ome_tiff import convert_ome_tiff

# VALIS registration
try:
    from valis import registration
    VALIS_AVAILABLE = True
except ImportError:
    VALIS_AVAILABLE = False
    warnings.warn("VALIS not available. Please install with: pip install valis")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for registration-only workflow."""
    parser = argparse.ArgumentParser(
        description="Registration-only WSI workflow: preprocess OME-TIFF and register images"
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
        default="./wsi_registration_only_output",
        help="Output directory (default: ./wsi_registration_only_output)"
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
    
    # Processing options
    parser.add_argument(
        "--compression",
        default="lzw",
        choices=["lzw", "jpeg", "jp2k"],
        help="Compression method for output images (default: lzw)"
    )
    
    parser.add_argument(
        "--save_overlays",
        action="store_true",
        help="Save registration overlay images for visual inspection"
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
    
    if not VALIS_AVAILABLE:
        errors.append("VALIS not available. Please install with: pip install valis")
    
    return errors


def register_wsi_images(he_path: str, orion_path: str, output_dir: Path, 
                       max_processed_dim: int = 2048, max_nonrigid_dim: int = 3000,
                       save_overlays: bool = False) -> dict:
    """
    Register WSI images using VALIS.
    
    Returns:
        Dictionary with registration results and metrics
    """
    logger.info("Starting WSI registration using VALIS")
    
    # Create temporary directories for VALIS
    temp_dir = output_dir / "temp"
    src_dir = temp_dir / "valis_input"
    dst_dir = temp_dir / "valis_output"
    src_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy input files to VALIS input directory
        he_temp_path = src_dir / "he_slide.tif"
        orion_temp_path = src_dir / "orion_slide.tif"
        
        logger.info(f"Copying H&E from {he_path} to {he_temp_path}")
        logger.info(f"Copying Orion from {orion_path} to {orion_temp_path}")
        
        shutil.copy2(he_path, str(he_temp_path))
        shutil.copy2(orion_path, str(orion_temp_path))
        
        # Create VALIS registrar
        registrar = registration.Valis(
            str(src_dir),
            str(dst_dir),
            reference_img_f="he_slide.tif",  # Use H&E as reference
            align_to_reference=True,
            imgs_ordered=True,
            max_processed_image_dim_px=max_processed_dim,
            max_non_rigid_registration_dim_px=max_nonrigid_dim,
            crop="reference",  # Crop to H&E reference
            check_for_reflections=False,  # Disable to speed up and avoid issues
        )
        
        # Perform registration
        logger.info("Running VALIS registration...")
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        
        # Create registered WSI directory
        registered_wsi_dir = output_dir / "registered_wsi"
        registered_wsi_dir.mkdir(exist_ok=True)
        
        # Copy registered results to final location
        valis_output_files = list(dst_dir.glob("*.ome.tiff"))
        
        registration_results = {
            'registration_success': True,
            'registered_files': [],
            'valis_output_dir': str(dst_dir),
            'registration_metrics': {}
        }
        
        for valis_file in valis_output_files:
            # Copy to final registered directory
            final_path = registered_wsi_dir / valis_file.name
            shutil.copy2(str(valis_file), str(final_path))
            registration_results['registered_files'].append(str(final_path))
            logger.info(f"Saved registered image: {final_path}")
        
        # Save registration metadata if available
        if hasattr(registrar, 'summary_df') and registrar.summary_df is not None:
            summary_path = output_dir / "registration_summary.csv"
            registrar.summary_df.to_csv(str(summary_path), index=False)
            logger.info(f"Saved registration summary: {summary_path}")
        
        # Create overlays if requested
        if save_overlays:
            logger.info("Creating registration overlay visualizations...")
            overlay_dir = output_dir / "registration_overlays"
            overlay_dir.mkdir(exist_ok=True)
            
            try:
                # This will depend on VALIS version - try to create overlays
                registrar.create_overlap_imgs()
                
                # Copy overlay images if they exist
                overlay_files = list(dst_dir.glob("*overlap*.jpg")) + list(dst_dir.glob("*overlay*.jpg"))
                for overlay_file in overlay_files:
                    final_overlay_path = overlay_dir / overlay_file.name
                    shutil.copy2(str(overlay_file), str(final_overlay_path))
                    logger.info(f"Saved overlay: {final_overlay_path}")
                    
            except Exception as e:
                logger.warning(f"Could not create overlays: {e}")
        
        logger.info("✅ Registration completed successfully!")
        return registration_results
        
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return {
            'registration_success': False,
            'error': str(e),
            'registered_files': []
        }


def main():
    """Main registration-only workflow function."""
    args = parse_arguments()
    
    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        print("Input validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print("=" * 70)
    print("WSI REGISTRATION-ONLY WORKFLOW: Preprocessing + Registration")
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
        
        # Convert Orion
        print(f"Converting Orion: {args.orion_wsi} -> {orion_processed}")
        orion_success = convert_ome_tiff(
            args.orion_wsi, 
            str(orion_processed), 
            args.max_preprocess_resolution,
            registration_channel=0  # Use channel 0 for registration
        )
        
        if not orion_success:
            print("❌ Failed to preprocess Orion image")
            sys.exit(1)
        
        print("✅ Preprocessing completed successfully!")
        
        # Step 2: Register images
        print("\n🔄 STEP 2: Running WSI registration")
        print("-" * 50)
        
        print("Registration Parameters:")
        print(f"  Max Processed Dimension: {args.max_processed_dim}")
        print(f"  Max Non-rigid Dimension: {args.max_nonrigid_dim}")
        print(f"  Save Overlays: {args.save_overlays}")
        print()
        
        registration_results = register_wsi_images(
            str(he_processed),
            str(orion_processed), 
            Path(args.output),
            args.max_processed_dim,
            args.max_nonrigid_dim,
            args.save_overlays
        )
        
        # Step 3: Report results
        print("\n" + "=" * 70)
        print("REGISTRATION RESULTS")
        print("=" * 70)
        
        if registration_results['registration_success']:
            print("✅ Registration completed successfully!")
            print()
            print("Preprocessing Results:")
            print(f"  H&E Converted: ✅ {args.he_wsi} -> {he_processed}")
            print(f"  Orion Converted: ✅ {args.orion_wsi} -> {orion_processed}")
            if args.keep_preprocessed:
                print(f"  Preprocessed files saved in: {temp_dir}")
            print()
            print("Registration Results:")
            print(f"  Registered Files: {len(registration_results['registered_files'])}")
            for file_path in registration_results['registered_files']:
                print(f"    - {file_path}")
            print()
            print("Output Structure:")
            print(f"  📁 {args.output}/")
            print(f"    ├── 📁 registered_wsi/          # Registered whole slide images")
            if args.keep_preprocessed:
                print(f"    ├── 📁 preprocessed_tiff/       # Converted TIFF files")
            if args.save_overlays:
                print(f"    ├── 📁 registration_overlays/   # Registration overlay images")
            print(f"    ├── 📁 temp/                     # VALIS temporary files")
            print(f"    └── 📄 registration_summary.csv  # Registration metrics")
            print()
            print("🎯 Registration completed! You can now proceed with core detection.")
            print("💡 Next steps:")
            print("   1. Examine the registered images in registered_wsi/")
            print("   2. If registration looks good, proceed with core detection")
            print("   3. Use the diagnostic tools to tune core detection parameters")
            
        else:
            print("❌ Registration failed!")
            error_msg = registration_results.get('error', 'Unknown error')
            print(f"Error: {error_msg}")
            sys.exit(1)
        
        # Save results summary
        results_summary = {
            'workflow_type': 'registration_only',
            'input_files': {
                'he_wsi': args.he_wsi,
                'orion_wsi': args.orion_wsi
            },
            'parameters': {
                'max_processed_dim': args.max_processed_dim,
                'max_nonrigid_dim': args.max_nonrigid_dim,
                'max_preprocess_resolution': args.max_preprocess_resolution,
                'compression': args.compression
            },
            'results': registration_results
        }
        
        summary_path = Path(args.output) / "registration_only_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"📄 Summary saved to: {summary_path}")
            
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
        if temp_dir and not args.keep_preprocessed and temp_dir.startswith("/tmp"):
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary preprocessing files")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")


if __name__ == "__main__":
    main() 