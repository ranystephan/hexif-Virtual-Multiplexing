#!/usr/bin/env python3
"""
TMA WSI Registration using VALIS

Clean, focused registration script for large TMA whole slide images.
Follows VALIS documentation best practices for images with 270+ cores.

Usage:
    python register_tma_wsi.py --he_wsi input_he.tif --orion_wsi input_orion.tif --output output_dir
"""

import argparse
import sys
import shutil
import logging
import json
import tempfile
from pathlib import Path

# VALIS registration
try:
    from valis import registration
    VALIS_AVAILABLE = True
except ImportError:
    VALIS_AVAILABLE = False
    print("❌ VALIS not available. Please install with: pip install valis")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Register TMA WSI images using VALIS (registration only, no core detection)"
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
        default="./tma_registration_output",
        help="Output directory (default: ./tma_registration_output)"
    )
    
    # VALIS parameters optimized for large TMA images
    parser.add_argument(
        "--max_processed_dim",
        type=int,
        default=1500,  # Larger for TMA with lots of cores
        help="Max dimension for processed images (default: 1500, VALIS default: 850)"
    )
    
    parser.add_argument(
        "--max_nonrigid_dim", 
        type=int,
        default=3000,  # Higher resolution for non-rigid
        help="Max dimension for non-rigid registration (default: 3000)"
    )
    
    # Cropping and compression options
    parser.add_argument(
        "--crop",
        choices=["reference", "overlap", "all"],
        default="reference",
        help="Cropping method (default: reference)"
    )
    
    parser.add_argument(
        "--compression",
        choices=["lzw", "jpeg", "jp2k"],
        default="lzw",
        help="Compression method (default: lzw - lossless)"
    )
    
    parser.add_argument(
        "--compression_quality",
        type=int,
        default=95,
        help="Compression quality for lossy methods (default: 95)"
    )
    
    # Processing options
    parser.add_argument(
        "--rigid_only",
        action="store_true",
        help="Perform only rigid registration (faster, less precise)"
    )
    
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate VALIS processing files"
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
    
    # Validate parameters
    if args.max_processed_dim < 500 or args.max_processed_dim > 4000:
        errors.append("max_processed_dim should be between 500-4000")
    
    if args.max_nonrigid_dim < 1000 or args.max_nonrigid_dim > 8000:
        errors.append("max_nonrigid_dim should be between 1000-8000")
    
    if args.compression_quality < 1 or args.compression_quality > 100:
        errors.append("compression_quality should be between 1-100")
    
    return errors


def register_tma_images(he_wsi_path, orion_wsi_path, output_dir, **params):
    """
    Register TMA WSI images using VALIS.
    
    Parameters:
        he_wsi_path: Path to H&E image
        orion_wsi_path: Path to Orion image  
        output_dir: Output directory
        **params: VALIS parameters
        
    Returns:
        dict: Registration results
    """
    
    # Setup directories
    output_path = Path(output_dir)
    slides_dir = output_path / "slides"  # Input slides for VALIS
    results_dir = output_path / "valis_results"  # VALIS intermediate results
    registered_dir = output_path / "registered"  # Final registered images
    
    for dir_path in [slides_dir, results_dir, registered_dir]:
        dir_path.mkdir(exist_ok=True)
    
    try:
        # Step 1: Prepare slides for VALIS
        logger.info("📁 Preparing slides for VALIS...")
        
        # VALIS expects consistent naming, H&E should be reference
        he_slide_path = slides_dir / "01_he_reference.tif"  
        orion_slide_path = slides_dir / "02_orion_target.tif"
        
        # Copy files to VALIS input directory
        logger.info(f"Copying H&E: {he_wsi_path} -> {he_slide_path}")
        shutil.copy2(he_wsi_path, he_slide_path)
        
        logger.info(f"Copying Orion: {orion_wsi_path} -> {orion_slide_path}")
        shutil.copy2(orion_wsi_path, orion_slide_path)
        
        # Step 2: Create VALIS registrar with optimized parameters
        logger.info("🔧 Initializing VALIS registrar...")
        logger.info(f"   Max processed dimension: {params['max_processed_dim']}")
        logger.info(f"   Max non-rigid dimension: {params['max_nonrigid_dim']}")
        logger.info(f"   Reference image: {he_slide_path.name}")
        logger.info(f"   Cropping method: {params['crop']}")
        
        registrar = registration.Valis(
            src_dir=str(slides_dir),
            dst_dir=str(results_dir),
            reference_img_f=he_slide_path.name,  # Use H&E as reference
            align_to_reference=True,              # Align Orion directly to H&E
            imgs_ordered=True,                    # Preserve order (H&E first)
            max_processed_image_dim_px=params['max_processed_dim'],
            max_non_rigid_registration_dim_px=params['max_nonrigid_dim'],
            crop=params['crop'],                  # Crop method
        )
        
        # Step 3: Perform registration
        logger.info("🔄 Starting VALIS registration...")
        logger.info("   This may take several minutes for large images...")
        
        # Register with error handling
        try:
            rigid_registrar, non_rigid_registrar, error_df = registrar.register()
            logger.info("✅ VALIS registration completed successfully!")
            
            # Log registration results
            if error_df is not None and not error_df.empty:
                logger.info("📊 Registration error summary:")
                for _, row in error_df.iterrows():
                    logger.info(f"   {row.get('img1', 'N/A')} -> {row.get('img2', 'N/A')}: "
                              f"Error = {row.get('tre', 'N/A'):.3f}")
            
        except Exception as e:
            logger.error(f"❌ VALIS registration failed: {e}")
            try:
                registration.kill_jvm()
            except:
                pass
            raise ValueError(f"Registration failed: {e}")
        
        # Step 4: Warp and save registered images
        logger.info("💾 Warping and saving registered images...")
        logger.info(f"   Compression: {params['compression']}")
        logger.info(f"   Quality: {params['compression_quality']}")
        logger.info(f"   Non-rigid: {not params['rigid_only']}")
        
        try:
            registrar.warp_and_save_slides(
                dst_dir=str(registered_dir),
                crop=params['crop'],
                non_rigid=not params['rigid_only'],  # Apply non-rigid if not rigid_only
                compression=params['compression'],
                Q=params['compression_quality']
            )
            
            logger.info("✅ Images warped and saved successfully!")
            
        except Exception as e:
            logger.error(f"❌ Image warping failed: {e}")
            try:
                registration.kill_jvm()
            except:
                pass
            raise ValueError(f"Warping failed: {e}")
        
        # Step 5: Clean up JVM
        try:
            registration.kill_jvm()
            logger.info("🧹 JVM cleaned up")
        except Exception as e:
            logger.warning(f"Warning: JVM cleanup failed: {e}")
        
        # Step 6: Verify output files
        expected_files = []
        for slide_path in [he_slide_path, orion_slide_path]:
            expected_file = registered_dir / f"{slide_path.stem}.ome.tiff"
            if expected_file.exists():
                expected_files.append(str(expected_file))
                logger.info(f"✅ Created: {expected_file}")
            else:
                raise FileNotFoundError(f"Expected output file not found: {expected_file}")
        
        # Step 7: Clean up intermediate files if not keeping them
        if not params['keep_intermediate']:
            logger.info("🧹 Cleaning up intermediate files...")
            try:
                shutil.rmtree(slides_dir)
                if not params['keep_intermediate']:
                    shutil.rmtree(results_dir)
            except Exception as e:
                logger.warning(f"Warning: Cleanup failed: {e}")
        
        # Step 8: Create summary
        summary = {
            "success": True,
            "registration_type": "rigid_only" if params['rigid_only'] else "rigid_and_nonrigid",
            "parameters": {
                "max_processed_dim": params['max_processed_dim'],
                "max_nonrigid_dim": params['max_nonrigid_dim'],
                "crop": params['crop'],
                "compression": params['compression'],
                "compression_quality": params['compression_quality'],
                "rigid_only": params['rigid_only']
            },
            "input_files": {
                "he_wsi": str(he_wsi_path),
                "orion_wsi": str(orion_wsi_path)
            },
            "output_files": expected_files,
            "output_directory": str(registered_dir)
        }
        
        # Save summary
        summary_path = output_path / "registration_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Registration summary saved: {summary_path}")
        
        return summary
        
    except Exception as e:
        # Ensure JVM cleanup on any error
        try:
            registration.kill_jvm()
        except:
            pass
        
        logger.error(f"Registration pipeline failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "input_files": {
                "he_wsi": str(he_wsi_path),
                "orion_wsi": str(orion_wsi_path)
            }
        }


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        print("❌ Input validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Print configuration
    print("=" * 80)
    print("🔬 TMA WSI REGISTRATION using VALIS")
    print("=" * 80)
    print(f"📁 H&E WSI: {args.he_wsi}")
    print(f"📁 Orion WSI: {args.orion_wsi}")
    print(f"📂 Output Directory: {args.output}")
    print()
    print("🔧 VALIS Parameters (optimized for large TMA images):")
    print(f"   Max Processed Dimension: {args.max_processed_dim} px")
    print(f"   Max Non-rigid Dimension: {args.max_nonrigid_dim} px") 
    print(f"   Cropping Method: {args.crop}")
    print(f"   Compression: {args.compression} (Quality: {args.compression_quality})")
    print(f"   Registration Mode: {'Rigid Only' if args.rigid_only else 'Rigid + Non-rigid'}")
    print("=" * 80)
    
    # Prepare parameters
    params = {
        'max_processed_dim': args.max_processed_dim,
        'max_nonrigid_dim': args.max_nonrigid_dim,
        'crop': args.crop,
        'compression': args.compression,
        'compression_quality': args.compression_quality,
        'rigid_only': args.rigid_only,
        'keep_intermediate': args.keep_intermediate
    }
    
    # Run registration
    results = register_tma_images(
        he_wsi_path=args.he_wsi,
        orion_wsi_path=args.orion_wsi,
        output_dir=args.output,
        **params
    )
    
    # Report results
    print("\n" + "=" * 80)
    print("📊 REGISTRATION RESULTS")
    print("=" * 80)
    
    if results.get("success"):
        print("✅ Registration completed successfully!")
        print()
        print("📄 Output Files:")
        for output_file in results['output_files']:
            print(f"  🖼️  {output_file}")
        print()
        print("📂 Directory Structure:")
        print(f"  {args.output}/")
        print(f"    ├── 📁 registered/              # Final registered images")
        if args.keep_intermediate:
            print(f"    ├── 📁 valis_results/           # VALIS intermediate results") 
            print(f"    ├── 📁 slides/                  # Input slides copy")
        print(f"    └── 📄 registration_summary.json # Registration summary")
        print()
        
        reg_type = results['parameters']['registration_type']
        print(f"🎯 Registration Type: {reg_type}")
        print(f"📐 Used Parameters: max_processed={results['parameters']['max_processed_dim']}px, "
              f"max_nonrigid={results['parameters']['max_nonrigid_dim']}px")
        print()
        print("✨ Registration Complete!")
        print("💡 Next Steps:")
        print("   1. Inspect registered images visually")
        print("   2. Use registered images for downstream analysis")
        print("   3. For core extraction, develop core detection separately")
        
    else:
        print("❌ Registration failed!")
        print(f"🔥 Error: {results.get('error', 'Unknown error')}")
        print()
        print("💡 Troubleshooting suggestions:")
        print("   - Try different max_processed_dim values (1000-2500)")
        print("   - Check if input images are valid")
        print("   - Try --rigid_only for faster testing")
        print("   - Use --keep_intermediate to inspect VALIS results")
        sys.exit(1)


if __name__ == "__main__":
    main() 