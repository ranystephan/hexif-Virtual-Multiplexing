#!/usr/bin/env python3
"""
Registration-Only Workflow for WSI Images

This script focuses ONLY on registering H&E and Orion WSI images using VALIS,
without attempting core detection or extraction. This allows you to verify
registration quality before tackling core pairing separately.

Usage:
    python run_registration_only.py --he_wsi input_he.tif --orion_wsi input_orion.tif --output output_dir
"""

import argparse
import sys
import pathlib
import shutil
import logging
import json
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
        description="Registration-only workflow for WSI images using VALIS"
    )
    
    # Required arguments
    parser.add_argument(
        "--he_wsi", 
        required=True,
        help="Path to H&E whole slide image (preprocessed TIFF)"
    )
    
    parser.add_argument(
        "--orion_wsi",
        required=True, 
        help="Path to Orion/multiplex whole slide image (preprocessed TIFF)"
    )
    
    parser.add_argument(
        "--output",
        default="./registration_only_output",
        help="Output directory (default: ./registration_only_output)"
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
    
    # Output options
    parser.add_argument(
        "--compression",
        default="lzw",
        choices=["lzw", "jpeg", "jp2k"],
        help="Compression method for output images (default: lzw)"
    )
    
    parser.add_argument(
        "--rigid_only",
        action="store_true",
        help="Perform only rigid registration (faster, less precise)"
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
    
    return errors


def register_wsi_only(he_wsi_path, orion_wsi_path, output_dir, max_processed_dim=2048, 
                      max_nonrigid_dim=3000, compression="lzw", rigid_only=False):
    """
    Register WSI images using VALIS without core detection.
    
    Returns:
        dict: Registration results with paths and success status
    """
    # Create output directories
    output_path = Path(output_dir)
    registered_dir = output_path / "registered_wsi"
    temp_dir = output_path / "temp"
    
    for dir_path in [registered_dir, temp_dir]:
        dir_path.mkdir(exist_ok=True)
    
    try:
        # Create temporary directory for VALIS input
        src_dir = temp_dir / "valis_input"
        dst_dir = temp_dir / "valis_output"
        src_dir.mkdir(exist_ok=True)
        dst_dir.mkdir(exist_ok=True)
        
        # Copy/link input files to VALIS input directory with consistent names
        he_temp_path = src_dir / "he_slide.tif"
        orion_temp_path = src_dir / "orion_slide.tif"
        
        logger.info(f"Preparing VALIS input files:")
        logger.info(f"  H&E: {he_wsi_path} -> {he_temp_path}")
        logger.info(f"  Orion: {orion_wsi_path} -> {orion_temp_path}")
        
        # Copy files (VALIS works better with local copies)
        shutil.copy2(he_wsi_path, str(he_temp_path))
        shutil.copy2(orion_wsi_path, str(orion_temp_path))
        
        # Create VALIS registrar
        logger.info("Initializing VALIS registration...")
        registrar = registration.Valis(
            str(src_dir),
            str(dst_dir),
            reference_img_f="he_slide.tif",  # Use H&E as reference
            align_to_reference=True,
            imgs_ordered=True,  # Preserve order
            max_processed_image_dim_px=max_processed_dim,
            max_non_rigid_registration_dim_px=max_nonrigid_dim if not rigid_only else None,
            crop="reference",  # Crop to H&E reference
        )
        
        # Perform registration
        logger.info("Starting VALIS registration...")
        try:
            if rigid_only:
                logger.info("Performing RIGID-ONLY registration...")
                rigid_registrar, non_rigid_registrar, error_df = registrar.register()
            else:
                logger.info("Performing RIGID + NON-RIGID registration...")
                rigid_registrar, non_rigid_registrar, error_df = registrar.register()
            
            logger.info("✅ VALIS registration completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ VALIS registration failed: {e}")
            try:
                registration.kill_jvm()
            except:
                pass
            raise
        
        # Warp and save registered images
        logger.info("Warping and saving registered images...")
        try:
            registrar.warp_and_save_slides(
                str(registered_dir),
                crop="reference",
                non_rigid=not rigid_only,  # Use non-rigid if not rigid_only
                compression=compression,
                Q=95  # High quality
            )
            
            # Clean up JVM
            registration.kill_jvm()
            logger.info("✅ Images warped and saved successfully!")
            
        except Exception as e:
            logger.error(f"❌ Image warping failed: {e}")
            try:
                registration.kill_jvm()
            except:
                pass
            raise
        
        # Check output files
        registered_he = registered_dir / "he_slide.ome.tiff"
        registered_orion = registered_dir / "orion_slide.ome.tiff"
        
        if not registered_he.exists():
            raise FileNotFoundError(f"Registered H&E not found: {registered_he}")
        
        if not registered_orion.exists():
            raise FileNotFoundError(f"Registered Orion not found: {registered_orion}")
        
        # Create summary
        summary = {
            "success": True,
            "registration_type": "rigid_only" if rigid_only else "rigid_and_nonrigid",
            "input_files": {
                "he_wsi": str(he_wsi_path),
                "orion_wsi": str(orion_wsi_path)
            },
            "output_files": {
                "registered_he": str(registered_he),
                "registered_orion": str(registered_orion)
            },
            "parameters": {
                "max_processed_dim": max_processed_dim,
                "max_nonrigid_dim": max_nonrigid_dim,
                "compression": compression,
                "rigid_only": rigid_only
            }
        }
        
        # Save summary
        summary_path = output_path / "registration_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Registration summary saved to: {summary_path}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "input_files": {
                "he_wsi": str(he_wsi_path),
                "orion_wsi": str(orion_wsi_path)
            }
        }
    finally:
        # Clean up temporary files
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info("🧹 Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")


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
    
    print("=" * 70)
    print("🔄 REGISTRATION-ONLY WORKFLOW")
    print("=" * 70)
    print(f"📁 H&E WSI: {args.he_wsi}")
    print(f"📁 Orion WSI: {args.orion_wsi}")
    print(f"📂 Output Directory: {args.output}")
    print(f"⚙️  Max Processed Dimension: {args.max_processed_dim}")
    print(f"⚙️  Max Non-rigid Dimension: {args.max_nonrigid_dim}")
    print(f"⚙️  Compression: {args.compression}")
    print(f"⚙️  Rigid Only: {args.rigid_only}")
    print("=" * 70)
    
    # Run registration
    results = register_wsi_only(
        he_wsi_path=args.he_wsi,
        orion_wsi_path=args.orion_wsi,
        output_dir=args.output,
        max_processed_dim=args.max_processed_dim,
        max_nonrigid_dim=args.max_nonrigid_dim,
        compression=args.compression,
        rigid_only=args.rigid_only
    )
    
    # Report results
    print("\n" + "=" * 70)
    print("📊 REGISTRATION RESULTS")
    print("=" * 70)
    
    if results.get("success"):
        print("✅ Registration completed successfully!")
        print()
        print("📄 Output Files:")
        print(f"  🖼️  Registered H&E: {results['output_files']['registered_he']}")
        print(f"  🖼️  Registered Orion: {results['output_files']['registered_orion']}")
        print()
        print("📂 Output Structure:")
        print(f"  {args.output}/")
        print(f"    ├── 📁 registered_wsi/")
        print(f"    │   ├── 📄 he_slide.ome.tiff")
        print(f"    │   └── 📄 orion_slide.ome.tiff")
        print(f"    └── 📄 registration_summary.json")
        print()
        print("🎯 Registration is now complete!")
        print("💡 Next steps:")
        print("   1. Inspect the registered images visually")
        print("   2. Develop core detection parameters separately") 
        print("   3. Create core pairing workflow once parameters are tuned")
        
    else:
        print("❌ Registration failed!")
        print(f"🔥 Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main() 