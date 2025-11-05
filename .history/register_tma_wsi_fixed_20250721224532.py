#!/usr/bin/env python3
"""
TMA WSI Registration using VALIS (with OME-TIFF preprocessing)

Clean, focused registration script for large TMA whole slide images.
Includes preprocessing step to handle OME-TIFF "tiled separate planes" format
that causes pyvips errors.

Usage:
    python register_tma_wsi_fixed.py --he_wsi input_he.ome.tiff --orion_wsi input_orion.ome.tiff --output output_dir
"""

import argparse
import sys
import shutil
import logging
import json
import tempfile
import numpy as np
import cv2
import warnings
from pathlib import Path
from tifffile import imread, imwrite

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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def convert_ome_tiff_for_valis(input_path: str, output_path: str, max_resolution: int = None, registration_channel: int = None) -> bool:
    """
    Convert OME-TIFF to VALIS-compatible format.
    Handles "tiled separate planes" format that causes pyvips errors.
    
    Args:
        input_path: Path to input OME-TIFF file
        output_path: Path to output TIFF file
        max_resolution: Maximum dimension to resize to (optional)
        registration_channel: Channel to use for registration (for multi-channel images)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Converting OME-TIFF: {input_path} -> {output_path}")
        
        # Read the image
        img = imread(input_path)
        logger.info(f"Original shape: {img.shape}, dtype: {img.dtype}")
        
        # Handle different image formats
        if img.ndim == 2:
            # Grayscale image
            processed_img = img
        elif img.ndim == 3:
            # Check if it's RGB, multi-channel, or other format
            if img.shape[2] <= 4 and img.shape[0] > img.shape[2] and img.shape[1] > img.shape[2]:
                # Likely (H, W, C) format with few channels - probably RGB
                processed_img = img
                logger.info("Detected RGB/RGBA format (H, W, C)")
            elif img.shape[0] <= 50 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                # Likely (C, H, W) format - multi-channel
                logger.info(f"Detected multi-channel format (C, H, W): {img.shape[0]} channels")
                processed_img = img  # Keep as is for multi-channel
            else:
                # Ambiguous - treat as RGB if 3 channels, otherwise keep as is
                if img.shape[2] == 3:
                    processed_img = img
                    logger.info("Treating as RGB format")
                else:
                    processed_img = img
                    logger.info(f"Keeping original format: {img.shape}")
        elif img.ndim == 4:
            # 4D image - could be (T, C, H, W) or (C, Z, H, W) etc.
            logger.info(f"4D image detected: {img.shape}")
            if img.shape[0] == 1:
                # Remove singleton first dimension
                processed_img = img[0]
                logger.info(f"Removed singleton dimension: {processed_img.shape}")
            else:
                # Take first timepoint/z-slice
                processed_img = img[0]
                logger.info(f"Taking first slice: {processed_img.shape}")
        else:
            logger.warning(f"Unsupported image dimensions: {img.ndim}D")
            processed_img = img
        
        # Auto-resize very large images to prevent memory issues
        auto_resize_threshold = 20000  # 20k pixels
        if max_resolution is None and max(processed_img.shape[-2:]) > auto_resize_threshold:
            logger.warning(f"Image is very large ({processed_img.shape[-2:]}), auto-resizing to {auto_resize_threshold}px to prevent memory/format issues")
            max_resolution = auto_resize_threshold
        
        # Resize if requested or auto-triggered
        if max_resolution and max(processed_img.shape[-2:]) > max_resolution:
            # Get spatial dimensions (last 2 dimensions)
            height, width = processed_img.shape[-2:]
            
            # Calculate new size
            scale = max_resolution / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            logger.info(f"Resizing from {height}x{width} to {new_height}x{new_width}")
            
            if processed_img.ndim == 2:
                # Grayscale
                processed_img = cv2.resize(processed_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            elif processed_img.ndim == 3:
                if processed_img.shape[0] < processed_img.shape[2]:
                    # (C, H, W) format
                    resized_channels = []
                    for c in range(processed_img.shape[0]):
                        channel = cv2.resize(processed_img[c], (new_width, new_height), interpolation=cv2.INTER_AREA)
                        resized_channels.append(channel)
                    processed_img = np.stack(resized_channels, axis=0)
                else:
                    # (H, W, C) format
                    processed_img = cv2.resize(processed_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Validate data before saving
        if not np.isfinite(processed_img).all():
            logger.warning("Image contains non-finite values, clipping...")
            processed_img = np.nan_to_num(processed_img, nan=0, posinf=255, neginf=0)
        
        # Ensure data is in valid range for uint8
        if processed_img.dtype != np.uint8:
            # Convert to uint8
            processed_img = ((processed_img - processed_img.min()) / (processed_img.max() - processed_img.min()) * 255).astype(np.uint8)
        
        processed_img = np.clip(processed_img, 0, 255)
        
        logger.info(f"Final processed shape: {processed_img.shape}, dtype: {processed_img.dtype}")
        logger.info(f"Data range: {processed_img.min()} - {processed_img.max()}")
        
        # Save as regular TIFF (not OME-TIFF to avoid pyvips issues)
        try:
            if processed_img.ndim == 3 and processed_img.shape[0] <= 50 and processed_img.shape[0] != 3:
                # Multi-channel format (not RGB) - save for registration
                logger.info(f"Detected multi-channel image with {processed_img.shape[0]} channels")
                
                # For registration: save single channel (DAPI/first channel) as grayscale
                if registration_channel is not None:
                    if registration_channel < processed_img.shape[0]:
                        single_channel = processed_img[registration_channel]
                        logger.info(f"Using channel {registration_channel} for registration")
                    else:
                        single_channel = processed_img[0]
                        logger.info(f"Channel {registration_channel} not available, using channel 0")
                else:
                    single_channel = processed_img[0]  # Default to first channel (usually DAPI)
                    logger.info("Using channel 0 (typically DAPI) for registration")
                
                # Save the single channel for registration
                imwrite(
                    output_path,
                    single_channel,
                    photometric='minisblack',
                    compression='lzw'
                )
                
                # Also save the full multi-channel version with a different name
                multi_channel_path = output_path.replace('.tif', '_multichannel.tif')
                logger.info(f"Saving full {processed_img.shape[0]}-channel version to: {multi_channel_path}")
                imwrite(
                    multi_channel_path,
                    processed_img,
                    photometric='minisblack',
                    compression='lzw'
                )
                
            elif processed_img.ndim == 3 and processed_img.shape[0] == 3:
                # For H&E images, convert to grayscale for better VALIS compatibility
                logger.info("Converting H&E to grayscale for better VALIS registration")
                
                # Handle (C, H, W) format
                processed_img_rgb = np.transpose(processed_img, (1, 2, 0))
                logger.info(f"Transposed to shape: {processed_img_rgb.shape}")
                
                # Convert RGB to grayscale using standard weights
                processed_img_gray = cv2.cvtColor(processed_img_rgb, cv2.COLOR_RGB2GRAY)
                logger.info(f"Converted to grayscale: {processed_img_gray.shape}")
                
                # Save grayscale version for registration
                imwrite(
                    output_path,
                    processed_img_gray,
                    photometric='minisblack',
                    compression='lzw'
                )
                
                # Also save RGB version for visualization if needed
                rgb_path = output_path.replace('.tif', '_rgb.tif')
                logger.info(f"Saving RGB version to: {rgb_path}")
                imwrite(
                    rgb_path,
                    processed_img_rgb,
                    compression='lzw'
                )
            else:
                # RGB or grayscale
                logger.info("Saving as standard TIFF")
                imwrite(
                    output_path,
                    processed_img,
                    compression='lzw'
                )
                
        except Exception as save_error:
            logger.warning(f"Failed to save with LZW compression: {save_error}")
            # Fallback: save without compression
            imwrite(output_path, processed_img.astype(np.uint8))
        
        logger.info(f"✅ Successfully converted to {output_path}")
        logger.info(f"Output shape: {processed_img.shape}, dtype: {processed_img.dtype}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to convert {input_path}: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Register TMA WSI images using VALIS (with OME-TIFF preprocessing)"
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
        default="./tma_registration_output",
        help="Output directory (default: ./tma_registration_output)"
    )
    
    # VALIS parameters optimized for large TMA images
    parser.add_argument(
        "--max_processed_dim",
        type=int,
        default=1500,
        help="Max dimension for processed images (default: 1500)"
    )
    
    parser.add_argument(
        "--max_nonrigid_dim", 
        type=int,
        default=3000,
        help="Max dimension for non-rigid registration (default: 3000)"
    )
    
    # Preprocessing parameters
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


def register_tma_images_with_preprocessing(he_wsi_path, orion_wsi_path, output_dir, **params):
    """
    Preprocess OME-TIFF files and register TMA WSI images using VALIS.
    
    Parameters:
        he_wsi_path: Path to H&E OME-TIFF image
        orion_wsi_path: Path to Orion OME-TIFF image
        output_dir: Output directory
        **params: VALIS and preprocessing parameters
        
    Returns:
        dict: Registration results
    """
    
    # Setup directories
    output_path = Path(output_dir)
    preprocessed_dir = output_path / "preprocessed"
    slides_dir = output_path / "slides"
    results_dir = output_path / "valis_results"
    registered_dir = output_path / "registered"
    
    for dir_path in [preprocessed_dir, slides_dir, results_dir, registered_dir]:
        dir_path.mkdir(exist_ok=True)
    
    temp_dir = None
    
    try:
        # Step 1: Preprocess OME-TIFF files to VALIS-compatible format
        logger.info("🔄 STEP 1: Preprocessing OME-TIFF files to VALIS-compatible format")
        logger.info("=" * 60)
        
        # Create preprocessing directory
        if params['keep_preprocessed']:
            preprocess_output_dir = preprocessed_dir
        else:
            temp_dir = tempfile.mkdtemp(prefix="tma_preprocess_")
            preprocess_output_dir = Path(temp_dir)
        
        he_processed = preprocess_output_dir / "he_processed.tif"
        orion_processed = preprocess_output_dir / "orion_processed.tif"
        
        # Preprocess H&E
        logger.info(f"Converting H&E: {he_wsi_path} -> {he_processed}")
        he_success = convert_ome_tiff_for_valis(
            he_wsi_path, 
            str(he_processed), 
            params.get('max_preprocess_resolution')
        )
        
        if not he_success:
            raise ValueError("Failed to preprocess H&E image")
        
        # Preprocess Orion (with DAPI channel for registration)
        logger.info(f"Converting Orion: {orion_wsi_path} -> {orion_processed}")
        orion_success = convert_ome_tiff_for_valis(
            orion_wsi_path, 
            str(orion_processed), 
            params.get('max_preprocess_resolution'),
            registration_channel=0  # Use channel 0 (typically DAPI) for registration
        )
        
        if not orion_success:
            raise ValueError("Failed to preprocess Orion image")
        
        logger.info("✅ Preprocessing completed successfully!")
        
        # Step 2: Prepare slides for VALIS
        logger.info("🔄 STEP 2: Preparing slides for VALIS registration")
        logger.info("=" * 60)
        
        # Copy preprocessed files to VALIS input directory with consistent names
        he_slide_path = slides_dir / "01_he_reference.tif"  
        orion_slide_path = slides_dir / "02_orion_target.tif"
        
        logger.info(f"Copying H&E: {he_processed} -> {he_slide_path}")
        shutil.copy2(he_processed, he_slide_path)
        
        logger.info(f"Copying Orion: {orion_processed} -> {orion_slide_path}")
        shutil.copy2(orion_processed, orion_slide_path)
        
        # Step 3: Create VALIS registrar
        logger.info("🔄 STEP 3: Initializing VALIS registrar")
        logger.info("=" * 60)
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
        
        # Step 4: Perform registration
        logger.info("🔄 STEP 4: Starting VALIS registration")
        logger.info("=" * 60)
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
        
        # Step 5: Warp and save registered images
        logger.info("🔄 STEP 5: Warping and saving registered images")
        logger.info("=" * 60)
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
        
        # Step 6: Clean up JVM
        try:
            registration.kill_jvm()
            logger.info("🧹 JVM cleaned up")
        except Exception as e:
            logger.warning(f"Warning: JVM cleanup failed: {e}")
        
        # Step 7: Verify output files
        expected_files = []
        for slide_path in [he_slide_path, orion_slide_path]:
            expected_file = registered_dir / f"{slide_path.stem}.ome.tiff"
            if expected_file.exists():
                expected_files.append(str(expected_file))
                logger.info(f"✅ Created: {expected_file}")
            else:
                raise FileNotFoundError(f"Expected output file not found: {expected_file}")
        
        # Step 8: Clean up intermediate files if not keeping them
        if not params['keep_intermediate']:
            logger.info("🧹 Cleaning up intermediate files...")
            try:
                shutil.rmtree(slides_dir)
                if not params['keep_intermediate']:
                    shutil.rmtree(results_dir)
            except Exception as e:
                logger.warning(f"Warning: Cleanup failed: {e}")
        
        # Clean up temp preprocessing directory
        if temp_dir and not params['keep_preprocessed']:
            try:
                shutil.rmtree(temp_dir)
                logger.info("🧹 Cleaned up temporary preprocessing files")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
        
        # Step 9: Create summary
        summary = {
            "success": True,
            "registration_type": "rigid_only" if params['rigid_only'] else "rigid_and_nonrigid",
            "parameters": {
                "max_processed_dim": params['max_processed_dim'],
                "max_nonrigid_dim": params['max_nonrigid_dim'],
                "crop": params['crop'],
                "compression": params['compression'],
                "compression_quality": params['compression_quality'],
                "rigid_only": params['rigid_only'],
                "max_preprocess_resolution": params.get('max_preprocess_resolution'),
                "keep_preprocessed": params['keep_preprocessed']
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
    
    finally:
        # Clean up temp directory if it exists
        if temp_dir and not params.get('keep_preprocessed', False):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")


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
    print("🔬 TMA WSI REGISTRATION using VALIS (with OME-TIFF preprocessing)")
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
    print()
    print("🔧 Preprocessing Parameters:")
    if args.max_preprocess_resolution:
        print(f"   Max Preprocess Resolution: {args.max_preprocess_resolution} px")
    else:
        print(f"   Max Preprocess Resolution: Auto (20000px)")
    print(f"   Keep Preprocessed Files: {args.keep_preprocessed}")
    print("=" * 80)
    
    # Prepare parameters
    params = {
        'max_processed_dim': args.max_processed_dim,
        'max_nonrigid_dim': args.max_nonrigid_dim,
        'crop': args.crop,
        'compression': args.compression,
        'compression_quality': args.compression_quality,
        'rigid_only': args.rigid_only,
        'keep_intermediate': args.keep_intermediate,
        'max_preprocess_resolution': args.max_preprocess_resolution,
        'keep_preprocessed': args.keep_preprocessed
    }
    
    # Run registration with preprocessing
    results = register_tma_images_with_preprocessing(
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
        if args.keep_preprocessed:
            print(f"    ├── 📁 preprocessed/            # Preprocessed TIFF files")
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
        print("   - Use --max_preprocess_resolution to resize very large images")
        print("   - Try --rigid_only for faster testing")
        print("   - Use --keep_intermediate and --keep_preprocessed to debug")
        sys.exit(1)


if __name__ == "__main__":
    main() 