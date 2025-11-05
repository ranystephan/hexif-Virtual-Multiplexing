#!/usr/bin/env python3
"""
OME-TIFF Preprocessor for VALIS Compatibility

This script converts OME-TIFF files that have "tiled separate planes" format
(which causes pyvips errors) to a VALIS-compatible format.

Usage:
    python preprocess_ome_tiff.py --input input.ome.tiff --output output.tif
    python preprocess_ome_tiff.py --input_dir /path/to/ome/files --output_dir /path/to/processed
"""

import argparse
import sys
from pathlib import Path
import logging
import numpy as np
from tifffile import imread, imwrite
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def convert_ome_tiff(input_path: str, output_path: str, max_resolution: int = None) -> bool:
    """
    Convert OME-TIFF to VALIS-compatible format.
    
    Args:
        input_path: Path to input OME-TIFF file
        output_path: Path to output TIFF file
        max_resolution: Maximum dimension to resize to (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Converting {input_path} to {output_path}")
        
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
        
        # Resize if requested
        if max_resolution and max(processed_img.shape[-2:]) > max_resolution:
            import cv2
            
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
        if processed_img.dtype == np.uint8:
            processed_img = np.clip(processed_img, 0, 255)
        else:
            # Convert to uint8 if needed
            processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        logger.info(f"Final processed shape: {processed_img.shape}, dtype: {processed_img.dtype}")
        logger.info(f"Data range: {processed_img.min()} - {processed_img.max()}")
        
        # Save as regular TIFF (not OME-TIFF to avoid pyvips issues)
        try:
            if processed_img.ndim == 3 and processed_img.shape[0] <= 50 and processed_img.shape[0] != 3:
                # Multi-channel format (not RGB) - save with proper axes metadata
                logger.info("Saving as multi-channel TIFF")
                imwrite(
                    output_path, 
                    processed_img,
                    photometric='minisblack',  # Prevent RGB interpretation
                    compression='lzw',
                    # Remove problematic metadata for now
                )
            elif processed_img.ndim == 3 and processed_img.shape[0] == 3:
                # Convert (C, H, W) to (H, W, C) for RGB
                logger.info("Converting to RGB format (H, W, C)")
                processed_img_rgb = np.transpose(processed_img, (1, 2, 0))
                imwrite(
                    output_path,
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
            logger.info("Trying without compression...")
            
            # Fallback: save without compression
            if processed_img.ndim == 3 and processed_img.shape[0] == 3:
                # Convert (C, H, W) to (H, W, C) for RGB
                processed_img_rgb = np.transpose(processed_img, (1, 2, 0))
                imwrite(output_path, processed_img_rgb)
            else:
                imwrite(output_path, processed_img)
        
        logger.info(f"✅ Successfully converted to {output_path}")
        logger.info(f"Output shape: {processed_img.shape}, dtype: {processed_img.dtype}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to convert {input_path}: {e}")
        return False


def convert_directory(input_dir: str, output_dir: str, max_resolution: int = None) -> dict:
    """
    Convert all OME-TIFF files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        max_resolution: Maximum dimension to resize to
        
    Returns:
        Dictionary with conversion results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return {'success': 0, 'failed': 0, 'files': []}
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find OME-TIFF files
    ome_files = list(input_path.glob("*.ome.tif*"))
    
    if not ome_files:
        logger.warning(f"No OME-TIFF files found in {input_path}")
        return {'success': 0, 'failed': 0, 'files': []}
    
    logger.info(f"Found {len(ome_files)} OME-TIFF files to convert")
    
    results = {'success': 0, 'failed': 0, 'files': []}
    
    for ome_file in ome_files:
        # Create output filename (remove .ome extension)
        if ome_file.name.endswith('.ome.tiff'):
            output_name = ome_file.name.replace('.ome.tiff', '.tif')
        elif ome_file.name.endswith('.ome.tif'):
            output_name = ome_file.name.replace('.ome.tif', '.tif')
        else:
            output_name = ome_file.name
        
        output_file = output_path / output_name
        
        success = convert_ome_tiff(str(ome_file), str(output_file), max_resolution)
        
        if success:
            results['success'] += 1
        else:
            results['failed'] += 1
        
        results['files'].append({
            'input': str(ome_file),
            'output': str(output_file),
            'success': success
        })
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert OME-TIFF files to VALIS-compatible format")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Input OME-TIFF file")
    group.add_argument("--input_dir", help="Input directory containing OME-TIFF files")
    
    # Output options
    parser.add_argument("--output", help="Output TIFF file (for single file conversion)")
    parser.add_argument("--output_dir", help="Output directory (for directory conversion)")
    
    # Processing options
    parser.add_argument("--max_resolution", type=int, 
                       help="Maximum dimension to resize to (optional)")
    
    args = parser.parse_args()
    
    try:
        if args.input:
            # Single file conversion
            if not args.output:
                # Generate output filename
                input_path = Path(args.input)
                if input_path.name.endswith('.ome.tiff'):
                    output_name = input_path.name.replace('.ome.tiff', '_converted.tif')
                elif input_path.name.endswith('.ome.tif'):
                    output_name = input_path.name.replace('.ome.tif', '_converted.tif')
                else:
                    output_name = input_path.stem + '_converted.tif'
                
                args.output = str(input_path.parent / output_name)
            
            success = convert_ome_tiff(args.input, args.output, args.max_resolution)
            
            if success:
                print(f"✅ Successfully converted {args.input} to {args.output}")
                sys.exit(0)
            else:
                print(f"❌ Failed to convert {args.input}")
                sys.exit(1)
        
        else:
            # Directory conversion
            if not args.output_dir:
                args.output_dir = str(Path(args.input_dir) / "converted")
            
            results = convert_directory(args.input_dir, args.output_dir, args.max_resolution)
            
            print(f"\n{'='*50}")
            print(f"CONVERSION SUMMARY")
            print(f"{'='*50}")
            print(f"Total files: {len(results['files'])}")
            print(f"Successful: {results['success']}")
            print(f"Failed: {results['failed']}")
            print(f"Output directory: {args.output_dir}")
            
            if results['failed'] > 0:
                print(f"\nFailed conversions:")
                for file_result in results['files']:
                    if not file_result['success']:
                        print(f"  ❌ {file_result['input']}")
            
            if results['success'] > 0:
                print(f"✅ Successfully converted {results['success']} files!")
                sys.exit(0)
            else:
                print(f"❌ No files were successfully converted")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 