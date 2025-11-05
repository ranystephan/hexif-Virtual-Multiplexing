#!/usr/bin/env python3
"""
valis_pairing_pipeline.py
--------------------------

Detect TMA cores in an H&E whole‑slide image, register the H&E and Orion
slides using VALIS, warp the H&E core coordinates to the Orion coordinate
system, and extract paired patches from both slides.  This script
requires the `valis` and `tifffile` libraries and a Java installation for
BioFormats.  It has been designed for whole‑slide images (WSI) that are
too large to be processed entirely in memory; VALIS performs the
registration on downsampled versions of the slides and then applies the
resulting transforms at full resolution.  After running this script
you will obtain a directory containing paired patches, quick‑look
overlays and a CSV summarising each pair.

Note
----
Registration of gigapixel slides can be memory intensive.  Use the
`--mem_gb` argument to allocate sufficient memory to the JVM and adjust
`--max_image_dim`, `--max_processed_dim` and `--max_non_rigid_dim` to
reduce memory usage during registration.  See the VALIS documentation
for details【251153547706828†L174-L224】.

Example
-------
```bash
python valis_pairing_pipeline.py \
    --he data/raw/TA118-HEraw.ome.tiff \
    --orion data/raw/TA118-Orionraw.ome.tiff \
    --out_dir paired_dataset_valis \
    --patch_size 2048 \
    --max_image_dim 1024 --max_processed_dim 512 --max_non_rigid_dim 2048
```

This will register the H&E and Orion slides, detect TMA cores on the
H&E slide, warp the core centres to the Orion slide, extract paired
patches of size 2048×2048 pixels and write the results to
`paired_dataset_valis`.
"""

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# External libraries that must be installed by the user
try:
    import tifffile
except ImportError as e:
    raise ImportError("tifffile must be installed to run this script") from e

try:
    from valis import registration, slide_io
except ImportError as e:
    raise ImportError("valis must be installed to run this script") from e

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError as e:
    raise ImportError("matplotlib must be installed to run this script") from e

logger = logging.getLogger(__name__)


def load_thumbnail(slide_path: str, thumb_size: int = 4096, use_channel0: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load a low‑resolution thumbnail from an OME‑TIFF slide using tifffile.

    Parameters
    ----------
    slide_path : str
        Path to the OME‑TIFF.
    thumb_size : int
        Target size of the largest dimension of the thumbnail.
    use_channel0 : bool
        For fluorescence images that may have multiple channels, force use of
        the first channel.

    Returns
    -------
    thumb : np.ndarray
        Grayscale thumbnail image.
    full_shape : Tuple[int, int]
        (height, width) of the full resolution image.
    """
    with tifffile.TiffFile(slide_path) as tif:
        series = tif.series[0]
        full_shape = series.shape[-2:]
        level = series.levels[-1]
        arr = level.asarray()
        if arr.ndim == 3:
            # OME ordering (samples, y, x)
            if arr.shape[0] < arr.shape[-1]:
                arr = arr[0] if use_channel0 else arr[0]
            else:
                arr = arr[..., 0] if use_channel0 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        thumb = arr.astype(np.uint8)
    # Upscale if smaller than requested
    h, w = thumb.shape
    if max(h, w) < thumb_size:
        scale = thumb_size / max(h, w)
        thumb = cv2.resize(thumb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return thumb, (int(full_shape[0]), int(full_shape[1]))


def detect_cores(
    image: np.ndarray,
    brightfield: bool = True,
    min_area: int = 100,
    circularity_thresh: float = 0.05
) -> List[Dict[str, object]]:
    """
    Detect approximate circular TMA cores in a grayscale thumbnail.

    Parameters
    ----------
    image : np.ndarray
        Grayscale thumbnail.
    brightfield : bool
        If True, uses inverse thresholding for brightfield slides.
        If False, uses thresholding for fluorescence slides.
    min_area : int
        Minimum contour area (pixels in the thumbnail) to consider a core.
    circularity_thresh : float
        Minimum circularity metric (4π * area / perimeter^2) to accept a contour.

    Returns
    -------
    List[Dict]
        Each dict contains 'center' (tuple) and 'area'.
    """
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    if not brightfield:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        blur = clahe.apply(blur)
    bs = max(31, int(min(blur.shape) // 50) * 2 + 1)
    thresh_type = cv2.THRESH_BINARY_INV if brightfield else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, bs, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cores = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < circularity_thresh:
            continue
        cores.append({'center': (cx, cy), 'area': area})
    return cores


def scale_centres(cores: List[Dict[str, object]], thumb_shape: Tuple[int, int], full_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Scale thumbnail coordinates back to full resolution (x, y)."""
    h_t, w_t = thumb_shape
    h_f, w_f = full_shape
    sx = w_f / w_t
    sy = h_f / h_t
    return [(int(x * sx), int(y * sy)) for (x, y) in [c['center'] for c in cores]]


def convert_ome_tiff_to_standard_tiff(input_path: str, output_path: str) -> bool:
    """
    Convert OME-TIFF file to standard TIFF format that VALIS can handle.
    
    Parameters
    ----------
    input_path : str
        Path to the input OME-TIFF file
    output_path : str
        Path for the output standard TIFF file
        
    Returns
    -------
    bool
        True if conversion successful, False otherwise
    """
    try:
        logger.info(f"Converting OME-TIFF to standard TIFF: {input_path} -> {output_path}")
        
        with tifffile.TiffFile(input_path) as tif:
            # Get the image data
            data = tif.asarray()
            logger.info(f"Original data shape: {data.shape}, dtype: {data.dtype}")
            
            # Handle different data types and shapes
            if data.ndim > 3:
                # For multi-dimensional arrays, take the first series/timepoint
                while data.ndim > 3:
                    data = data[0]
            
            # Handle channel-first vs channel-last arrangement
            if data.ndim == 3:
                # If first dimension is small (likely channels), rearrange to channel-last
                if data.shape[0] <= 20 and data.shape[0] < min(data.shape[1], data.shape[2]):
                    logger.info(f"Rearranging from channel-first {data.shape} to channel-last")
                    data = np.transpose(data, (1, 2, 0))
                    logger.info(f"New shape after transpose: {data.shape}")
            
            # Ensure data is in a format VALIS can handle
            if data.dtype == np.float64:
                # Convert float64 to float32 to save space
                data = data.astype(np.float32)
            elif data.dtype in [np.int32, np.int64]:
                # Convert large integers to uint16 if possible
                if data.max() <= 65535 and data.min() >= 0:
                    data = data.astype(np.uint16)
                else:
                    data = data.astype(np.float32)
            
            logger.info(f"Final data shape: {data.shape}, dtype: {data.dtype}")
            
            # Try different TIFF writing approaches
            write_success = False
            
            # Method 1: Standard approach with compression
            try:
                tifffile.imwrite(
                    output_path, 
                    data, 
                    compression='zlib',
                    planarconfig='contig',
                )
                write_success = True
                logger.info("Successfully wrote TIFF using standard method")
            except Exception as e1:
                logger.warning(f"Standard TIFF write failed: {e1}")
                
                # Method 2: Simple approach without compression
                try:
                    tifffile.imwrite(output_path, data)
                    write_success = True
                    logger.info("Successfully wrote TIFF using simple method")
                except Exception as e2:
                    logger.warning(f"Simple TIFF write failed: {e2}")
                    
                    # Method 3: Convert to uint8 and try again
                    try:
                        if data.dtype != np.uint8:
                            logger.info("Converting to uint8 and trying again")
                            # Normalize to 0-255 range
                            data_min, data_max = data.min(), data.max()
                            if data_max > data_min:
                                data_normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                            else:
                                data_normalized = np.zeros_like(data, dtype=np.uint8)
                            
                            tifffile.imwrite(output_path, data_normalized)
                            write_success = True
                            logger.info("Successfully wrote TIFF after uint8 conversion")
                    except Exception as e3:
                        logger.error(f"All TIFF write methods failed: {e1}, {e2}, {e3}")
            
            if write_success:
                logger.info(f"Successfully converted to standard TIFF: {output_path}")
                return True
            else:
                return False
            
    except Exception as e:
        logger.error(f"Failed to convert OME-TIFF file: {e}")
        return False


def ensure_valis_compatible_files(he_path: Path, orion_path: Path, temp_dir: Path) -> Tuple[Path, Path]:
    """
    Ensure files are in a VALIS-compatible format, converting if necessary.
    
    Parameters
    ----------
    he_path : Path
        Path to H&E file
    orion_path : Path  
        Path to Orion file
    temp_dir : Path
        Directory to store converted files
        
    Returns
    -------
    Tuple[Path, Path]
        Paths to VALIS-compatible files (may be original or converted)
    """
    def needs_conversion(file_path: Path) -> bool:
        """Check if file needs conversion based on format characteristics."""
        try:
            with tifffile.TiffFile(str(file_path)) as tif:
                # Check for characteristics that cause pyvips issues
                tags = tif.pages[0].tags
                
                # Check for tiled format with separate planes
                if hasattr(tags, 'get'):
                    planar_config = tags.get('PlanarConfiguration', None)
                    if planar_config and planar_config.value == 2:  # Separate planes
                        logger.info(f"File {file_path.name} has separate planes configuration")
                        return True
                        
                    # Check for complex tile configuration
                    tile_width = tags.get('TileWidth', None)
                    tile_length = tags.get('TileLength', None)
                    if tile_width and tile_length:
                        logger.info(f"File {file_path.name} is tiled ({tile_width.value}x{tile_length.value})")
                        return True
                
                # Check data type - some formats cause issues
                data_sample = tif.asarray(out='memmap')
                if data_sample.ndim > 3:
                    logger.info(f"File {file_path.name} has {data_sample.ndim} dimensions")
                    return True
                    
        except Exception as e:
            logger.warning(f"Could not analyze file {file_path.name}: {e}")
            # If we can't analyze it, assume it needs conversion
            return True
            
        return False
    
    he_final = he_path
    orion_final = orion_path
    
    # Check and convert H&E file if needed
    if needs_conversion(he_path):
        he_converted = temp_dir / f"converted_{he_path.name}"
        if convert_ome_tiff_to_standard_tiff(str(he_path), str(he_converted)):
            he_final = he_converted
        else:
            logger.warning(f"Failed to convert H&E file, using original: {he_path}")
    
    # Check and convert Orion file if needed  
    if needs_conversion(orion_path):
        orion_converted = temp_dir / f"converted_{orion_path.name}"
        if convert_ome_tiff_to_standard_tiff(str(orion_path), str(orion_converted)):
            orion_final = orion_converted
        else:
            logger.warning(f"Failed to convert Orion file, using original: {orion_path}")
    
    return he_final, orion_final


def extract_region(slide_path: str, center: Tuple[int, int], patch_size: int = 2048) -> np.ndarray:
    """
    Extract a square patch centred at `center` from the full resolution slide.

    Parameters
    ----------
    slide_path : str
        Path to the OME‑TIFF slide.
    center : Tuple[int, int]
        (x, y) coordinate of the patch centre in the slide.
    patch_size : int
        Side length of the patch to extract.

    Returns
    -------
    np.ndarray
        Extracted patch as a numpy array.  May be multichannel if the slide
        contains multiple channels.
    """
    half = patch_size // 2
    cx, cy = center
    with tifffile.TiffFile(slide_path) as tif:
        series = tif.series[0]
        h_f, w_f = series.shape[-2:]
        left = max(0, cx - half)
        top = max(0, cy - half)
        right = min(w_f, cx + half)
        bottom = min(h_f, cy + half)
        arr = series.asarray(region=(top, left, bottom, right))
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr


def create_overlay(he_patch: np.ndarray, or_patch: np.ndarray, max_side: int = 512) -> np.ndarray:
    """
    Create a quick‑look overlay mapping the H&E patch to red and the Orion patch
    to green.  The overlay is resized so that its longest side is
    `max_side` pixels for easy viewing.
    """
    def to_gray(p):
        if p.ndim == 3:
            if p.shape[2] >= 1:
                return cv2.normalize(p[..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    he_gray = to_gray(he_patch)
    or_gray = to_gray(or_patch)
    if he_gray.shape != or_gray.shape:
        or_gray = cv2.resize(or_gray, he_gray.shape[::-1], interpolation=cv2.INTER_AREA)
    overlay = np.zeros((*he_gray.shape, 3), dtype=np.uint8)
    overlay[..., 2] = he_gray
    overlay[..., 1] = or_gray
    h, w = overlay.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return overlay


def create_registration_overview(he_slide, or_slide, he_centres_arr, warped_or_xy, pairs, output_path: str, max_dim: int = 2048):
    """
    Create a full-slide registration overview showing detected cores and their matches.
    
    Parameters
    ----------
    he_slide : valis slide object
        H&E slide object from VALIS
    or_slide : valis slide object  
        Orion slide object from VALIS
    he_centres_arr : np.ndarray
        Array of H&E core centers (x, y)
    warped_or_xy : np.ndarray
        Array of warped Orion coordinates (x, y)
    pairs : List[Tuple[int, int]]
        List of (orion_idx, he_idx) pairs
    output_path : str
        Path to save the overview image
    max_dim : int
        Maximum dimension for the overview image
    """
    try:
        # Get thumbnails from both slides
        he_thumb = he_slide.get_thumbnail(max_dim)
        or_thumb = or_slide.get_thumbnail(max_dim)
        
        # Convert to RGB if needed
        if he_thumb.ndim == 3 and he_thumb.shape[2] > 3:
            he_thumb = he_thumb[:, :, :3]
        if or_thumb.ndim == 3 and or_thumb.shape[2] > 3:
            or_thumb = or_thumb[:, :, :3]
            
        # Normalize to uint8
        he_thumb = cv2.normalize(he_thumb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        or_thumb = cv2.normalize(or_thumb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Ensure both thumbnails are the same size
        target_size = (max_dim, max_dim)
        he_thumb_resized = cv2.resize(he_thumb, target_size, interpolation=cv2.INTER_AREA)
        or_thumb_resized = cv2.resize(or_thumb, target_size, interpolation=cv2.INTER_AREA)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle('VALIS Registration Overview', fontsize=16)
        
        # Calculate scaling factors for coordinate display
        he_full_dims = he_slide.slide_dimensions_wh
        or_full_dims = or_slide.slide_dimensions_wh
        he_scale_x = max_dim / he_full_dims[0]
        he_scale_y = max_dim / he_full_dims[1]
        or_scale_x = max_dim / or_full_dims[0] 
        or_scale_y = max_dim / or_full_dims[1]
        
        # Plot 1: H&E slide with detected cores
        axes[0,0].imshow(he_thumb_resized)
        axes[0,0].set_title(f'H&E Slide - {len(he_centres_arr)} Detected Cores')
        if len(he_centres_arr) > 0:
            he_x_scaled = he_centres_arr[:, 0] * he_scale_x
            he_y_scaled = he_centres_arr[:, 1] * he_scale_y
            axes[0,0].scatter(he_x_scaled, he_y_scaled, c='red', s=20, alpha=0.7, marker='o')
        axes[0,0].axis('off')
        
        # Plot 2: Orion slide with warped core positions
        axes[0,1].imshow(or_thumb_resized)
        axes[0,1].set_title(f'Orion Slide - {len(warped_or_xy)} Warped Positions')
        if len(warped_or_xy) > 0:
            or_x_scaled = warped_or_xy[:, 0] * or_scale_x
            or_y_scaled = warped_or_xy[:, 1] * or_scale_y
            axes[0,1].scatter(or_x_scaled, or_y_scaled, c='green', s=20, alpha=0.7, marker='s')
        axes[0,1].axis('off')
        
        # Plot 3: Registration overlay (H&E in red channel, Orion in green)
        overlay = np.zeros((*he_thumb_resized.shape[:2], 3), dtype=np.uint8)
        if he_thumb_resized.ndim == 3:
            overlay[:, :, 2] = cv2.cvtColor(he_thumb_resized, cv2.COLOR_RGB2GRAY)
        else:
            overlay[:, :, 2] = he_thumb_resized
        if or_thumb_resized.ndim == 3:
            overlay[:, :, 1] = cv2.cvtColor(or_thumb_resized, cv2.COLOR_RGB2GRAY)
        else:
            overlay[:, :, 1] = or_thumb_resized
        
        axes[1,0].imshow(overlay)
        axes[1,0].set_title('Registration Overlay (H&E=Red, Orion=Green)')
        axes[1,0].axis('off')
        
        # Plot 4: Matched pairs with connecting lines
        axes[1,1].imshow(he_thumb_resized, alpha=0.7)
        axes[1,1].set_title(f'Matched Pairs - {len(pairs)} Valid Matches')
        
        if len(pairs) > 0:
            for or_idx, he_idx in pairs:
                if or_idx < len(warped_or_xy) and he_idx < len(he_centres_arr):
                    he_x = he_centres_arr[he_idx, 0] * he_scale_x
                    he_y = he_centres_arr[he_idx, 1] * he_scale_y
                    or_x = warped_or_xy[or_idx, 0] * or_scale_x
                    or_y = warped_or_xy[or_idx, 1] * or_scale_y
                    
                    # Draw line connecting matched pairs
                    axes[1,1].plot([he_x, or_x], [he_y, or_y], 'b-', alpha=0.5, linewidth=1)
                    axes[1,1].scatter(he_x, he_y, c='red', s=30, marker='o')
                    axes[1,1].scatter(or_x, or_y, c='green', s=30, marker='s')
        
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Registration overview saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create registration overview: {e}")
        # Create a simple text-based summary instead
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.text(0.5, 0.7, f'Registration Summary', ha='center', va='center', fontsize=16, weight='bold')
            ax.text(0.5, 0.6, f'H&E Cores Detected: {len(he_centres_arr)}', ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.5, f'Warped Positions: {len(warped_or_xy)}', ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.4, f'Successful Pairs: {len(pairs)}', ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.2, f'Error creating detailed overview: {str(e)[:100]}...', ha='center', va='center', fontsize=10, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e2:
            logger.error(f"Failed to create fallback overview: {e2}")


def pair_points(or_points: np.ndarray, he_points: np.ndarray, max_dist_factor: float = 3.0) -> List[Tuple[int, int]]:
    """
    Pair Orion points to H&E points using the Hungarian algorithm and reject
    pairs that are too far apart relative to the median inter‑core spacing.
    """
    if or_points.size == 0 or he_points.size == 0:
        logger.warning(f"Cannot pair points: or_points.size={or_points.size}, he_points.size={he_points.size}")
        return []
    
    # Remove any invalid coordinates (NaN, inf)
    valid_or_mask = np.isfinite(or_points).all(axis=1)
    valid_he_mask = np.isfinite(he_points).all(axis=1)
    
    valid_or_points = or_points[valid_or_mask]
    valid_he_points = he_points[valid_he_mask]
    
    logger.info(f"Valid Orion points: {len(valid_or_points)}/{len(or_points)}")
    logger.info(f"Valid H&E points: {len(valid_he_points)}/{len(he_points)}")
    
    if len(valid_or_points) == 0 or len(valid_he_points) == 0:
        logger.warning("No valid points remaining after filtering")
        return []
    
    # Calculate distance matrix
    diff = valid_or_points[:, None, :] - valid_he_points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    logger.info(f"Distance matrix shape: {dists.shape}")
    logger.info(f"Distance range: {dists.min():.1f} - {dists.max():.1f}")
    
    # Calculate typical spacing between H&E cores
    if len(valid_he_points) >= 2:
        kd = cKDTree(valid_he_points)
        nn_dists, _ = kd.query(valid_he_points, k=min(3, len(valid_he_points)))
        if nn_dists.shape[1] > 1:
            spacing = np.median(nn_dists[:, 1])
        else:
            spacing = np.median(dists)
    else:
        spacing = np.median(dists)
    
    logger.info(f"Estimated core spacing: {spacing:.1f}")
    max_allowed_dist = max_dist_factor * spacing
    logger.info(f"Maximum allowed pairing distance: {max_allowed_dist:.1f}")
    
    # Use Hungarian algorithm for optimal assignment
    try:
        row_ind, col_ind = linear_sum_assignment(dists)
        logger.info(f"Hungarian algorithm found {len(row_ind)} potential pairs")
    except Exception as e:
        logger.error(f"Hungarian algorithm failed: {e}")
        return []
    
    # Filter pairs by distance threshold
    pairs = []
    valid_or_indices = np.where(valid_or_mask)[0]
    valid_he_indices = np.where(valid_he_mask)[0]
    
    for r, c in zip(row_ind, col_ind):
        dist = dists[r, c]
        if dist <= max_allowed_dist:
            orig_or_idx = valid_or_indices[r]
            orig_he_idx = valid_he_indices[c]
            pairs.append((orig_or_idx, orig_he_idx))
            logger.debug(f"Pair accepted: or_idx={orig_or_idx}, he_idx={orig_he_idx}, dist={dist:.1f}")
        else:
            logger.debug(f"Pair rejected: dist={dist:.1f} > max_allowed={max_allowed_dist:.1f}")
    
    logger.info(f"Final pairs after distance filtering: {len(pairs)}/{len(row_ind)}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Pair TMA cores in H&E and Orion slides using VALIS")
    parser.add_argument("--he", required=True, help="Path to H&E OME‑TIFF slide")
    parser.add_argument("--orion", required=True, help="Path to Orion OME‑TIFF slide")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--patch_size", type=int, default=2048, help="Patch size for extraction")
    parser.add_argument("--thumb_size", type=int, default=4096, help="Thumbnail size for core detection")
    parser.add_argument("--min_core_area", type=int, default=100, help="Minimum contour area on thumbnail")
    parser.add_argument("--circularity_thresh", type=float, default=0.05, help="Minimum contour circularity")
    parser.add_argument("--max_dist_factor", type=float, default=3.0, help="Maximum pairing distance factor")
    parser.add_argument("--mem_gb", type=int, default=16, help="Memory to allocate to BioFormats JVM (GB)")
    parser.add_argument("--max_image_dim", type=int, default=1024, help="Maximum slide dimension to save (valis max_image_dim_px)")
    parser.add_argument("--max_processed_dim", type=int, default=512, help="Maximum dimension for feature detection (valis max_processed_image_dim_px)")
    parser.add_argument("--max_non_rigid_dim", type=int, default=2048, help="Maximum dimension for non‑rigid registration (valis max_non_rigid_registration_dim_px)")
    parser.add_argument("--non_rigid", action="store_true", help="Perform non‑rigid registration (default rigid only)")
    parser.add_argument("--verbose", action="store_true", help="Increase logging output")
    parser.add_argument("--keep_converted", action="store_true", help="Keep converted TIFF files for debugging (default: delete after processing)")
    parser.add_argument("--create_overview", action="store_true", default=True, help="Create registration overview visualization (default: True)")
    args = parser.parse_args()
    # Configure logging - use INFO level to get detailed output for debugging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    out = Path(args.out_dir)
    patch_dir = out / "patches"
    overlay_dir = out / "qc_overlays"
    registration_qc_dir = out / "registration_qc"
    patch_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    registration_qc_dir.mkdir(exist_ok=True)
    # Verify input files exist
    he_path = Path(args.he)
    orion_path = Path(args.orion)
    if not he_path.exists():
        raise FileNotFoundError(f"H&E file not found: {he_path}")
    if not orion_path.exists():
        raise FileNotFoundError(f"Orion file not found: {orion_path}")
    
    # Create output directories
    valis_output_dir = out / "valis_output"
    valis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for file conversion if needed
    conversion_temp_dir = out / "temp_converted_files"
    conversion_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure files are VALIS-compatible, converting if necessary
    logger.info("Checking if files need conversion for VALIS compatibility...")
    he_final_path, orion_final_path = ensure_valis_compatible_files(he_path, orion_path, conversion_temp_dir)
    
    # Check if both files are in the same directory - if so, use direct approach
    if he_final_path.parent == orion_final_path.parent:
        logger.info("Files are in the same directory, using direct approach")
        src_dir = he_final_path.parent
        he_name = he_final_path.name
        orion_name = orion_final_path.name
        
        # Start JVM for BioFormats
        slide_io.init_jvm(mem_gb=args.mem_gb)
        try:
            # Test file accessibility before VALIS registration
            logger.info("Testing file accessibility...")
            try:
                # Try to open files with tifffile to verify they're readable
                with tifffile.TiffFile(str(he_final_path)) as tif:
                    logger.info(f"H&E file readable: shape={tif.series[0].shape}")
                with tifffile.TiffFile(str(orion_final_path)) as tif:
                    logger.info(f"Orion file readable: shape={tif.series[0].shape}")
            except Exception as e:
                logger.error(f"File accessibility test failed: {e}")
                raise
            
            # Change to source directory to ensure VALIS can find files
            original_cwd = os.getcwd()
            logger.info(f"Changing working directory from {original_cwd} to {src_dir}")
            os.chdir(str(src_dir))
            
            try:
                # Verify files exist in current directory
                if not Path(he_name).exists():
                    raise FileNotFoundError(f"H&E file not found in working directory: {he_name}")
                if not Path(orion_name).exists():
                    raise FileNotFoundError(f"Orion file not found in working directory: {orion_name}")
                
                # Initialize VALIS registrar with current directory as source
                logger.info(f"Initializing VALIS with src_dir='.', dst_dir={valis_output_dir}")
                logger.info(f"Image list: {[he_name, orion_name]}")
                logger.info(f"Current working directory: {os.getcwd()}")
                
                reg = registration.Valis(
                    src_dir=".",
                    dst_dir=str(valis_output_dir),
                    img_list=[he_name, orion_name],
                    imgs_ordered=True,
                    reference_img_f=he_name,
                    max_image_dim_px=args.max_image_dim,
                    max_processed_image_dim_px=args.max_processed_dim,
                    max_non_rigid_registration_dim_px=args.max_non_rigid_dim,
                    non_rigid_registrar_cls=None if not args.non_rigid else registration.Valis.non_rigid_registrar_cls,
                )
                # Register slides (rigid then optional non‑rigid)
                logger.info("Starting VALIS registration...")
                try:
                    reg.register()
                    logger.info("VALIS registration completed successfully")
                except Exception as e:
                    error_msg = str(e)
                    if "tiled separate planes not supported" in error_msg or "unable to write to memory" in error_msg:
                        logger.error(f"VALIS registration failed due to OME-TIFF format incompatibility: {e}")
                        logger.error("The OME-TIFF files use a format that pyvips cannot handle.")
                        logger.error("This is a known limitation with certain OME-TIFF file formats.")
                        logger.error("")
                        logger.error("SUGGESTED SOLUTIONS:")
                        logger.error("1. Convert OME-TIFF files to standard TIFF format using ImageJ/FIJI:")
                        logger.error("   - Open files in ImageJ/FIJI")
                        logger.error("   - File > Save As > TIFF")
                        logger.error("2. Use bioformats2raw + raw2ometiff to convert to a compatible format")
                        logger.error("3. Try reducing max_image_dim and max_processed_dim parameters")
                        logger.error("4. Consider using a different image registration tool")
                    else:
                        logger.error(f"VALIS registration failed: {e}")
                        logger.error("This may be due to large file size or memory constraints")
                    raise
                
                # Retrieve slide objects
                he_slide = reg.get_slide(he_name)
                or_slide = reg.get_slide(orion_name)
                
            finally:
                # Always restore original working directory
                logger.info(f"Restoring working directory to {original_cwd}")
                os.chdir(original_cwd)
            
        finally:
            slide_io.kill_jvm()
    else:
        # Files are in different directories, use temporary directory approach
        logger.info("Files are in different directories, using temporary directory approach")
        # Create a temporary directory and copy files there with symlinks to avoid copying large files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            he_name = he_final_path.name
            orion_name = orion_final_path.name
            tmp_he_path = tmp_dir / he_name
            tmp_orion_path = tmp_dir / orion_name
            
            # Create symbolic links instead of copying large files
            try:
                tmp_he_path.symlink_to(he_final_path.absolute())
                tmp_orion_path.symlink_to(orion_final_path.absolute())
            except OSError:
                # If symlinks fail (e.g., on Windows), copy the files
                logger.warning("Symbolic links failed, copying files (this may take time for large files)")
                shutil.copy2(he_final_path, tmp_he_path)
                shutil.copy2(orion_final_path, tmp_orion_path)
            
            # Verify the temporary files exist
            if not tmp_he_path.exists():
                raise FileNotFoundError(f"Failed to create temporary H&E file: {tmp_he_path}")
            if not tmp_orion_path.exists():
                raise FileNotFoundError(f"Failed to create temporary Orion file: {tmp_orion_path}")
            
            logger.info(f"Working with files in temporary directory: {tmp_dir}")
            logger.info(f"H&E: {tmp_he_path}")
            logger.info(f"Orion: {tmp_orion_path}")
            
            # Start JVM for BioFormats
            slide_io.init_jvm(mem_gb=args.mem_gb)
            try:
                # Test file accessibility before VALIS registration
                logger.info("Testing file accessibility...")
                try:
                    # Try to open files with tifffile to verify they're readable
                    with tifffile.TiffFile(str(tmp_he_path)) as tif:
                        logger.info(f"H&E file readable: shape={tif.series[0].shape}")
                    with tifffile.TiffFile(str(tmp_orion_path)) as tif:
                        logger.info(f"Orion file readable: shape={tif.series[0].shape}")
                except Exception as e:
                    logger.error(f"File accessibility test failed: {e}")
                    raise
                
                # Change to temporary directory to ensure VALIS can find files
                original_cwd = os.getcwd()
                logger.info(f"Changing working directory from {original_cwd} to {tmp_dir}")
                os.chdir(str(tmp_dir))
                
                try:
                    # Verify files exist in current directory
                    if not Path(he_name).exists():
                        raise FileNotFoundError(f"H&E file not found in working directory: {he_name}")
                    if not Path(orion_name).exists():
                        raise FileNotFoundError(f"Orion file not found in working directory: {orion_name}")
                    
                    # Initialize VALIS registrar with current directory as source
                    logger.info(f"Initializing VALIS with src_dir='.', dst_dir={valis_output_dir}")
                    logger.info(f"Image list: {[he_name, orion_name]}")
                    logger.info(f"Current working directory: {os.getcwd()}")
                    
                    # Force BioFormats reader to handle OME-TIFF files that pyvips can't read
                    reg = registration.Valis(
                        src_dir=".",
                        dst_dir=str(valis_output_dir),
                        img_list=[he_name, orion_name],
                        imgs_ordered=True,
                        reference_img_f=he_name,
                        max_image_dim_px=args.max_image_dim,
                        max_processed_image_dim_px=args.max_processed_dim,
                        max_non_rigid_registration_dim_px=args.max_non_rigid_dim,
                        non_rigid_registrar_cls=None if not args.non_rigid else registration.Valis.non_rigid_registrar_cls,
                    )
                    
                    # Override the default reader to use BioFormats for OME-TIFF files
                    logger.info("Configuring VALIS to use BioFormats reader for OME-TIFF files")
                    reg.reader_dict = {he_name: slide_io.BioFormatsSlideReader, 
                                     orion_name: slide_io.BioFormatsSlideReader}
                    # Register slides (rigid then optional non‑rigid)
                    logger.info("Starting VALIS registration...")
                    try:
                        reg.register()
                        logger.info("VALIS registration completed successfully")
                    except Exception as e:
                        error_msg = str(e)
                        if "tiled separate planes not supported" in error_msg or "unable to write to memory" in error_msg:
                            logger.error(f"VALIS registration failed due to OME-TIFF format incompatibility: {e}")
                            logger.error("The OME-TIFF files use a format that pyvips cannot handle.")
                            logger.error("This is a known limitation with certain OME-TIFF file formats.")
                            logger.error("")
                            logger.error("SUGGESTED SOLUTIONS:")
                            logger.error("1. Convert OME-TIFF files to standard TIFF format using ImageJ/FIJI:")
                            logger.error("   - Open files in ImageJ/FIJI")
                            logger.error("   - File > Save As > TIFF")
                            logger.error("2. Use bioformats2raw + raw2ometiff to convert to a compatible format")
                            logger.error("3. Try reducing max_image_dim and max_processed_dim parameters")
                            logger.error("4. Consider using a different image registration tool")
                        else:
                            logger.error(f"VALIS registration failed: {e}")
                            logger.error("This may be due to large file size or memory constraints")
                        raise
                    
                    # Retrieve slide objects
                    he_slide = reg.get_slide(he_name)
                    or_slide = reg.get_slide(orion_name)
                    
                finally:
                    # Always restore original working directory
                    logger.info(f"Restoring working directory to {original_cwd}")
                    os.chdir(original_cwd)
                
            finally:
                slide_io.kill_jvm()
    
    # Common processing code continues here (moved outside the conditional blocks)
    # Detect cores on H&E slide
    logger.info("Detecting cores on H&E thumbnail...")
    he_thumb, he_full_shape = load_thumbnail(args.he, thumb_size=args.thumb_size, use_channel0=False)
    he_cores = detect_cores(he_thumb, brightfield=True, min_area=args.min_core_area, circularity_thresh=args.circularity_thresh)
    he_centres = scale_centres(he_cores, he_thumb.shape, he_full_shape)
    logger.info(f"Detected {len(he_centres)} H&E cores")
    he_centres_arr = np.array(he_centres, dtype=float)
    # Warp H&E core coordinates to Orion slide coordinate system
    if len(he_centres_arr) > 0:
        logger.info(f"Warping {len(he_centres_arr)} H&E coordinates to Orion coordinate system...")
        try:
            # Points are (x,y) but VALIS expects (x,y) array of floats
            warped_or_xy = he_slide.warp_xy_from_to(
                xy=he_centres_arr,
                to_slide_obj=or_slide,
                src_pt_level=0,
                dst_slide_level=0,
                non_rigid=args.non_rigid
            )
            logger.info(f"Successfully warped coordinates. Shape: {warped_or_xy.shape}")
            
            # Check for invalid warped coordinates
            valid_mask = np.isfinite(warped_or_xy).all(axis=1)
            valid_count = np.sum(valid_mask)
            logger.info(f"Valid warped coordinates: {valid_count}/{len(warped_or_xy)}")
            
            if valid_count < len(warped_or_xy):
                logger.warning(f"Some coordinates failed to warp properly: {len(warped_or_xy) - valid_count} invalid")
                
        except Exception as e:
            logger.error(f"Failed to warp coordinates: {e}")
            warped_or_xy = np.empty((0, 2))
    else:
        warped_or_xy = np.empty((0, 2))
        logger.warning("No H&E centres detected, cannot perform warping")
    # Create registration overview visualization
    if args.create_overview:
        overview_path = registration_qc_dir / "registration_overview.png"
        create_registration_overview(he_slide, or_slide, he_centres_arr, warped_or_xy, [], str(overview_path))
    
    # Compute pairing distances and filter out improbable matches
    logger.info("Starting core pairing process...")
    pairs = pair_points(warped_or_xy, he_centres_arr, max_dist_factor=args.max_dist_factor)
    logger.info(f"Created {len(pairs)} pairs using Hungarian assignment")
    
    # Update overview with pairs if we have any
    if args.create_overview and len(pairs) > 0:
        overview_with_pairs_path = registration_qc_dir / "registration_overview_with_pairs.png"
        create_registration_overview(he_slide, or_slide, he_centres_arr, warped_or_xy, pairs, str(overview_with_pairs_path))
    # Extract and save patches
    csv_rows = []
    for idx, (or_idx, he_idx) in enumerate(pairs, 1):
        he_center = he_centres_arr[he_idx]
        or_center = warped_or_xy[or_idx]
        # Extract full resolution patches
        he_patch = extract_region(args.he, (int(he_center[0]), int(he_center[1])), patch_size=args.patch_size)
        or_patch = extract_region(args.orion, (int(or_center[0]), int(or_center[1])), patch_size=args.patch_size)
        if he_patch is None or or_patch is None:
            logger.warning(f"Skipping pair {idx} due to extraction failure")
            continue
        he_patch_name = f"core_{idx:03d}_he.tiff"
        or_patch_name = f"core_{idx:03d}_orion.tiff"
        he_patch_path = patch_dir / he_patch_name
        or_patch_path = patch_dir / or_patch_name
        tifffile.imwrite(str(he_patch_path), he_patch, compression="zlib")
        tifffile.imwrite(str(or_patch_path), or_patch, compression="zlib")
        overlay = create_overlay(he_patch, or_patch, max_side=512)
        overlay_path = overlay_dir / f"core_{idx:03d}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        dist = float(np.linalg.norm(or_center - he_center))
        csv_rows.append([idx, str(he_patch_path), str(or_patch_path), str(overlay_path), dist])
        logger.info(f"Processed pair {idx}/{len(pairs)}")
    # Save summary CSV
    csv_path = out / "paired_core_info.csv"
    import csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pair_index", "he_patch_path", "orion_patch_path", "overlay_path", "pair_distance"])
        writer.writerows(csv_rows)
    logger.info(f"Saved summary CSV to {csv_path}")
    
    # Log final statistics
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"H&E cores detected: {len(he_centres_arr)}")
    logger.info(f"Coordinates successfully warped: {len(warped_or_xy)}")
    logger.info(f"Valid coordinate pairs found: {len(pairs)}")
    logger.info(f"Patches extracted: {len(csv_rows)}")
    logger.info(f"Success rate: {len(pairs)/max(len(he_centres_arr), 1)*100:.1f}%")
    if len(pairs) == 0:
        logger.warning("No valid pairs were created. Consider:")
        logger.warning("1. Adjusting --max_dist_factor (current: {:.1f})".format(args.max_dist_factor))
        logger.warning("2. Checking registration quality in the overview images")
        logger.warning("3. Verifying core detection parameters")
        logger.warning("4. Using --non_rigid flag for better registration")
    logger.info("=" * 60)
    
    # Clean up temporary converted files
    if conversion_temp_dir.exists() and not args.keep_converted:
        try:
            shutil.rmtree(conversion_temp_dir)
            logger.info("Cleaned up temporary converted files")
        except Exception as e:
            logger.warning(f"Could not clean up temporary files: {e}")
    elif conversion_temp_dir.exists() and args.keep_converted:
        logger.info(f"Converted files kept for debugging in: {conversion_temp_dir}")
    
    logger.info(f"Registration QC images saved in: {registration_qc_dir}")
    logger.info(f"Individual core overlays saved in: {overlay_dir}")
    logger.info(f"Extracted patches saved in: {patch_dir}")


if __name__ == "__main__":
    main()