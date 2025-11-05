import os
import csv
import math
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional

# Try importing tifffile; fall back to PIL if unavailable
try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    from PIL import Image
    TIFF_AVAILABLE = False

# Try importing pycpd for coherent point drift; otherwise we'll use a simple Procrustes alignment
try:
    from pycpd import AffineRegistration
    CPD_AVAILABLE = True
except Exception:
    CPD_AVAILABLE = False

from scipy.optimize import linear_sum_assignment

def load_thumbnail(slide_path: str, thumb_size: int = 4096, use_channel0: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load a low‑resolution thumbnail from an OME‑TIFF slide.
    If `tifffile` is available, the lowest pyramid level is returned.
    Otherwise, the image is opened via Pillow and resized down.

    Parameters
    ----------
    slide_path : str
        Path to the OME‑TIFF.
    thumb_size : int
        Target size of the largest dimension of the thumbnail.
    use_channel0 : bool
        For fluorescence images that may have multiple channels, force use of the first channel.

    Returns
    -------
    thumb : np.ndarray
        Grayscale thumbnail image.
    full_shape : Tuple[int, int]
        (height, width) of the full resolution image.
    """
    if TIFF_AVAILABLE:
        # Read via tifffile; this handles OME pyramids robustly
        with tifffile.TiffFile(slide_path) as tif:
            series = tif.series[0]
            full_shape = series.shape[-2:]  # (height, width)
            # Use the lowest resolution level
            level = series.levels[-1]
            arr = level.asarray()
            # arr might be (samples, h, w) or (h, w, samples)
            if arr.ndim == 3:
                # (samples, y, x) ordering (OME)
                if arr.shape[0] < arr.shape[-1]:
                    arr = arr[0] if use_channel0 else arr[0]
                else:
                    arr = arr[..., 0] if use_channel0 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            thumb = arr.astype(np.uint8)
        # Resize up if the smallest dimension is too small
        h, w = thumb.shape
        if max(h, w) < thumb_size:
            scale = thumb_size / max(h, w)
            thumb = cv2.resize(thumb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        # Fallback: open via PIL and downsample
        img = Image.open(slide_path)
        img_arr = np.array(img)
        # Convert to grayscale
        if img_arr.ndim == 3:
            img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_arr
        full_shape = img_gray.shape
        h, w = img_gray.shape
        scale = thumb_size / max(h, w) if max(h, w) > thumb_size else 1.0
        thumb = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return thumb, full_shape

def detect_cores(
    image: np.ndarray,
    brightfield: bool = True,
    min_area: int = 100,
    circularity_thresh: float = 0.1
) -> List[Dict[str, object]]:
    """
    Detect approximate circular cores in a grayscale thumbnail.

    Uses adaptive thresholding followed by morphological opening and closing to generate a binary mask.
    Contours are extracted and filtered by area and circularity.

    Parameters
    ----------
    image : np.ndarray
        Grayscale thumbnail.
    brightfield : bool
        If True, uses inverse thresholding appropriate for brightfield images.
        If False, uses thresholding for fluorescence images.
    min_area : int
        Minimum contour area (in pixel^2 of the thumbnail) to consider a core.
    circularity_thresh : float
        Minimum circularity metric (4π * area / perimeter^2) to accept a contour.

    Returns
    -------
    cores : List[Dict]
        List of detected cores, each with keys 'center' (tuple of ints) and 'area'.
    """
    # Normalize intensity and blur
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply CLAHE for fluorescent images to enhance contrast
    if not brightfield:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        blur = clahe.apply(blur)
    # Adaptive thresholding parameters
    bs = max(31, int(min(blur.shape) // 50) * 2 + 1)
    thresh_type = cv2.THRESH_BINARY_INV if brightfield else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, bs, 2
    )
    # Morphological clean up
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
        circularity = 4 * math.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < circularity_thresh:
            continue
        cores.append({'center': (cx, cy), 'area': area})
    return cores

def scale_centres(cores: List[Dict[str, object]], thumb_shape: Tuple[int, int], full_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Scale thumbnail coordinates back to full resolution."""
    h_t, w_t = thumb_shape
    h_f, w_f = full_shape
    sx = w_f / w_t
    sy = h_f / h_t
    return [(int(x * sx), int(y * sy)) for (x, y) in [c['center'] for c in cores]]

def procrustes_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Align the moving point set A to the fixed point set B using a similarity transform (scale + rotation + translation).
    When A and B have different numbers of points, we use a subset-based approach.
    Returns the transformed version of A.
    """
    A = A.astype(float)
    B = B.astype(float)
    
    # Handle different sized point sets by using a subset approach
    if len(A) != len(B):
        # Use the smaller set size for initial alignment
        min_size = min(len(A), len(B))
        if min_size < 3:
            # Not enough points for reliable alignment, use simple translation
            centroid_A = A.mean(axis=0)
            centroid_B = B.mean(axis=0)
            return A + (centroid_B - centroid_A)
        
        # For different sized sets, we'll compute the transformation using centroids and scale
        # This is a simplified approach that works when point sets have different sizes
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)
        
        # Estimate scale from the spread of points
        spread_A = np.std(A, axis=0)
        spread_B = np.std(B, axis=0)
        scale = np.mean(spread_B / (spread_A + 1e-6))
        
        # Apply simple similarity transform: scale + translation
        A_centered = A - centroid_A
        A_aligned = A_centered * scale + centroid_B
        return A_aligned
    
    # Original Procrustes for same-sized point sets
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    A_c = A - centroid_A
    B_c = B - centroid_B
    # Compute covariance matrix and SVD
    H = A_c.T @ B_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure a proper rotation (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_A = np.sum(A_c ** 2)
    # Scale
    scale = np.sum(S) / var_A if var_A > 0 else 1.0
    # Apply transformation
    A_aligned = (A_c @ R) * scale + centroid_B
    return A_aligned

def align_points(or_points: np.ndarray, he_points: np.ndarray) -> np.ndarray:
    """
    Align Orion core centres to H&E using either CPD (if available) or Procrustes alignment.
    Returns the aligned Orion points in H&E space.
    """
    if CPD_AVAILABLE and len(or_points) > 0 and len(he_points) > 0:
        reg = AffineRegistration(X=he_points, Y=or_points)
        TY, (s, R, t) = reg.register()
        return TY
    else:
        return procrustes_align(or_points, he_points)

def pair_points(
    or_points: np.ndarray,
    he_points: np.ndarray,
    max_dist_factor: float = 3.0
) -> List[Tuple[int, int]]:
    """
    Pair points between two sets using the Hungarian algorithm.

    A distance matrix of size (n_orion, n_he) is computed.
    Pairs with distances greater than `max_dist_factor * median_spacing` are rejected.

    Returns a list of tuples (i, j) indexing the matched Orion and H&E points.
    """
    if or_points.size == 0 or he_points.size == 0:
        return []
    # Compute distance matrix
    diff = or_points[:, None, :] - he_points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Estimate typical spacing from he_points (median nearest neighbour distance)
    from scipy.spatial import cKDTree
    kd = cKDTree(he_points)
    nn_dists, _ = kd.query(he_points, k=2)
    spacing = np.median(nn_dists[:, 1]) if len(he_points) >= 2 else np.median(dists)
    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(dists)
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if dists[r, c] <= max_dist_factor * spacing:
            pairs.append((r, c))
    return pairs

def extract_region(
    slide_path: str,
    center: Tuple[int, int],
    patch_size: int = 2048
) -> Optional[np.ndarray]:
    """
    Extract a square patch of size `patch_size`×`patch_size` centred at `center` from the full resolution slide.
    Assumes `tifffile` is available.  Returns None on failure.
    """
    if not TIFF_AVAILABLE:
        raise RuntimeError('tifffile is required for full‑resolution extraction')
    half = patch_size // 2
    cx, cy = center
    try:
        with tifffile.TiffFile(slide_path) as tif:
            series = tif.series[0]
            full_shape = series.shape[-2:]
            h_f, w_f = full_shape
            left = max(0, cx - half)
            top = max(0, cy - half)
            right = min(w_f, cx + half)
            bottom = min(h_f, cy + half)
            arr = series.asarray(region=(top, left, bottom, right))
            if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
            return arr
    except Exception as e:
        print(f'Failed to extract region: {e}')
        return None

def create_overlay(he_patch: np.ndarray, or_patch: np.ndarray, max_side: int = 512) -> np.ndarray:
    """
    Create a quick‑look overlay by mapping the H&E patch to the red channel and the Orion patch to the green channel.
    Optionally downsample to `max_side` for storage efficiency.
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

def process_slides(
    he_path: str,
    orion_path: str,
    out_dir: str = 'paired_dataset',
    thumb_size: int = 4096,
    patch_size: int = 2048,
    min_core_area: int = 100,
    circularity_thresh: float = 0.1,
    max_dist_factor: float = 3.0
) -> None:
    """
    Run the full pipeline on the provided H&E and Orion slides.

    Parameters
    ----------
    he_path, orion_path : str
        Paths to the OME‑TIFF images.
    out_dir : str
        Root output directory where patches, overlays and CSV will be written.
    thumb_size : int
        Target size of the thumbnail used for detection.
    patch_size : int
        Side length (in pixels) of the full‑resolution extracted patches.
    min_core_area : int
        Minimum area of detected contours on the thumbnail.
    circularity_thresh : float
        Minimum circularity metric for contour acceptance.
    max_dist_factor : float
        Maximum allowed distance (relative to median core spacing) for pairing.
    """
    os.makedirs(out_dir, exist_ok=True)
    patch_dir = os.path.join(out_dir, 'patches')
    overlay_dir = os.path.join(out_dir, 'qc_overlays')
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    print('Loading H&E thumbnail...')
    he_thumb, he_full_shape = load_thumbnail(he_path, thumb_size=thumb_size, use_channel0=False)
    print('Detecting H&E cores...')
    he_cores = detect_cores(he_thumb, brightfield=True, min_area=min_core_area, circularity_thresh=circularity_thresh)
    he_centres = scale_centres(he_cores, he_thumb.shape, he_full_shape)
    print(f'Detected {len(he_centres)} H&E cores')
    print('Loading Orion thumbnail...')
    or_thumb, or_full_shape = load_thumbnail(orion_path, thumb_size=thumb_size, use_channel0=True)
    print('Detecting Orion cores...')
    or_cores = detect_cores(or_thumb, brightfield=False, min_area=min_core_area, circularity_thresh=circularity_thresh)
    or_centres = scale_centres(or_cores, or_thumb.shape, or_full_shape)
    print(f'Detected {len(or_centres)} Orion cores')
    he_centres_arr = np.array(he_centres)
    or_centres_arr = np.array(or_centres)
    if len(he_centres_arr) > 0 and len(or_centres_arr) > 0:
        print('Estimating global transformation...')
        or_aligned = align_points(or_centres_arr, he_centres_arr)
    else:
        or_aligned = or_centres_arr
    print('Pairing cores...')
    pairs = pair_points(or_aligned, he_centres_arr, max_dist_factor=max_dist_factor)
    print(f'Found {len(pairs)} pairs')
    csv_rows = []
    for idx, (or_idx, he_idx) in enumerate(pairs, 1):
        he_center = he_centres_arr[he_idx]
        or_center = or_centres_arr[or_idx]
        he_patch = extract_region(he_path, tuple(he_center), patch_size=patch_size)
        or_patch = extract_region(orion_path, tuple(or_center), patch_size=patch_size)
        if he_patch is None or or_patch is None:
            print(f'Skipping pair {idx} due to extraction failure')
            continue
        he_filename = f'core_{idx:03d}_he.tiff'
        or_filename = f'core_{idx:03d}_orion.tiff'
        he_patch_path = os.path.join(patch_dir, he_filename)
        or_patch_path = os.path.join(patch_dir, or_filename)
        tifffile.imwrite(he_patch_path, he_patch, compression='zlib')
        tifffile.imwrite(or_patch_path, or_patch, compression='zlib')
        overlay = create_overlay(he_patch, or_patch, max_side=512)
        overlay_name = f'core_{idx:03d}_overlay.png'
        overlay_path = os.path.join(overlay_dir, overlay_name)
        cv2.imwrite(overlay_path, overlay)
        csv_rows.append([idx, he_patch_path, or_patch_path, overlay_path, float(np.linalg.norm(or_aligned[or_idx] - he_centres_arr[he_idx]))])
        print(f'Processed pair {idx}/{len(pairs)}')
    csv_path = os.path.join(out_dir, 'paired_core_info.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pair_index', 'he_patch_path', 'orion_patch_path', 'overlay_path', 'pair_distance'])
        writer.writerows(csv_rows)
    print(f'Wrote summary CSV to {csv_path}')

# Example usage
if __name__ == "__main__":
    he_slide = 'data/raw/TA118-HEraw.ome.tiff'
    orion_slide = 'data/raw/TA118-Orionraw.ome.tiff'
    process_slides(he_slide, orion_slide, out_dir='paired_dataset_aug18')
