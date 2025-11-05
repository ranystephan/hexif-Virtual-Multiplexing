#!/usr/bin/env python3
"""
tma_pairing_pipeline.py
-----------------------
Detect and pair TMA cores from H&E and Orion whole‑slide images.

This version reads thumbnails exclusively via tifffile, which supports OME‑TIFF.

python tma_pairing_pipeline_jul29.py \
    --he data/raw/TA118-HEraw.ome.tiff \
    --orion data/raw/TA118-Orionraw.ome.tiff \
    --out_dir paired_cores_jul29 \
    --register

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
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tifffile import TiffFile, imwrite, imread

try:
    from valis import registration, slide_io
    VALIS_AVAILABLE = True
except ImportError:
    VALIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility functions (unchanged)
# ---------------------------------------------------------------------------

def estimate_core_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    distances, _ = nbrs.kneighbors(points)
    return float(np.median(distances[:, 1]))

def align_orion_to_he(or_centres: np.ndarray, he_centres: np.ndarray) -> np.ndarray:
    or_arr = or_centres.astype(float)
    he_arr = he_centres.astype(float)
    or_mean = or_arr.mean(axis=0)
    he_mean = he_arr.mean(axis=0)
    or_centered = or_arr - or_mean
    he_centered = he_arr - he_mean
    _, _, vt_or = np.linalg.svd(or_centered, full_matrices=False)
    _, _, vt_he = np.linalg.svd(he_centered, full_matrices=False)
    angle_or = np.arctan2(vt_or[0, 1], vt_or[0, 0])
    angle_he = np.arctan2(vt_he[0, 1], vt_he[0, 0])
    angle_diff = angle_or - angle_he
    cos_a = np.cos(-angle_diff)
    sin_a = np.sin(-angle_diff)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    or_proj = or_centered @ vt_or[0]
    he_proj = he_centered @ vt_he[0]
    std_or = or_proj.std() if or_proj.size > 1 else 1.0
    std_he = he_proj.std() if he_proj.size > 1 else 1.0
    scale = std_or / std_he if std_he > 0 else 1.0
    aligned = ((or_centered @ R) / scale) + he_mean
    return aligned

def cluster_grid_order(points: np.ndarray, spacing: float) -> Tuple[List[Tuple[int, int]], List[int]]:
    if len(points) == 0:
        return [], []
    pts = points.copy()
    y_coords = pts[:, 1]
    est_rows = max(1, int(round((y_coords.max() - y_coords.min()) / spacing)) + 1)
    kmeans = KMeans(n_clusters=est_rows, random_state=0)
    row_labels = kmeans.fit_predict(y_coords.reshape(-1, 1))
    label_order = sorted(np.unique(row_labels), key=lambda lbl: y_coords[row_labels == lbl].mean())
    ordered_pts = []
    ordered_indices = []
    for lbl in label_order:
        indices = np.where(row_labels == lbl)[0]
        row_pts = pts[indices]
        sorted_idx = indices[np.argsort(row_pts[:, 0])]
        for i in sorted_idx:
            ordered_pts.append(tuple(map(int, pts[i])))
            ordered_indices.append(int(i))
    return ordered_pts, ordered_indices

# ---------------------------------------------------------------------------
# Detector updated to read thumbnails via tifffile
# ---------------------------------------------------------------------------

class ThumbnailCoreDetector:
    def __init__(self, thumb_size: int = 4096, min_core_area: int = 100):  # CHANGED: lowered default
        self.thumb_size = thumb_size
        self.min_core_area = min_core_area

    def load_thumbnail(self, slide_path: str, use_channel0: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Load the lowest-resolution level of an OME‑TIFF as a grayscale thumbnail.
        If use_channel0 is True, always take the first channel; otherwise, if the
        image has three channels, convert it to grayscale.
        Returns (thumbnail, full_image_shape).
        """
        with TiffFile(slide_path) as tif:
            series = tif.series[0]
            # Get full-resolution shape (height, width)
            full_shape = series.shape[-2:]
            full_shape = (int(full_shape[0]), int(full_shape[1]))
            # Use the lowest-resolution level available
            level = series.levels[-1]
            arr = level.asarray()
            # arr may be (samples, h, w) or (h, w)
            if arr.ndim == 3:
                # If the first axis is smaller than the last (typical OME),
                # treat it as (samples, y, x)
                if arr.shape[0] < arr.shape[-1]:
                    arr = arr[0] if use_channel0 else arr[0]
                else:
                    # (y, x, samples)
                    arr = arr[..., 0] if use_channel0 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            thumb = arr.astype(np.uint8)
        # upscale to thumb_size if needed
        h, w = thumb.shape
        if max(h, w) < self.thumb_size:
            scale = self.thumb_size / max(h, w)
            thumb = cv2.resize(thumb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Loaded thumbnail from {slide_path}: {thumb.shape}, full size: {full_shape}")
        return thumb, full_shape

    def detect_cores(
        self, image: np.ndarray, brightfield: bool = True, min_area: int = None,
        circularity_thresh: float = 0.05  # CHANGED: lowered default
    ) -> List[Dict]:
        min_area = min_area or self.min_core_area
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        if not brightfield:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            blur = clahe.apply(blur)
        bs = max(31, int(min(blur.shape) // 50) * 2 + 1)
        thresh_type = cv2.THRESH_BINARY_INV if brightfield else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            bs, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cores = []
        
        # ADDED: logging for debugging
        logger.info(f"Found {len(contours)} total contours")
        area_filtered = 0
        circularity_filtered = 0
        moment_filtered = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                area_filtered += 1
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                moment_filtered += 1
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            if circularity < circularity_thresh:  # CHANGED: use parameter
                circularity_filtered += 1
                continue
            cores.append({'center': (cx, cy), 'area': area})
        
        # ADDED: logging results
        logger.info(f"Filtering: {area_filtered} by area, {moment_filtered} by moments, {circularity_filtered} by circularity")
        logger.info(f"Final cores detected: {len(cores)}")
        
        return cores

    def scale_cores(self, cores: List[Dict], thumb_shape: Tuple[int, int], full_shape: Tuple[int, int]) -> List[Dict]:
        sx = full_shape[1] / thumb_shape[1]
        sy = full_shape[0] / thumb_shape[0]
        scaled = []
        for c in cores:
            x, y = c['center']
            scaled.append({'center': (int(x * sx), int(y * sy)), 'area': c['area']})
        return scaled

# ---------------------------------------------------------------------------
# Patch extractor, registration and preview helpers (unchanged)
# ---------------------------------------------------------------------------

class PatchExtractor:
    def __init__(self, core_size: int = 2048):
        self.core_size = core_size

    def compute_region(self, center: Tuple[int, int], slide_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x, y = center
        h, w = slide_shape
        half = self.core_size // 2
        left = max(0, x - half)
        top = max(0, y - half)
        right = min(w, left + self.core_size)
        bottom = min(h, top + self.core_size)
        if right - left < self.core_size:
            left = max(0, right - self.core_size)
        if bottom - top < self.core_size:
            top = max(0, bottom - self.core_size)
        return top, bottom, left, right

    def extract(self, slide_path: str, region: Tuple[int, int, int, int]) -> np.ndarray:
        t, b, l, r = region
        try:
            with TiffFile(slide_path) as tif:
                arr = tif.series[0].asarray()
                if arr.ndim == 3:
                    if arr.shape[0] < arr.shape[-1]:
                        arr = np.transpose(arr, (1, 2, 0))
                sub = arr[t:b, l:r]
                return sub
        except Exception as e:
            logger.error(f"Failed to extract region from {slide_path}: {e}")
            return None

def make_preview_png(he_patch: np.ndarray, or_patch: np.ndarray, png_path: Path, max_side: int = 512):
    he_gray = cv2.normalize(he_patch[..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    or_gray = cv2.normalize(or_patch[..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = np.zeros((*he_gray.shape, 3), dtype=np.uint8)
    overlay[..., 2] = he_gray
    overlay[..., 1] = or_gray
    h, w = overlay.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(png_path), overlay)

def register_pair(he_path: str, or_path: str, reg_dir: Path, prev_dir: Path, max_dim: int = 1024) -> str:
    he_name = Path(he_path).name
    or_name = Path(or_path).name
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copy(he_path, tmp / he_name)
        shutil.copy(or_path, tmp / or_name)
        reg = registration.Valis(
            src_dir=str(tmp),
            dst_dir=str(tmp),
            img_list=[he_name, or_name],
            imgs_ordered=True,
            reference_img_f=he_name,
            max_processed_image_dim_px=max_dim
        )
        reg.register(reader_cls=slide_io.BioFormatsSlideReader)
        warped_path = reg_dir / f"{Path(or_path).stem}_warped.tiff"
        reg.warp_and_save_slides(str(warped_path), crop="reference", pyramid=False)
        he_patch = imread(he_path)
        warped_patch = imread(warped_path)
        make_preview_png(he_patch, warped_patch, prev_dir / f"{Path(or_path).stem}_preview.png")
        return str(warped_path)
    except Exception as e:
        logger.error(f"VALIS registration failed: {e}")
        return or_path
    finally:
        shutil.rmtree(tmp)

# ---------------------------------------------------------------------------
# Main function (unchanged)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pair TMA cores between H&E and Orion slides")
    parser.add_argument("--he", required=True, help="Path to H&E OME‑TIFF")
    parser.add_argument("--orion", required=True, help="Path to Orion OME‑TIFF")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--thumb_size", type=int, default=4096)
    parser.add_argument("--core_size", type=int, default=2048)
    parser.add_argument("--min_core_area", type=int, default=2000)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--mem_gb", type=int, default=32)
    parser.add_argument("--max_processed_dim", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    out = Path(args.out_dir)
    patch_dir = out / "patches"
    reg_dir = out / "registered"
    prev_dir = out / "preview_png"
    patch_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(exist_ok=True)
    if args.register:
        if not VALIS_AVAILABLE:
            raise RuntimeError("VALIS is not available; install valis-wsi or run without --register")
        reg_dir.mkdir(exist_ok=True)

    detector = ThumbnailCoreDetector(args.thumb_size, args.min_core_area)

    # Detect H&E cores
    he_thumb, he_full = detector.load_thumbnail(args.he)
    he_core_objs = detector.detect_cores(he_thumb, brightfield=True)
    he_scaled = detector.scale_cores(he_core_objs, he_thumb.shape, he_full)
    he_centres = np.array([c['center'] for c in he_scaled])
    spacing = estimate_core_spacing(he_centres)

    # Detect Orion cores
    or_thumb, or_full = detector.load_thumbnail(args.orion, use_channel0=True)
    or_core_objs = detector.detect_cores(or_thumb, brightfield=False)
    or_scaled = detector.scale_cores(or_core_objs, or_thumb.shape, or_full)
    or_centres = np.array([c['center'] for c in or_scaled])

    # Align and order
    if len(he_centres) > 0 and len(or_centres) > 0:
        or_aligned = align_orion_to_he(or_centres, he_centres)
    else:
        or_aligned = or_centres.copy()

    he_order, he_indices = cluster_grid_order(he_centres, spacing)
    or_order_aligned, or_indices = cluster_grid_order(or_aligned, spacing)
    num_pairs = min(len(he_order), len(or_order_aligned))
    pairs = []
    for i in range(num_pairs):
        he_idx = he_indices[i]
        or_idx = or_indices[i]
        pairs.append((he_centres[he_idx], or_centres[or_idx]))

    logger.info(f"Found {len(pairs)} paired cores.")

    extractor = PatchExtractor(args.core_size)
    if args.register:
        slide_io.init_jvm(mem_gb=args.mem_gb)

    try:
        for idx, (he_c, or_c) in enumerate(pairs, 1):
            logger.info(f"Processing pair {idx}/{len(pairs)}: HE {he_c} ⇔ Orion {or_c}")
            he_reg = extractor.compute_region(tuple(he_c), he_full)
            or_reg = extractor.compute_region(tuple(or_c), or_full)
            he_patch = extractor.extract(args.he, he_reg)
            or_patch = extractor.extract(args.orion, or_reg)
            if he_patch is None or or_patch is None:
                logger.warning("Skipping pair due to extraction failure")
                continue
            he_path = patch_dir / f"core_{idx:03d}_he.tiff"
            or_path = patch_dir / f"core_{idx:03d}_orion.tiff"
            imwrite(he_path, he_patch, compression="zlib")
            imwrite(or_path, or_patch, compression="zlib")
            make_preview_png(he_patch, or_patch, prev_dir / f"core_{idx:03d}_preview.png")
            if args.register:
                register_pair(str(he_path), str(or_path), reg_dir, prev_dir, args.max_processed_dim)
    finally:
        if args.register:
            slide_io.kill_jvm()

    logger.info("Completed.")

if __name__ == "__main__":
    main()
