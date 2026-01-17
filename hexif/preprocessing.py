import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from skimage import measure

try:
    import pyvips
    HAS_VIPS = True
except ImportError:
    HAS_VIPS = False
    logging.warning("pyvips not found. Some functionality will be disabled.")

try:
    from valis import registration, slide_io
    HAS_VALIS = True
except ImportError:
    HAS_VALIS = False
    logging.warning("valis not found. Registration functionality will be disabled.")


# --- VIPS helpers ---
_VIPS2NP = {
    'uchar':    np.uint8,
    'char':     np.int8,
    'ushort':   np.uint16,
    'short':    np.int16,
    'uint':     np.uint32,
    'int':      np.int32,
    'float':    np.float32,
    'double':   np.float64,
    'complex':  np.complex64,
    'dpcomplex':np.complex128,
}

def vips_to_numpy(img) -> np.ndarray:
    """pyvips.Image -> np.ndarray (H, W, C) with correct dtype mapping."""
    if not HAS_VIPS:
        raise ImportError("pyvips is required for this function")
        
    fmt = img.format
    if fmt not in _VIPS2NP:
        raise RuntimeError(f"Unhandled VIPS pixel format: {fmt}")
    dtype = _VIPS2NP[fmt]
    mem = img.write_to_memory()
    arr = np.frombuffer(mem, dtype=dtype)
    arr = arr.reshape(img.height, img.width, img.bands)
    return arr

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))


def register_slides(he_path: str, orion_path: str, output_dir: str):
    """Registers H&E and Orion slides using Valis."""
    if not HAS_VALIS:
        raise ImportError("valis is required for registration")
        
    slide_src_dir = os.path.dirname(he_path)
    
    # Create a temporary list for valis
    img_list = {
        he_path: "HE",
        orion_path: "Orion"
    }
    
    registrar = registration.Valis(
        slide_src_dir,
        dst_dir=output_dir,
        img_list=img_list,
        imgs_ordered=True,
        reference_img_f=he_path,
        align_to_reference=True,
    )
    
    rigid_registrar, non_rigid_registrar, error_df = registrar.register(
        reader_cls=slide_io.BioFormatsSlideReader
    )
    
    registrar.warp_and_save_slides(output_dir)
    return output_dir


def detect_cores(image_path: str, thumb_size: int = 4000) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Detects tissue cores in a slide.
    Returns list of bounding boxes (min_row, min_col, max_row, max_col) and the thumb image used.
    """
    if HAS_VALIS:
        # Use valis reader if available as it's robust
        reader_cls = slide_io.get_slide_reader(image_path)
        reader = reader_cls(src_f=image_path)
        
        # Pick level close to thumb_size
        w0, h0 = reader.metadata.slide_dimensions[0]
        long_side = max(w0, h0)
        down_approx = max(1, int(round(long_side / thumb_size)))
        ratios = [w0 / w for (w, h) in reader.metadata.slide_dimensions]
        level = int(np.argmin([abs(r - down_approx) for r in ratios]))
        
        img = reader.slide2image(level=level)
        full_w, full_h = w0, h0
    elif HAS_VIPS:
        # Fallback to pyvips
        img_vips = pyvips.Image.new_from_file(image_path, access="random")
        full_w, full_h = img_vips.width, img_vips.height
        scale = thumb_size / max(full_w, full_h)
        img_vips_thumb = img_vips.resize(scale)
        img = vips_to_numpy(img_vips_thumb)
        # Ensure 3 channels for H&E
        if img.shape[-1] > 3: img = img[..., :3]
    else:
        # Fallback to opencv (might fail for large OME-TIFFs)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        full_h, full_w = img.shape[:2] # Assume it loaded full size if it worked

    # Process for core detection
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    laplace = cv2.Laplacian(gray, ddepth=cv2.CV_32F)
    laplace = cv2.normalize(laplace, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(laplace, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    labels = measure.label(opened, connectivity=2)
    regions = measure.regionprops(labels)
    
    bboxes = []
    areas = [r.area for r in regions if r.area > 10]
    if areas:
        q1 = np.percentile(areas, 25)
        for r in regions:
            if r.area > q1:
                bboxes.append(r.bbox)
                
    # Scale bboxes back to full resolution
    thumb_h, thumb_w = img.shape[:2]
    scale_x = full_w / float(thumb_w)
    scale_y = full_h / float(thumb_h)
    
    full_bboxes = []
    for (min_r, min_c, max_r, max_c) in bboxes:
        full_bboxes.append((
            int(min_r * scale_y),
            int(min_c * scale_x),
            int(max_r * scale_y),
            int(max_c * scale_x)
        ))
        
    return full_bboxes, img


def extract_patches(he_path: str, orion_path: Optional[str], bboxes: List[Tuple[int, int, int, int]], 
                   output_dir: str, target_size: int = 2048):
    """
    Extracts cores from slides and saves as .npy files.
    """
    if not HAS_VIPS:
        raise ImportError("pyvips is required for extraction")
        
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    he_slide = pyvips.Image.new_from_file(he_path, access="random")
    full_w, full_h = he_slide.width, he_slide.height
    
    ori_slide = None
    ori_pages = None
    ori_is_multi_page = False
    
    if orion_path:
        ori_slide = pyvips.Image.new_from_file(orion_path, access="random")
        # Check layout
        fields = ori_slide.get_fields()
        n_pages = ori_slide.get("n-pages") if "n-pages" in fields else 1
        if n_pages >= 20:
            ori_is_multi_page = True
            ori_pages = [pyvips.Image.new_from_file(orion_path, page=i, access="random") for i in range(20)]
            
    saved_count = 0
    
    for idx, (min_r_f, min_c_f, max_r_f, max_c_f) in enumerate(bboxes):
        # Calculate center in full res
        cx = (min_c_f + max_c_f) / 2.0
        cy = (min_r_f + max_r_f) / 2.0
        
        half_w = target_size // 2
        half_h = target_size // 2
        
        x0 = clamp(int(cx - half_w), 0, full_w - target_size)
        y0 = clamp(int(cy - half_h), 0, full_h - target_size)
        
        try:
            # Crop H&E
            he_crop = he_slide.crop(x0, y0, target_size, target_size)
            he_np = vips_to_numpy(he_crop)
            
            if he_np.ndim == 2:
                he_np = np.stack([he_np]*3, axis=-1)
            if he_np.shape[-1] > 3:
                he_np = he_np[..., :3]
                
            # Normalize H&E
            if he_np.dtype == np.uint8:
                he_np = he_np.astype(np.float32) / 255.0
            elif he_np.dtype == np.uint16:
                he_np = he_np.astype(np.float32) / 65535.0
            else:
                he_np = he_np.astype(np.float32)
                
            base = f"core_{idx:03d}"
            np.save(out_dir / f"{base}_HE.npy", he_np, allow_pickle=False)
            
            # Crop Orion if available
            if orion_path:
                if ori_is_multi_page:
                    chans = []
                    for p in ori_pages:
                        roi = p.crop(x0, y0, target_size, target_size)
                        a = vips_to_numpy(roi)
                        if a.ndim == 3 and a.shape[-1] == 1:
                            a = a[..., 0]
                        
                        if a.dtype == np.uint16:
                            a = a.astype(np.float32) / 65535.0
                        elif a.dtype == np.uint8:
                            a = a.astype(np.float32) / 255.0
                        else:
                            a = a.astype(np.float32)
                        chans.append(a)
                    ori_np = np.stack(chans, axis=-1)
                else:
                    roi = ori_slide.crop(x0, y0, target_size, target_size)
                    ori_np = vips_to_numpy(roi)
                    if ori_np.dtype == np.uint16:
                        ori_np = ori_np.astype(np.float32) / 65535.0
                    elif ori_np.dtype == np.uint8:
                        ori_np = ori_np.astype(np.float32) / 255.0
                    else:
                        ori_np = ori_np.astype(np.float32)
                        
                np.save(out_dir / f"{base}_ORION.npy", ori_np, allow_pickle=False)
                
            saved_count += 1
            
        except Exception as e:
            logging.error(f"Error extracting core {idx}: {e}")
            
    logging.info(f"Extracted {saved_count}/{len(bboxes)} cores to {output_dir}")
