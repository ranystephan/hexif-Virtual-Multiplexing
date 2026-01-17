#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
import logging
from pathlib import Path
from hexif.inference import HexifPredictor
from hexif.preprocessing import detect_cores

def process_file(file_path: Path, output_dir: Path, predictor: HexifPredictor, 
                 detect_tissue: bool = True):
    logging.info(f"Processing {file_path}...")
    
    # 1. Load Image or Detect Cores
    # If it's a large slide (tiff, svs), we likely need to detect cores first.
    # If it's a standard image (png, jpg), we might treat it as a single ROI.
    
    ext = file_path.suffix.lower()
    is_wsi = ext in ['.tiff', '.tif', '.svs', '.ndpi', '.vms', '.vmu']
    
    cores = [] # List of (image_np, name_suffix)
    
    if is_wsi and detect_tissue:
        logging.info("Detected WSI format, attempting core detection...")
        try:
            # We use detect_cores which returns bboxes on a downsampled image
            # But we actually need the full resolution crops.
            # So we should probably use the logic from hexif.preprocessing.extract_patches
            # But extract_patches saves to disk. We might want to keep in memory or save temp.
            # For simplicity and robustness, let's re-use the extraction logic but adapt it here
            # or update hexif.preprocessing to allow returning arrays.
            
            # Actually, `detect_cores` returns bboxes. We can then crop using pyvips.
            bboxes, thumb = detect_cores(str(file_path))
            logging.info(f"Found {len(bboxes)} cores.")
            
            import pyvips
            slide = pyvips.Image.new_from_file(str(file_path), access="random")
            
            for i, (min_r, min_c, max_r, max_c) in enumerate(bboxes):
                # Clamp coordinates
                h, w = max_r - min_r, max_c - min_c
                # Optional: Ensure minimum size or fixed size? 
                # The training used 2048x2048 cores.
                # If these are variable size, the model can handle it via sliding window.
                
                crop = slide.crop(min_c, min_r, w, h)
                
                # Convert to numpy
                mem = crop.write_to_memory()
                # Determine format
                # This part can be tricky without the helper.
                # Let's import vips_to_numpy from preprocessing if possible, but it's internal.
                # We'll trust the user installed dependencies.
                
                # Simplified: assume standard RGB 8-bit for H&E
                # If not, vips might give us something else.
                # Let's force sRGB 8-bit
                if crop.interpretation != 'srgb':
                    crop = crop.colourspace('srgb')
                
                img_np = np.ndarray(buffer=mem, dtype=np.uint8, 
                                   shape=[crop.height, crop.width, crop.bands])
                
                if img_np.shape[2] > 3:
                    img_np = img_np[..., :3]
                    
                cores.append((img_np, f"_core_{i:03d}"))
                
        except Exception as e:
            logging.error(f"Failed to process WSI {file_path}: {e}")
            return
    else:
        # Standard image reading
        img = cv2.imread(str(file_path))
        if img is None:
            logging.warning(f"Could not read {file_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cores.append((img, ""))

    # 2. Predict per core
    for core_img, suffix in cores:
        logging.info(f"Predicting on region {suffix if suffix else 'main'} (shape {core_img.shape})...")
        
        # Predict
        pred_linear = predictor.predict_image(core_img)
        
        # Save output
        out_name = file_path.stem + suffix
        
        # Save raw predictions (npy)
        np.save(output_dir / f"{out_name}_pred.npy", pred_linear)
        
        # Save diagnostic plot
        predictor.save_diagnostics(core_img, pred_linear, output_dir / f"{out_name}_diagnostic.png")
        
        logging.info(f"Saved results to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run HEXIF inference on H&E images.")
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results.")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.pth).")
    parser.add_argument("--scaler_path", required=True, help="Path to scaler config (.json).")
    parser.add_argument("--no_tissue_detect", action="store_true", help="Disable automatic tissue detection for WSI.")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu).")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    predictor = HexifPredictor(args.model_path, args.scaler_path, args.device)
    
    inp = Path(args.input)
    if inp.is_file():
        process_file(inp, out_dir, predictor, not args.no_tissue_detect)
    elif inp.is_dir():
        # Process all images in dir
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.svs'}
        for f in inp.iterdir():
            if f.suffix.lower() in exts:
                process_file(f, out_dir, predictor, not args.no_tissue_detect)
    else:
        logging.error(f"Input {inp} not found.")

if __name__ == "__main__":
    main()
