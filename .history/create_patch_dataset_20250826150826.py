#!/usr/bin/env python3
"""
Pre-extract patches from large numpy files for fast training.
This converts the slow "load entire file for each patch" approach
to a fast "load small pre-extracted patches" approach.
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def extract_patches_from_core(he_path, ori_path, output_dir, basename, 
                            patch_size=128, num_patches=200, seed=42):
    """Extract random patches from a single core pair."""
    
    # Load full images
    he = np.load(he_path).astype(np.float32)
    ori = np.load(ori_path).astype(np.float32)
    
    # Normalize
    if he.max() > 1.0:
        he = he / 255.0
    if ori.ndim == 3 and ori.shape[0] == 20:
        ori = np.transpose(ori, (1, 2, 0))
    if ori.max() > 1.0:
        ori = ori / 255.0
    
    H, W = he.shape[:2]
    patches_extracted = 0
    
    # Set seed for reproducible patch extraction
    rng = random.Random(seed + hash(basename))
    
    for i in range(num_patches * 3):  # Try more to get enough valid patches
        if patches_extracted >= num_patches:
            break
            
        # Random crop coordinates
        if H > patch_size and W > patch_size:
            y = rng.randint(0, H - patch_size)
            x = rng.randint(0, W - patch_size)
        else:
            y, x = 0, 0
        
        # Extract patches
        he_patch = he[y:y+patch_size, x:x+patch_size, :]
        ori_patch = ori[y:y+patch_size, x:x+patch_size, :]
        
        # Skip if patch is too small or has issues
        if he_patch.shape[:2] != (patch_size, patch_size):
            continue
        if ori_patch.shape[:2] != (patch_size, patch_size):
            continue
        if he_patch.mean() < 0.01:  # Skip very dark patches
            continue
            
        # Save patch pair
        patch_name = f"{basename}_patch_{patches_extracted:04d}"
        np.save(output_dir / f"{patch_name}_HE.npy", he_patch)
        np.save(output_dir / f"{patch_name}_ORION.npy", ori_patch)
        patches_extracted += 1
    
    return patches_extracted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="core_patches_npy")
    parser.add_argument("--output_dir", default="fast_patches")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--patches_per_core", type=int, default=100)
    parser.add_argument("--max_cores", type=int, default=50, 
                       help="Maximum cores to process (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all core pairs
    he_files = list(input_dir.glob("core_*_HE.npy"))
    he_files = he_files[:args.max_cores]  # Limit for testing
    
    print(f"Processing {len(he_files)} cores...")
    print(f"Extracting {args.patches_per_core} patches per core")
    print(f"Total patches: {len(he_files) * args.patches_per_core}")
    
    total_patches = 0
    start_time = time.time()
    
    for he_file in tqdm(he_files, desc="Processing cores"):
        basename = he_file.stem.replace("_HE", "")
        ori_file = input_dir / f"{basename}_ORION.npy"
        
        if not ori_file.exists():
            print(f"Warning: Missing ORION file for {basename}")
            continue
        
        patches_extracted = extract_patches_from_core(
            he_file, ori_file, output_dir, basename,
            patch_size=args.patch_size,
            num_patches=args.patches_per_core,
            seed=args.seed
        )
        
        total_patches += patches_extracted
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Extracted {total_patches} total patches")
    print(f"Average: {total_patches/elapsed:.1f} patches/sec")
    print(f"Saved to: {output_dir.resolve()}")
    
    # Test loading speed
    print("\nTesting patch loading speed...")
    patch_files = list(output_dir.glob("*_HE.npy"))[:10]
    
    start = time.time()
    for pf in patch_files:
        he_patch = np.load(pf)
        ori_patch = np.load(pf.parent / pf.name.replace("_HE", "_ORION"))
    load_time = time.time() - start
    
    print(f"Loading 10 patches: {load_time:.3f}s = {10/load_time:.1f} patches/sec")
    print(f"Expected batch speed (32 patches): {32/10*load_time:.3f}s per batch")
    print("This should be 100x faster than before!")

if __name__ == "__main__":
    main()
