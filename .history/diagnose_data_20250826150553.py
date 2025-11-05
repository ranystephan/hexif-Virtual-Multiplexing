#!/usr/bin/env python3
"""
Diagnose data loading performance issues.
"""

import time
import numpy as np
from pathlib import Path

def diagnose_data_loading(pairs_dir):
    pairs_dir = Path(pairs_dir)
    
    # Find first few files
    he_files = list(pairs_dir.glob("core_*_HE.npy"))[:5]
    ori_files = [pairs_dir / f.name.replace("_HE", "_ORION") for f in he_files]
    
    print(f"Testing with {len(he_files)} file pairs...")
    
    for i, (he_path, ori_path) in enumerate(zip(he_files, ori_files)):
        print(f"\n=== File {i+1}: {he_path.name} ===")
        
        # Check file sizes
        he_size = he_path.stat().st_size / (1024*1024)  # MB
        ori_size = ori_path.stat().st_size / (1024*1024)  # MB
        print(f"File sizes: HE={he_size:.1f}MB, ORION={ori_size:.1f}MB")
        
        # Test loading times
        print("Testing load methods:")
        
        # Method 1: Regular load
        start = time.time()
        he = np.load(he_path)
        he_load_time = time.time() - start
        print(f"  HE regular load: {he_load_time:.2f}s, shape={he.shape}")
        
        start = time.time()
        ori = np.load(ori_path)
        ori_load_time = time.time() - start
        print(f"  ORION regular load: {ori_load_time:.2f}s, shape={ori.shape}")
        
        # Method 2: Memory-mapped load
        start = time.time()
        he_mmap = np.load(he_path, mmap_mode='r')
        he_mmap_time = time.time() - start
        print(f"  HE mmap load: {he_mmap_time:.2f}s")
        
        start = time.time()
        ori_mmap = np.load(ori_path, mmap_mode='r')
        ori_mmap_time = time.time() - start
        print(f"  ORION mmap load: {ori_mmap_time:.2f}s")
        
        # Test array access
        start = time.time()
        patch = he[:128, :128, :]
        access_time = time.time() - start
        print(f"  Array access (128x128 patch): {access_time:.3f}s")
        
        # Test copy operations
        start = time.time()
        he_copy = he.astype(np.float32)
        copy_time = time.time() - start
        print(f"  Type conversion: {copy_time:.2f}s")
        
        total_time = he_load_time + ori_load_time + copy_time
        print(f"  TOTAL per file pair: {total_time:.2f}s")
        
        if i == 0:  # Only test first file in detail
            break
    
    print(f"\n=== Summary ===")
    print(f"If loading 2 files takes >1 second, that's your bottleneck!")
    print(f"With 32 batch size, you're loading 64 files per batch.")
    print(f"Expected time per batch: {total_time * 32:.1f}s")

if __name__ == "__main__":
    diagnose_data_loading("core_patches_npy")
