#!/usr/bin/env python3
"""
Test script to understand TIF file format and fix SPACEc loading issues
"""

import tifffile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Define paths
root_path = Path.cwd().parent.parent
data_path = root_path / 'data' / 'nomad_data' / 'CODEX'
output_dir = root_path / 'output' / 'spacec_analysis'

# Load markers
marker_file = data_path / 'MarkerList.txt'
with open(marker_file, 'r') as f:
    markers = [line.strip() for line in f.readlines()]

print(f"Loaded {len(markers)} markers")

# Find TMA directories
tma_dirs = [d for d in data_path.iterdir() if d.is_dir() and 'TMA' in d.name]
print(f"Found {len(tma_dirs)} TMA directories")

# Test first ccRCC TMA
for tma_dir in tma_dirs:
    if 'ccRCC' in tma_dir.name:
        print(f"\nTesting TMA: {tma_dir.name}")
        
        bestfocus_dir = tma_dir / 'bestFocus'
        if bestfocus_dir.exists():
            tif_files = list(bestfocus_dir.glob('*.tif'))
            if tif_files:
                sample_tif = tif_files[0]
                print(f"Testing file: {sample_tif.name}")
                
                # Load TIF file
                img = tifffile.imread(str(sample_tif))
                print(f"Image shape: {img.shape}")
                print(f"Image dtype: {img.dtype}")
                print(f"Image min/max: {img.min()}/{img.max()}")
                
                # Check channel count
                if img.shape[0] == len(markers):
                    print("✓ Correct number of channels")
                else:
                    print(f"⚠ Mismatch: {len(markers)} markers vs {img.shape[0]} channels")
                
                # Test different channel arrangements
                print("\nTesting channel arrangements:")
                
                # Original format (C, H, W)
                print(f"Original (C, H, W): {img.shape}")
                
                # Try transposing to (H, W, C)
                if img.ndim == 3:
                    img_hwc = np.transpose(img, (1, 2, 0))
                    print(f"Transposed (H, W, C): {img_hwc.shape}")
                    
                    # Check if this matches expected format
                    if img_hwc.shape[2] == len(markers):
                        print("✓ Transposed format matches marker count")
                    else:
                        print("⚠ Transposed format still doesn't match")
                
                # Test saving in different formats
                print("\nTesting format conversion...")
                
                # Save as individual channels
                test_dir = output_dir / 'test_channels'
                test_dir.mkdir(exist_ok=True)
                
                for i, marker in enumerate(markers[:5]):  # Test first 5 markers
                    channel_img = img[i]
                    channel_file = test_dir / f"{marker}.tif"
                    tifffile.imwrite(str(channel_file), channel_img)
                    print(f"  Saved {marker} channel: {channel_img.shape}")
                
                # Save as RGB composite (first 3 channels)
                if img.shape[0] >= 3:
                    rgb_img = img[:3]
                    rgb_file = test_dir / 'rgb_composite.tif'
                    tifffile.imwrite(str(rgb_file), rgb_img)
                    print(f"  Saved RGB composite: {rgb_img.shape}")
                
                break

print("\nTest completed. Check output/test_channels/ for converted files.") 