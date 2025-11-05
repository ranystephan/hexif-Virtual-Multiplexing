#!/usr/bin/env python3
"""
Test script to verify CODEX processing setup
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if required packages can be imported"""
    print("Testing package imports...")
    
    # Test basic packages
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError:
        print("✗ numpy not available")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError:
        print("✗ pandas not available")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError:
        print("✗ matplotlib not available")
        return False
    
    # Test optional packages
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("⚠ OpenCV not available (optional)")
    
    try:
        from skimage import measure, morphology, filters
        print("✓ scikit-image imported successfully")
    except ImportError:
        print("⚠ scikit-image not available (optional)")
    
    try:
        import scanpy as sc
        print("✓ scanpy imported successfully")
    except ImportError:
        print("⚠ scanpy not available (optional)")
    
    try:
        import spacec as sp
        print("✓ SPACEc imported successfully")
    except ImportError:
        print("⚠ SPACEc not available (optional)")
    
    return True

def test_data_structure():
    """Test if data structure is correct"""
    print("\nTesting data structure...")
    
    data_root = Path("data/nomad_data/CODEX")
    
    if not data_root.exists():
        print(f"✗ Data directory not found: {data_root}")
        return False
    
    print(f"✓ Data directory found: {data_root}")
    
    # Check for marker list
    marker_file = data_root / "MarkerList.txt"
    if marker_file.exists():
        print(f"✓ Marker list found: {marker_file}")
        with open(marker_file, 'r') as f:
            markers = [line.strip() for line in f.readlines() if line.strip()]
        print(f"  Found {len(markers)} markers")
    else:
        print(f"⚠ Marker list not found: {marker_file}")
    
    # Check for TMA directories
    tma_dirs = [d for d in data_root.iterdir() if d.is_dir() and "TMA" in d.name]
    if tma_dirs:
        print(f"✓ Found {len(tma_dirs)} TMA directories:")
        for tma_dir in tma_dirs[:5]:  # Show first 5
            print(f"  - {tma_dir.name}")
        if len(tma_dirs) > 5:
            print(f"  ... and {len(tma_dirs) - 5} more")
    else:
        print("⚠ No TMA directories found")
    
    return True

def test_output_directory():
    """Test if output directory can be created"""
    print("\nTesting output directory...")
    
    output_root = Path("data/nomad_data/CODEX/processed")
    
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory ready: {output_root}")
        return True
    except Exception as e:
        print(f"✗ Cannot create output directory: {e}")
        return False

def main():
    """Main test function"""
    print("CODEX Processing Setup Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data structure
    data_ok = test_data_structure()
    
    # Test output directory
    output_ok = test_output_directory()
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"Data structure: {'✓' if data_ok else '✗'}")
    print(f"Output directory: {'✓' if output_ok else '✗'}")
    
    if imports_ok and data_ok and output_ok:
        print("\n🎉 Setup looks good! You can run the processing scripts.")
        print("\nNext steps:")
        print("1. Install SPACEc: pip install git+https://github.com/yuqiyuqitan/SPACEc.git")
        print("2. Run simple pipeline: python simple_codex_processor.py")
        print("3. Or run full pipeline: python codex_processing_pipeline.py")
    else:
        print("\n⚠ Some issues found. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check data directory structure")
        print("3. Ensure write permissions for output directory")

if __name__ == "__main__":
    main() 