#!/usr/bin/env python3
"""
Test script to verify SPACEc installation and basic functionality
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if required packages can be imported"""
    print("Testing package imports...")
    
    # Test basic packages
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    # Test SPACEc
    try:
        import spacec as sp
        print("✓ SPACEc imported successfully")
        print(f"  SPACEc version: {sp.__version__ if hasattr(sp, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"✗ SPACEc import failed: {e}")
        print("  Install with: pip install git+https://github.com/yuqiyuqitan/SPACEc.git")
        return False
    
    # Test scanpy
    try:
        import scanpy as sc
        print("✓ Scanpy imported successfully")
        print(f"  Scanpy version: {sc.__version__}")
    except ImportError as e:
        print(f"✗ Scanpy import failed: {e}")
        print("  Install with: pip install scanpy")
        return False
    
    # Test cell segmentation dependencies
    try:
        import cellpose
        print("✓ Cellpose imported successfully")
    except ImportError as e:
        print(f"⚠ Cellpose import failed: {e}")
        print("  Install with: pip install cellpose")
    
    try:
        import deepcell
        print("✓ DeepCell imported successfully")
    except ImportError as e:
        print(f"⚠ DeepCell import failed: {e}")
        print("  Install with: pip install deepcell")
    
    return True

def test_spacec_functions():
    """Test basic SPACEc function availability"""
    print("\nTesting SPACEc function availability...")
    
    try:
        import spacec as sp
        
        # Check if key functions exist
        functions_to_check = [
            'hf.downscale_tissue',
            'tl.label_tissue',
            'tl.save_labelled_tissue',
            'tl.cell_segmentation',
            'tl.extract_features'
        ]
        
        for func_name in functions_to_check:
            try:
                # Try to access the function
                module_path = func_name.split('.')
                obj = sp
                for attr in module_path:
                    obj = getattr(obj, attr)
                print(f"✓ {func_name} available")
            except AttributeError:
                print(f"✗ {func_name} not available")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing SPACEc functions: {e}")
        return False

def test_data_structure():
    """Test if data structure is accessible"""
    print("\nTesting data structure...")
    
    from pathlib import Path
    
    # Check data directory
    data_root = Path("../../data/nomad_data/CODEX")
    if data_root.exists():
        print(f"✓ Data directory found: {data_root}")
        
        # Check marker list
        marker_file = data_root / "MarkerList.txt"
        if marker_file.exists():
            print(f"✓ Marker list found: {marker_file}")
            try:
                with open(marker_file, 'r') as f:
                    markers = [line.strip() for line in f.readlines() if line.strip()]
                print(f"  Loaded {len(markers)} markers")
            except Exception as e:
                print(f"✗ Error reading marker file: {e}")
        else:
            print(f"✗ Marker list not found: {marker_file}")
        
        # Check TMA directories
        tma_dirs = [d for d in data_root.iterdir() if d.is_dir() and "TMA" in d.name]
        if tma_dirs:
            print(f"✓ Found {len(tma_dirs)} TMA directories")
            for tma_dir in tma_dirs[:3]:  # Show first 3
                print(f"  - {tma_dir.name}")
            if len(tma_dirs) > 3:
                print(f"  ... and {len(tma_dirs) - 3} more")
        else:
            print("✗ No TMA directories found")
    else:
        print(f"✗ Data directory not found: {data_root}")
        return False
    
    return True

def main():
    """Main test function"""
    print("SPACEc Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test SPACEc functions
    functions_ok = test_spacec_functions()
    
    # Test data structure
    data_ok = test_data_structure()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if imports_ok and functions_ok and data_ok:
        print("✓ All tests passed! SPACEc is ready to use.")
        print("\nYou can now run the SPACEc pipeline:")
        print("python spacec_codex_processor.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        
        if not imports_ok:
            print("\nTo fix import issues:")
            print("pip install -r requirements.txt")
        
        if not functions_ok:
            print("\nTo fix SPACEc function issues:")
            print("pip install git+https://github.com/yuqiyuqitan/SPACEc.git")
        
        if not data_ok:
            print("\nTo fix data structure issues:")
            print("Ensure your CODEX data is in the correct directory structure")

if __name__ == "__main__":
    main() 