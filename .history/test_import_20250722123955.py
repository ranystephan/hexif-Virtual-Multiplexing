#!/usr/bin/env python3
"""
Simple test to isolate the import issue.
"""

try:
    print("Testing individual imports...")
    
    print("1. Testing numpy...")
    import numpy as np
    print("✅ numpy OK")
    
    print("2. Testing opencv...")
    import cv2
    print("✅ opencv OK")
    
    print("3. Testing skimage...")
    from skimage import filters, morphology, measure, feature, segmentation
    print("✅ skimage OK")
    
    print("4. Testing scipy...")
    from scipy import ndimage, spatial
    print("✅ scipy OK")
    
    print("5. Testing other imports...")
    import matplotlib.pyplot as plt
    from tifffile import imread, imwrite
    from pathlib import Path
    import logging
    from typing import Dict, List, Tuple, Optional, Union
    from dataclasses import dataclass
    import json
    import warnings
    print("✅ All basic imports OK")
    
    print("6. Testing core_detector import...")
    import sys
    sys.path.append('core_first_pipeline')
    
    # First try to import the module
    import core_detector as cd_module
    print("✅ core_detector module imported")
    
    # Then try to get the class
    CoreDetector = getattr(cd_module, 'CoreDetector')
    print("✅ CoreDetector class found")
    
    CoreDetectionConfig = getattr(cd_module, 'CoreDetectionConfig') 
    print("✅ CoreDetectionConfig class found")
    
    print("🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc() 