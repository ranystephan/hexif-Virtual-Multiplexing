#!/usr/bin/env python3
"""
Test script to verify image loading works correctly with multi-channel Orion data.
"""

import sys
import pathlib
from registration_pipeline import ImagePreprocessor
from tifffile import imread

def test_image_loading(input_dir: str):
    """Test image loading with a few sample images."""
    input_path = pathlib.Path(input_dir)
    
    # Find a few sample images
    he_files = list(input_path.glob("*_HE.tif"))[:3]  # Test first 3
    
    print(f"Testing image loading with {len(he_files)} sample images...")
    
    for he_file in he_files:
        # Extract core ID
        filename = he_file.stem
        if filename.endswith("_HE"):
            core_id = filename[:-3]
        else:
            core_id = filename.replace("_HE.tif", "")
        
        # Find corresponding Orion file
        orion_file = input_path / f"{core_id}_Orion.tif"
        
        if not orion_file.exists():
            print(f"❌ Missing Orion file for {core_id}")
            continue
        
        print(f"\n🔍 Testing {core_id}:")
        
        try:
            # Test raw loading
            he_raw = imread(str(he_file))
            orion_raw = imread(str(orion_file))
            
            print(f"  Raw H&E shape: {he_raw.shape}, dtype: {he_raw.dtype}")
            print(f"  Raw Orion shape: {orion_raw.shape}, dtype: {orion_raw.dtype}")
            
            # Test H&E preprocessing
            try:
                he_processed = ImagePreprocessor.load_and_preprocess_image(str(he_file))
                print(f"  ✅ H&E processed shape: {he_processed.shape}, dtype: {he_processed.dtype}")
            except Exception as e:
                print(f"  ❌ H&E processing failed: {e}")
            
            # Test Orion DAPI extraction
            try:
                dapi_processed = ImagePreprocessor.extract_dapi_channel(str(orion_file), dapi_channel=0)
                print(f"  ✅ DAPI processed shape: {dapi_processed.shape}, dtype: {dapi_processed.dtype}")
                
                # Test different channels
                if orion_raw.shape[0] > 1:
                    dapi_ch1 = ImagePreprocessor.extract_dapi_channel(str(orion_file), dapi_channel=1)
                    print(f"  ✅ Channel 1 processed shape: {dapi_ch1.shape}, dtype: {dapi_ch1.dtype}")
                    
            except Exception as e:
                print(f"  ❌ Orion processing failed: {e}")
        
        except Exception as e:
            print(f"  ❌ Raw loading failed: {e}")
    
    print(f"\n✅ Image loading test completed!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_image_loading.py <input_directory>")
        print("Example: python test_image_loading.py data/paired_core_extracted_TA118/")
        sys.exit(1)
    
    test_image_loading(sys.argv[1]) 