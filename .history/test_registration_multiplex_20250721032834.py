#!/usr/bin/env python3
"""
Test script for the multiplex registration pipeline to verify that
all channels are being preserved in the Orion patches.
"""

import os
import pathlib
import numpy as np
from registration_pipeline_multiplex import RegistrationConfig, RegistrationPipelineMultiplex

def test_multiplex_patches(pairs_dir: str):
    """Test that the multiplex patches have the correct number of channels."""
    
    pairs_path = pathlib.Path(pairs_dir)
    if not pairs_path.exists():
        print(f"❌ Pairs directory does not exist: {pairs_dir}")
        return False
    
    # Find all Orion patches
    orion_files = list(pairs_path.glob("*_ORION.npy"))
    he_files = list(pairs_path.glob("*_HE.npy"))
    
    if not orion_files:
        print(f"❌ No Orion patches found in {pairs_dir}")
        return False
    
    print(f"📊 Found {len(orion_files)} Orion patches and {len(he_files)} H&E patches")
    
    # Test a sample of patches
    sample_size = min(5, len(orion_files))
    success_count = 0
    
    for i, orion_file in enumerate(orion_files[:sample_size]):
        # Find corresponding H&E file
        base_name = orion_file.stem.replace("_ORION", "")
        he_file = pairs_path / f"{base_name}_HE.npy"
        
        if he_file.exists():
            try:
                orion_patch = np.load(orion_file)
                he_patch = np.load(he_file)
                
                print(f"Patch {i+1} ({base_name}):")
                print(f"  🟢 Orion shape: {orion_patch.shape}")
                print(f"  🔵 H&E shape: {he_patch.shape}")
                
                # Check if Orion has multiple channels
                if orion_patch.ndim == 3 and orion_patch.shape[2] > 1:
                    print(f"  ✅ SUCCESS: Multi-channel Orion patch with {orion_patch.shape[2]} channels!")
                    success_count += 1
                elif orion_patch.ndim == 2:
                    print(f"  ⚠️  WARNING: Single-channel Orion patch (2D shape)")
                elif orion_patch.ndim == 3 and orion_patch.shape[2] == 1:
                    print(f"  ⚠️  WARNING: Single-channel Orion patch (3D with 1 channel)")
                else:
                    print(f"  ❓ UNKNOWN: Unusual Orion patch shape")
                    
                # Validate H&E shape
                if he_patch.ndim == 3 and he_patch.shape[2] == 3:
                    print(f"  ✅ H&E patch has correct RGB shape")
                else:
                    print(f"  ⚠️  H&E patch has unexpected shape: {he_patch.shape}")
                    
                print()
                
            except Exception as e:
                print(f"  ❌ ERROR loading patch {base_name}: {e}")
        else:
            print(f"  ❌ No matching H&E file for {orion_file}")
    
    print(f"🎯 SUMMARY: {success_count}/{sample_size} patches have multi-channel Orion data")
    
    if success_count > 0:
        print("✅ SUCCESS: Multi-channel registration is working!")
        return True
    else:
        print("❌ FAILURE: No multi-channel patches found")
        return False

def run_test_pipeline(input_dir: str, output_dir: str):
    """Run a test of the multiplex registration pipeline."""
    
    print("🚀 Testing Multiplex Registration Pipeline")
    print("=" * 50)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return False
    
    try:
        # Configure the pipeline
        config = RegistrationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            he_suffix="_HE.tif",
            orion_suffix="_Orion.tif",
            patch_size=256,
            stride=256,
            num_workers=1,  # Use single worker for testing
            max_failures_before_stop=5
        )
        
        print(f"📂 Input directory: {input_dir}")
        print(f"📁 Output directory: {output_dir}")
        print()
        
        # Create and run pipeline
        pipeline = RegistrationPipelineMultiplex(config)
        results = pipeline.run()
        
        print("\n🎉 Pipeline Results:")
        print(f"  Total pairs: {results['total_image_pairs']}")
        print(f"  Successful registrations: {results['successful_registrations']}")
        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Training pairs directory: {results['training_pairs_directory']}")
        print()
        
        # Test the generated patches
        if results['training_pairs_directory']:
            return test_multiplex_patches(results['training_pairs_directory'])
        else:
            print("❌ No training pairs generated")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False

def main():
    """Main function to run tests."""
    
    # Option 1: Test existing patches (if you already have them)
    existing_pairs_dir = "./registration_output_multiplex/training_pairs_multiplex"
    if os.path.exists(existing_pairs_dir):
        print("🔍 Testing existing patches...")
        test_multiplex_patches(existing_pairs_dir)
        print()
    
    # Option 2: Run full pipeline test (uncomment and update paths)
    # test_input_dir = "/path/to/your/test/images"
    # test_output_dir = "./test_registration_output"
    # 
    # if os.path.exists(test_input_dir):
    #     run_test_pipeline(test_input_dir, test_output_dir)

if __name__ == "__main__":
    main() 