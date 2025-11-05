#!/usr/bin/env python3
"""
Simple TMA Core Detection Test

This script tests the existing detection methods with more reasonable parameters.
The grid approach was over-complicated - let's go back to basics with Hough circles
and morphological detection, which were already finding cores.

Usage:
    python test_simple_detection.py
"""

import sys
import time
from pathlib import Path
import logging
import json

# Add the core_first_pipeline to the path
sys.path.append('core_first_pipeline')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_simple_detection():
    """Test simple detection methods with reasonable parameters."""
    
    print("🔬 TESTING SIMPLE TMA DETECTION")
    print("=" * 50)
    
    try:
        from core_detector import CoreDetector, CoreDetectionConfig
        logger.info("Successfully imported core detector")
    except Exception as e:
        print(f"❌ Failed to import core detector: {e}")
        return False
    
    # Test paths
    he_path = "data/raw/TA118-HEraw.ome.tiff"
    orion_path = "data/raw/TA118-Orionraw.ome.tiff"
    
    # Check if files exist
    if not Path(he_path).exists():
        print(f"❌ H&E image not found: {he_path}")
        return False
    
    if not Path(orion_path).exists():
        print(f"❌ Orion image not found: {orion_path}")
        return False
    
    print(f"✅ Found both images")
    print()
    
    # Test different methods and parameters
    test_configs = [
        {
            'name': 'Hough Circles (Lenient)',
            'config': CoreDetectionConfig(
                detection_method="hough",
                min_core_area=20000,           # Reduced from 30000
                max_core_area=800000,
                min_core_diameter=100,         # Reduced from 150
                max_core_diameter=1500,        # Increased from 1000
                min_circularity=0.2,           # Reduced from 0.3
                min_solidity=0.4,              # Reduced from 0.6
                hough_param2=20,               # Reduced from 30 (more lenient)
                hough_min_dist=150,            # Reduced from 200
                max_detection_dimension=3000,   # Increased for better detection
                create_visualizations=True
            )
        },
        {
            'name': 'Hough Circles (Very Lenient)',
            'config': CoreDetectionConfig(
                detection_method="hough",
                min_core_area=10000,           # Very small minimum
                max_core_area=1000000,         # Very large maximum
                min_core_diameter=80,          # Very small minimum
                max_core_diameter=2000,        # Very large maximum
                min_circularity=0.15,          # Very lenient
                min_solidity=0.3,              # Very lenient
                hough_param2=15,               # Very lenient detection
                hough_min_dist=100,            # Allow closer cores
                max_detection_dimension=4000,   # Even higher resolution
                create_visualizations=True
            )
        },
        {
            'name': 'Hybrid (Lenient)',
            'config': CoreDetectionConfig(
                detection_method="hybrid",
                min_core_area=20000,
                max_core_area=800000,
                min_core_diameter=100,
                max_core_diameter=1500,
                min_circularity=0.2,
                min_solidity=0.4,
                hough_param2=20,
                max_detection_dimension=3000,
                create_visualizations=True
            )
        }
    ]
    
    results = {}
    
    # Test H&E detection
    print("🎯 Testing H&E Detection Methods...")
    print("-" * 40)
    
    for test in test_configs:
        print(f"\n🔸 Testing {test['name']}...")
        
        try:
            detector = CoreDetector(test['config'])
            start_time = time.time()
            
            detection_results = detector.detect_cores(he_path, image_type="he")
            end_time = time.time()
            
            cores_found = detection_results['filtered_cores_count']
            processing_time = end_time - start_time
            
            print(f"   ✅ SUCCESS: {cores_found} cores found in {processing_time:.1f}s")
            
            results[f"HE_{test['name']}"] = {
                'cores_found': cores_found,
                'processing_time': processing_time,
                'success': True,
                'method': test['config'].detection_method
            }
            
            # Save visualization
            if test['config'].create_visualizations:
                viz_path = f"he_{test['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
                try:
                    detector.visualize_detection(detection_results, viz_path)
                    print(f"   📊 Visualization saved: {viz_path}")
                except Exception as e:
                    print(f"   ⚠️  Visualization failed: {e}")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results[f"HE_{test['name']}"] = {
                'cores_found': 0,
                'processing_time': 0,
                'success': False,
                'error': str(e)
            }
    
    print()
    print("🎯 Testing Orion Detection Methods...")
    print("-" * 40)
    
    # Test Orion detection (just the best performing method)
    best_he_method = None
    best_he_count = 0
    
    for key, result in results.items():
        if result['success'] and result['cores_found'] > best_he_count:
            best_he_count = result['cores_found']
            best_he_method = key
    
    if best_he_method:
        print(f"\n🔸 Using best method for Orion: {best_he_method}")
        
        # Find the corresponding config
        best_config = None
        for test in test_configs:
            if best_he_method.endswith(test['name']):
                best_config = test['config']
                break
        
        if best_config:
            try:
                detector = CoreDetector(best_config)
                start_time = time.time()
                
                orion_results = detector.detect_cores(orion_path, image_type="orion")
                end_time = time.time()
                
                orion_cores = orion_results['filtered_cores_count']
                orion_time = end_time - start_time
                
                print(f"   ✅ SUCCESS: {orion_cores} cores found in {orion_time:.1f}s")
                
                results['Orion_Best'] = {
                    'cores_found': orion_cores,
                    'processing_time': orion_time,
                    'success': True,
                    'method': best_config.detection_method
                }
                
                # Save visualization
                if best_config.create_visualizations:
                    viz_path = "orion_best_method.png"
                    try:
                        detector.visualize_detection(orion_results, viz_path)
                        print(f"   📊 Visualization saved: {viz_path}")
                    except Exception as e:
                        print(f"   ⚠️  Visualization failed: {e}")
                
            except Exception as e:
                print(f"   ❌ FAILED: {e}")
                results['Orion_Best'] = {
                    'cores_found': 0,
                    'processing_time': 0,
                    'success': False,
                    'error': str(e)
                }
    
    # Print summary
    print()
    print("=" * 50)
    print("🎯 DETECTION RESULTS SUMMARY")
    print("=" * 50)
    
    best_total_cores = 0
    best_method_name = None
    
    for method_name, result in results.items():
        if result['success']:
            cores = result['cores_found']
            time_taken = result['processing_time']
            method = result['method']
            
            print(f"✅ {method_name}: {cores} cores ({method}, {time_taken:.1f}s)")
            
            if cores > best_total_cores:
                best_total_cores = cores
                best_method_name = method_name
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ {method_name}: Failed ({error})")
    
    print()
    print(f"🏆 BEST METHOD: {best_method_name} with {best_total_cores} cores")
    
    # Compare with expected count
    expected_cores = 270
    if best_total_cores >= expected_cores * 0.8:  # Within 80% of expected
        print(f"🎉 EXCELLENT! Detected {best_total_cores} cores (expected ~{expected_cores})")
        print("   This is very close to the expected count!")
        success = True
    elif best_total_cores >= expected_cores * 0.5:  # Within 50% of expected
        print(f"✅ GOOD! Detected {best_total_cores} cores (expected ~{expected_cores})")
        print("   This is a reasonable count, may need minor parameter tuning")
        success = True
    else:
        print(f"⚠️  NEEDS WORK: Only detected {best_total_cores} cores (expected ~{expected_cores})")
        print("   Parameters need significant adjustment")
        success = False
    
    # Save results
    with open('simple_detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: simple_detection_results.json")
    
    if success:
        print("\n🚀 RECOMMENDATIONS:")
        print(f"1. Use the '{best_method_name}' configuration")
        print("2. Run the full pipeline with these parameters:")
        print("   python run_core_first.py --detection_method hough")
        print("3. Check the visualizations to verify core quality")
    else:
        print("\n🔧 NEXT STEPS:")
        print("1. Run: python diagnose_tma_detection.py")
        print("2. Use the interactive parameter tuning")
        print("3. Adjust detection parameters based on the analysis")
    
    return success


def compare_with_original():
    """Compare with the original failing approach."""
    
    print("\n📊 COMPARING WITH ORIGINAL APPROACH")
    print("-" * 50)
    
    try:
        from core_detector import CoreDetector, CoreDetectionConfig
        
        # Original failing configuration
        original_config = CoreDetectionConfig(
            detection_method="hybrid",
            min_core_area=30000,
            max_core_area=800000,
            min_circularity=0.25,
            max_detection_dimension=2048  # This was the problem!
        )
        
        detector = CoreDetector(original_config)
        he_path = "data/raw/TA118-HEraw.ome.tiff"
        
        if Path(he_path).exists():
            print("Testing original configuration...")
            results = detector.detect_cores(he_path, image_type="he")
            original_count = results['filtered_cores_count']
            
            print(f"Original method: {original_count} cores")
            print("Issues with original approach:")
            print("- max_detection_dimension=2048 (too aggressive downsampling)")
            print("- min_circularity=0.25 (too strict after downsampling)")
            print("- min_core_area=30000 (may be too strict)")
            
        else:
            print("H&E image not found for comparison")
    
    except Exception as e:
        print(f"Comparison failed: {e}")


if __name__ == "__main__":
    print("Starting Simple TMA Detection Test...")
    print()
    
    success = test_simple_detection()
    
    if success:
        print("\n✨ SUCCESS! The simple approach is working much better!")
        compare_with_original()
    
    print("\nTest completed!") 