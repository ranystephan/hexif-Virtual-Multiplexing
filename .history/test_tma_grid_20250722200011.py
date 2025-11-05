#!/usr/bin/env python3
"""
Test TMA Grid-Based Detection

This script tests the new TMA grid-based detection approach to see if we can
detect the expected ~270 cores that the user knows exist in their TMA.

Usage:
    python test_tma_grid.py
"""

import sys
import time
from pathlib import Path
import logging

# Add the core_first_pipeline to the path
sys.path.append('core_first_pipeline')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tma_grid_detection():
    """Test the TMA grid detection on both H&E and Orion images."""
    
    print("🔬 TESTING TMA GRID-BASED DETECTION")
    print("=" * 50)
    
    # Import modules
    try:
        from tma_grid_detector import TMAGridDetector, TMAGridConfig
        logger.info("Successfully imported TMA grid detector")
    except Exception as e:
        print(f"❌ Failed to import TMA grid detector: {e}")
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
    
    print(f"✅ Found both images:")
    print(f"   H&E: {he_path}")
    print(f"   Orion: {orion_path}")
    print()
    
    # Create configuration
    config = TMAGridConfig(
        expected_core_diameter_range=(800, 1200),  # Adjust based on your TMA
        expected_spacing_range=(1000, 1600),       # Adjust based on your TMA
        grid_detection_downsample=8,               # Balance between speed and accuracy
        template_match_threshold=0.3,              # More lenient for better coverage
        autocorr_peak_threshold=0.2,               # Lower threshold for more peaks
        enable_visualizations=True,
        save_debug_images=True
    )
    
    # Create detector
    detector = TMAGridDetector(config)
    print("✅ TMA grid detector initialized")
    print()
    
    # Test H&E detection
    print("🎯 Testing H&E Grid Detection...")
    print("-" * 30)
    
    start_time = time.time()
    try:
        he_results = detector.detect_tma_grid(he_path, image_type="he")
        he_time = time.time() - start_time
        
        if he_results['success']:
            print(f"✅ H&E Detection SUCCESS!")
            print(f"   Cores detected: {len(he_results['cores'])}")
            print(f"   Processing time: {he_time:.1f} seconds")
            print(f"   Grid parameters: {he_results.get('grid_parameters', 'N/A')}")
            
            # Create visualization
            if config.enable_visualizations:
                viz_path = "he_tma_grid_detection.png"
                detector.visualize_grid_detection(he_results, viz_path)
                print(f"   Visualization saved: {viz_path}")
        else:
            print(f"❌ H&E Detection FAILED")
            print(f"   Error: {he_results.get('error', 'Unknown error')}")
            print(f"   Processing time: {he_time:.1f} seconds")
            
    except Exception as e:
        print(f"❌ H&E Detection ERROR: {e}")
        import traceback
        traceback.print_exc()
        he_results = None
    
    print()
    
    # Test Orion detection
    print("🎯 Testing Orion Grid Detection...")
    print("-" * 30)
    
    start_time = time.time()
    try:
        orion_results = detector.detect_tma_grid(orion_path, image_type="orion")
        orion_time = time.time() - start_time
        
        if orion_results['success']:
            print(f"✅ Orion Detection SUCCESS!")
            print(f"   Cores detected: {len(orion_results['cores'])}")
            print(f"   Processing time: {orion_time:.1f} seconds")
            print(f"   Grid parameters: {orion_results.get('grid_parameters', 'N/A')}")
            
            # Create visualization
            if config.enable_visualizations:
                viz_path = "orion_tma_grid_detection.png"
                detector.visualize_grid_detection(orion_results, viz_path)
                print(f"   Visualization saved: {viz_path}")
        else:
            print(f"❌ Orion Detection FAILED")
            print(f"   Error: {orion_results.get('error', 'Unknown error')}")
            print(f"   Processing time: {orion_time:.1f} seconds")
            
    except Exception as e:
        print(f"❌ Orion Detection ERROR: {e}")
        import traceback
        traceback.print_exc()
        orion_results = None
    
    print()
    
    # Test grid-based matching if both detections succeeded
    if (he_results and he_results['success'] and 
        orion_results and orion_results['success']):
        
        print("🎯 Testing Grid-Based Matching...")
        print("-" * 30)
        
        try:
            from tma_grid_matcher import TMAGridMatcher, TMAGridMatchingConfig
            
            # Create matching configuration
            match_config = TMAGridMatchingConfig(
                max_grid_shift=3,
                min_match_confidence=0.3,  # More lenient
                save_visualizations=True
            )
            
            # Create matcher and perform matching
            matcher = TMAGridMatcher(match_config)
            match_results = matcher.match_grid_cores(he_results, orion_results)
            
            if match_results['success']:
                matches = len(match_results['matches'])
                he_cores = len(he_results['cores'])
                orion_cores = len(orion_results['cores'])
                
                print(f"✅ Grid Matching SUCCESS!")
                print(f"   Matched pairs: {matches}")
                print(f"   H&E match rate: {matches/he_cores:.1%}")
                print(f"   Orion match rate: {matches/orion_cores:.1%}")
                print(f"   Average distance: {match_results['matching_statistics']['average_distance']:.1f} px")
                print(f"   Average confidence: {match_results['matching_statistics']['average_confidence']:.3f}")
                
                # Create visualization
                if match_config.save_visualizations:
                    viz_path = "tma_grid_matching.png"
                    matcher.visualize_grid_matching(match_results, he_results, orion_results, viz_path)
                    print(f"   Visualization saved: {viz_path}")
            else:
                print(f"❌ Grid Matching FAILED")
                print(f"   Error: {match_results.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"❌ Grid Matching ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 50)
    print("🎯 SUMMARY")
    print("=" * 50)
    
    success = False
    if he_results and he_results['success']:
        he_count = len(he_results['cores'])
        print(f"✅ H&E: {he_count} cores detected")
        
        if he_count >= 200:  # We expect ~270 cores
            print(f"   🎉 Great! Detected {he_count} cores (expected ~270)")
            success = True
        else:
            print(f"   ⚠️  Only {he_count} cores detected (expected ~270)")
    else:
        print("❌ H&E: Detection failed")
    
    if orion_results and orion_results['success']:
        orion_count = len(orion_results['cores'])
        print(f"✅ Orion: {orion_count} cores detected")
        
        if orion_count >= 200:
            print(f"   🎉 Great! Detected {orion_count} cores (expected ~270)")
            if success:  # Both succeeded
                success = True
        else:
            print(f"   ⚠️  Only {orion_count} cores detected (expected ~270)")
    else:
        print("❌ Orion: Detection failed")
    
    if success:
        print("\n🎉 TMA GRID DETECTION IS WORKING!")
        print("   The new approach successfully detects most/all cores")
        print("   You can now use 'python run_core_first.py' with --detection_method tma_grid")
    else:
        print("\n🔧 NEEDS ADJUSTMENT:")
        print("   The grid detection parameters may need tuning")
        print("   Try adjusting expected_core_diameter_range and expected_spacing_range")
        print("   Or lowering template_match_threshold and autocorr_peak_threshold")
    
    return success


def test_traditional_vs_grid():
    """Compare traditional detection vs grid detection."""
    
    print("\n🔄 COMPARING TRADITIONAL VS GRID DETECTION")
    print("=" * 50)
    
    try:
        from core_detector import CoreDetector, CoreDetectionConfig
        
        # Test traditional hybrid method
        print("Testing traditional hybrid detection...")
        
        traditional_config = CoreDetectionConfig(
            detection_method="hybrid",
            min_core_area=30000,
            max_core_area=800000,
            min_circularity=0.25
        )
        
        traditional_detector = CoreDetector(traditional_config)
        
        he_path = "data/raw/TA118-HEraw.ome.tiff"
        if Path(he_path).exists():
            traditional_results = traditional_detector.detect_cores(he_path, image_type="he")
            traditional_count = traditional_results['filtered_cores_count']
            
            print(f"Traditional method: {traditional_count} cores")
        else:
            print("H&E image not found for comparison")
            traditional_count = 0
        
        # Test grid method
        print("Testing grid detection...")
        
        from tma_grid_detector import TMAGridDetector, TMAGridConfig
        
        grid_config = TMAGridConfig(
            expected_core_diameter_range=(800, 1200),
            expected_spacing_range=(1000, 1600),
            template_match_threshold=0.3
        )
        
        grid_detector = TMAGridDetector(grid_config)
        
        if Path(he_path).exists():
            grid_results = grid_detector.detect_tma_grid(he_path, image_type="he")
            
            if grid_results['success']:
                grid_count = len(grid_results['cores'])
            else:
                grid_count = 0
            
            print(f"Grid method: {grid_count} cores")
        
        print(f"\n📊 COMPARISON:")
        print(f"   Traditional: {traditional_count} cores")
        print(f"   Grid-based:  {grid_count} cores")
        print(f"   Improvement: {grid_count - traditional_count:+d} cores ({((grid_count/max(traditional_count, 1)) - 1)*100:+.1f}%)")
        
        if grid_count > traditional_count * 2:
            print("   🎉 HUGE IMPROVEMENT! Grid-based detection is much better!")
        elif grid_count > traditional_count:
            print("   ✅ IMPROVEMENT! Grid-based detection found more cores!")
        else:
            print("   ⚠️  Grid detection needs tuning")
    
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting TMA Grid Detection Tests...")
    print()
    
    # Run main test
    success = test_tma_grid_detection()
    
    # Run comparison if main test worked
    if success:
        test_traditional_vs_grid()
    
    print("\nTest completed!")
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Run: python run_core_first.py --detection_method tma_grid")
        print("2. Check the visualizations to verify quality")
        print("3. Adjust parameters if needed")
        print("4. Proceed with your analysis!")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check the error messages above")
        print("2. Adjust TMAGridConfig parameters")
        print("3. Ensure your images are valid TMA images")
        print("4. Try with different threshold values") 