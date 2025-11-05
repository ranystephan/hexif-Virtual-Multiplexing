#!/usr/bin/env python3
"""
Run Core-First TMA Processing Pipeline

Simple command-line script to run the core-first pipeline on your TA118 data.
This will detect, match, and extract tissue cores from H&E and Orion images.

Usage:
    python run_core_first.py
    
    # Or with custom paths:
    python run_core_first.py --he_path path/to/he.tiff --orion_path path/to/orion.tiff

Requirements:
- Place your images in data/raw/ as TA118-HEraw.ome.tiff and TA118-Orionraw.ome.tiff
- Or specify custom paths using command line arguments
"""

import sys
import argparse
import time
from pathlib import Path

# Add the core_first_pipeline to the path
sys.path.append('core_first_pipeline')


def main():
    """Run the core-first pipeline with command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Run Core-First TMA Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_core_first.py                                    # Use default paths
  python run_core_first.py --output_dir custom_output        # Custom output
  python run_core_first.py --he_path he.tiff --orion_path orion.tiff  # Custom images
        """
    )
    
    parser.add_argument(
        "--he_path", 
        default="data/raw/TA118-HEraw.ome.tiff",
        help="Path to H&E image (default: data/raw/TA118-HEraw.ome.tiff)"
    )
    
    parser.add_argument(
        "--orion_path",
        default="data/raw/TA118-Orionraw.ome.tiff", 
        help="Path to Orion image (default: data/raw/TA118-Orionraw.ome.tiff)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="core_first_output",
        help="Output directory (default: core_first_output)"
    )
    
    parser.add_argument(
        "--min_cores",
        type=int,
        default=5,
        help="Minimum number of cores required (default: 5)"
    )
    
    parser.add_argument(
        "--no_visualizations",
        action="store_true",
        help="Skip creating visualizations (faster processing)"
    )
    
    parser.add_argument(
        "--detection_method",
        choices=["morphology", "hough", "contours", "hybrid"],
        default="hybrid",
        help="Core detection method (default: hybrid)"
    )
    
    parser.add_argument(
        "--matching_method", 
        choices=["hungarian", "nearest_neighbor", "greedy"],
        default="hungarian",
        help="Core matching method (default: hungarian)"
    )
    
    args = parser.parse_args()
    
    print("🚀 CORE-FIRST TMA PROCESSING PIPELINE")
    print("=" * 50)
    print(f"H&E Image: {args.he_path}")
    print(f"Orion Image: {args.orion_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Detection Method: {args.detection_method}")
    print(f"Matching Method: {args.matching_method}")
    print()
    
    # Check if input files exist
    he_path = Path(args.he_path)
    orion_path = Path(args.orion_path)
    
    if not he_path.exists():
        print(f"❌ H&E image not found: {he_path}")
        print("💡 Make sure your H&E image is at the specified path")
        return False
    
    if not orion_path.exists():
        print(f"❌ Orion image not found: {orion_path}")
        print("💡 Make sure your Orion image is at the specified path")
        return False
    
    # Check image properties
    try:
        from tifffile import imread
        
        he_img = imread(str(he_path))
        orion_img = imread(str(orion_path))
        
        print(f"✅ H&E loaded: {he_img.shape}, {he_img.dtype}")
        print(f"✅ Orion loaded: {orion_img.shape}, {orion_img.dtype}")
        
        if orion_img.ndim == 3 and orion_img.shape[0] > 10:
            print(f"📊 Orion has {orion_img.shape[0]} channels - all will be preserved")
        
        print()
        
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return False
    
    # Import and configure pipeline
    try:
        from core_first_pipeline import CoreFirstPipeline, CoreFirstPipelineConfig
        from core_first_pipeline import CoreDetectionConfig, CoreMatchingConfig
        
        # Configure detection
        detection_config = CoreDetectionConfig(
            detection_method=args.detection_method,
            min_core_area=30000,
            max_core_area=800000,
            min_circularity=0.25,
            create_visualizations=not args.no_visualizations
        )
        
        # Configure matching
        matching_config = CoreMatchingConfig(
            matching_method=args.matching_method,
            max_distance_threshold=300.0,
            min_match_confidence=0.4,
            save_visualizations=not args.no_visualizations
        )
        
        # Configure main pipeline
        config = CoreFirstPipelineConfig(
            he_image_path=str(he_path),
            orion_image_path=str(orion_path),
            output_dir=args.output_dir,
            create_visualizations=not args.no_visualizations,
            save_intermediate_results=True,
            create_paired_cores=True,
            min_cores_required=args.min_cores,
            detection_config=detection_config,
            matching_config=matching_config
        )
        
    except Exception as e:
        print(f"❌ Error importing pipeline: {e}")
        print("💡 Make sure you're running from the project root directory")
        return False
    
    # Run pipeline
    print("🎯 Starting Core-First Processing...")
    print("-" * 30)
    
    start_time = time.time()
    
    try:
        pipeline = CoreFirstPipeline(config)
        results = pipeline.run()
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Report results
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print("🎯 PROCESSING COMPLETE")
    print("=" * 50)
    
    if results['success']:
        # Extract key metrics
        detection = results['processing_stages']['detection']
        matching = results['processing_stages']['matching']
        extraction = results['processing_stages']['extraction']
        
        he_cores = detection['he_detection']['filtered_cores_count']
        orion_cores = detection['orion_detection']['filtered_cores_count']
        matched_pairs = matching['matching_data']['high_quality_matches']
        
        print(f"✅ SUCCESS! Pipeline completed in {total_time:.1f} seconds")
        print()
        print(f"📊 RESULTS SUMMARY:")
        print(f"   • H&E cores detected: {he_cores}")
        print(f"   • Orion cores detected: {orion_cores}")
        print(f"   • Successfully matched pairs: {matched_pairs}")
        print(f"   • Match rate: {matched_pairs/min(he_cores, orion_cores)*100:.1f}%")
        print()
        print(f"📁 OUTPUT DIRECTORY: {args.output_dir}")
        print(f"   • Individual cores: {args.output_dir}/extracted_cores/")
        print(f"   • Paired core info: {args.output_dir}/paired_cores.csv")
        print(f"   • Full report: {args.output_dir}/pipeline_report.md")
        
        if not args.no_visualizations:
            print(f"   • Visualizations: {args.output_dir}/visualizations/")
        
        print()
        print("🎉 NEXT STEPS:")
        print("1. Review the visualizations to check quality")
        print("2. Use extracted cores for downstream analysis")
        print("3. Integrate with your existing ML/analysis workflows")
        
        # Show channel info if available
        orion_extraction = extraction['orion_extraction']
        if orion_extraction['channel_info']:
            channel_count = orion_extraction['channel_info']['num_channels']
            print(f"4. All {channel_count} Orion channels have been preserved!")
        
        return True
        
    else:
        print(f"❌ PIPELINE FAILED: {results.get('error', 'Unknown error')}")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("1. Check that your images are valid TMA images")
        print("2. Try adjusting detection parameters (lower min_core_area)")
        print("3. Review the error message above for specific issues")
        print("4. Run 'python test_core_first.py' for debugging")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 