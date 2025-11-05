#!/usr/bin/env python3
"""
Test script for the enhanced VALIS pairing pipeline.
This script provides a simple way to test the pipeline with different parameters.
"""

import subprocess
import sys
from pathlib import Path

def run_valis_test():
    """Run the enhanced VALIS pipeline with optimized parameters."""
    
    # Define paths - adjust these for your system
    he_path = "data/raw/TA118-HEraw.ome.tiff"
    orion_path = "data/raw/TA118-Orionraw.ome.tiff"
    output_dir = "paired_dataset_valis_enhanced"
    
    # Build command with enhanced parameters
    cmd = [
        "python", "valis_pairing_pipeline.py",
        "--he", he_path,
        "--orion", orion_path, 
        "--out_dir", output_dir,
        "--patch_size", "2048",
        "--max_image_dim", "1024",
        "--max_processed_dim", "512", 
        "--max_non_rigid_dim", "2048",
        "--max_cores", "300",  # Limit to 300 cores as requested
        "--max_dist_factor", "5.0",  # More lenient distance factor
        "--auto_non_rigid",  # Automatically try non-rigid if needed
        "--create_overview",  # Generate visualization
        "--verbose"
    ]
    
    print("Running enhanced VALIS pipeline...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("=" * 80)
        print("Pipeline completed successfully!")
        
        # Print summary of outputs
        output_path = Path(output_dir)
        if output_path.exists():
            print(f"\nOutput directory: {output_path}")
            
            # Check for key output files
            key_files = [
                "paired_core_info.csv",
                "registration_qc/registration_overview.png",
                "registration_qc/registration_overview_with_pairs.png",
                "patches/",
                "qc_overlays/"
            ]
            
            for file_path in key_files:
                full_path = output_path / file_path
                if full_path.exists():
                    if full_path.is_dir():
                        count = len(list(full_path.glob("*")))
                        print(f"✓ {file_path} ({count} files)")
                    else:
                        print(f"✓ {file_path}")
                else:
                    print(f"✗ {file_path} (missing)")
        
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    run_valis_test()
