#!/usr/bin/env python3
"""
Registration Pipeline Runner

This script provides a simple interface to run the registration pipeline
for preparing training datasets for H&E to multiplex protein prediction.

Usage:
    python run_registration.py --input_dir /path/to/images --output_dir ./output
    python run_registration.py --config registration_config.yaml
"""

import argparse
import yaml
import pathlib
import sys
from registration_pipeline import RegistrationConfig, RegistrationPipeline


def load_config_from_yaml(config_path: str) -> RegistrationConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return RegistrationConfig(**config_dict)


def create_default_config(output_path: str = "registration_config.yaml"):
    """Create a default configuration file."""
    config = RegistrationConfig(
        input_dir="/path/to/your/image/pairs",
        output_dir="./registration_output",
        he_suffix="_HE.tif",
        orion_suffix="_Orion.tif",
        patch_size=256,
        stride=256,
        num_workers=4
    )
    
    # Convert to dictionary
    config_dict = {
        'input_dir': config.input_dir,
        'output_dir': config.output_dir,
        'he_suffix': config.he_suffix,
        'orion_suffix': config.orion_suffix,
        'max_processed_image_dim_px': config.max_processed_image_dim_px,
        'max_non_rigid_registration_dim_px': config.max_non_rigid_registration_dim_px,
        'reference_img': config.reference_img,
        'patch_size': config.patch_size,
        'stride': config.stride,
        'min_background_threshold': config.min_background_threshold,
        'min_ssim_threshold': config.min_ssim_threshold,
        'min_ncc_threshold': config.min_ncc_threshold,
        'min_mi_threshold': config.min_mi_threshold,
        'num_workers': config.num_workers,
        'compression': config.compression,
        'compression_quality': config.compression_quality,
        'save_ome_tiff': config.save_ome_tiff,
        'save_npy_pairs': config.save_npy_pairs,
        'save_quality_plots': config.save_quality_plots
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to {output_path}")
    print("Please edit the configuration file and update the paths before running.")


def main():
    parser = argparse.ArgumentParser(description="Run registration pipeline for H&E to multiplex protein prediction")
    parser.add_argument("--input_dir", type=str, help="Directory containing H&E and Orion image pairs")
    parser.add_argument("--output_dir", type=str, default="./registration_output", 
                       help="Output directory for registration results")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--create_config", type=str, help="Create default configuration file")
    parser.add_argument("--he_suffix", type=str, default="_HE.tif", help="Suffix for H&E images")
    parser.add_argument("--orion_suffix", type=str, default="_Orion.tif", help="Suffix for Orion images")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of training patches")
    parser.add_argument("--stride", type=int, default=256, help="Stride for patch extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--max_failures", type=int, default=50, help="Maximum failures before stopping")
    parser.add_argument("--analyze_first", action="store_true", help="Analyze images before registration")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without running")
    
    args = parser.parse_args()
    
    # Handle configuration file creation
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # Load configuration
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        if not args.input_dir:
            print("Error: Either --input_dir or --config must be specified")
            parser.print_help()
            sys.exit(1)
        
        config = RegistrationConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            he_suffix=args.he_suffix,
            orion_suffix=args.orion_suffix,
            patch_size=args.patch_size,
            stride=args.stride,
            num_workers=args.num_workers
        )
    
    # Validate input directory
    input_path = pathlib.Path(config.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {config.input_dir} does not exist")
        sys.exit(1)
    
    # Find image pairs
    he_files = list(input_path.glob(f"*{config.he_suffix}"))
    print(f"Found {len(he_files)} H&E images in {config.input_dir}")
    
    if args.dry_run:
        print("\nImage pairs that would be processed:")
        for he_file in he_files:
            core_id = he_file.stem.replace(config.he_suffix, "")
            orion_file = input_path / f"{core_id}{config.orion_suffix}"
            status = "✓" if orion_file.exists() else "✗"
            print(f"  {status} {he_file.name} <-> {orion_file.name}")
        
        if not he_files:
            print("  No H&E images found with the specified suffix")
        return
    
    # Create and run pipeline
    try:
        pipeline = RegistrationPipeline(config)
        results = pipeline.run()
        
        print("\n" + "="*50)
        print("REGISTRATION PIPELINE COMPLETED")
        print("="*50)
        print(f"Total image pairs: {results['total_image_pairs']}")
        print(f"Successful registrations: {results['successful_registrations']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Quality metrics:")
        print(f"  Mean SSIM: {results['quality_metrics']['mean_ssim']:.4f}")
        print(f"  Mean NCC: {results['quality_metrics']['mean_ncc']:.4f}")
        print(f"  Mean Mutual Info: {results['quality_metrics']['mean_mutual_info']:.4f}")
        print(f"  Passed quality thresholds: {results['quality_metrics']['passed_quality_thresholds']}")
        print(f"\nOutput directory: {results['output_directory']}")
        if results['training_pairs_directory']:
            print(f"Training pairs: {results['training_pairs_directory']}")
        
        print("\nNext steps:")
        print("1. Check quality plots in quality_plots/ directory")
        print("2. Review registration_quality.csv for detailed metrics")
        print("3. Use training pairs for model training")
        
    except Exception as e:
        print(f"Error running registration pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 