#!/usr/bin/env python3
"""
Example Registration Pipeline Usage

This script demonstrates how to use the registration pipeline with your existing
project structure and data. It shows different ways to configure and run the
pipeline for H&E to multiplex protein prediction.
"""

import os
import pathlib
import sys
from registration_pipeline import RegistrationConfig, RegistrationPipeline
from registration_dataset import create_data_loaders, integrate_with_rosie


def example_basic_registration():
    """Example 1: Basic registration with default settings."""
    print("="*60)
    print("EXAMPLE 1: Basic Registration")
    print("="*60)
    
    # Configure the pipeline
    config = RegistrationConfig(
        input_dir="./example_data/image_pairs",  # Update this path
        output_dir="./registration_output",
        he_suffix="_HE.tif",
        orion_suffix="_Orion.tif",
        patch_size=256,
        stride=256,
        num_workers=4
    )
    
    # Run the pipeline
    try:
        pipeline = RegistrationPipeline(config)
        results = pipeline.run()
        
        print(f"✓ Registration completed successfully!")
        print(f"  Total pairs: {results['total_image_pairs']}")
        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Training pairs: {results['training_pairs_directory']}")
        
    except Exception as e:
        print(f"✗ Registration failed: {e}")
        print("  Make sure you have image pairs in the specified directory")


def example_advanced_registration():
    """Example 2: Advanced registration with custom settings."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Advanced Registration")
    print("="*60)
    
    # Advanced configuration
    config = RegistrationConfig(
        input_dir="./example_data/image_pairs",  # Update this path
        output_dir="./registration_output_advanced",
        he_suffix="_HE.tif",
        orion_suffix="_Orion.tif",
        patch_size=512,  # Larger patches
        stride=256,      # Overlapping patches
        num_workers=8,   # More parallel workers
        max_processed_image_dim_px=2048,  # Higher resolution
        min_background_threshold=20,      # Higher threshold
        min_ssim_threshold=0.4,           # Stricter quality control
        min_ncc_threshold=0.3,
        min_mi_threshold=0.6,
        save_ome_tiff=True,
        save_npy_pairs=True,
        save_quality_plots=True
    )
    
    # Run the pipeline
    try:
        pipeline = RegistrationPipeline(config)
        results = pipeline.run()
        
        print(f"✓ Advanced registration completed!")
        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Quality metrics:")
        print(f"    Mean SSIM: {results['quality_metrics']['mean_ssim']:.4f}")
        print(f"    Mean NCC: {results['quality_metrics']['mean_ncc']:.4f}")
        print(f"    Passed quality thresholds: {results['quality_metrics']['passed_quality_thresholds']}")
        
    except Exception as e:
        print(f"✗ Advanced registration failed: {e}")


def example_dataset_creation():
    """Example 3: Create training datasets from registered images."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Training Dataset Creation")
    print("="*60)
    
    pairs_dir = "./registration_output/training_pairs"
    
    if not os.path.exists(pairs_dir):
        print(f"✗ Training pairs directory not found: {pairs_dir}")
        print("  Run registration pipeline first to generate training pairs")
        return
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            pairs_dir=pairs_dir,
            batch_size=16,
            train_split=0.8,
            val_split=0.1,
            num_workers=2
        )
        
        print(f"✓ Training datasets created successfully!")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        
        # Test loading a batch
        print("\n  Testing data loading...")
        for batch_idx, (he_batch, orion_batch) in enumerate(train_loader):
            print(f"    Batch {batch_idx}: H&E {he_batch.shape}, Orion {orion_batch.shape}")
            if batch_idx >= 2:  # Just test first few batches
                break
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")


def example_rosie_integration():
    """Example 4: Integration with ROSIE training pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 4: ROSIE Integration")
    print("="*60)
    
    pairs_dir = "./registration_output/training_pairs"
    
    if not os.path.exists(pairs_dir):
        print(f"✗ Training pairs directory not found: {pairs_dir}")
        return
    
    try:
        # ROSIE configuration (from your existing train.py)
        rosie_config = {
            'BATCH_SIZE': 32,
            'PATCH_SIZE': 256,
            'NUM_WORKERS': 4,
            'LEARNING_RATE': 1e-4,
            'EVAL_INTERVAL': 3000,
            'PATIENCE': 75000
        }
        
        # Create data loaders compatible with ROSIE
        train_loader, val_loader, test_loader = integrate_with_rosie(
            pairs_dir=pairs_dir,
            rosie_config=rosie_config
        )
        
        print(f"✓ ROSIE integration successful!")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Show how to use in training loop
        print("\n  Example training loop integration:")
        print("  ```python")
        print("  # In your training script")
        print("  for epoch in range(num_epochs):")
        print("      for batch_idx, (he_batch, orion_batch) in enumerate(train_loader):")
        print("          # he_batch: [B, 3, H, W] - H&E images")
        print("          # orion_batch: [B, 1, H, W] - Orion images")
        print("          # ... your training code here")
        print("  ```")
        
    except Exception as e:
        print(f"✗ ROSIE integration failed: {e}")


def example_quality_control():
    """Example 5: Quality control and analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Quality Control")
    print("="*60)
    
    quality_file = "./registration_output/registration_quality.csv"
    
    if not os.path.exists(quality_file):
        print(f"✗ Quality file not found: {quality_file}")
        print("  Run registration pipeline first to generate quality metrics")
        return
    
    try:
        import pandas as pd
        
        # Load quality metrics
        quality_df = pd.read_csv(quality_file)
        
        print(f"✓ Quality analysis:")
        print(f"  Total cores: {len(quality_df)}")
        print(f"  Successful registrations: {quality_df['success'].sum() if 'success' in quality_df.columns else 'N/A'}")
        
        if 'ssim' in quality_df.columns:
            print(f"  Mean SSIM: {quality_df['ssim'].mean():.4f}")
            print(f"  SSIM range: [{quality_df['ssim'].min():.4f}, {quality_df['ssim'].max():.4f}]")
        
        if 'ncc' in quality_df.columns:
            print(f"  Mean NCC: {quality_df['ncc'].mean():.4f}")
            print(f"  NCC range: [{quality_df['ncc'].min():.4f}, {quality_df['ncc'].max():.4f}]")
        
        # Show failed registrations
        if 'error' in quality_df.columns:
            failed = quality_df[quality_df['error'].notna()]
            if len(failed) > 0:
                print(f"\n  Failed registrations ({len(failed)}):")
                for _, row in failed.head(3).iterrows():
                    print(f"    {row['core_id']}: {row['error']}")
        
        print(f"\n  Quality plots saved to: ./registration_output/quality_plots/")
        
    except Exception as e:
        print(f"✗ Quality analysis failed: {e}")


def create_sample_data_structure():
    """Create sample data structure for testing."""
    print("\n" + "="*60)
    print("CREATING SAMPLE DATA STRUCTURE")
    print("="*60)
    
    sample_dir = pathlib.Path("./example_data/image_pairs")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created sample directory: {sample_dir}")
    print(f"  Please add your image pairs with the following naming convention:")
    print(f"    core001_HE.tif")
    print(f"    core001_Orion.tif")
    print(f"    core002_HE.tif")
    print(f"    core002_Orion.tif")
    print(f"    ...")
    print(f"  Then update the input_dir paths in the examples above.")


def main():
    """Run all examples."""
    print("REGISTRATION PIPELINE EXAMPLES")
    print("="*60)
    print("This script demonstrates how to use the registration pipeline")
    print("for H&E to multiplex protein prediction.")
    print()
    
    # Create sample data structure
    create_sample_data_structure()
    
    # Run examples (commented out since we don't have actual data)
    print("\n" + "="*60)
    print("EXAMPLE RUNS (commented out - requires actual data)")
    print("="*60)
    
    print("To run the examples:")
    print("1. Add your image pairs to ./example_data/image_pairs/")
    print("2. Uncomment the example functions below")
    print("3. Run: python example_registration.py")
    
    # Uncomment these lines when you have actual data:
    # example_basic_registration()
    # example_advanced_registration()
    # example_dataset_creation()
    # example_rosie_integration()
    # example_quality_control()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Install dependencies: pip install -r requirements_registration.txt")
    print("2. Add your image pairs to the example_data directory")
    print("3. Run the registration pipeline")
    print("4. Use the generated training pairs for model training")
    print("5. Integrate with your existing ROSIE training pipeline")


if __name__ == "__main__":
    main() 