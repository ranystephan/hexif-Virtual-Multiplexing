# Registration Pipeline for H&E to Multiplex Protein Prediction

This registration pipeline provides a comprehensive solution for aligning H&E images with multiplex protein images (Orion, CODEX, etc.) to prepare training datasets for in-silico staining models.

## Overview

The pipeline uses VALIS (Vectra Automated Lesion Image Segmentation) for robust image registration and provides:

- **Per-core registration** using VALIS with rigid + non-rigid alignment
- **Multi-channel warping** for complete Orion stacks
- **Training dataset preparation** with automatic tiling and patch extraction
- **Quality control** with comprehensive metrics and visualization
- **Integration** with existing ROSIE baseline training pipeline

## Features

### 🎯 Core Functionality
- **VALIS-based registration** with configurable parameters
- **Automatic image pair detection** based on naming conventions
- **Parallel processing** for efficient batch registration
- **Quality assessment** using SSIM, NCC, and mutual information metrics

### 📊 Quality Control
- **Registration quality plots** showing metric distributions
- **Detailed quality reports** in CSV format
- **Configurable quality thresholds** for filtering results
- **Visual overlays** for manual inspection

### 🧩 Training Dataset Preparation
- **Automatic patch extraction** with configurable size and stride
- **Background filtering** to remove empty patches
- **Multiple output formats** (OME-TIFF, NumPy arrays)
- **PyTorch dataset integration** for seamless training

## Installation

### 1. Install Dependencies

```bash
# Install registration-specific requirements
pip install -r requirements_registration.txt

# Or install core dependencies manually
pip install valis opencv-python scikit-image tifffile pandas numpy matplotlib
```

### 2. Verify Installation

```bash
# Test VALIS installation
python -c "from valis import registration; print('VALIS installed successfully')"

# Test the registration pipeline
python -c "from registration_pipeline import RegistrationConfig; print('Pipeline ready')"
```

## Quick Start

### 1. Prepare Your Data

Organize your image pairs in a directory with the following structure:

```
input_directory/
├── core001_HE.tif      # H&E image
├── core001_Orion.tif   # Orion/multiplex image
├── core002_HE.tif
├── core002_Orion.tif
└── ...
```

### 2. Run Registration Pipeline

#### Option A: Command Line Interface

```bash
# Basic usage
python run_registration.py --input_dir /path/to/images --output_dir ./registration_output

# With custom parameters
python run_registration.py \
    --input_dir /path/to/images \
    --output_dir ./registration_output \
    --patch_size 256 \
    --stride 256 \
    --num_workers 4

# Dry run to see what would be processed
python run_registration.py --input_dir /path/to/images --dry_run
```

#### Option B: Configuration File

```bash
# Create default configuration
python run_registration.py --create_config my_config.yaml

# Edit the configuration file, then run
python run_registration.py --config my_config.yaml
```

#### Option C: Python Script

```python
from registration_pipeline import RegistrationConfig, RegistrationPipeline

# Configure the pipeline
config = RegistrationConfig(
    input_dir="/path/to/your/image/pairs",
    output_dir="./registration_output",
    he_suffix="_HE.tif",
    orion_suffix="_Orion.tif",
    patch_size=256,
    stride=256,
    num_workers=4
)

# Run the pipeline
pipeline = RegistrationPipeline(config)
results = pipeline.run()

print(f"Success rate: {results['success_rate']:.2%}")
```

### 3. Check Results

After running the pipeline, you'll find:

```
registration_output/
├── registration_quality.csv      # Quality metrics for each core
├── pipeline_summary.json         # Overall pipeline summary
├── quality_plots/
│   └── registration_quality.png  # Quality control plots
├── registered_images/            # Warped images (if saved)
└── training_pairs/              # Training dataset
    ├── core001_patch_0000_HE.npy
    ├── core001_patch_0000_ORION.npy
    ├── core001_patch_0001_HE.npy
    ├── core001_patch_0001_ORION.npy
    └── ...
```

## Configuration

### RegistrationConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dir` | - | Directory containing image pairs |
| `output_dir` | - | Output directory for results |
| `he_suffix` | `"_HE.tif"` | Suffix for H&E images |
| `orion_suffix` | `"_Orion.tif"` | Suffix for Orion images |
| `patch_size` | `256` | Size of training patches |
| `stride` | `256` | Stride for patch extraction |
| `num_workers` | `4` | Number of parallel workers |
| `max_processed_image_dim_px` | `1024` | Max image dimension for VALIS |
| `min_background_threshold` | `10` | Minimum signal for valid patches |
| `min_ssim_threshold` | `0.3` | Minimum SSIM for quality control |
| `min_ncc_threshold` | `0.2` | Minimum NCC for quality control |
| `min_mi_threshold` | `0.5` | Minimum mutual info for quality control |

### Example Configuration File

```yaml
input_dir: "/path/to/your/image/pairs"
output_dir: "./registration_output"
he_suffix: "_HE.tif"
orion_suffix: "_Orion.tif"
patch_size: 256
stride: 256
num_workers: 4
max_processed_image_dim_px: 1024
max_non_rigid_registration_dim_px: 1500
min_background_threshold: 10
min_ssim_threshold: 0.3
min_ncc_threshold: 0.2
min_mi_threshold: 0.5
save_ome_tiff: true
save_npy_pairs: true
save_quality_plots: true
```

## Integration with ROSIE Training Pipeline

### 1. Use Registration Dataset

```python
from registration_dataset import create_data_loaders, integrate_with_rosie

# Create data loaders for training
train_loader, val_loader, test_loader = create_data_loaders(
    pairs_dir="./registration_output/training_pairs",
    batch_size=32,
    train_split=0.8,
    val_split=0.1,
    num_workers=4
)

# Or integrate with existing ROSIE config
rosie_config = {
    'BATCH_SIZE': 32,
    'PATCH_SIZE': 256,
    'NUM_WORKERS': 4
}

train_loader, val_loader, test_loader = integrate_with_rosie(
    pairs_dir="./registration_output/training_pairs",
    rosie_config=rosie_config
)
```

### 2. Modify ROSIE Training Script

Update your existing `rosie-main/train.py` to use the registration dataset:

```python
# Add to imports
from registration_dataset import RegistrationDataset, get_default_transforms

# Replace dataset creation
input_transform, target_transform = get_default_transforms(
    patch_size=PATCH_SIZE,
    augment=True,
    normalize=True
)

train_dataset = RegistrationDataset(
    pairs_dir="./registration_output/training_pairs",
    transform=input_transform,
    target_transform=target_transform,
    patch_size=PATCH_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)
```

## Quality Control

### Understanding Quality Metrics

1. **SSIM (Structural Similarity Index)**: Measures structural similarity between H&E and warped Orion
   - Range: [-1, 1], higher is better
   - Threshold: ≥ 0.3 recommended

2. **NCC (Normalized Cross-Correlation)**: Measures linear correlation
   - Range: [-1, 1], higher is better
   - Threshold: ≥ 0.2 recommended

3. **Mutual Information**: Measures information shared between images
   - Range: [0, ∞), higher is better
   - Threshold: ≥ 0.5 recommended

### Quality Control Workflow

1. **Run registration pipeline**
2. **Check quality plots** in `quality_plots/registration_quality.png`
3. **Review metrics** in `registration_quality.csv`
4. **Filter results** based on quality thresholds
5. **Manually inspect** failed registrations if needed

### Troubleshooting Poor Registration

- **Increase image resolution**: Reduce `max_processed_image_dim_px`
- **Adjust VALIS parameters**: Modify registration settings
- **Check image quality**: Ensure images are not corrupted
- **Verify image pairs**: Confirm H&E and Orion correspond to same tissue

## Advanced Usage

### Multi-Channel Orion Images

For multi-channel Orion images, use the `MultiChannelRegistrationDataset`:

```python
from registration_dataset import MultiChannelRegistrationDataset

# Select specific channels
dataset = MultiChannelRegistrationDataset(
    pairs_dir="./registration_output/training_pairs",
    orion_channels=[0, 2, 5],  # Use channels 0, 2, and 5
    channel_names=["DAPI", "CD3", "CD8"]
)
```

### Custom Transforms

```python
from torchvision import transforms
from registration_dataset import RegistrationDataset

# Custom transforms
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = RegistrationDataset(
    pairs_dir="./registration_output/training_pairs",
    transform=custom_transform
)
```

### Batch Processing Multiple Datasets

```python
import pathlib
from registration_pipeline import RegistrationConfig, RegistrationPipeline

# Process multiple datasets
datasets = [
    ("dataset1", "/path/to/dataset1"),
    ("dataset2", "/path/to/dataset2"),
    ("dataset3", "/path/to/dataset3")
]

for name, input_dir in datasets:
    config = RegistrationConfig(
        input_dir=input_dir,
        output_dir=f"./registration_output/{name}",
        patch_size=256,
        num_workers=4
    )
    
    pipeline = RegistrationPipeline(config)
    results = pipeline.run()
    
    print(f"{name}: {results['success_rate']:.2%} success rate")
```

## Performance Optimization

### Memory Management

- **Reduce batch size** if running out of memory
- **Use fewer workers** on memory-constrained systems
- **Process in smaller batches** for large datasets

### Speed Optimization

- **Increase num_workers** (up to CPU core count)
- **Use SSD storage** for faster I/O
- **Reduce image resolution** if quality allows
- **Use GPU acceleration** for training (not registration)

### Parallel Processing

The pipeline automatically uses parallel processing for registration:

```python
# Configure parallel processing
config = RegistrationConfig(
    num_workers=8,  # Use 8 parallel workers
    # ... other parameters
)
```

## Troubleshooting

### Common Issues

1. **VALIS Import Error**
   ```bash
   pip install valis
   # If issues persist, try:
   conda install -c conda-forge valis
   ```

2. **Memory Issues**
   - Reduce `max_processed_image_dim_px`
   - Reduce `num_workers`
   - Process smaller batches

3. **Poor Registration Quality**
   - Check image quality and format
   - Verify image pairs correspond to same tissue
   - Adjust VALIS parameters
   - Increase image resolution

4. **No Training Pairs Generated**
   - Check `min_background_threshold`
   - Verify image loading and preprocessing
   - Check file permissions and paths

### Getting Help

1. **Check logs**: Look for error messages in console output
2. **Review quality metrics**: Check `registration_quality.csv`
3. **Inspect intermediate files**: Look in temporary directories
4. **Test with small dataset**: Start with 2-3 image pairs

## Contributing

To contribute to the registration pipeline:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests** for new functionality
5. **Submit a pull request**

## License

This registration pipeline is part of the hexif project and follows the same license terms.

## Citation

If you use this registration pipeline in your research, please cite:

```bibtex
@software{hexif_registration,
  title={Registration Pipeline for H&E to Multiplex Protein Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/hexif}
}
``` 