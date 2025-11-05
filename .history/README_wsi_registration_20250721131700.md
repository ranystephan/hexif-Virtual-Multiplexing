# WSI Registration and Core Extraction Pipeline

This pipeline provides a comprehensive solution for registering whole slide images (WSI) of Tissue Microarrays (TMAs) and automatically extracting paired tissue cores. It's specifically designed for H&E to multiplex protein prediction workflows where you have full TMA slides rather than individual core images.

## Features

- **Full WSI Registration**: Uses VALIS to register complete TMA slides between H&E and multiplex (Orion/CODEX) images
- **Automated Core Detection**: Intelligent tissue core segmentation using morphological analysis
- **Spatial Core Pairing**: Robust matching of cores between registered slides based on spatial correspondence
- **Organized Output**: Creates structured folders with paired cores ready for downstream analysis
- **Quality Control**: Comprehensive visualization and metrics for registration and core detection quality
- **Multi-channel Support**: Handles multi-channel protein images while preserving all channel information

## Requirements

Install the required dependencies:

```bash
pip install valis
pip install tifffile
pip install scikit-image
pip install opencv-python
pip install matplotlib
pip install pandas
pip install numpy
```

## Usage

### Basic Usage

```bash
python run_wsi_registration.py \
    --he_wsi /path/to/your/he_tma.tif \
    --orion_wsi /path/to/your/orion_tma.tif \
    --output ./wsi_output
```

### Advanced Usage with Custom Parameters

```bash
python run_wsi_registration.py \
    --he_wsi /path/to/your/he_tma.tif \
    --orion_wsi /path/to/your/orion_tma.tif \
    --output ./wsi_output \
    --max_processed_dim 3000 \
    --core_min_area 75000 \
    --core_max_area 400000 \
    --core_circularity 0.5 \
    --expected_core_diameter 350 \
    --core_padding 75
```

### Command Line Arguments

#### Required Arguments
- `--he_wsi`: Path to H&E whole slide image
- `--orion_wsi`: Path to Orion/multiplex whole slide image

#### Optional Arguments
- `--output`: Output directory (default: ./wsi_registration_output)
- `--max_processed_dim`: Maximum dimension for processed images (default: 2048)
- `--max_nonrigid_dim`: Maximum dimension for non-rigid registration (default: 3000)
- `--core_min_area`: Minimum area for valid cores in pixels (default: 50000)
- `--core_max_area`: Maximum area for valid cores in pixels (default: 500000)
- `--core_circularity`: Minimum circularity threshold for cores (default: 0.4)
- `--expected_core_diameter`: Expected core diameter in pixels (default: 400)
- `--core_padding`: Padding around cores in pixels (default: 50)
- `--compression`: Compression method for output images (default: lzw)
- `--no_plots`: Disable generation of quality control plots

## Programmatic Usage

```python
from registration_pipeline_wsi import WSIRegistrationConfig, WSIRegistrationPipeline

# Configure pipeline
config = WSIRegistrationConfig(
    he_wsi_path="/path/to/he_tma.tif",
    orion_wsi_path="/path/to/orion_tma.tif", 
    output_dir="./wsi_output",
    max_processed_image_dim_px=2048,
    core_min_area=50000,
    core_max_area=500000,
    core_circularity_threshold=0.4,
    expected_core_diameter=400
)

# Run pipeline
pipeline = WSIRegistrationPipeline(config)
results = pipeline.run()

# Check results
if results['success']:
    print(f"Extracted {results['cores_extracted']} core pairs")
    print(f"Output saved to: {config.output_dir}")
```

## Output Structure

The pipeline creates a structured output directory:

```
wsi_registration_output/
├── registered_wsi/                    # Registered whole slide images
│   ├── he_slide.ome.tiff             # Registered H&E slide
│   └── orion_slide.ome.tiff          # Registered Orion slide
├── extracted_cores/                   # Individual core pairs
│   ├── core_001/
│   │   ├── core_001_HE.tif           # H&E core image
│   │   ├── core_001_Orion.ome.tif    # Orion core (all channels)
│   │   └── core_001_metadata.json    # Core metadata
│   ├── core_002/
│   │   └── ...
│   └── core_XXX/
├── quality_plots/                     # Visualization plots
│   ├── he_core_detection.png         # H&E core detection results
│   ├── orion_core_detection.png      # Orion core detection results
│   └── core_matching.png             # Core pairing visualization
└── wsi_pipeline_summary.json         # Complete pipeline summary
```

## Core Detection Algorithm

The pipeline uses a sophisticated multi-step approach for core detection:

1. **Preprocessing**: Gaussian smoothing to reduce noise
2. **Thresholding**: Otsu thresholding to separate tissue from background
3. **Morphological Operations**: Remove small objects and fill holes
4. **Connected Component Analysis**: Identify individual tissue regions
5. **Filtering**: Apply area, circularity, and size constraints
6. **Validation**: Check core properties against expected parameters

### Core Detection Parameters

- **`core_min_area`**: Minimum area in pixels (default: 50,000)
- **`core_max_area`**: Maximum area in pixels (default: 500,000)  
- **`core_circularity_threshold`**: Minimum circularity (4π×area/perimeter², default: 0.4)
- **`expected_core_diameter`**: Expected diameter for validation (default: 400 pixels)
- **`core_padding`**: Extra padding around detected boundaries (default: 50 pixels)

## Core Matching Strategy

The pipeline matches cores between H&E and Orion slides using:

1. **Spatial Correspondence**: After registration, cores should be in similar positions
2. **Distance-Based Assignment**: Hungarian algorithm-style matching based on centroid distances
3. **Validation**: Only accept matches within reasonable distance thresholds
4. **Ordering**: Consistent ordering to ensure reproducible pairing

## Registration Parameters

Built on VALIS registration with optimized parameters for TMA slides:

- **`max_processed_image_dim_px`**: Controls resolution of images used for feature detection (default: 2048)
- **`max_non_rigid_registration_dim_px`**: Controls resolution for non-rigid registration (default: 3000)
- Uses H&E as reference image for consistent orientation
- Applies both rigid and non-rigid transformations for optimal alignment

## Quality Control

The pipeline provides comprehensive quality control:

### Visualizations
- Core detection overlays showing detected boundaries
- Core matching visualization with paired numbering
- Registration quality plots and metrics

### Metrics
- Number of cores detected in each slide
- Number of successfully matched pairs
- Registration error statistics
- Core extraction success rates

### Output Validation
- Automatic validation of extracted core dimensions
- Metadata preservation for traceability
- Error logging for failed extractions

## Troubleshooting

### Common Issues

1. **No cores detected**
   - Adjust `core_min_area` and `core_max_area` parameters
   - Lower `core_circularity_threshold` for irregular cores
   - Check image quality and contrast

2. **Poor core matching**
   - Increase `expected_core_diameter` tolerance
   - Verify registration quality
   - Check for tissue artifacts or damage

3. **Registration failure**
   - Increase `max_processed_image_dim_px` for better feature detection
   - Ensure images have sufficient overlap
   - Check for significant differences in staining or imaging conditions

4. **Memory issues**
   - Reduce `max_processed_image_dim_px` and `max_non_rigid_registration_dim_px`
   - Process slides at lower resolution first to estimate parameters

### Parameter Tuning Guidelines

- **For small cores (< 300 pixels diameter)**: Reduce `core_min_area` to ~30,000
- **For large cores (> 500 pixels diameter)**: Increase `core_max_area` to ~750,000
- **For irregular tissue shapes**: Lower `core_circularity_threshold` to 0.2-0.3
- **For high-resolution slides**: Increase `max_processed_image_dim_px` to 3000-4000

## Integration with Existing Workflows

This WSI pipeline is designed to integrate with the existing patch-based registration pipeline:

1. **Use WSI pipeline** for initial TMA slide registration and core extraction
2. **Use patch-based pipeline** for training dataset preparation from extracted cores
3. **Combine outputs** for comprehensive H&E to protein prediction workflows

## Example Workflow

```bash
# Step 1: Extract paired cores from TMA slides
python run_wsi_registration.py \
    --he_wsi TMA_slide_HE.tif \
    --orion_wsi TMA_slide_Orion.tif \
    --output ./extracted_cores

# Step 2: Process individual cores for training (using existing pipeline)
python run_registration.py \
    --input_dir ./extracted_cores/extracted_cores \
    --output_dir ./training_data \
    --patch_size 256
```

## Performance Considerations

- **Processing Time**: 10-30 minutes per TMA slide pair depending on size and resolution
- **Memory Usage**: 4-16 GB RAM depending on slide size and parameters  
- **Storage**: Output requires ~2-5x the input slide size depending on core count
- **Parallelization**: Currently single-threaded but can process multiple slide pairs in parallel

## Citation

If you use this pipeline in your research, please cite the VALIS registration method:

Gatenbee, C.D., et al. "Virtual alignment of pathology image series for multi-gigapixel whole slide images." Nature Communications 14, 4502 (2023).

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your VALIS installation
3. Review the quality control plots for diagnostic information
4. Adjust parameters based on your specific TMA characteristics 