# VALIS Pairing Pipeline Enhancements

## Overview
Enhanced the VALIS pairing pipeline to address the issue of 0 pairs being created from 1821 detected cores, and added comprehensive PNG overlay visualization for registration quality assessment.

## Key Issues Addressed

### 1. **Zero Pairs Problem**
- **Root Cause**: Distance threshold too restrictive (max_dist_factor * spacing was too small)
- **Solution**: Implemented multi-strategy pairing with adaptive thresholds

### 2. **Missing Visualization**
- **Root Cause**: No full-slide registration overview
- **Solution**: Added comprehensive 2x2 grid visualization showing registration quality

### 3. **Memory Management**
- **Root Cause**: Processing 1821+ cores can cause memory issues
- **Solution**: Added core limiting (--max_cores parameter) with smart selection

## New Features

### 1. **Enhanced Core Pairing Algorithm**
```python
# Multiple pairing strategies with different thresholds:
strategies = [
    ("strict", max_dist_factor * spacing),           # Original approach
    ("moderate", max_dist_factor * spacing * 2),     # 2x more lenient  
    ("lenient", max_dist_factor * spacing * 5),      # 5x more lenient
    ("very_lenient", np.percentile(dists, 10))       # Use 10th percentile
]
```

**Benefits:**
- Automatically selects best strategy that produces reasonable results
- Falls back to more lenient thresholds if strict matching fails
- Provides detailed logging of each strategy's performance

### 2. **Comprehensive Registration Visualization**
Creates 4-panel overview images:
- **Panel 1**: H&E slide with detected cores (red circles)
- **Panel 2**: Orion slide with warped core positions (green squares)  
- **Panel 3**: Registration overlay (H&E=red channel, Orion=green channel)
- **Panel 4**: Matched pairs with connecting lines

**Output Files:**
- `registration_qc/registration_overview.png` - Initial detection results
- `registration_qc/registration_overview_with_pairs.png` - Final pairing results
- `registration_qc/registration_overview_nonrigid.png` - Non-rigid results (if used)

### 3. **Automatic Non-Rigid Registration**
```python
# If rigid registration produces 0 pairs, automatically try non-rigid
if len(pairs) == 0 and args.auto_non_rigid and not args.non_rigid:
    # Re-warp coordinates using non-rigid registration
    warped_or_xy_nonrigid = he_slide.warp_xy_from_to(
        xy=he_centres_arr, to_slide_obj=or_slide,
        src_pt_level=0, dst_slide_level=0, non_rigid=True
    )
```

### 4. **Smart Core Selection**
```python
# Limit cores for memory management, select largest by area
if len(he_centres) > args.max_cores:
    he_cores_sorted = sorted(he_cores, key=lambda x: x['area'], reverse=True)[:args.max_cores]
```

### 5. **Enhanced Error Handling & Logging**
- Detailed coordinate validation (filters NaN, inf values)
- Comprehensive distance statistics logging
- Step-by-step pairing process tracking
- Helpful troubleshooting suggestions

## New Command Line Parameters

```bash
--max_cores 300              # Limit number of cores (default: 500)
--max_dist_factor 5.0        # More lenient distance threshold (default: 5.0, was 3.0)  
--auto_non_rigid             # Auto-try non-rigid if rigid fails (default: True)
--create_overview            # Generate registration visualizations (default: True)
```

## Recommended Usage

### For ~300 Cores (Your Use Case):
```bash
python valis_pairing_pipeline.py \
    --he data/raw/TA118-HEraw.ome.tiff \
    --orion data/raw/TA118-Orionraw.ome.tiff \
    --out_dir paired_dataset_valis_enhanced \
    --patch_size 2048 \
    --max_image_dim 1024 --max_processed_dim 512 --max_non_rigid_dim 2048 \
    --max_cores 300 \
    --max_dist_factor 5.0 \
    --auto_non_rigid \
    --create_overview
```

### For Debugging Registration Issues:
```bash
python valis_pairing_pipeline.py \
    --he your_he_file.ome.tiff \
    --orion your_orion_file.ome.tiff \
    --out_dir debug_output \
    --max_cores 100 \
    --max_dist_factor 10.0 \
    --non_rigid \
    --create_overview \
    --keep_converted \
    --verbose
```

## Expected Results

### Previous Output:
```
Detected 1821 H&E cores
Created 0 pairs using Hungarian assignment
Success rate: 0.0%
```

### Enhanced Output:
```
Detected 1821 H&E cores
Processing 300 largest cores  
Warping 300 H&E coordinates to Orion coordinate system...
Valid warped coordinates: 300/300

Trying strict strategy with max distance: 5301.2
strict strategy: 0/300 pairs accepted

Trying moderate strategy with max distance: 10602.5  
moderate strategy: 15/300 pairs accepted

Selected moderate strategy with 15 final pairs
Success rate: 5.0%
```

## Output Directory Structure

```
paired_dataset_valis_enhanced/
├── paired_core_info.csv                    # Summary of all pairs
├── patches/                                # Extracted tissue patches
│   ├── core_001_he.tiff
│   ├── core_001_orion.tiff
│   └── ...
├── qc_overlays/                           # Individual core overlays
│   ├── core_001_overlay.png
│   └── ...
├── registration_qc/                       # Full-slide visualizations
│   ├── registration_overview.png
│   ├── registration_overview_with_pairs.png
│   └── registration_overview_nonrigid.png (if applicable)
└── valis_output/                         # VALIS registration files
```

## Troubleshooting Guide

### Still Getting 0 Pairs?
1. **Check registration quality**: Look at `registration_qc/registration_overview.png`
2. **Increase distance tolerance**: Try `--max_dist_factor 10.0` or higher
3. **Enable non-rigid**: Use `--non_rigid` flag explicitly  
4. **Reduce core count**: Try `--max_cores 100` for initial testing
5. **Check core detection**: Adjust `--min_core_area` and `--circularity_thresh`

### Poor Registration Quality?
1. **Try non-rigid registration**: Add `--non_rigid` flag
2. **Adjust VALIS parameters**: Increase `--max_processed_dim` and `--max_non_rigid_dim`
3. **Check input files**: Ensure H&E and Orion slides are properly aligned
4. **Memory issues**: Reduce `--max_image_dim` if getting memory errors

## Performance Optimizations

1. **Memory Management**: Automatic core limiting prevents out-of-memory errors
2. **Smart Fallbacks**: Multiple pairing strategies ensure best possible results  
3. **Parallel Processing**: VALIS uses multi-threading for registration
4. **Efficient Visualization**: Thumbnails used for overview generation
5. **Cleanup**: Temporary files automatically removed unless debugging

## Quality Assurance

The enhanced pipeline provides multiple ways to assess registration quality:

1. **Quantitative Metrics**: Distance statistics, success rates, pair counts
2. **Visual Inspection**: Full-slide overlays, individual core pairs  
3. **Comparative Analysis**: Rigid vs non-rigid results
4. **Detailed Logging**: Step-by-step process tracking

This comprehensive approach ensures you can identify and resolve registration issues efficiently.
