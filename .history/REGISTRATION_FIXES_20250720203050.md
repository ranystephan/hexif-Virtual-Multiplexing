# Registration Pipeline Fixes and Improvements

## Issues Identified and Fixed

Based on your error logs, I've identified and fixed several critical issues:

### 1. **VALIS Feature Detection Failures**
**Error**: `'NoneType' object has no attribute 'shape'`
**Cause**: VALIS couldn't detect features in some images
**Fix**: Added comprehensive image validation and better error handling

### 2. **OpenCV Resize Errors** 
**Error**: `OpenCV(4.12.0) ... error: (-215:Assertion failed) inv_scale_x > 0 in function 'resize'`
**Cause**: Images with problematic dimensions or extreme aspect ratios
**Fix**: Added dimension validation and aspect ratio checks

### 3. **JSON Serialization Error**
**Error**: `Object of type int64 is not JSON serializable`
**Cause**: Numpy data types in summary output
**Fix**: Added type conversion for JSON serialization

## Key Improvements Made

### 1. **Enhanced Image Validation**
- Minimum size checks (100x100 pixels)
- Dimension validation
- Aspect ratio analysis
- Intensity distribution checks
- Detection of problematic images before processing

### 2. **Better Error Handling**
- Graceful failure handling for individual cores
- JVM cleanup on errors
- Specific error messages for different failure types
- Configurable failure thresholds

### 3. **Image Analysis Tool**
- New `analyze_images.py` script to pre-screen images
- Identifies problematic images before registration
- Generates detailed reports and recommendations

### 4. **Improved Configuration**
- Added `max_failures_before_stop` parameter
- Better error handling options
- Pre-analysis capabilities

## How to Use the Fixed Pipeline

### Step 1: Analyze Your Images First
```bash
# Analyze images to identify problems
python analyze_images.py --input_dir /path/to/your/images --output_dir ./analysis_results

# This will show you:
# - Which images are problematic
# - Common issues in your dataset
# - Expected success rate
```

### Step 2: Run Registration with Better Error Handling
```bash
# Basic usage with improved error handling
python run_registration.py \
    --input_dir /path/to/your/images \
    --output_dir ./registration_output \
    --max_failures 20 \
    --num_workers 2

# Or with pre-analysis
python run_registration.py \
    --input_dir /path/to/your/images \
    --output_dir ./registration_output \
    --analyze_first \
    --max_failures 20
```

### Step 3: Check Results
The pipeline now provides better feedback:
- Clear error messages for failed registrations
- Stops early if too many failures occur
- Detailed quality reports
- Recommendations for improvement

## Configuration Options for Problem Images

### For Images with Extreme Aspect Ratios
```python
config = RegistrationConfig(
    input_dir="/path/to/images",
    output_dir="./registration_output",
    max_processed_image_dim_px=512,  # Reduce if memory issues
    max_failures_before_stop=10,     # Stop early if many failures
    num_workers=1                    # Reduce parallelism for stability
)
```

### For Low-Quality Images
```python
config = RegistrationConfig(
    min_ssim_threshold=0.1,      # Lower quality thresholds
    min_ncc_threshold=0.1,
    min_mi_threshold=0.2,
    skip_failed_registrations=True
)
```

## Troubleshooting Common Issues

### Issue: Many "Feature detection failed" errors
**Solution**: Your images may lack sufficient features for VALIS
```bash
# Pre-analyze to identify problematic images
python analyze_images.py --input_dir /path/to/images

# Filter out problematic images or adjust VALIS parameters
```

### Issue: OpenCV resize errors
**Solution**: Images have problematic dimensions
```bash
# The analysis tool will identify these
# Consider preprocessing images to standard sizes
```

### Issue: Very low success rate
**Solution**: Dataset may not be suitable for VALIS registration
```bash
# Run analysis first
python analyze_images.py --input_dir /path/to/images

# Check the recommendations in the output
# Consider image preprocessing or alternative registration methods
```

## Expected Behavior Now

1. **Better Failure Handling**: Failed registrations won't crash the entire pipeline
2. **Early Stopping**: Pipeline stops if too many consecutive failures occur
3. **Detailed Logging**: Clear error messages for each failure type
4. **Quality Reports**: Better metrics and visualization
5. **Pre-screening**: Option to analyze images before processing

## Sample Output

```
INFO - Found 50 H&E images in /path/to/images
INFO - Registering core reg001
INFO - Registering core reg002
ERROR - Registration failed for core reg003: VALIS feature detection failed - images may lack sufficient features
INFO - Registering core reg004
ERROR - Registration failed for core reg005: OpenCV resize error - image dimensions may be problematic
...
INFO - Registration completed: 35 successful, 15 failed
INFO - Success rate: 70.0%
```

## Next Steps

1. **Run the analysis tool** on your dataset first
2. **Use the improved pipeline** with better error handling
3. **Review the quality reports** to understand which images work best
4. **Consider preprocessing** problematic images if needed
5. **Adjust parameters** based on your specific dataset characteristics

The pipeline is now much more robust and should handle your dataset better, providing clear feedback about what works and what doesn't. 