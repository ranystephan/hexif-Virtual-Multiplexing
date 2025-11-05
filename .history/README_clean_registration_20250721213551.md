# Clean TMA WSI Registration using VALIS

This is a **clean, focused registration script** built from scratch following VALIS documentation. It's optimized for large TMA images with 270+ cores and **does no core detection** - just pure VALIS registration.

## 🎯 Key Features

- **Clean VALIS implementation** following official documentation
- **Optimized for large TMA images** (270+ cores) 
- **No preprocessing dependencies** - works with your existing images
- **Proper parameter tuning** for TMA-sized images
- **Comprehensive error handling** and JVM cleanup
- **Detailed logging** and progress reporting

## 📋 Requirements

```bash
pip install valis
```

That's it! No other dependencies needed.

## 🚀 Basic Usage

### Quick Start

```bash
python register_tma_wsi.py \
    --he_wsi data/raw/TA118-HEraw.ome.tiff \
    --orion_wsi data/raw/TA118-Orionraw.ome.tiff \
    --output data/registered_TA118
```

### With Custom Parameters

```bash
python register_tma_wsi.py \
    --he_wsi data/raw/TA118-HEraw.ome.tiff \
    --orion_wsi data/raw/TA118-Orionraw.ome.tiff \
    --output data/registered_TA118 \
    --max_processed_dim 2000 \
    --max_nonrigid_dim 4000 \
    --compression lzw
```

## ⚙️ Parameters Optimized for TMA Images

Based on VALIS documentation for large images with 270+ cores:

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `--max_processed_dim` | 1500 | 500-4000 | Resolution for rigid registration |
| `--max_nonrigid_dim` | 3000 | 1000-8000 | Resolution for non-rigid registration |
| `--crop` | reference | reference/overlap/all | How to crop final images |
| `--compression` | lzw | lzw/jpeg/jp2k | Output compression method |

### Parameter Guidelines from VALIS Docs:

- **For TMA images with lots of cores**: Use **higher** `max_processed_dim` (1500-2500)
- **For images with empty space**: Use **larger** values so tissue is high enough resolution
- **For tight tissue images**: Use **smaller** values (1000-1500)

## 📊 Output Structure

```
data/registered_TA118/
├── registered/
│   ├── 01_he_reference.ome.tiff    # Registered H&E (reference)
│   └── 02_orion_target.ome.tiff    # Registered Orion (aligned to H&E)
└── registration_summary.json       # Summary with parameters and results
```

## 🔧 Command Line Options

### Required
- `--he_wsi`: Path to H&E WSI file
- `--orion_wsi`: Path to Orion WSI file

### Optional
- `--output`: Output directory (default: `./tma_registration_output`)
- `--max_processed_dim`: Max dimension for processed images (default: 1500)
- `--max_nonrigid_dim`: Max dimension for non-rigid (default: 3000)
- `--crop`: Cropping method - `reference`/`overlap`/`all` (default: `reference`)
- `--compression`: Compression - `lzw`/`jpeg`/`jp2k` (default: `lzw`)
- `--compression_quality`: Quality for lossy compression (default: 95)
- `--rigid_only`: Skip non-rigid registration (faster)
- `--keep_intermediate`: Keep VALIS intermediate files

## 📖 Usage Examples

### 1. Quick Test (Rigid Only)
```bash
python register_tma_wsi.py \
    --he_wsi data/raw/TA118-HEraw.ome.tiff \
    --orion_wsi data/raw/TA118-Orionraw.ome.tiff \
    --output data/test_registration \
    --rigid_only
```

### 2. High Quality Registration
```bash
python register_tma_wsi.py \
    --he_wsi data/raw/TA118-HEraw.ome.tiff \
    --orion_wsi data/raw/TA118-Orionraw.ome.tiff \
    --output data/high_quality_registration \
    --max_processed_dim 2000 \
    --max_nonrigid_dim 4000 \
    --compression lzw
```

### 3. Debug Mode (Keep Intermediate Files)
```bash
python register_tma_wsi.py \
    --he_wsi data/raw/TA118-HEraw.ome.tiff \
    --orion_wsi data/raw/TA118-Orionraw.ome.tiff \
    --output data/debug_registration \
    --keep_intermediate
```

## 🔍 Parameter Tuning Guide

### If registration fails or is poor:

1. **Try different `max_processed_dim` values:**
   ```bash
   # For very large images with lots of empty space
   --max_processed_dim 2500
   
   # For images with tight tissue packing  
   --max_processed_dim 1000
   ```

2. **Test with rigid-only first:**
   ```bash
   --rigid_only  # Much faster, helps diagnose issues
   ```

3. **Adjust non-rigid resolution:**
   ```bash
   --max_nonrigid_dim 4000  # Higher quality but slower
   --max_nonrigid_dim 2000  # Faster but lower quality
   ```

## ✅ Expected Results

### Success Indicators:
- ✅ Script completes without errors
- ✅ Two registered `.ome.tiff` files created
- ✅ Registration summary JSON created
- ✅ Log shows "Registration completed successfully"

### Output Files:
- `01_he_reference.ome.tiff` - Registered H&E (reference image)
- `02_orion_target.ome.tiff` - Registered Orion (aligned to H&E)
- `registration_summary.json` - Complete summary with parameters

## 🚨 Troubleshooting

### Common Issues:

1. **"Registration failed"**:
   - Try `--rigid_only` first
   - Adjust `--max_processed_dim` (try 1000, 1500, 2000)
   - Use `--keep_intermediate` to inspect VALIS results

2. **"Images too large"**:
   - Decrease `--max_processed_dim` to 1000-1200
   - Use `--rigid_only` for faster processing

3. **"Poor alignment"**:
   - Increase `--max_processed_dim` to 2000-2500
   - Check if input images have sufficient overlap

## 🔗 Integration with Your Workflow

This script produces **standard VALIS-registered images** that can be used:

1. **For visual inspection** (using your existing tools)
2. **As input to core detection** (once you've tuned parameters)
3. **For downstream analysis** (QuPath, ImageJ, etc.)
4. **With existing pipelines** (just replace input images with registered ones)

## 💡 Next Steps After Registration

1. **Visual inspection**: Check registration quality
2. **Core detection tuning**: Use registered images as input
3. **Pipeline integration**: Replace original images with registered ones
4. **Quality assessment**: Measure registration metrics if needed

This clean approach separates registration from core detection, making it easier to debug and optimize each step independently! 