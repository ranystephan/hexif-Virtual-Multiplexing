# Registration and Core Detection Diagnosis Workflow

## 🚨 Problem Identified

Your original workflow (`run_wsi_workflow.py`) was failing because:

1. **✅ Registration is WORKING** - VALIS successfully registered your images
2. **❌ Core detection is FAILING** - Detected 100 cores in H&E vs 0 in Orion (should be 273+)
3. **❌ Core detection parameters are WRONG** for your specific TMA

## 🎯 New Approach: Separate Registration from Core Detection

Instead of running everything together, let's **separate the workflows**:

1. **First**: Get registration working perfectly ✅
2. **Then**: Tune core detection parameters separately 🔧
3. **Finally**: Combine them once parameters are optimized 🎯

---

## 📋 Step-by-Step Workflow

### Step 1: Registration Only

Use the **registration-only workflow** to verify VALIS registration works:

```bash
python run_registration_only.py \
    --he_wsi data/raw/TA118-HEraw.ome.tiff \
    --orion_wsi data/raw/TA118-Orionraw.ome.tiff \
    --output data/registration_only_TA118
```

**What this does:**
- ✅ Preprocesses OME-TIFF files (same as before)
- ✅ Runs VALIS registration (same as before)
- ✅ Saves registered images
- ❌ **Skips core detection entirely**

**Output:**
```
data/registration_only_TA118/
├── registered_wsi/
│   ├── he_slide.ome.tiff      # Registered H&E
│   └── orion_slide.ome.tiff   # Registered Orion
└── registration_summary.json   # Summary with metrics
```

### Step 2: Inspect Registration Quality

Verify the registration worked properly:

```bash
python inspect_registration.py \
    --registered_dir data/registration_only_TA118/registered_wsi \
    --output data/registration_inspection_TA118
```

**What this does:**
- 📊 Calculates SSIM, NCC, MAE, RMSE metrics
- 🎨 Creates overlay visualizations
- ✅ Assesses registration quality

**Look for:**
- SSIM > 0.6 (good registration)
- NCC > 0.5 (good correlation) 
- Visual alignment in overlay plots

### Step 3: Tune Core Detection Parameters

Now tune core detection parameters separately:

```bash
python tune_core_detection.py \
    --he_wsi data/registration_only_TA118/registered_wsi/he_slide.ome.tiff \
    --orion_wsi data/registration_only_TA118/registered_wsi/orion_slide.ome.tiff \
    --output data/core_detection_tuning_TA118
```

**What this does:**
- 🔍 Tests different parameter combinations
- 📊 Shows how many cores each parameter set detects
- 🎨 Visualizes detection results
- 💡 Helps you find parameters that detect ~273 cores

**Key parameters to adjust:**
- `--min_area` (try 10000, 20000, 30000)
- `--max_area` (try 300000, 500000, 800000)
- `--circularity` (try 0.2, 0.3, 0.4)
- `--orion_channel` (try 0, 1, 2 for different channels)

### Step 4: Test Different Orion Channels

The Orion channel selection is critical! Try different channels:

```bash
# Try DAPI channel (usually channel 0)
python tune_core_detection.py \
    --orion_wsi data/registration_only_TA118/registered_wsi/orion_slide.ome.tiff \
    --orion_channel 0 \
    --output data/core_tuning_orion_ch0

# Try other channels
python tune_core_detection.py \
    --orion_wsi data/registration_only_TA118/registered_wsi/orion_slide.ome.tiff \
    --orion_channel 1 \
    --output data/core_tuning_orion_ch1
```

---

## 🔧 Likely Issues with Your Core Detection

Based on your log output, here are the probable issues:

### 1. **Area Parameters Too Restrictive**
```
Current: min_area=50000, max_area=500000
Try: min_area=10000, max_area=800000
```

### 2. **Wrong Orion Channel**
```
Current: Using channel 0 (DAPI) 
Try: Different channels (1, 2, 3...) might have better contrast
```

### 3. **Spatial Dimension Mismatch**
```
Issue: H&E (12362, 20000) vs Orion (10550, 20000)
Solution: This is handled by resizing, but may affect core detection
```

### 4. **Circularity Too Strict**
```
Current: circularity=0.4
Try: circularity=0.2 or 0.3 (cores might not be perfectly round)
```

---

## 📊 Expected Parameter Ranges for TMA with 273+ Cores

Based on typical TMA images:

| Parameter | Current Value | Suggested Range | Why |
|-----------|---------------|-----------------|-----|
| `min_area` | 50,000 | 10,000-30,000 | Cores might be smaller than expected |
| `max_area` | 500,000 | 300,000-1,000,000 | Cores might be larger than expected |
| `circularity` | 0.4 | 0.2-0.4 | TMA cores often not perfectly circular |
| `orion_channel` | 0 | 0-5 | Try different channels for better contrast |

---

## 🎯 Quick Commands to Run

### Run Everything Step by Step:

```bash
# Step 1: Registration only
python run_registration_only.py \
    --he_wsi data/raw/TA118-HEraw.ome.tiff \
    --orion_wsi data/raw/TA118-Orionraw.ome.tiff \
    --output data/reg_only_TA118

# Step 2: Inspect registration
python inspect_registration.py \
    --registered_dir data/reg_only_TA118/registered_wsi \
    --output data/reg_inspection_TA118

# Step 3: Tune core detection with permissive parameters
python tune_core_detection.py \
    --he_wsi data/reg_only_TA118/registered_wsi/he_slide.ome.tiff \
    --orion_wsi data/reg_only_TA118/registered_wsi/orion_slide.ome.tiff \
    --min_area 10000 \
    --max_area 800000 \
    --circularity 0.2 \
    --orion_channel 0 \
    --output data/core_tuning_TA118_permissive

# Step 4: Try different Orion channels
python tune_core_detection.py \
    --he_wsi data/reg_only_TA118/registered_wsi/he_slide.ome.tiff \
    --orion_wsi data/reg_only_TA118/registered_wsi/orion_slide.ome.tiff \
    --min_area 15000 \
    --max_area 600000 \
    --circularity 0.3 \
    --orion_channel 1 \
    --output data/core_tuning_TA118_ch1
```

---

## 💡 Benefits of This Approach

1. **✅ Isolates Issues** - You can verify registration works before tackling core detection
2. **🔧 Easier Debugging** - Tune parameters visually with immediate feedback
3. **⚡ Faster Iteration** - No need to re-run expensive registration every time
4. **🎯 Better Results** - Parameters optimized specifically for your data
5. **📊 Visual Feedback** - See exactly what each parameter combination detects

---

## 🎯 Success Criteria

You'll know you're ready to proceed when:

- **Registration Quality**: SSIM > 0.6, good visual alignment
- **H&E Core Detection**: Detects ~273 cores with reasonable parameters
- **Orion Core Detection**: Detects ~273 cores with the right channel/parameters
- **Core Matching**: Similar core counts and spatial distribution

---

## 🚀 Once Parameters Are Tuned

When you find good parameters, you can:

1. **Update** the original `run_wsi_workflow.py` with your optimized parameters
2. **Create** a custom workflow with your specific settings
3. **Proceed** with confidence to full core extraction and pairing

**Example with tuned parameters:**
```bash
python run_wsi_workflow.py \
    --he_wsi data/raw/TA118-HEraw.ome.tiff \
    --orion_wsi data/raw/TA118-Orionraw.ome.tiff \
    --output data/final_extraction_TA118 \
    --min_area 15000 \
    --max_area 600000 \
    --circularity 0.3 \
    --orion_channel 1
```

This systematic approach will get you to working core extraction much faster! 🎯 