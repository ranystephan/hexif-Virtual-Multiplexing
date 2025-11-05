#!/bin/bash

# Optimized H&E to Orion Training Launch Script
# This script uses optimized parameters for much faster and better training

echo "Starting OPTIMIZED H&E to Orion training..."
echo "Key improvements:"
echo "  - Smaller patches (256x256) for better GPU utilization"
echo "  - Proper batch size (16-32) instead of 2"
echo "  - Simplified data loading without complex caching"
echo "  - Better loss function weighting"
echo "  - Gradient clipping for stable training"
echo "  - Efficient model architecture"
echo ""

# Set CUDA device if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    export CUDA_VISIBLE_DEVICES=0
else
    echo "No GPU detected, using CPU"
fi

# Default parameters (modify as needed)
PAIRS_DIR=${1:-"core_patches_npy"}
OUTPUT_DIR=${2:-"orion_optimized_model"}
EPOCHS=${3:-100}
BATCH_SIZE=${4:-16}

# Check if pairs directory exists
if [ ! -d "$PAIRS_DIR" ]; then
    echo "ERROR: Pairs directory '$PAIRS_DIR' not found!"
    echo "Usage: $0 [pairs_dir] [output_dir] [epochs] [batch_size]"
    exit 1
fi

# Count available data
HE_FILES=$(find "$PAIRS_DIR" -name "*_HE.npy" | wc -l)
ORION_FILES=$(find "$PAIRS_DIR" -name "*_ORION.npy" | wc -l)

echo "Found $HE_FILES H&E files and $ORION_FILES Orion files"

if [ "$HE_FILES" -eq 0 ] || [ "$ORION_FILES" -eq 0 ]; then
    echo "ERROR: No data files found in $PAIRS_DIR"
    exit 1
fi

# Calculate optimal batch size based on available GPU memory
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    if [ "$GPU_MEM" -gt 20000 ]; then
        BATCH_SIZE=32
        echo "Large GPU detected (${GPU_MEM}MB), using batch size 32"
    elif [ "$GPU_MEM" -gt 10000 ]; then
        BATCH_SIZE=16
        echo "Medium GPU detected (${GPU_MEM}MB), using batch size 16"
    else
        BATCH_SIZE=8
        echo "Small GPU detected (${GPU_MEM}MB), using batch size 8"
    fi
fi

echo ""
echo "Training configuration:"
echo "  Pairs directory: $PAIRS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Patch size: 256x256 (optimized for speed)"
echo "  Patches per image: 64 (more smaller patches)"
echo "  Learning rate: 2e-4 (optimized)"
echo ""

# Run the optimized training script
python train_orion_patches_optimized.py \
    --pairs_dir "$PAIRS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 2e-4 \
    --weight_decay 1e-5 \
    --patch_size 256 \
    --patches_per_image 64 \
    --patches_per_image_val 32 \
    --base_features 32 \
    --use_amp \
    --num_workers 4 \
    --max_grad_norm 1.0 \
    --save_every 10 \
    --seed 42

echo ""
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To monitor training progress:"
echo "  tail -f $OUTPUT_DIR/train.log"
echo ""
echo "To view training metrics:"
echo "  cat $OUTPUT_DIR/training_log.csv"
