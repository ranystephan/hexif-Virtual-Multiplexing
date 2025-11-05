#!/bin/bash

# ULTRA-FAST Training - Addresses Data Loading Bottleneck
# Based on diagnostic: 58.9s per batch data loading vs 0.1s model forward

echo "=== ULTRA-FAST Training - Data Loading Optimized ==="
echo "🚨 CRITICAL FIXES for 58.9s data loading bottleneck:"
echo "- Reduced patch size: 384→256 (44% less data per patch)"
echo "- Reduced patches per image: 32→8 (75% fewer patches)"
echo "- Simplified preprocessing (no percentile normalization)"
echo "- Increased workers: 16→32"
echo "- Smaller effective batch size for faster loading"
echo ""

# Set optimal environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=4

python train_spatial_orion_unet.py \
    --pairs_dir core_patches_npy \
    --output_dir runs/orion_ultra_fast \
    --epochs 15 \
    --patch_size 256 \
    --batch_size 16 \
    --val_batch_size 8 \
    --learning_rate 4e-4 \
    --weight_decay 1e-4 \
    --patches_per_image 8 \
    --patches_per_image_val 4 \
    --num_workers 32 \
    --mse_weight 0.5 \
    --l1_weight 0.7 \
    --boundary_weight 0.4 \
    --focal_weight 0.3 \
    --use_amp \
    --use_boundary_guidance \
    --noise_removal \
    --sanity_check_freq 3 \
    --val_split 0.1

echo ""
echo "Expected improvements:"
echo "- Data loading: 58.9s → ~5-10s per batch (6-12x faster)"
echo "- Epoch time: 2.4hrs → ~5-10 minutes (15-30x faster)" 
echo "- Total patches: 8,160 → 2,040 (75% reduction)"
echo "- Memory per patch: 384²×20 → 256²×20 (44% reduction)"
