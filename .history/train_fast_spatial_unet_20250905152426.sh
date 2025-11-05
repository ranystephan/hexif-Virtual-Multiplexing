#!/bin/bash

# Fast Spatial U-Net Training Configuration
# Optimized for speed and better convergence

echo "Starting Fast Spatial U-Net Training..."
echo "Optimizations:"
echo "- Larger patch size (384) for more spatial context"
echo "- Reduced patches per image (32) for speed"
echo "- Higher learning rate (2e-4) for faster convergence"
echo "- Mixed precision training enabled"
echo "- Improved loss weighting"
echo "- Gradient clipping for stability"

python train_spatial_orion_unet.py \
    --pairs_dir core_patches_npy \
    --output_dir runs/orion_spatial_fast \
    --epochs 30 \
    --patch_size 384 \
    --batch_size 16 \
    --val_batch_size 8 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --patches_per_image 32 \
    --patches_per_image_val 16 \
    --mse_weight 0.7 \
    --l1_weight 0.5 \
    --boundary_weight 0.2 \
    --focal_weight 0.1 \
    --use_amp \
    --use_boundary_guidance \
    --noise_removal \
    --sanity_check_freq 2 \
    --num_workers 8

echo "Training completed!"
