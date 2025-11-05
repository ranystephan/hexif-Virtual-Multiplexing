#!/bin/bash

# ANTI-COLLAPSE Training - Fixes Model Collapse Issue
# Based on diagnostic: Model predicting uniform 0.49, zero spatial variance, vanished gradients

echo "=== ANTI-COLLAPSE Training ==="
echo "🚨 CRITICAL FIXES for model collapse:"
echo "- Model stuck at uniform ~0.49 predictions"
echo "- Spatial variance: 0.000002 (essentially zero)"
echo "- Vanished gradients: 0.000000"
echo ""
echo "🎯 Anti-collapse strategies:"
echo "- Higher learning rate: 1e-4 → 5e-4 (escape local minimum)"
echo "- Anti-collapse loss: Penalizes uniform predictions"
echo "- Variance regularization: Forces spatial diversity"
echo "- Better initialization with Xavier/He"
echo "- Gradient clipping for stability"
echo ""

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

python train_spatial_orion_unet.py \
    --pairs_dir core_patches_npy \
    --output_dir runs_sep6v1/orion_anti_collapse \
    --epochs 20 \
    --patch_size 256 \
    --batch_size 8 \
    --val_batch_size 4 \
    --learning_rate 5e-4 \
    --weight_decay 1e-5 \
    --patches_per_image 16 \
    --patches_per_image_val 8 \
    --num_workers 16 \
    --mse_weight 0.8 \
    --l1_weight 0.4 \
    --boundary_weight 0.1 \
    --focal_weight 0.2 \
    --variance_weight 0.5 \
    --use_amp \
    --use_boundary_guidance \
    --noise_removal \
    --sanity_check_freq 2 \
    --val_split 0.15

echo ""
echo "Expected improvements:"
echo "- Predictions should spread beyond [0.45, 0.55] range"
echo "- Spatial variance should increase from 0.000002 to >0.01"
echo "- Loss should decrease from 0.23 to <0.15"
echo "- Visual patterns should appear by epoch 3-5"
echo ""
echo "Key anti-collapse features:"
echo "- variance_weight=0.5: Strong penalty for uniform predictions"
echo "- Higher LR: 5e-4 to escape local minimum"
echo "- More patches: 16 per image for diverse gradients"
