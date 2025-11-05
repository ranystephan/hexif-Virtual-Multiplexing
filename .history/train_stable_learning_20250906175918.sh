#!/bin/bash

# STABLE LEARNING Training - Fixes Unstable Validation & Poor Convergence
# Based on analysis: unstable val loss (0.43-0.54), poor training convergence (0.077 improvement)

echo "=== STABLE LEARNING Training ==="
echo "🚨 CRITICAL FIXES for training instability:"
echo "- Unstable validation loss: 0.43-0.54 (highly erratic)"
echo "- Poor training convergence: only 0.077 improvement over 20 epochs"
echo "- Training-validation gap: 0.287 vs 0.43 (significant overfitting)"
echo ""
echo "🎯 Stability improvements:"
echo "- Reduced learning rate: 5e-4 → 1e-4 (much more conservative)"
echo "- Simplified loss function: Remove conflicting objectives"
echo "- Better batch size: 8→12 per GPU for more stable gradients"
echo "- Conservative scheduler: More patience, smaller reductions"
echo "- Better regularization: Higher dropout, proper weight decay"
echo "- Larger validation set: 15%→20% for better validation signals"
echo ""

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

python train_spatial_orion_unet.py \
    --pairs_dir core_patches_npy \
    --output_dir runs_sep6v1/orion_stable_learning \
    --epochs 30 \
    --patch_size 256 \
    --batch_size 16 \
    --val_batch_size 8 \
    --learning_rate 2e-4 \
    --weight_decay 2e-4 \
    --patches_per_image 12 \
    --patches_per_image_val 8 \
    --num_workers 16 \
    --mse_weight 0.8 \
    --l1_weight 0.3 \
    --boundary_weight 0.1 \
    --focal_weight 0.0 \
    --variance_weight 0.1 \
    --use_amp \
    --use_boundary_guidance \
    --noise_removal \
    --sanity_check_freq 2 \
    --val_split 0.20

echo ""
echo "Expected stability improvements:"
echo "- Validation loss should be more stable (range <0.05 instead of 0.11)"
echo "- Training loss should improve more: >0.10 instead of 0.077"
echo "- Training-validation gap should be smaller: <0.10 instead of 0.14"
echo "- Learning curve should be smoother with clear downward trend"
echo ""
echo "Key stability features:"
echo "- Lower LR (1e-4): More stable gradient updates"
echo "- Simplified loss: Fewer conflicting objectives"
echo "- Better batch size: More stable gradient estimates"
echo "- Conservative variance penalty: 0.1 instead of 0.5"
echo "- Disabled focal loss: Removes complexity"
