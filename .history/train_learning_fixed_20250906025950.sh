#!/bin/bash

# LEARNING-FIXED Training - Addresses Model Not Learning Issues
# Based on common problems: loss plateau, poor convergence

echo "=== LEARNING-FIXED Training ==="
echo "🎯 FIXES for model not learning (loss plateau at 0.23-0.30):"
echo "- Lower learning rate: 4e-4 → 1e-4 (more stable)"
echo "- Restore robust normalization (simple min-max might be too weak)"
echo "- Reduce loss complexity (focus on MSE+L1)"
echo "- Add gradient clipping"
echo "- Longer training with patience"
echo "- Better initialization"
echo ""

# Set environment for stable training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

python train_spatial_orion_unet.py \
    --pairs_dir core_patches_npy \
    --output_dir runs/orion_learning_fixed \
    --epochs 25 \
    --patch_size 256 \
    --batch_size 12 \
    --val_batch_size 6 \
    --learning_rate 1e-4 \
    --weight_decay 5e-5 \
    --patches_per_image 12 \
    --patches_per_image_val 6 \
    --num_workers 16 \
    --mse_weight 1.0 \
    --l1_weight 0.3 \
    --boundary_weight 0.0 \
    --focal_weight 0.0 \
    --use_amp \
    --use_boundary_guidance \
    --sanity_check_freq 2 \
    --val_split 0.15

echo ""
echo "Key changes for better learning:"
echo "- Conservative LR: 1e-4 (vs 4e-4) for stable convergence"
echo "- Simplified loss: MSE + L1 only (remove complex components)"
echo "- More patches: 12 per image (vs 8) for better gradients"
echo "- Smaller batches: 12 (vs 16) for more gradient updates"
echo "- Longer training: 25 epochs with patience"
