#!/bin/bash

# Ultra-Fast Multi-GPU Spatial U-Net Training
# Optimized for 4x TITAN Xp GPUs

echo "=== Ultra-Fast Multi-GPU Training Configuration ==="
echo "Optimizations:"
echo "- 4x GPU utilization with DataParallel"
echo "- Large effective batch size (32x4=128)"
echo "- Increased data loading workers (16)"
echo "- Memory prefetching enabled"
echo "- Aggressive patch reduction for speed"
echo "- Mixed precision training"
echo ""

# Set CUDA visible devices explicitly
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable optimized CUDA operations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python train_spatial_orion_unet.py \
    --pairs_dir core_patches_npy \
    --output_dir runs/orion_multi_gpu_fast \
    --epochs 20 \
    --patch_size 256 \
    --batch_size 32 \
    --val_batch_size 16 \
    --learning_rate 3e-4 \
    --weight_decay 1e-4 \
    --patches_per_image 16 \
    --patches_per_image_val 8 \
    --num_workers 16 \
    --mse_weight 0.6 \
    --l1_weight 0.6 \
    --boundary_weight 0.3 \
    --focal_weight 0.2 \
    --use_amp \
    --use_boundary_guidance \
    --noise_removal \
    --sanity_check_freq 2 \
    --val_split 0.15

echo ""
echo "Expected improvements:"
echo "- Training time: ~2.4hrs → ~15-20min per epoch"
echo "- GPU utilization: All 4 GPUs should be >80%"
echo "- Memory usage: ~3GB per GPU"
echo "- Convergence: Visible improvements by epoch 5"
