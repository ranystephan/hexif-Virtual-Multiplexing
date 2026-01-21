#!/usr/bin/env bash
set -euo pipefail

# Fine-tune starting from an existing checkpoint, focusing on TA118 channels.
# Update paths before running.

PAIRS_DIR="data/TA118/TA118_core_patches_npy"
OUTPUT_DIR="runs/ta118_finetune_jan18"
BASE_CKPT="runs/nov5/focal_l1_plateau/best_model.pth"
SCALER_PATH="runs/nov5/focal_l1_plateau/orion_scaler.json"
CHANNEL_STATS="hexif/experiment/results_ta118/channel_summary.json"

# Train (single GPU example). For multi-GPU, use torchrun.
python train.py \
  --pairs_dir "${PAIRS_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 12 \
  --batch_size 8 \
  --val_batch_size 8 \
  --lr 1e-4 \
  --patch_size 224 \
  --scaler_path "${SCALER_PATH}" \
  --resume_checkpoint "${BASE_CKPT}" \
  --loss_type l1 \
  --pos_frac 0.75 \
  --pos_threshold 0.10 \
  --samples_per_core 96 \
  --resample_tries 12 \
  --channel_stats_json "${CHANNEL_STATS}" \
  --channel_weight_power 1.2 \
  --channel_weight_clip 6.0 \
  --channel_sampling_temperature 0.6 \
  --speckle_boost_topk 6 \
  --w_presence 0.35 \
  --use_focal_presence \
  --use_presence_head

# Evaluate on TA118 patches (update MODEL_PATH/SCALER_PATH inside run_ta118.py if needed).
python hexif/experiment/run_ta118.py
