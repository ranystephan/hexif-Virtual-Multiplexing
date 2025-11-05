#!/usr/bin/env python3
"""
H&E → Orion (20-channel) Patch-based Training with per-channel metrics, grid/rand sampling, and optional K-fold.

Data layout (pairs_dir):
  core_###_HE.npy     -> float32 (H, W, 3) in [0,1] or [0,255]
  core_###_ORION.npy  -> float32 (H, W, 20) or (20, H, W)

Examples
--------
# Simple 80/20 split; random train, grid val; AMP; sample PNG every 10 epochs
python train_orion_patches.py \
  --pairs_dir core_patches_npy \
  --output_dir orion_patches_model \
  --epochs 60 --batch_size 256 --learning_rate 1e-4 \
  --use_amp --patch_size 128 --label_patch 0 \
  --patches_per_image 4096 --patches_per_image_val 1024 \
  --num_workers 8

# 5-fold CV (train on folds != 0, validate on fold 0)
python train_orion_patches.py --kfolds 5 --fold 0 ...

Notes
-----
- Train uses random sampling (better morphology variety).
- Val uses grid tiling by default (uniform coverage); set --val_sampling random to match train.
- Per-epoch CSVs:
   * metrics_overall.csv            (train/val totals)
   * val_per_channel.csv            (20 rows/epoch; per-channel MSE/MAE/SSIM-loss)
- PNG every N epochs (default 10): predictions_epoch_XX.png
"""

import os
import json
import time
import math
import random
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------ Repro & Logging ------------------------ #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"

    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logs → {log_file}")


def robust_norm(img: np.ndarray, p1=1, p99=99, eps=1e-6):
    lo, hi = np.percentile(img, (p1, p99))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo + eps), 0, 1).astype(np.float32)


# ------------------------ Dataset ------------------------ #

class OrionPatchDataset(Dataset):
    """
    Patch sampler over paired cores with two modes:

    - sampling='random': length = #cores * patches_per_image; each __getitem__ draws a fresh random crop
    - sampling='grid'  : deterministic tiling with stride; length = total grid positions across cores

    Expects *_HE.npy (H,W,3) and *_ORION.npy ((H,W,20) or (20,H,W)).
    """
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        patches_per_image: int = 4096,
        patch_size: int = 128,
        label_patch: int = 0,
        augment: bool = True,
        he_color_jitter: bool = False,
        sampling: str = "random",         # "random" or "grid"
        grid_stride: int = None,          # default: patch_size
        advanced_augment: bool = False,   # use advanced augmentation
    ):
        assert sampling in ("random", "grid")
        self.pairs_dir = Path(pairs_dir)
        self.basenames = basenames
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.label_patch = label_patch
        self.augment = augment
        self.he_color_jitter = he_color_jitter
        self.sampling = sampling
        self.grid_stride = grid_stride or (patch_size // 2)  # Default to 50% overlap
        self.advanced_augment = advanced_augment

        # Pair paths
        self.he_paths  = [self.pairs_dir / f"{b}_HE.npy"    for b in basenames]
        self.ori_paths = [self.pairs_dir / f"{b}_ORION.npy" for b in basenames]
        for hp, op in zip(self.he_paths, self.ori_paths):
            if not hp.exists() or not op.exists():
                raise FileNotFoundError(f"Pair missing: {hp} / {op}")
        
        # Smart caching: cache most frequently accessed cores
        self._cache = {}
        self._cache_hits = {}  # Track access frequency
        self._max_cache_size = min(10, len(basenames))  # Cache top 10 cores only

        # Shapes - load lazily to avoid blocking during init
        self._shapes = []
        logging.info(f"Initializing dataset with {len(self.ori_paths)} files...")
        for i, op in enumerate(self.ori_paths):
            if i % 50 == 0:  # Progress logging
                logging.info(f"Loading shape info: {i}/{len(self.ori_paths)}")
            try:
                arr = np.load(op, mmap_mode="r")
                if arr.ndim == 3 and arr.shape[0] == 20:   # (20,H,W)
                    H, W = arr.shape[1], arr.shape[2]
                elif arr.ndim == 3 and arr.shape[2] == 20: # (H,W,20)
                    H, W = arr.shape[0], arr.shape[1]
                else:
                    raise RuntimeError(f"Unexpected ORION shape {arr.shape} for {op}")
                self._shapes.append((H, W))
            except Exception as e:
                logging.error(f"Error loading {op}: {e}")
                raise

        # Indexing
        if self.sampling == "random":
            self._len = len(self.basenames) * self.patches_per_image
            self._grid = None
        else:
            # Precompute grid positions across cores
            logging.info(f"Precomputing grid positions for {self.sampling} sampling...")
            ps, st = self.patch_size, self.grid_stride
            grid = []
            total_positions = 0
            for i, (H, W) in enumerate(self._shapes):
                # Ensure at least one position
                ys = [0] if H <= ps else list(range(0, max(1, H - ps) + 1, st))
                xs = [0] if W <= ps else list(range(0, max(1, W - ps) + 1, st))
                positions_this_core = len(ys) * len(xs)
                total_positions += positions_this_core
                if i % 50 == 0:  # Progress logging
                    logging.info(f"Grid computation: {i}/{len(self._shapes)} cores, {total_positions} positions so far")
                for y in ys:
                    for x in xs:
                        grid.append((i, y, x))
            self._grid = grid
            self._len = len(grid)
            logging.info(f"Grid computation complete: {len(grid)} total positions")

    def __len__(self):
        return self._len

    @staticmethod
    def _pad_if_needed(img, tgt_h, tgt_w):
        H, W = img.shape[:2]
        if H >= tgt_h and W >= tgt_w:
            return img
        pad_h = max(0, tgt_h - H)
        pad_w = max(0, tgt_w - W)
        if img.ndim == 3:
            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        else:
            return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')

    def _load_pair(self, idx_core: int):
        # Smart caching: keep frequently accessed cores in memory
        if idx_core in self._cache:
            self._cache_hits[idx_core] = self._cache_hits.get(idx_core, 0) + 1
            return self._cache[idx_core]

        # Load from disk without forcing full-array materialization
        he = np.load(self.he_paths[idx_core], mmap_mode='r')
        ori = np.load(self.ori_paths[idx_core], mmap_mode='r')

        # Transpose Orion to (H,W,20) lazily if needed
        if ori.ndim == 3 and ori.shape[0] == 20:
            ori = np.transpose(ori, (1, 2, 0))

        # Cache management: only cache if we have space or if this core is accessed frequently
        self._cache_hits[idx_core] = self._cache_hits.get(idx_core, 0) + 1

        if len(self._cache) < self._max_cache_size:
            # Cache it - we have space (store views; avoids extra copies)
            self._cache[idx_core] = (he, ori)
        elif self._cache_hits[idx_core] > 2:  # Accessed multiple times
            # Evict least frequently used core
            if self._cache_hits:
                lfu_core = min(self._cache_hits.keys(), key=lambda k: self._cache_hits[k] if k in self._cache else 0)
                if lfu_core in self._cache:
                    del self._cache[lfu_core]
            self._cache[idx_core] = (he, ori)

        return he, ori

    @staticmethod
    def _rand_crop_coords(H, W, patch_size):
        if H <= patch_size or W <= patch_size:
            return 0, 0
        y0 = random.randint(0, H - patch_size)
        x0 = random.randint(0, W - patch_size)
        return y0, x0

    def _augment_sync(self, he_patch, ori_patch):
        """Enhanced augmentation preserving fine structures"""
        # Geometric augmentations (structure-preserving)
        if random.random() < 0.5:
            he_patch = np.flip(he_patch, axis=0).copy()
            ori_patch = np.flip(ori_patch, axis=0).copy()
        if random.random() < 0.5:
            he_patch = np.flip(he_patch, axis=1).copy()
            ori_patch = np.flip(ori_patch, axis=1).copy()
        
        # 0/90/180/270 rotations
        k = random.randint(0, 3)
        if k:
            he_patch = np.rot90(he_patch, k, axes=(0, 1)).copy()
            ori_patch = np.rot90(ori_patch, k, axes=(0, 1)).copy()
        
        # Advanced color augmentation (H&E only) - more careful to preserve tissue structure
        if self.he_color_jitter:
            # Brightness with structure preservation
            if random.random() < 0.4:  # Reduced probability to avoid over-augmentation
                factor = 0.85 + 0.3 * random.random()  # More conservative range
                he_patch = np.clip(he_patch * factor, 0, 1)
            
            # Contrast enhancement preserving edge information
            if random.random() < 0.4:
                mean = he_patch.mean(axis=(0, 1), keepdims=True)
                factor = 0.8 + 0.4 * random.random()  # More conservative
                he_patch = np.clip((he_patch - mean) * factor + mean, 0, 1)
            
            # Saturation adjustment (helps with stain variation)
            if random.random() < 0.3:
                # Convert to HSV-like adjustment
                gray = np.mean(he_patch, axis=2, keepdims=True)
                factor = 0.7 + 0.6 * random.random()
                he_patch = gray + factor * (he_patch - gray)
                he_patch = np.clip(he_patch, 0, 1)
            
            # Gaussian noise (very mild to simulate acquisition noise)
            if random.random() < 0.2:
                noise = np.random.normal(0, 0.01, he_patch.shape).astype(np.float32)
                he_patch = np.clip(he_patch + noise, 0, 1)
        
        return he_patch, ori_patch
    
    def _advanced_augment_sync(self, he_patch, ori_patch):
        """Advanced augmentation specifically designed for small structure preservation"""
        # Start with basic augmentations
        he_patch, ori_patch = self._augment_sync(he_patch, ori_patch)
        
        # Elastic deformation (very mild to preserve small structures)
        if random.random() < 0.15:  # Low probability for subtle deformation
            from scipy.ndimage import map_coordinates
            import numpy as np
            
            # Create deformation field
            shape = he_patch.shape[:2]
            dx = np.random.uniform(-2, 2, size=shape).astype(np.float32)
            dy = np.random.uniform(-2, 2, size=shape).astype(np.float32)
            
            # Apply Gaussian smoothing to make deformation more natural
            from scipy.ndimage import gaussian_filter
            dx = gaussian_filter(dx, sigma=3)
            dy = gaussian_filter(dy, sigma=3)
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # Apply deformation to both patches
            try:
                for c in range(he_patch.shape[2]):
                    he_patch[:, :, c] = map_coordinates(he_patch[:, :, c], indices, 
                                                      order=1, mode='reflect').reshape(shape)
                for c in range(ori_patch.shape[2]):
                    ori_patch[:, :, c] = map_coordinates(ori_patch[:, :, c], indices, 
                                                       order=1, mode='reflect').reshape(shape)
            except:
                pass  # Skip if deformation fails
        
        # Small random crops and padding (to simulate different scales)
        if random.random() < 0.2:
            crop_size = random.randint(max(1, int(0.9 * min(he_patch.shape[:2]))), 
                                     min(he_patch.shape[:2]))
            if crop_size < min(he_patch.shape[:2]):
                # Random crop
                h, w = he_patch.shape[:2]
                top = random.randint(0, h - crop_size)
                left = random.randint(0, w - crop_size)
                
                he_crop = he_patch[top:top+crop_size, left:left+crop_size]
                ori_crop = ori_patch[top:top+crop_size, left:left+crop_size]
                
                # Resize back to original size
                from scipy.ndimage import zoom
                scale_h = h / crop_size
                scale_w = w / crop_size
                
                he_patch = zoom(he_crop, (scale_h, scale_w, 1), order=1)
                ori_patch = zoom(ori_crop, (scale_h, scale_w, 1), order=1)
        
        return he_patch, ori_patch

    def __getitem__(self, idx: int):
        ps = self.patch_size

        if self.sampling == "random":
            idx_core = idx // self.patches_per_image
            he, ori = self._load_pair(idx_core)
            he = self._pad_if_needed(he, ps, ps)
            ori = self._pad_if_needed(ori, ps, ps)
            H, W = he.shape[:2]
            y0, x0 = self._rand_crop_coords(H, W, ps)
        else:
            idx_core, y0, x0 = self._grid[idx]
            he, ori = self._load_pair(idx_core)
            he = self._pad_if_needed(he, ps, ps)
            ori = self._pad_if_needed(ori, ps, ps)

        he_patch  = he[y0:y0+ps, x0:x0+ps, :]     # (ps,ps,3)
        ori_patch = ori[y0:y0+ps, x0:x0+ps, :]    # (ps,ps,20)

        # Convert dtypes and scale lazily per-patch to avoid full-image scans
        if he_patch.dtype == np.uint8:
            he_patch = he_patch.astype(np.float32) / 255.0
        elif he_patch.dtype != np.float32:
            he_patch = he_patch.astype(np.float32)
        # If float but in 0..255 range, patch-normalize
        if he_patch.max(initial=0.0) > 1.5:
            he_patch = he_patch / 255.0

        if ori_patch.dtype == np.uint8:
            ori_patch = ori_patch.astype(np.float32) / 255.0
        elif ori_patch.dtype != np.float32:
            ori_patch = ori_patch.astype(np.float32)
        if ori_patch.max(initial=0.0) > 1.5:
            ori_patch = ori_patch / 255.0

        if self.augment and self.sampling == "random":
            if self.advanced_augment:
                he_patch, ori_patch = self._advanced_augment_sync(he_patch, ori_patch)
            else:
                he_patch, ori_patch = self._augment_sync(he_patch, ori_patch)

        he_t  = torch.from_numpy(he_patch.transpose(2, 0, 1))   # (3,ps,ps)
        ori_t = torch.from_numpy(ori_patch.transpose(2, 0, 1))  # (20,ps,ps)
        return he_t, ori_t


def discover_basenames(pairs_dir: str) -> List[str]:
    pairs_dir = Path(pairs_dir)
    bases = []
    for hef in sorted(pairs_dir.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (pairs_dir / f"{base}_ORION.npy").exists():
            bases.append(base)
    return bases


def split_train_val_simple(bases: List[str], val_frac=0.2, seed=42):
    rng = random.Random(seed)
    b = bases[:]
    rng.shuffle(b)
    n_val = int(round(len(b)*val_frac))
    return b[n_val:], b[:n_val]


def kfold_split(bases: List[str], kfolds: int, fold: int, seed=42):
    rng = random.Random(seed)
    b = bases[:]
    rng.shuffle(b)
    folds = [b[i::kfolds] for i in range(kfolds)]
    val_b = folds[fold]
    train_b = [x for i,f in enumerate(folds) if i != fold for x in f]
    return train_b, val_b


# ------------------------ Advanced Model Components ------------------------ #

class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) from CBAM"""
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) from CBAM"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class DenseBlock(nn.Module):
    """Dense Block for better feature reuse"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
            )
            self.layers.append(layer)
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """Transition layer to reduce feature map size"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.pool = nn.AvgPool2d(2, stride=2)
        
    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


class EnhancedEncoder(nn.Module):
    """Enhanced encoder with dense blocks and attention"""
    def __init__(self, in_ch, out_ch, use_dense=True):
        super().__init__()
        self.use_dense = use_dense
        
        if use_dense:
            self.dense_block = DenseBlock(in_ch, growth_rate=16, num_layers=3)
            dense_out = in_ch + 3 * 16  # in_ch + num_layers * growth_rate
            self.transition = nn.Conv2d(dense_out, out_ch, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(out_ch)
            
        self.attention = CBAM(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.use_dense:
            x = self.dense_block(x)
            x = self.transition(x)
        else:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            
        x = self.attention(x)
        skip = x
        x = self.pool(x)
        return x, skip


class EnhancedDecoder(nn.Module):
    """Enhanced decoder with attention and residual connections"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        
        # Multi-scale feature fusion
        self.conv1 = nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        # Residual connection
        self.residual = nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 1) if (in_ch // 2 + skip_ch) != out_ch else nn.Identity()
        
        self.attention = CBAM(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        concat_feat = torch.cat([x, skip], dim=1)
        
        # Main path
        out = self.relu(self.bn1(self.conv1(concat_feat)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        
        # Residual connection
        residual = self.residual(concat_feat)
        out = out + residual
        
        # Apply attention
        out = self.attention(out)
        
        return out


class EdgeEnhancementModule(nn.Module):
    """Module to enhance edge detection for fine details"""
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Sobel-like filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, x):
        # Apply edge enhancement
        enhanced = self.edge_conv(x)
        
        # Ensure Sobel filters match input dtype and device for AMP compatibility
        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device)
        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device)
        
        # Compute edge maps for first channel as reference
        x_gray = torch.mean(x, dim=1, keepdim=True)
        edge_x = F.conv2d(x_gray, sobel_x, padding=1)
        edge_y = F.conv2d(x_gray, sobel_y, padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2)
        
        # Broadcast edge information to all channels
        edges = edges.expand(-1, x.size(1), -1, -1)
        
        # Combine original features with edge-enhanced features
        return enhanced + x * (1 + 0.1 * edges)


class HE2OrionUNetAdvanced(nn.Module):
    """Advanced UNet with attention, dense blocks, and multi-scale features"""
    def __init__(self, in_channels=3, out_channels=20, base=64):
        super().__init__()
        
        # Encoder with dense blocks and attention
        self.enc1 = EnhancedEncoder(in_channels, base, use_dense=False)  # First layer regular
        self.enc2 = EnhancedEncoder(base, base*2, use_dense=True)
        self.enc3 = EnhancedEncoder(base*2, base*4, use_dense=True)
        self.enc4 = EnhancedEncoder(base*4, base*8, use_dense=True)

        # Enhanced bottleneck with attention
        self.bottleneck = nn.Sequential(
            DenseBlock(base*8, growth_rate=32, num_layers=4),
            nn.Conv2d(base*8 + 4*32, base*16, 1),  # Transition
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            CBAM(base*16),
            nn.Dropout2d(0.2),
        )

        # Enhanced decoder with attention and residual connections
        self.dec4 = EnhancedDecoder(base*16, base*8, base*8)
        self.dec3 = EnhancedDecoder(base*8, base*4, base*4)
        self.dec2 = EnhancedDecoder(base*4, base*2, base*2)
        self.dec1 = EnhancedDecoder(base*2, base, base)
        
        # Edge enhancement for fine details
        self.edge_enhance = EdgeEnhancementModule(base)
        
        # Multi-scale output heads for deep supervision
        self.out_main = nn.Conv2d(base, out_channels, 1)
        self.out_aux1 = nn.Conv2d(base*2, out_channels, 1)  # From dec2
        self.out_aux2 = nn.Conv2d(base*4, out_channels, 1)  # From dec3
        
        # Final refinement layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Encoder
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decoder with multi-scale outputs
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        aux2 = torch.sigmoid(self.out_aux2(d3))  # Auxiliary output
        
        d2 = self.dec2(d3, s2)
        aux1 = torch.sigmoid(self.out_aux1(d2))  # Auxiliary output
        
        d1 = self.dec1(d2, s1)
        
        # Edge enhancement for fine details
        d1 = self.edge_enhance(d1)
        
        # Main output
        main_out = torch.sigmoid(self.out_main(d1))
        
        # Multi-scale fusion during training
        if self.training:
            # Upsample auxiliary outputs to match main output size
            aux2_up = F.interpolate(aux2, size=main_out.shape[2:], mode='bilinear', align_corners=False)
            aux1_up = F.interpolate(aux1, size=main_out.shape[2:], mode='bilinear', align_corners=False)
            
            # Combine multi-scale features
            combined = torch.cat([main_out, aux1_up, aux2_up], dim=1)
            final_out = torch.sigmoid(self.final_conv(combined))
            
            return final_out, main_out, aux1, aux2
        else:
            return main_out


# Keep the original model as fallback
class HE2OrionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=20, base=64):
        super().__init__()
        self.enc1 = EnhancedEncoder(in_channels, base, use_dense=False)
        self.enc2 = EnhancedEncoder(base, base*2, use_dense=False)
        self.enc3 = EnhancedEncoder(base*2, base*4, use_dense=False)
        self.enc4 = EnhancedEncoder(base*4, base*8, use_dense=False)

        self.bot = nn.Sequential(
            nn.Conv2d(base*8, base*16, 3, padding=1),
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*16, base*16, 3, padding=1),
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            CBAM(base*16),
            nn.Dropout2d(0.2),
        )

        self.dec4 = EnhancedDecoder(base*16, base*8, base*8)
        self.dec3 = EnhancedDecoder(base*8, base*4, base*4)
        self.dec2 = EnhancedDecoder(base*4, base*2, base*2)
        self.dec1 = EnhancedDecoder(base*2, base, base)
        self.out  = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b = self.bot(x4)
        x = self.dec4(b, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.out(x)
        return torch.sigmoid(x)


# ------------------------ Advanced Loss Functions ------------------------ #

class FocalLoss(nn.Module):
    """Modified Focal Loss for regression targets (continuous values 0-1)"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        # For regression targets, use L1 loss with focal weighting
        # Focus on hard examples where prediction differs significantly from target
        diff = torch.abs(pred - target)
        
        # Focal weight based on prediction difficulty
        # Higher weight for larger errors (harder examples)
        focal_weight = self.alpha * (diff ** self.gamma)
        
        # Apply focal weighting to L1 loss
        focal_loss = focal_weight * diff
        return focal_loss.mean()


class EdgeLoss(nn.Module):
    """Edge-aware loss to enhance boundary detection"""
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, pred, target):
        # Ensure Sobel filters match input dtype and device for AMP compatibility
        sobel_x = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sobel_y = self.sobel_y.to(dtype=pred.dtype, device=pred.device)
        
        # Compute edges for each channel
        edge_loss = 0
        for i in range(pred.shape[1]):
            pred_ch = pred[:, i:i+1, :, :]
            target_ch = target[:, i:i+1, :, :]
            
            # Compute gradients
            pred_edge_x = F.conv2d(pred_ch, sobel_x, padding=1)
            pred_edge_y = F.conv2d(pred_ch, sobel_y, padding=1)
            target_edge_x = F.conv2d(target_ch, sobel_x, padding=1)
            target_edge_y = F.conv2d(target_ch, sobel_y, padding=1)
            
            # Compute edge magnitudes
            pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
            target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)
            
            # L1 loss on edges
            edge_loss += F.l1_loss(pred_edge, target_edge)
            
        return edge_loss / pred.shape[1]


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss, good for imbalanced data"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class AdvancedCombinedLoss(nn.Module):
    """
    Advanced loss combining multiple loss functions for better fine detail detection:
    - MSE/L1 for basic reconstruction
    - Focal loss for hard examples
    - Edge loss for boundary preservation
    - Tversky loss for small object detection
    - SSIM for structural similarity
    - Multi-scale supervision
    """
    def __init__(self, mse_w=1.0, l1_w=0.5, focal_w=0.8, edge_w=0.3, tversky_w=0.4, ssim_w=0.2, label_crop: int = 0):
        super().__init__()
        self.mse_w = mse_w
        self.l1_w = l1_w
        self.focal_w = focal_w
        self.edge_w = edge_w
        self.tversky_w = tversky_w
        self.ssim_w = ssim_w
        self.label_crop = label_crop
        
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.edge_loss = EdgeLoss()
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, pred, target, aux_outputs=None):
        # Handle multi-scale outputs from advanced model
        if isinstance(pred, tuple):
            main_pred, aux1, aux2 = pred[0], pred[2], pred[3]  # Skip main_out duplicate
            pred = pred[0]  # Use final output for main loss
        else:
            main_pred = pred
            aux1 = aux2 = None
            
        # Apply label crop if specified
        if self.label_crop and self.label_crop < pred.shape[-1]:
            c = self.label_crop
            ps = pred.shape[-1]
            s = (ps - c) // 2
            e = s + c
            pred = pred[..., s:e, s:e]
            target = target[..., s:e, s:e]
            if aux1 is not None:
                aux1 = F.interpolate(aux1, size=(c, c), mode='bilinear', align_corners=False)
            if aux2 is not None:
                aux2 = F.interpolate(aux2, size=(c, c), mode='bilinear', align_corners=False)

        # Main losses
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        focal = self.focal_loss(pred, target)
        edge = self.edge_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        ssim_loss = 1.0 - self._ssim(pred, target)
        
        # Main loss combination
        main_loss = (self.mse_w * mse + 
                    self.l1_w * mae + 
                    self.focal_w * focal + 
                    self.edge_w * edge + 
                    self.tversky_w * tversky + 
                    self.ssim_w * ssim_loss)
        
        total_loss = main_loss
        
        # Add auxiliary losses for deep supervision (if available)
        if aux1 is not None:
            aux1_loss = (0.4 * F.mse_loss(aux1, target) + 
                        0.3 * self.focal_loss(aux1, target) + 
                        0.2 * self.tversky_loss(aux1, target))
            total_loss = total_loss + 0.3 * aux1_loss
            
        if aux2 is not None:
            aux2_loss = (0.4 * F.mse_loss(aux2, target) + 
                        0.3 * self.focal_loss(aux2, target) + 
                        0.2 * self.tversky_loss(aux2, target))
            total_loss = total_loss + 0.2 * aux2_loss

        metrics = {
            "mse": float(mse),
            "mae": float(mae), 
            "focal": float(focal),
            "edge": float(edge),
            "tversky": float(tversky),
            "ssim_loss": float(ssim_loss),
            "total": float(total_loss)
        }
        
        return total_loss, metrics

    @staticmethod
    def _ssim(x, y, window=11):
        mu_x = F.avg_pool2d(x, window, 1, window//2)
        mu_y = F.avg_pool2d(y, window, 1, window//2)
        mu_x2, mu_y2 = mu_x.pow(2), mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x2 = F.avg_pool2d(x*x, window, 1, window//2) - mu_x2
        sigma_y2 = F.avg_pool2d(y*y, window, 1, window//2) - mu_y2
        sigma_xy = F.avg_pool2d(x*y, window, 1, window//2) - mu_xy
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-12)
        return ssim_map.mean()


# Keep original loss as fallback
class CombinedLoss(nn.Module):
    """MSE + L1 + (1-SSIM) on full patch or center label_crop×label_crop."""
    def __init__(self, mse_w=1.0, l1_w=0.5, ssim_w=0.3, label_crop: int = 0):
        super().__init__()
        self.mse_w, self.l1_w, self.ssim_w = mse_w, l1_w, ssim_w
        self.label_crop = label_crop

    def forward(self, pred, target):
        # Handle multi-output case
        if isinstance(pred, tuple):
            pred = pred[0]  # Use main output
            
        if self.label_crop and self.label_crop < pred.shape[-1]:
            c = self.label_crop
            ps = pred.shape[-1]
            s = (ps - c) // 2
            e = s + c
            pred = pred[..., s:e, s:e]
            target = target[..., s:e, s:e]

        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        ssim_loss = 1.0 - self._ssim(pred, target)
        total = self.mse_w*mse + self.l1_w*mae + self.ssim_w*ssim_loss
        return total, {"mse": float(mse), "mae": float(mae), "ssim_loss": float(ssim_loss), "total": float(total)}

    @staticmethod
    def _ssim(x, y, window=11):
        mu_x = F.avg_pool2d(x, window, 1, window//2)
        mu_y = F.avg_pool2d(y, window, 1, window//2)
        mu_x2, mu_y2 = mu_x.pow(2), mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x2 = F.avg_pool2d(x*x, window, 1, window//2) - mu_x2
        sigma_y2 = F.avg_pool2d(y*y, window, 1, window//2) - mu_y2
        sigma_xy = F.avg_pool2d(x*y, window, 1, window//2) - mu_xy
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-12)
        return ssim_map.mean()


# ------------------------ Viz Helpers ------------------------ #

def _make_tint_cmap(rgb, name):
    cdict = {
        'red':   ((0, 0, 0), (1, rgb[0], rgb[0])),
        'green': ((0, 0, 0), (1, rgb[1], rgb[1])),
        'blue':  ((0, 0, 0), (1, rgb[2], rgb[2])),
    }
    return LinearSegmentedColormap(name, cdict)

BASE_COLORS = [
    (0.10,0.55,0.95), (0.95,0.30,0.30), (0.10,0.80,0.35), (0.85,0.50,0.10), (0.65,0.30,0.85),
    (0.15,0.85,0.85), (0.95,0.65,0.15), (0.95,0.15,0.65), (0.35,0.95,0.20), (0.25,0.75,0.95),
    (0.95,0.45,0.45), (0.35,0.85,0.55), (0.85,0.35,0.15), (0.55,0.35,0.85), (0.20,0.90,0.75),
    (0.95,0.80,0.20), (0.85,0.20,0.55), (0.20,0.65,0.95), (0.60,0.90,0.25), (0.75,0.30,0.95),
]
CMAPS = [_make_tint_cmap(rgb, f"m{i}") for i, rgb in enumerate(BASE_COLORS)]


@torch.no_grad()
def save_sample_grid(
    model,
    val_loader,
    device,
    out_png: Path,
    label_crop: int = 0,
    marker_names: List[str] = None,
    gamma: float = 0.85,
):
    """Save H&E + Orion max, then 20×(GT,Pred) per-channel with tint colormaps."""
    model.eval()
    os.makedirs(out_png.parent, exist_ok=True)

    for he, ori in val_loader:
        he = he.to(device, non_blocking=True)
        ori = ori.to(device, non_blocking=True)
        if device.type == 'cuda':
            he = he.contiguous(memory_format=torch.channels_last)
        pred = model(he)
        if label_crop and label_crop < pred.shape[-1]:
            c = label_crop
            ps = pred.shape[-1]
            s = (ps - c) // 2
            e = s + c
            pred_vis = pred[:, :, s:e, s:e]
            ori_vis  = ori[:, :, s:e, s:e]
            he_vis   = he
        else:
            pred_vis, ori_vis, he_vis = pred, ori, he
        he_np = he_vis[0].float().cpu().numpy().transpose(1,2,0)
        gt_np = ori_vis[0].float().cpu().numpy()
        pr_np = pred_vis[0].float().cpu().numpy()
        break

    he_disp = he_np.copy()
    for c in range(min(3, he_disp.shape[2])):
        he_disp[..., c] = robust_norm(he_disp[..., c])
    max_proj = robust_norm(gt_np.max(axis=0))

    rows, cols = 21, 2
    fig_h = 3 + 20*1.6
    fig, ax = plt.subplots(rows, cols, figsize=(9, fig_h))
    ax[0,0].imshow(he_disp); ax[0,0].set_title("H&E"); ax[0,0].axis('off')
    ax[0,1].imshow(max_proj**gamma, cmap='gray'); ax[0,1].set_title("Orion max-proj"); ax[0,1].axis('off')
    for i in range(20):
        r = i+1
        gt_ch = robust_norm(gt_np[i])**gamma
        pr_ch = robust_norm(pr_np[i])**gamma
        title = marker_names[i] if marker_names and i < len(marker_names) else f"Marker {i}"
        cmap = CMAPS[i]
        ax[r,0].imshow(gt_ch, cmap=cmap, vmin=0, vmax=1); ax[r,0].set_title(f"{title} — GT", color=BASE_COLORS[i], fontsize=9); ax[r,0].axis('off')
        ax[r,1].imshow(pr_ch, cmap=cmap, vmin=0, vmax=1); ax[r,1].set_title(f"{title} — Pred", color=BASE_COLORS[i], fontsize=9); ax[r,1].axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close('all')


# ------------------------ Train / Validate ------------------------ #

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, gradient_accumulation_steps=1):
    model.train()
    running = {"loss":0., "mse":0., "mae":0., "ssim_loss":0.}
    n = 0
    epoch_start_time = time.time()
    logging.info(f"Starting training epoch with {len(loader)} batches (grad accum: {gradient_accumulation_steps})...")
    
    for batch_idx, (he, ori) in enumerate(loader):
        if batch_idx == 0:
            logging.info("First batch loaded successfully!")
        
        # Progress logging with timing
        if batch_idx % 50 == 0 and batch_idx > 0:
            elapsed = time.time() - epoch_start_time
            batches_per_sec = batch_idx / elapsed
            eta_minutes = (len(loader) - batch_idx) / batches_per_sec / 60 if batches_per_sec > 0 else 0
            logging.info(f"Batch {batch_idx}/{len(loader)} | Speed: {batches_per_sec:.2f} batch/sec | ETA: {eta_minutes:.1f}min")
        
        he = he.to(device, non_blocking=True)
        ori = ori.to(device, non_blocking=True)

        # Gradient accumulation
        if scaler is not None:
            device_type = 'cuda'
            amp_dtype = torch.float16
            with torch.amp.autocast(device_type, enabled=True, dtype=amp_dtype):
                pred = model(he)
                loss, metrics = criterion(pred, ori)
                loss = loss / gradient_accumulation_steps  # Scale loss
            scaler.scale(loss).backward()
        else:
            # Support AMP on MPS/CPU without GradScaler
            if device.type in ("mps", "cpu") and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
                with torch.amp.autocast('mps', enabled=True, dtype=torch.float16):
                    pred = model(he)
                    loss, metrics = criterion(pred, ori)
                    loss = loss / gradient_accumulation_steps
                loss.backward()
            else:
                pred = model(he)
                loss, metrics = criterion(pred, ori)
                loss = loss / gradient_accumulation_steps
                loss.backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = he.size(0)
        n += bs
        running["loss"] += metrics["total"] * bs
        running["mse"]  += metrics["mse"]   * bs
        running["mae"]  += metrics["mae"]   * bs
        running["ssim_loss"] += metrics["ssim_loss"] * bs

    # Final update if there are remaining gradients
    if len(loader) % gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for k in running:
        running[k] /= max(1, n)
    return running


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Returns:
      overall: dict(loss, mse, mae, ssim_loss)
      per_ch: dict with keys 'mse', 'mae', 'ssim_loss' each -> np.array(20,)
    """
    model.eval()
    # overall
    running = {"loss":0., "mse":0., "mae":0., "ssim_loss":0.}
    n = 0
    # per-channel accumulators (weighted by batch size)
    mse_c_sum = torch.zeros(20, dtype=torch.float64, device=device)
    mae_c_sum = torch.zeros(20, dtype=torch.float64, device=device)
    ssim_c_sum = torch.zeros(20, dtype=torch.float64, device=device)
    n_samples = 0

    for he, ori in loader:
        he = he.to(device, non_blocking=True)
        ori = ori.to(device, non_blocking=True)
        pred = model(he)
        loss, metrics = criterion(pred, ori)

        bs = he.size(0)
        n += bs
        running["loss"] += metrics["total"] * bs
        running["mse"]  += metrics["mse"]   * bs
        running["mae"]  += metrics["mae"]   * bs
        running["ssim_loss"] += metrics["ssim_loss"] * bs

        # Per-channel MSE/MAE (mean across B,H,W)
        diff = pred - ori  # (B,20,H,W)
        mse_c = (diff**2).mean(dim=(0,2,3))  # (20,)
        mae_c = diff.abs().mean(dim=(0,2,3)) # (20,)

        # Per-channel SSIM-loss
        ssim_losses = []
        for c in range(20):
            ssim_c = CombinedLoss._ssim(pred[:, c:c+1], ori[:, c:c+1])
            ssim_losses.append(1.0 - float(ssim_c))
        ssim_c = torch.tensor(ssim_losses, dtype=torch.float64, device=device)

        mse_c_sum += mse_c.to(dtype=torch.float64) * bs
        mae_c_sum += mae_c.to(dtype=torch.float64) * bs
        ssim_c_sum += ssim_c * bs
        n_samples += bs

    for k in running:
        running[k] /= max(1, n)

    per_ch = {
        "mse":  (mse_c_sum / max(1, n_samples)).detach().cpu().numpy(),
        "mae":  (mae_c_sum / max(1, n_samples)).detach().cpu().numpy(),
        "ssim": (ssim_c_sum / max(1, n_samples)).detach().cpu().numpy(),  # this is (1-SSIM) averaged
    }
    return running, per_ch


def append_csv(path: Path, headers: List[str], values: List):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(headers) + "\n")
        row = []
        for v in values:
            if isinstance(v, (float, int, np.floating, np.integer)):
                row.append(f"{v}")
            else:
                row.append(str(v))
        f.write(",".join(row) + "\n")


# ------------------------ Main ------------------------ #

def make_loaders(args, device):
    bases = discover_basenames(args.pairs_dir)
    if not bases:
        raise RuntimeError(f"No pairs found in {args.pairs_dir}")
    logging.info(f"Discovered {len(bases)} paired cores")

    if args.kfolds > 1:
        train_b, val_b = kfold_split(bases, args.kfolds, args.fold, seed=args.seed)
        logging.info(f"K-fold {args.kfolds}, fold={args.fold} → Train: {len(train_b)} | Val: {len(val_b)}")
    else:
        train_b, val_b = split_train_val_simple(bases, val_frac=args.val_split, seed=args.seed)
        logging.info(f"Train cores: {len(train_b)} | Val cores: {len(val_b)}")

    train_ds = OrionPatchDataset(
        args.pairs_dir, train_b,
        patches_per_image=args.patches_per_image,
        patch_size=args.patch_size,
        label_patch=args.label_patch,
        augment=True,
        he_color_jitter=args.he_color_jitter,
        sampling=args.train_sampling,
        grid_stride=args.grid_stride,
        advanced_augment=args.advanced_augment,
    )
    val_ds = OrionPatchDataset(
        args.pairs_dir, val_b,
        patches_per_image=args.patches_per_image_val,
        patch_size=args.patch_size,
        label_patch=args.label_patch,
        augment=False,
        he_color_jitter=False,
        sampling=args.val_sampling,
        grid_stride=args.grid_stride,
    )

    # Optimized worker settings: trust user-provided workers; tune pin_memory by device
    total_patches = len(train_ds)
    safe_workers = max(0, args.num_workers)
    pin_mem = (device.type == 'cuda')
    logging.info(f"Dataset size: {total_patches} patches, using {safe_workers} workers (pin_memory={pin_mem})")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(args.train_sampling=="random"),
        num_workers=safe_workers, pin_memory=pin_mem, drop_last=True,
        persistent_workers=(safe_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, safe_workers//2), pin_memory=pin_mem, drop_last=False,
        persistent_workers=(safe_workers > 0),
    )
    logging.info(f"Using {safe_workers} workers for training, {max(1, safe_workers//2)} for validation")
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str, default="core_patches_npy")
    p.add_argument("--output_dir", type=str, default="orion_patches_model")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=2)  # Very small batch size for 512x512 patches
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--label_patch", type=int, default=0, help="center-crop size for loss; 0=full patch")
    p.add_argument("--patches_per_image", type=int, default=16)  # Much fewer patches since they're larger
    p.add_argument("--patches_per_image_val", type=int, default=8)
    p.add_argument("--base_features", type=int, default=32)  # Reduce model size for 512x512 patches
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulate gradients over N steps")
    p.add_argument("--he_color_jitter", action="store_true")
    p.add_argument("--save_every", type=int, default=10, help="save sample PNG every N epochs")
    p.add_argument("--seed", type=int, default=42)

    # sampling
    p.add_argument("--train_sampling", type=str, default="random", choices=["random","grid"])
    p.add_argument("--val_sampling",   type=str, default="grid",   choices=["random","grid"])
    p.add_argument("--grid_stride", type=int, default=256, help="stride for grid sampling; default=patch_size//2 for overlap")

    # k-fold CV
    p.add_argument("--kfolds", type=int, default=1, help=">1 enables k-fold splitting")
    p.add_argument("--fold",   type=int, default=0, help="validation fold index (0..kfolds-1)")
    
    # Advanced model options
    p.add_argument("--use_advanced_model", action="store_true", help="Use advanced UNet with attention and dense blocks")
    p.add_argument("--use_advanced_loss", action="store_true", help="Use advanced loss with focal, edge, and Tversky components")
    p.add_argument("--advanced_augment", action="store_true", help="Use advanced data augmentation for small structures")

    args = p.parse_args()
    outdir = Path(args.output_dir)
    setup_logging(outdir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"))
    logging.info(f"Device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Log model configuration
    if args.use_advanced_model:
        logging.info("Using ADVANCED UNet with attention, dense blocks, and edge enhancement")
    else:
        logging.info("Using STANDARD UNet with attention")
        
    if args.use_advanced_loss:
        logging.info("Using ADVANCED loss with focal, edge, and Tversky components")
    else:
        logging.info("Using STANDARD combined loss")

    # Save config
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    train_loader, val_loader = make_loaders(args, device)

    # Model selection
    if args.use_advanced_model:
        net = HE2OrionUNetAdvanced(in_channels=3, out_channels=20, base=args.base_features)
        logging.info("Created advanced model with multi-scale supervision")
    else:
        net = HE2OrionUNet(in_channels=3, out_channels=20, base=args.base_features)
        logging.info("Created standard enhanced model")
        
    total_params = sum(p.numel() for p in net.parameters())
    logging.info(f"Model params: {total_params:,}")

    if torch.cuda.device_count() > 1:
        logging.info(f"Detected {torch.cuda.device_count()} GPUs → DataParallel")
        net = nn.DataParallel(net)
    net = net.to(device)
    if device.type == 'cuda':
        net = net.to(memory_format=torch.channels_last)

    # Loss function selection
    if args.use_advanced_loss:
        criterion = AdvancedCombinedLoss(
            mse_w=1.0, l1_w=0.5, focal_w=0.8, edge_w=0.3, 
            tversky_w=0.4, ssim_w=0.2, label_crop=args.label_patch
        )
        logging.info("Using advanced loss function")
    else:
        criterion = CombinedLoss(label_crop=args.label_patch)
        logging.info("Using standard combined loss")
        
    # Enhanced optimizer settings for better convergence
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    
    # More sophisticated learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.01
    )
    
    # AMP scaler only on CUDA; use autocast without scaler on MPS/CPU when enabled
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and device.type == 'cuda'))

    # CSVs
    overall_csv = outdir / "metrics_overall.csv"
    perch_csv   = outdir / "val_per_channel.csv"

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        trn = train_one_epoch(net, train_loader, criterion, optimizer, device, scaler=scaler, 
                             gradient_accumulation_steps=args.gradient_accumulation_steps)
        val_overall, val_perch = validate(net, val_loader, criterion, device)
        dt = time.time() - t0

        # Step cosine scheduler per epoch (no metric)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train: loss={trn['loss']:.4f} mse={trn['mse']:.4f} mae={trn['mae']:.4f} ssim={trn['ssim_loss']:.4f} | "
            f"val:   loss={val_overall['loss']:.4f} mse={val_overall['mse']:.4f} mae={val_overall['mae']:.4f} ssim={val_overall['ssim_loss']:.4f} | "
            f"lr={lr:.2e} | {dt:.1f}s"
        )
        history.append({"epoch": epoch, "train": trn, "val": val_overall, "lr": lr, "time_sec": dt})

        # Write overall CSV
        append_csv(
            overall_csv,
            headers=["epoch","split","loss","mse","mae","ssim_loss","lr","time_sec"],
            values=[epoch,"train",trn["loss"],trn["mse"],trn["mae"],trn["ssim_loss"],lr,dt],
        )
        append_csv(
            overall_csv,
            headers=["epoch","split","loss","mse","mae","ssim_loss","lr","time_sec"],
            values=[epoch,"val",val_overall["loss"],val_overall["mse"],val_overall["mae"],val_overall["ssim_loss"],lr,dt],
        )

        # Write per-channel CSV (20 rows)
        for ch in range(20):
            append_csv(
                perch_csv,
                headers=["epoch","channel","mse","mae","ssim_loss"],
                values=[epoch, ch, float(val_perch["mse"][ch]), float(val_perch["mae"][ch]), float(val_perch["ssim"][ch])],
            )

        # Best checkpoint
        if val_overall["loss"] < best_val:
            best_val = val_overall["loss"]
            state = {
                "epoch": epoch,
                "model": (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": best_val,
                "args": vars(args),
                "history": history,
            }
            torch.save(state, outdir / "best_model.pth")
            logging.info(f"  → New best saved (val_loss={best_val:.4f})")

        # Periodic checkpoint + PNG
        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = {
                "epoch": epoch,
                "model": (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_overall["loss"],
                "args": vars(args),
                "history": history,
            }
            ckpt_path = outdir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(state, ckpt_path)
            logging.info(f"  → Checkpoint saved: {ckpt_path}")

            png_path = outdir / f"predictions_epoch_{epoch}.png"
            try:
                save_sample_grid(
                    model=net, val_loader=val_loader, device=device,
                    out_png=png_path, label_crop=args.label_patch,
                    marker_names=[f"Marker {i}" for i in range(20)],
                    gamma=0.85,
                )
                logging.info(f"  → Sample predictions saved: {png_path}")
            except Exception as e:
                logging.warning(f"  → Could not save sample predictions: {e}")

    final = {
        "epoch": args.epochs,
        "model": (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val,
        "args": vars(args),
        "history": history,
    }
    torch.save(final, outdir / "final_model.pth")
    logging.info(f"Training complete. Best val loss: {best_val:.4f}")
    logging.info(f"Artifacts in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
