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
        self.grid_stride = grid_stride or patch_size

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
        
        # Load from disk
        he = np.load(self.he_paths[idx_core], mmap_mode='r').astype(np.float32)
        ori = np.load(self.ori_paths[idx_core], mmap_mode='r').astype(np.float32)
        
        # Normalize HE
        if he.max() > 1.0:
            he = he / 255.0
        
        # Transpose and normalize Orion
        if ori.ndim == 3 and ori.shape[0] == 20:
            ori = np.transpose(ori, (1, 2, 0))
        if ori.max() > 1.0:
            ori = ori / 255.0
        
        # Cache management: only cache if we have space or if this core is accessed frequently
        self._cache_hits[idx_core] = self._cache_hits.get(idx_core, 0) + 1
        
        if len(self._cache) < self._max_cache_size:
            # Cache it - we have space
            self._cache[idx_core] = (he.copy(), ori.copy())
        elif self._cache_hits[idx_core] > 2:  # Accessed multiple times
            # Evict least frequently used core
            if self._cache_hits:
                lfu_core = min(self._cache_hits.keys(), key=lambda k: self._cache_hits[k] if k in self._cache else 0)
                if lfu_core in self._cache:
                    del self._cache[lfu_core]
            self._cache[idx_core] = (he.copy(), ori.copy())
        
        return he, ori

    @staticmethod
    def _rand_crop_coords(H, W, patch_size):
        if H <= patch_size or W <= patch_size:
            return 0, 0
        y0 = random.randint(0, H - patch_size)
        x0 = random.randint(0, W - patch_size)
        return y0, x0

    def _augment_sync(self, he_patch, ori_patch):
        # Flips
        if random.random() < 0.5:
            he_patch = np.flip(he_patch, axis=0).copy()
            ori_patch = np.flip(ori_patch, axis=0).copy()
        if random.random() < 0.5:
            he_patch = np.flip(he_patch, axis=1).copy()
            ori_patch = np.flip(ori_patch, axis=1).copy()
        # 0/90/180/270
        k = random.randint(0, 3)
        if k:
            he_patch = np.rot90(he_patch, k, axes=(0, 1)).copy()
            ori_patch = np.rot90(ori_patch, k, axes=(0, 1)).copy()
        # Mild color jitter (HE only)
        if self.he_color_jitter:
            if random.random() < 0.5:  # brightness
                factor = 0.9 + 0.2 * random.random()
                he_patch = np.clip(he_patch * factor, 0, 1)
            if random.random() < 0.5:  # contrast
                mean = he_patch.mean(axis=(0, 1), keepdims=True)
                factor = 0.9 + 0.2 * random.random()
                he_patch = np.clip((he_patch - mean) * factor + mean, 0, 1)
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

        if self.augment and self.sampling == "random":
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


# ------------------------ Model ------------------------ #

class UNetEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class UNetDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv1 = nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class HE2OrionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=20, base=64):  # Increased base features
        super().__init__()
        self.enc1 = UNetEncoder(in_channels, base)
        self.enc2 = UNetEncoder(base, base*2)
        self.enc3 = UNetEncoder(base*2, base*4)
        self.enc4 = UNetEncoder(base*4, base*8)

        # Enhanced bottleneck like your working model
        self.bot = nn.Sequential(
            nn.Conv2d(base*8, base*16, 3, padding=1),
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*16, base*16, 3, padding=1),
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),  # Add dropout for regularization
        )

        self.dec4 = UNetDecoder(base*16, base*8, base*8)
        self.dec3 = UNetDecoder(base*8, base*4, base*4)
        self.dec2 = UNetDecoder(base*4, base*2, base*2)
        self.dec1 = UNetDecoder(base*2, base, base)
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


# ------------------------ Loss ------------------------ #

class CombinedLoss(nn.Module):
    """MSE + L1 + (1-SSIM) on full patch or center label_crop×label_crop."""
    def __init__(self, mse_w=1.0, l1_w=0.5, ssim_w=0.3, label_crop: int = 0):
        super().__init__()
        self.mse_w, self.l1_w, self.ssim_w = mse_w, l1_w, ssim_w
        self.label_crop = label_crop

    def forward(self, pred, target):
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

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running = {"loss":0., "mse":0., "mae":0., "ssim_loss":0.}
    n = 0
    epoch_start_time = time.time()
    logging.info(f"Starting training epoch with {len(loader)} batches...")
    
    for batch_idx, (he, ori) in enumerate(loader):
        batch_start_time = time.time()
        
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

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda', enabled=True):
                pred = model(he)
                loss, metrics = criterion(pred, ori)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(he)
            loss, metrics = criterion(pred, ori)
            loss.backward()
            optimizer.step()

        bs = he.size(0)
        n += bs
        running["loss"] += metrics["total"] * bs
        running["mse"]  += metrics["mse"]   * bs
        running["mae"]  += metrics["mae"]   * bs
        running["ssim_loss"] += metrics["ssim_loss"] * bs

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

def make_loaders(args):
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

    # Optimized worker settings based on dataset size and caching
    total_patches = len(train_ds)
    if total_patches > 100000:
        safe_workers = 2  # Large dataset: minimal workers
    else:
        safe_workers = min(4, args.num_workers)  # Smaller dataset: more workers OK
    
    logging.info(f"Dataset size: {total_patches} patches, using {safe_workers} workers")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(args.train_sampling=="random"),
        num_workers=safe_workers, pin_memory=True, drop_last=True,
        persistent_workers=(safe_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, safe_workers//2), pin_memory=True, drop_last=False,
        persistent_workers=(safe_workers > 0),
    )
    logging.info(f"Using {safe_workers} workers for training, {max(1, safe_workers//2)} for validation")
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str, default="core_patches_npy")
    p.add_argument("--output_dir", type=str, default="orion_patches_model")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)  # Much smaller batch size for 512x512 patches
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--label_patch", type=int, default=0, help="center-crop size for loss; 0=full patch")
    p.add_argument("--patches_per_image", type=int, default=16)  # Much fewer patches since they're larger
    p.add_argument("--patches_per_image_val", type=int, default=8)
    p.add_argument("--base_features", type=int, default=64)
    p.add_argument("--use_amp", action="store_true")
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

    args = p.parse_args()
    outdir = Path(args.output_dir)
    setup_logging(outdir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Save config
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    train_loader, val_loader = make_loaders(args)

    # Model
    net = HE2OrionUNet(in_channels=3, out_channels=20, base=args.base_features)
    total_params = sum(p.numel() for p in net.parameters())
    logging.info(f"Model params: {total_params:,}")

    if torch.cuda.device_count() > 1:
        logging.info(f"Detected {torch.cuda.device_count()} GPUs → DataParallel")
        net = nn.DataParallel(net)
    net = net.to(device)

    # Loss, Optim, Sched (enhanced like your working model)
    criterion = CombinedLoss(label_crop=args.label_patch)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=0.5, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and torch.cuda.is_available()))

    # CSVs
    overall_csv = outdir / "metrics_overall.csv"
    perch_csv   = outdir / "val_per_channel.csv"

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        trn = train_one_epoch(net, train_loader, criterion, optimizer, device, scaler=scaler)
        val_overall, val_perch = validate(net, val_loader, criterion, device)
        dt = time.time() - t0

        scheduler.step(val_overall["loss"])
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
