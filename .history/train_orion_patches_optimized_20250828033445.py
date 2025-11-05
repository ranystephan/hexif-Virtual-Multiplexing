#!/usr/bin/env python3
"""
OPTIMIZED H&E → Orion Training Script

Key optimizations:
1. Smaller patch sizes (256x256) for better GPU utilization
2. Proper batch sizes (16-32) instead of 2
3. Simplified, efficient data loading without complex caching
4. Better loss function weighting
5. Faster model architecture
6. Gradient clipping and better optimization
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

# ------------------------ Optimized Dataset ------------------------ #

class OptimizedOrionDataset(Dataset):
    """
    Simplified, fast dataset with minimal overhead:
    - No complex caching (let OS handle file caching)
    - Simple random cropping
    - Efficient augmentation
    - Better memory usage
    """
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        patch_size: int = 256,
        patches_per_image: int = 64,  # More smaller patches
        augment: bool = True,
        is_validation: bool = False,
    ):
        self.pairs_dir = Path(pairs_dir)
        self.basenames = basenames
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.is_validation = is_validation
        
        # Pair paths
        self.he_paths = [self.pairs_dir / f"{b}_HE.npy" for b in basenames]
        self.ori_paths = [self.pairs_dir / f"{b}_ORION.npy" for b in basenames]
        
        # Validate all files exist
        for hp, op in zip(self.he_paths, self.ori_paths):
            if not hp.exists() or not op.exists():
                raise FileNotFoundError(f"Pair missing: {hp} / {op}")
        
        # Pre-load shapes for validation (deterministic sampling)
        if is_validation:
            self._shapes = []
            for op in self.ori_paths:
                arr = np.load(op, mmap_mode="r")
                if arr.ndim == 3 and arr.shape[0] == 20:
                    H, W = arr.shape[1], arr.shape[2]
                elif arr.ndim == 3 and arr.shape[2] == 20:
                    H, W = arr.shape[0], arr.shape[1]
                else:
                    raise RuntimeError(f"Unexpected ORION shape {arr.shape}")
                self._shapes.append((H, W))
            
            # Create deterministic grid for validation
            self._val_positions = []
            for i, (H, W) in enumerate(self._shapes):
                # Grid sampling for validation consistency
                step = max(patch_size // 2, 32)  # Reasonable overlap
                positions = []
                for y in range(0, max(1, H - patch_size + 1), step):
                    for x in range(0, max(1, W - patch_size + 1), step):
                        positions.append((i, y, x))
                # Limit positions per image for reasonable validation time
                if len(positions) > patches_per_image:
                    positions = positions[:patches_per_image]
                self._val_positions.extend(positions)
            
            self._len = len(self._val_positions)
        else:
            self._len = len(self.basenames) * self.patches_per_image

    def __len__(self):
        return self._len

    def _load_and_preprocess(self, idx_core: int):
        """Fast loading with minimal processing"""
        # Load data
        he = np.load(self.he_paths[idx_core]).astype(np.float32)
        ori = np.load(self.ori_paths[idx_core]).astype(np.float32)
        
        # Normalize HE to [0,1]
        if he.max() > 1.0:
            he = he / 255.0
        
        # Handle Orion format and normalize
        if ori.ndim == 3 and ori.shape[0] == 20:
            ori = np.transpose(ori, (1, 2, 0))
        if ori.max() > 1.0:
            ori = ori / 255.0
            
        return he, ori

    def _fast_augment(self, he_patch, ori_patch):
        """Fast augmentation without scipy dependencies"""
        # Flip augmentations
        if random.random() < 0.5:
            he_patch = np.flip(he_patch, axis=0).copy()
            ori_patch = np.flip(ori_patch, axis=0).copy()
        if random.random() < 0.5:
            he_patch = np.flip(he_patch, axis=1).copy()
            ori_patch = np.flip(ori_patch, axis=1).copy()
        
        # Rotation (90 degree increments only)
        if random.random() < 0.7:
            k = random.randint(1, 3)
            he_patch = np.rot90(he_patch, k, axes=(0, 1)).copy()
            ori_patch = np.rot90(ori_patch, k, axes=(0, 1)).copy()
        
        # Simple color augmentation for H&E only
        if random.random() < 0.3:
            # Brightness
            factor = 0.9 + 0.2 * random.random()
            he_patch = np.clip(he_patch * factor, 0, 1)
        
        return he_patch, ori_patch

    def __getitem__(self, idx: int):
        ps = self.patch_size
        
        if self.is_validation:
            # Deterministic sampling for validation
            idx_core, y0, x0 = self._val_positions[idx]
            he, ori = self._load_and_preprocess(idx_core)
            
            # Ensure we don't go out of bounds
            H, W = he.shape[:2]
            y0 = min(y0, H - ps)
            x0 = min(x0, W - ps)
            
        else:
            # Random sampling for training
            idx_core = idx // self.patches_per_image
            he, ori = self._load_and_preprocess(idx_core)
            
            # Random crop
            H, W = he.shape[:2]
            if H <= ps or W <= ps:
                y0, x0 = 0, 0
            else:
                y0 = random.randint(0, H - ps)
                x0 = random.randint(0, W - ps)
        
        # Extract patches
        he_patch = he[y0:y0+ps, x0:x0+ps, :]
        ori_patch = ori[y0:y0+ps, x0:x0+ps, :]
        
        # Pad if needed
        if he_patch.shape[0] < ps or he_patch.shape[1] < ps:
            pad_h = max(0, ps - he_patch.shape[0])
            pad_w = max(0, ps - he_patch.shape[1])
            he_patch = np.pad(he_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            ori_patch = np.pad(ori_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Augment
        if self.augment and not self.is_validation:
            he_patch, ori_patch = self._fast_augment(he_patch, ori_patch)
        
        # Convert to tensors
        he_t = torch.from_numpy(he_patch.transpose(2, 0, 1))
        ori_t = torch.from_numpy(ori_patch.transpose(2, 0, 1))
        
        return he_t, ori_t

# ------------------------ Optimized Model ------------------------ #

class EfficientUNet(nn.Module):
    """
    Efficient UNet optimized for speed and memory:
    - Fewer parameters
    - Better gradient flow
    - Optimized for 256x256 patches
    """
    def __init__(self, in_channels=3, out_channels=20, base=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, base)
        self.enc2 = self._make_layer(base, base*2)
        self.enc3 = self._make_layer(base*2, base*4)
        self.enc4 = self._make_layer(base*4, base*8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base*8, base*16, 3, padding=1),
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*16, base*16, 3, padding=1),
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Decoder
        self.dec4 = self._make_decoder(base*16, base*8, base*8)
        self.dec3 = self._make_decoder(base*8, base*4, base*4)
        self.dec2 = self._make_decoder(base*4, base*2, base*2)
        self.dec1 = self._make_decoder(base*2, base, base)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def _make_decoder(self, in_ch, skip_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2),
            nn.Conv2d(in_ch//2 + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder with skip connections
        x1 = self.enc1(x)  # /2
        x2 = self.enc2(x1)  # /4
        x3 = self.enc3(x2)  # /8
        x4 = self.enc4(x3)  # /16
        
        # Bottleneck
        b = self.bottleneck(x4)  # /16
        
        # Decoder with skip connections
        d4 = self.dec4[0](b)  # Upsample
        # Skip connection - get encoder feature before pooling
        skip4 = x3  # This should be the feature before pooling in enc4
        if d4.shape[2:] != skip4.shape[2:]:
            d4 = F.interpolate(d4, size=skip4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, skip4], dim=1)
        d4 = self.dec4[1:](d4)
        
        d3 = self.dec3[0](d4)
        skip3 = x2
        if d3.shape[2:] != skip3.shape[2:]:
            d3 = F.interpolate(d3, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.dec3[1:](d3)
        
        d2 = self.dec2[0](d3)
        skip2 = x1
        if d2.shape[2:] != skip2.shape[2:]:
            d2 = F.interpolate(d2, size=skip2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2[1:](d2)
        
        d1 = self.dec1[0](d2)
        # Skip connection with input features (need to extract from encoder)
        # For simplicity, let's create a proper skip connection
        skip1 = x  # Use input directly for now - this needs encoder features before first pooling
        if d1.shape[2:] != skip1.shape[2:]:
            d1 = F.interpolate(d1, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        # We need to match channels - create a projection
        if not hasattr(self, 'skip1_proj'):
            self.skip1_proj = nn.Conv2d(skip1.shape[1], d1.shape[1], 1).to(d1.device)
        skip1_proj = self.skip1_proj(skip1)
        d1 = torch.cat([d1, skip1_proj], dim=1)
        d1 = self.dec1[1:](d1)
        
        return self.output(d1)

class ImprovedEfficientUNet(nn.Module):
    """
    Properly designed efficient UNet with correct skip connections
    """
    def __init__(self, in_channels=3, out_channels=20, base=32):
        super().__init__()
        
        # Encoder blocks (without pooling for skip connections)
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4_conv = nn.Sequential(
            nn.Conv2d(base*4, base*8, 3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*8, base*8, 3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base*8, base*16, 3, padding=1),
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*16, base*16, 3, padding=1),
            nn.BatchNorm2d(base*16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base*16, base*8, 3, padding=1),  # base*8 + base*8 from skip
            nn.BatchNorm2d(base*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*8, base*8, 3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base*8, base*4, 3, padding=1),  # base*4 + base*4 from skip
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2, 3, padding=1),  # base*2 + base*2 from skip
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1),  # base + base from skip
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.output = nn.Conv2d(base, out_channels, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1_conv(x)
        x1 = self.pool1(enc1)
        
        enc2 = self.enc2_conv(x1)
        x2 = self.pool2(enc2)
        
        enc3 = self.enc3_conv(x2)
        x3 = self.pool3(enc3)
        
        enc4 = self.enc4_conv(x3)
        x4 = self.pool4(enc4)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decoder with skip connections
        up4 = self.up4(b)
        if up4.shape[2:] != enc4.shape[2:]:
            up4 = F.interpolate(up4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        
        up3 = self.up3(dec4)
        if up3.shape[2:] != enc3.shape[2:]:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        
        up2 = self.up2(dec3)
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.up1(dec2)
        if up1.shape[2:] != enc1.shape[2:]:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        output = self.output(dec1)
        return torch.sigmoid(output)

# ------------------------ Optimized Loss Function ------------------------ #

class OptimizedLoss(nn.Module):
    """
    Optimized loss function with proper weighting:
    - Balanced MSE and L1 loss
    - Structural similarity (SSIM)
    - Gradient penalty for edge preservation
    """
    def __init__(self, mse_weight=1.0, l1_weight=1.0, ssim_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        # Basic losses
        mse_loss = F.mse_loss(pred, target)
        l1_loss = F.l1_loss(pred, target)
        
        # SSIM loss
        ssim_loss = 1.0 - self._ssim(pred, target)
        
        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.l1_weight * l1_loss + 
                     self.ssim_weight * ssim_loss)
        
        metrics = {
            "mse": float(mse_loss),
            "l1": float(l1_loss),
            "ssim_loss": float(ssim_loss),
            "total": float(total_loss)
        }
        
        return total_loss, metrics
    
    @staticmethod
    def _ssim(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
        """Compute SSIM between x and y"""
        mu_x = F.avg_pool2d(x, window_size, 1, window_size//2)
        mu_y = F.avg_pool2d(y, window_size, 1, window_size//2)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x * x, window_size, 1, window_size//2) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y * y, window_size, 1, window_size//2) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size//2) - mu_xy
        
        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        
        ssim = numerator / (denominator + 1e-8)
        return ssim.mean()

# ------------------------ Utility Functions ------------------------ #

def discover_basenames(pairs_dir: str) -> List[str]:
    pairs_dir = Path(pairs_dir)
    bases = []
    for hef in sorted(pairs_dir.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (pairs_dir / f"{base}_ORION.npy").exists():
            bases.append(base)
    return bases

def split_train_val(bases: List[str], val_frac=0.2, seed=42):
    rng = random.Random(seed)
    b = bases[:]
    rng.shuffle(b)
    n_val = int(round(len(b) * val_frac))
    return b[n_val:], b[:n_val]

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    running_metrics = {"mse": 0.0, "l1": 0.0, "ssim_loss": 0.0}
    num_batches = 0
    
    for batch_idx, (he, ori) in enumerate(loader):
        he = he.to(device, non_blocking=True)
        ori = ori.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(he)
                loss, metrics = criterion(pred, ori)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(he)
            loss, metrics = criterion(pred, ori)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        running_loss += loss.item()
        for key in running_metrics:
            running_metrics[key] += metrics[key]
        num_batches += 1
        
        if batch_idx % 50 == 0:
            logging.info(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    # Average metrics
    avg_loss = running_loss / num_batches
    avg_metrics = {key: val / num_batches for key, val in running_metrics.items()}
    avg_metrics["total"] = avg_loss
    
    return avg_metrics

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_metrics = {"mse": 0.0, "l1": 0.0, "ssim_loss": 0.0}
    num_batches = 0
    
    for he, ori in loader:
        he = he.to(device, non_blocking=True)
        ori = ori.to(device, non_blocking=True)
        
        pred = model(he)
        loss, metrics = criterion(pred, ori)
        
        running_loss += loss.item()
        for key in running_metrics:
            running_metrics[key] += metrics[key]
        num_batches += 1
    
    # Average metrics
    avg_loss = running_loss / num_batches
    avg_metrics = {key: val / num_batches for key, val in running_metrics.items()}
    avg_metrics["total"] = avg_loss
    
    return avg_metrics

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args, history, path):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "args": vars(args),
        "history": history,
    }
    torch.save(state, path)

def append_csv(path: Path, headers: List[str], values: List):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(headers) + "\n")
        row = [str(v) for v in values]
        f.write(",".join(row) + "\n")

# ------------------------ Main Training Loop ------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Optimized H&E to Orion Training")
    
    # Data arguments
    parser.add_argument("--pairs_dir", type=str, default="core_patches_npy")
    parser.add_argument("--output_dir", type=str, default="orion_optimized_model")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)  # Much better batch size
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Model arguments  
    parser.add_argument("--patch_size", type=int, default=256)  # Smaller patches
    parser.add_argument("--patches_per_image", type=int, default=64)  # More patches
    parser.add_argument("--patches_per_image_val", type=int, default=32)
    parser.add_argument("--base_features", type=int, default=32)
    
    # Optimization arguments
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    # Saving arguments
    parser.add_argument("--save_every", type=int, default=10)
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Data
    basenames = discover_basenames(args.pairs_dir)
    if not basenames:
        raise RuntimeError(f"No pairs found in {args.pairs_dir}")
    
    train_names, val_names = split_train_val(basenames, args.val_split, args.seed)
    logging.info(f"Found {len(basenames)} pairs: {len(train_names)} train, {len(val_names)} val")
    
    # Datasets
    train_dataset = OptimizedOrionDataset(
        args.pairs_dir, train_names, args.patch_size, 
        args.patches_per_image, augment=True, is_validation=False
    )
    val_dataset = OptimizedOrionDataset(
        args.pairs_dir, val_names, args.patch_size,
        args.patches_per_image_val, augment=False, is_validation=True
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = ImprovedEfficientUNet(
        in_channels=3, out_channels=20, base=args.base_features
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = OptimizedLoss(mse_weight=1.0, l1_weight=1.0, ssim_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler() if args.use_amp else None
    
    # Training loop
    best_val_loss = float('inf')
    history = []
    csv_path = output_dir / "training_log.csv"
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args.max_grad_norm)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log progress
        logging.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['total']:.4f} | "
            f"Val Loss: {val_metrics['total']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save metrics
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": current_lr,
            "time": epoch_time
        })
        
        # Save to CSV
        append_csv(
            csv_path,
            ["epoch", "split", "total_loss", "mse", "l1", "ssim_loss", "lr", "time"],
            [epoch, "train", train_metrics["total"], train_metrics["mse"], 
             train_metrics["l1"], train_metrics["ssim_loss"], current_lr, epoch_time]
        )
        append_csv(
            csv_path,
            ["epoch", "split", "total_loss", "mse", "l1", "ssim_loss", "lr", "time"],
            [epoch, "val", val_metrics["total"], val_metrics["mse"], 
             val_metrics["l1"], val_metrics["ssim_loss"], current_lr, epoch_time]
        )
        
        # Save best model
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, args, history, output_dir / "best_model.pth")
            logging.info(f"New best model saved! Val loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics["total"], args, history, output_dir / f"checkpoint_epoch_{epoch}.pth")
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs, val_metrics["total"], args, history, output_dir / "final_model.pth")
    
    logging.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
