#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Testing Script for Trained H&E → ORION Model

This script tests the ConvNeXt-based model trained with model_new.py on:
- Validation patches (not seen during training)
- Multiple zoom levels
- All 20 ORION markers

Usage:
    python test_orion_model.py \
        --model_path runs_octt15/orion_center_ddp_cw12/best_model.pth \
        --pairs_dir core_patches_npy \
        --output_dir test_results \
        --zoom_levels 0.5 1.0 1.5 2.0
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
from PIL import Image

# Try to import MS-SSIM if available
try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except:
    HAS_MSSSIM = False

# Try to import timm
try:
    import timm
    HAS_TIMM = True
except:
    HAS_TIMM = False


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "test.log"
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logs → {log_file}")


# ======================= Model and Scaler (from model_new.py) =======================

class QuantileScaler:
    def __init__(self, q_low=1.0, q_high=99.5, C=20):
        self.q_low = q_low
        self.q_high = q_high
        self.qlo = np.zeros(C, dtype=np.float32)
        self.qhi = np.ones(C, dtype=np.float32)
        self.C = C

    def to_dict(self) -> Dict:
        return {"q_low": self.q_low, "q_high": self.q_high,
                "qlo": self.qlo.tolist(), "qhi": self.qhi.tolist(), "C": self.C}

    @classmethod
    def from_dict(cls, d: Dict):
        obj = cls(d.get("q_low", 1.0), d.get("q_high", 99.5), d.get("C", 20))
        obj.qlo = np.array(d["qlo"], dtype=np.float32)
        obj.qhi = np.array(d["qhi"], dtype=np.float32)
        return obj

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text()))

    def inverse_transform_log(self, log_values: np.ndarray) -> np.ndarray:
        """Convert from log1p space back to original intensity space"""
        C = self.C
        out = np.zeros_like(log_values, dtype=np.float32)
        for c in range(C):
            # Reverse: log1p -> expm1 -> denormalize
            lin = np.expm1(log_values[c])  # (H, W)
            out[c] = lin * (self.qhi[c] - self.qlo[c] + 1e-6) + self.qlo[c]
        return out


class ConvNeXtUNet(nn.Module):
    def __init__(self, out_ch: int = 20, base_ch: int = 192, softplus_beta: float = 1.0):
        super().__init__()
        assert HAS_TIMM, "timm is required. pip install timm"
        self.enc = timm.create_model(
            'convnext_tiny', pretrained=True, features_only=True, out_indices=(0,1,2,3)
        )
        enc_chs = self.enc.feature_info.channels()
        self.lats = nn.ModuleList([nn.Conv2d(c, base_ch, 1) for c in enc_chs])
        self.smooth3 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.smooth1 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.smooth0 = nn.Sequential(nn.Conv2d(base_ch, base_ch//2, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(base_ch//2, out_ch, 1)
        self.softplus = nn.Softplus(beta=softplus_beta)

    def forward(self, x):
        feats = self.enc(x)
        f3 = self.lats[3](feats[3])
        f2 = self._upsum(f3, self.lats[2](feats[2]))
        f2 = self.smooth3(f2)
        f1 = self._upsum(f2, self.lats[1](feats[1]))
        f1 = self.smooth2(f1)
        f0 = self._upsum(f1, self.lats[0](feats[0]))
        f0 = self.smooth1(f0)
        up = F.interpolate(f0, size=x.shape[-2:], mode='bilinear', align_corners=False)
        up = self.smooth0(up)
        y = self.out(up)
        y = self.softplus(y)  # log1p domain is >=0
        return y

    @staticmethod
    def _up(x, size_hw):
        return F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)

    def _upsum(self, x_small, x_skip):
        x_up = self._up(x_small, x_skip.shape[-2:])
        return x_up + x_skip


# ======================= Dataset =======================

def _np_to_float01(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype in (np.uint16, np.int16):
        a = a.astype(np.float32)
        if a.max(initial=0.0) > 1.5:
            a = a / (np.percentile(a, 99.9) + 1e-6)
    elif a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.max(initial=0.0) > 1.5:
        a = a / 255.0
    return a


def discover_basenames(pairs_dir: str) -> List[str]:
    d = Path(pairs_dir)
    out = []
    for hef in sorted(d.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (d / f"{base}_ORION.npy").exists():
            out.append(base)
    return out


def split_train_val(bases: List[str], val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    b = list(bases)
    rng.shuffle(b)
    n_val = max(1, int(round(len(b) * val_frac)))
    return b[n_val:], b[:n_val]


class TestOrionDataset(Dataset):
    """Dataset for testing with zoom levels."""
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        scaler: QuantileScaler,
        patch_size: int = 224,
        zoom_level: float = 1.0,
        grid_stride: int = 112,
    ):
        self.dir = Path(pairs_dir)
        self.basenames = basenames
        self.ps = patch_size
        self.zoom = zoom_level
        self.grid_stride = grid_stride
        self.scaler = scaler
        self.C = scaler.C

        self.he_paths = [self.dir / f"{b}_HE.npy" for b in basenames]
        self.or_paths = [self.dir / f"{b}_ORION.npy" for b in basenames]
        
        # Verify files exist
        for hp, op in zip(self.he_paths, self.or_paths):
            if not hp.exists() or not op.exists():
                raise FileNotFoundError(f"Missing pair: {hp} / {op}")

        # Get shapes
        self.shapes: List[Tuple[int, int]] = []
        for op in self.or_paths:
            arr = np.load(op, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[0] == self.C:
                H, W = arr.shape[1], arr.shape[2]
            elif arr.ndim == 3 and arr.shape[2] == self.C:
                H, W = arr.shape[0], arr.shape[1]
            else:
                raise RuntimeError(f"Unexpected Orion shape {arr.shape} for {op}")
            self.shapes.append((H, W))

        # Build grid
        ps, st = self.ps, self.grid_stride
        grid = []
        for i, (H, W) in enumerate(self.shapes):
            ys = [0] if H <= ps else list(range(0, max(1, H - ps) + 1, st))
            xs = [0] if W <= ps else list(range(0, max(1, W - ps) + 1, st))
            for y in ys:
                for x in xs:
                    grid.append((i, y, x))
        self.grid = grid
        self._len = len(grid)

        # Transforms (eval mode only for testing)
        self.tf_eval = T.Compose([
            T.ToPILImage(),
            T.Resize(self.ps, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return self._len

    def _load_pair(self, idx_core: int):
        he = np.load(self.he_paths[idx_core], mmap_mode='r')
        orion = np.load(self.or_paths[idx_core], mmap_mode='r')
        if orion.ndim == 3 and orion.shape[0] == self.C:
            orion = np.transpose(orion, (1, 2, 0))
        he = _np_to_float01(he)
        orion = _np_to_float01(orion)
        return he, orion

    def _scale_to_log(self, or_patch: np.ndarray) -> np.ndarray:
        C = self.C
        out = np.zeros_like(or_patch, dtype=np.float32)
        for c in range(C):
            x = (or_patch[..., c] - self.scaler.qlo[c]) / (self.scaler.qhi[c] - self.scaler.qlo[c] + 1e-6)
            x = np.clip(x, 0, None)
            out[..., c] = np.log1p(x)
        return out

    def __getitem__(self, idx: int):
        ps = self.ps
        core_idx, y0, x0 = self.grid[idx]
        he, orion = self._load_pair(core_idx)
        
        # Apply zoom by adjusting crop size
        actual_ps = int(ps / self.zoom)
        actual_ps = min(actual_ps, min(he.shape[0] - y0, he.shape[1] - x0))
        
        he_crop = he[y0:y0+actual_ps, x0:x0+actual_ps, :]
        or_crop = orion[y0:y0+actual_ps, x0:x0+actual_ps, :].copy()
        
        # Scale and convert target
        or_log = self._scale_to_log(or_crop)
        
        # Convert H&E to tensor (resize will handle zoom)
        he_img = (he_crop*255).astype(np.uint8)
        he_t = self.tf_eval(he_img)
        
        # Also resize target to match
        or_log_resized = np.zeros((self.C, ps, ps), dtype=np.float32)
        for c in range(self.C):
            or_log_resized[c] = np.array(
                T.ToPILImage()(or_log[..., c]).resize((ps, ps), resample=Image.Resampling.BILINEAR)
            )
        
        info = {
            "y0": y0, "x0": x0, 
            "core_idx": core_idx, 
            "basename": self.basenames[core_idx],
            "zoom": self.zoom
        }
        
        return {
            "he": he_t, 
            "tgt_log": torch.from_numpy(or_log_resized), 
            "info": info
        }


# ======================= Visualization =======================

FLUOR_COLORS = [
    (0.0, 0.5, 1.0),   # Blue
    (0.0, 1.0, 0.0),   # Green
    (1.0, 0.0, 0.0),   # Red
    (1.0, 1.0, 0.0),   # Yellow
    (1.0, 0.0, 1.0),   # Magenta
    (0.0, 1.0, 1.0),   # Cyan
    (1.0, 0.5, 0.0),   # Orange
    (0.5, 0.0, 1.0),   # Purple
    (0.0, 0.8, 0.4),   # Teal
    (1.0, 0.2, 0.6),   # Pink
    (0.6, 1.0, 0.2),   # Lime
    (0.8, 0.4, 0.0),   # Brown
    (0.4, 0.6, 1.0),   # Light Blue
    (1.0, 0.8, 0.2),   # Gold
    (0.6, 0.0, 0.6),   # Maroon
    (0.0, 0.6, 0.8),   # Steel Blue
    (0.8, 0.2, 0.4),   # Crimson
    (0.2, 0.8, 0.6),   # Sea Green
    (0.9, 0.6, 0.1),   # Dark Orange
    (0.3, 0.3, 0.9),   # Royal Blue
]

def create_fluorescence_colormap(color):
    colors = [(0, 0, 0), color]
    return LinearSegmentedColormap.from_list('fluor', colors, N=256)

FLUOR_CMAPS = [create_fluorescence_colormap(color) for color in FLUOR_COLORS]


def denormalize_he_tensor(he_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized H&E tensor back to displayable image."""
    # Handle both batched and unbatched tensors
    if he_tensor.ndim == 4:
        he_tensor = he_tensor[0]  # Take first sample if batched
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if he_tensor.device.type != 'cpu':
        mean = mean.to(he_tensor.device)
        std = std.to(he_tensor.device)
    
    denorm = he_tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    
    # Now should be (3, H, W) -> (H, W, 3)
    img = denorm.cpu().numpy().transpose(1, 2, 0)
    return img


def visualize_predictions(he_tensor, pred_log, tgt_log, basename, zoom, output_path, num_channels=6):
    """Create visualization comparing predictions to ground truth."""
    he_img = denormalize_he_tensor(he_tensor.squeeze(0))
    pred_np = pred_log.cpu().numpy()[0]  # (20, H, W)
    tgt_np = tgt_log.cpu().numpy()[0]    # (20, H, W)
    
    # Show first num_channels markers
    fig, axes = plt.subplots(nrows=num_channels, ncols=3, figsize=(9, num_channels * 3))
    if num_channels == 1:
        axes = axes.reshape(1, -1)
    
    for c in range(num_channels):
        cmap = FLUOR_CMAPS[c % len(FLUOR_CMAPS)]
        color = FLUOR_COLORS[c % len(FLUOR_COLORS)]
        
        # H&E
        axes[c, 0].imshow(he_img)
        axes[c, 0].set_title(f"H&E - Marker {c}", fontsize=9)
        axes[c, 0].axis('off')
        
        # Ground Truth
        # Convert from log space for visualization
        tgt_vis = np.expm1(tgt_np[c])
        tgt_vis = np.clip(tgt_vis, 0, 1)
        axes[c, 1].imshow(tgt_vis, cmap=cmap, vmin=0, vmax=1)
        axes[c, 1].set_title(f"GT Marker {c}", fontsize=9, color=color)
        axes[c, 1].axis('off')
        
        # Prediction
        pred_vis = np.expm1(pred_np[c])
        pred_vis = np.clip(pred_vis, 0, 1)
        axes[c, 2].imshow(pred_vis, cmap=cmap, vmin=0, vmax=1)
        axes[c, 2].set_title(f"Pred Marker {c}", fontsize=9, color=color)
        axes[c, 2].axis('off')
    
    plt.suptitle(f"{basename} | Zoom {zoom:.1f}x", fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_all_markers_grid(he_tensor, pred_log, tgt_log, basename, zoom, output_path):
    """Create comprehensive grid showing all 20 markers."""
    he_img = denormalize_he_tensor(he_tensor.squeeze(0))
    pred_np = pred_log.cpu().numpy()[0]  # (20, H, W)
    tgt_np = tgt_log.cpu().numpy()[0]    # (20, H, W)
    
    # Create 5x4 grid (20 markers, each showing GT vs Pred side by side)
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(5, 5, hspace=0.3, wspace=0.3)
    
    # First cell: H&E
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(he_img)
    ax.set_title("H&E Input", fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Remaining cells: markers
    for c in range(20):
        row = (c + 1) // 5
        col = (c + 1) % 5
        
        ax = fig.add_subplot(gs[row, col])
        
        cmap = FLUOR_CMAPS[c % len(FLUOR_CMAPS)]
        color = FLUOR_COLORS[c % len(FLUOR_COLORS)]
        
        # Show GT and Pred as side-by-side composite
        tgt_vis = np.expm1(np.clip(tgt_np[c], 0, 10))
        pred_vis = np.expm1(np.clip(pred_np[c], 0, 10))
        
        # Normalize each independently for better visualization
        tgt_vis = (tgt_vis - tgt_vis.min()) / (tgt_vis.max() - tgt_vis.min() + 1e-8)
        pred_vis = (pred_vis - pred_vis.min()) / (pred_vis.max() - pred_vis.min() + 1e-8)
        
        # Concatenate horizontally
        combined = np.concatenate([tgt_vis, pred_vis], axis=1)
        
        ax.imshow(combined, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"M{c:02d}: GT|Pred", fontsize=8, color=color)
        ax.axis('off')
        
        # Add divider line
        H, W = tgt_vis.shape
        ax.axvline(x=W-0.5, color='white', linewidth=2)
    
    plt.suptitle(f"{basename} | Zoom {zoom:.1f}x | All 20 Markers (Left=GT, Right=Pred)", 
                 fontsize=14, y=0.995, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ======================= Metrics =======================

def compute_metrics(pred_log, tgt_log):
    """Compute comprehensive metrics between prediction and target."""
    pred_np = pred_log.cpu().numpy()
    tgt_np = tgt_log.cpu().numpy()
    
    # Convert from log space to linear, with clipping to prevent overflow
    # Clip log values to reasonable range before expm1 (e^10 ≈ 22026 is large enough)
    pred_np_clipped = np.clip(pred_np, 0, 10)
    tgt_np_clipped = np.clip(tgt_np, 0, 10)
    
    pred_lin = np.expm1(pred_np_clipped)
    tgt_lin = np.expm1(tgt_np_clipped)
    
    # Additional safety: clip to prevent any remaining issues
    pred_lin = np.clip(pred_lin, 0, 1e6)
    tgt_lin = np.clip(tgt_lin, 0, 1e6)
    
    metrics = {}
    
    # Per-channel metrics
    C = pred_np.shape[1]
    for c in range(C):
        pred_c = pred_lin[:, c].flatten()
        tgt_c = tgt_lin[:, c].flatten()
        
        # Check for valid data
        if not (np.all(np.isfinite(pred_c)) and np.all(np.isfinite(tgt_c))):
            metrics[f'marker_{c:02d}_mse'] = float('nan')
            metrics[f'marker_{c:02d}_mae'] = float('nan')
            metrics[f'marker_{c:02d}_corr'] = float('nan')
            continue
        
        mse = mean_squared_error(tgt_c, pred_c)
        mae = mean_absolute_error(tgt_c, pred_c)
        
        # Pearson correlation
        if tgt_c.std() > 1e-8 and pred_c.std() > 1e-8:
            try:
                corr, _ = pearsonr(tgt_c, pred_c)
            except:
                corr = 0.0
        else:
            corr = 0.0
        
        metrics[f'marker_{c:02d}_mse'] = float(mse)
        metrics[f'marker_{c:02d}_mae'] = float(mae)
        metrics[f'marker_{c:02d}_corr'] = float(corr)
    
    # Overall metrics
    pred_flat = pred_lin.flatten()
    tgt_flat = tgt_lin.flatten()
    
    if np.all(np.isfinite(pred_flat)) and np.all(np.isfinite(tgt_flat)):
        metrics['overall_mse'] = float(mean_squared_error(tgt_flat, pred_flat))
        metrics['overall_mae'] = float(mean_absolute_error(tgt_flat, pred_flat))
        
        if tgt_flat.std() > 1e-8 and pred_flat.std() > 1e-8:
            try:
                corr, _ = pearsonr(tgt_flat, pred_flat)
                metrics['overall_corr'] = float(corr)
            except:
                metrics['overall_corr'] = 0.0
        else:
            metrics['overall_corr'] = 0.0
    else:
        metrics['overall_mse'] = float('nan')
        metrics['overall_mae'] = float('nan')
        metrics['overall_corr'] = float('nan')
    
    return metrics


# ======================= Main Testing =======================

@torch.no_grad()
def test_model(args):
    """Main testing function."""
    outdir = Path(args.output_dir)
    setup_logging(outdir)
    
    logging.info("="*80)
    logging.info("Testing ORION Model")
    logging.info("="*80)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Load scaler
    if 'scaler' in checkpoint:
        scaler = QuantileScaler.from_dict(checkpoint['scaler'])
        logging.info("Loaded scaler from checkpoint")
    else:
        # Try to load from separate file
        scaler_path = Path(args.model_path).parent / "orion_scaler.json"
        if scaler_path.exists():
            scaler = QuantileScaler.load(scaler_path)
            logging.info(f"Loaded scaler from {scaler_path}")
        else:
            raise ValueError("Could not find scaler in checkpoint or as separate file")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create model
    model = ConvNeXtUNet(out_ch=20, base_ch=192, softplus_beta=1.0)
    
    # Load weights (handle DDP wrapper if present)
    state_dict = checkpoint['model']
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k.replace('module.', '', 1)] = v
        state_dict = new_state
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    logging.info(f"Model loaded from epoch {epoch}")
    
    # Discover cores and split
    all_bases = discover_basenames(args.pairs_dir)
    logging.info(f"Found {len(all_bases)} total cores")
    
    train_bases, val_bases = split_train_val(all_bases, val_frac=args.val_split, seed=42)
    logging.info(f"Using {len(val_bases)} validation cores for testing")
    logging.info(f"Validation cores: {val_bases}")
    
    # Test on each zoom level
    all_results = []
    
    for zoom in args.zoom_levels:
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing at zoom level {zoom:.2f}x")
        logging.info(f"{'='*60}")
        
        # Create dataset
        test_ds = TestOrionDataset(
            args.pairs_dir,
            val_bases,
            scaler,
            patch_size=args.patch_size,
            zoom_level=zoom,
            grid_stride=args.grid_stride,
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        )
        
        logging.info(f"Testing on {len(test_ds)} patches")
        
        # Run inference
        zoom_dir = outdir / f"zoom_{zoom:.2f}x"
        zoom_dir.mkdir(parents=True, exist_ok=True)
        
        all_metrics = []
        sample_count = 0
        
        for batch_idx, batch in enumerate(test_loader):
            he = batch['he'].to(device)
            tgt_log = batch['tgt_log'].to(device)
            info = batch['info']
            
            # Forward pass
            pred_log = model(he)
            
            # Compute metrics
            metrics = compute_metrics(pred_log, tgt_log)
            metrics['zoom'] = zoom
            metrics['basename'] = info['basename'][0]
            metrics['y0'] = info['y0'][0].item()
            metrics['x0'] = info['x0'][0].item()
            all_metrics.append(metrics)
            
            # Save visualizations for first few samples
            if sample_count < args.num_visualizations:
                basename = info['basename'][0]
                
                # Detailed visualization (first 6 markers)
                vis_path = zoom_dir / f"{basename}_y{info['y0'][0]}_x{info['x0'][0]}_detail.png"
                visualize_predictions(he, pred_log, tgt_log, basename, zoom, vis_path, num_channels=6)
                
                # All markers grid
                vis_path_all = zoom_dir / f"{basename}_y{info['y0'][0]}_x{info['x0'][0]}_all.png"
                visualize_all_markers_grid(he, pred_log, tgt_log, basename, zoom, vis_path_all)
                
                sample_count += 1
                logging.info(f"  Saved visualizations for {basename} at ({info['y0'][0]}, {info['x0'][0]})")
        
        # Save metrics
        df_metrics = pd.DataFrame(all_metrics)
        csv_path = zoom_dir / "metrics.csv"
        df_metrics.to_csv(csv_path, index=False)
        logging.info(f"Saved metrics to {csv_path}")
        
        # Compute summary statistics
        summary = {
            'zoom': zoom,
            'num_patches': len(all_metrics),
            'overall_mse_mean': df_metrics['overall_mse'].mean(),
            'overall_mse_std': df_metrics['overall_mse'].std(),
            'overall_mae_mean': df_metrics['overall_mae'].mean(),
            'overall_mae_std': df_metrics['overall_mae'].std(),
            'overall_corr_mean': df_metrics['overall_corr'].mean(),
            'overall_corr_std': df_metrics['overall_corr'].std(),
        }
        
        # Per-marker summary
        for c in range(20):
            summary[f'marker_{c:02d}_mse_mean'] = df_metrics[f'marker_{c:02d}_mse'].mean()
            summary[f'marker_{c:02d}_mae_mean'] = df_metrics[f'marker_{c:02d}_mae'].mean()
            summary[f'marker_{c:02d}_corr_mean'] = df_metrics[f'marker_{c:02d}_corr'].mean()
        
        all_results.append(summary)
        
        logging.info(f"\nSummary for zoom {zoom:.2f}x:")
        logging.info(f"  Overall MSE: {summary['overall_mse_mean']:.6f} ± {summary['overall_mse_std']:.6f}")
        logging.info(f"  Overall MAE: {summary['overall_mae_mean']:.6f} ± {summary['overall_mae_std']:.6f}")
        logging.info(f"  Overall Corr: {summary['overall_corr_mean']:.4f} ± {summary['overall_corr_std']:.4f}")
    
    # Save overall summary
    df_summary = pd.DataFrame(all_results)
    summary_path = outdir / "summary_all_zooms.csv"
    df_summary.to_csv(summary_path, index=False)
    logging.info(f"\nSaved overall summary to {summary_path}")
    
    # Create comparison plots
    create_comparison_plots(df_summary, outdir)
    
    logging.info("\n" + "="*80)
    logging.info("Testing complete!")
    logging.info(f"Results saved to {outdir.resolve()}")
    logging.info("="*80)


def create_comparison_plots(df_summary, outdir):
    """Create comparison plots across zoom levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    zooms = df_summary['zoom'].values
    
    # MSE
    axes[0].errorbar(zooms, df_summary['overall_mse_mean'], 
                     yerr=df_summary['overall_mse_std'], 
                     marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0].set_xlabel('Zoom Level', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title('MSE vs Zoom Level', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].errorbar(zooms, df_summary['overall_mae_mean'], 
                     yerr=df_summary['overall_mae_std'], 
                     marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Zoom Level', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1].set_title('MAE vs Zoom Level', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Correlation
    axes[2].errorbar(zooms, df_summary['overall_corr_mean'], 
                     yerr=df_summary['overall_corr_std'], 
                     marker='^', capsize=5, linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Zoom Level', fontsize=12)
    axes[2].set_ylabel('Pearson Correlation', fontsize=12)
    axes[2].set_title('Correlation vs Zoom Level', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(outdir / "comparison_across_zooms.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved comparison plot to {outdir / 'comparison_across_zooms.png'}")


def main():
    parser = argparse.ArgumentParser(description="Test trained H&E → ORION model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint (.pth file)")
    parser.add_argument("--pairs_dir", type=str, default="core_patches_npy",
                       help="Directory containing HE and ORION .npy files")
    parser.add_argument("--output_dir", type=str, default="test_results",
                       help="Directory to save test results")
    parser.add_argument("--patch_size", type=int, default=224,
                       help="Patch size (should match training)")
    parser.add_argument("--grid_stride", type=int, default=112,
                       help="Stride for grid sampling")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--zoom_levels", type=float, nargs='+', 
                       default=[0.5, 1.0, 1.5, 2.0],
                       help="List of zoom levels to test (e.g., 0.5 1.0 1.5 2.0)")
    parser.add_argument("--num_visualizations", type=int, default=5,
                       help="Number of sample visualizations to save per zoom level")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split fraction (must match training)")
    
    args = parser.parse_args()
    
    test_model(args)


if __name__ == "__main__":
    main()

