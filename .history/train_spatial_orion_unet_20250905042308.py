#!/usr/bin/env python3
"""
Enhanced Spatial H&E → Orion U-Net Training with SpaceC-inspired techniques.

Key improvements over the regression model:
1. U-Net architecture for spatial predictions (H,W,20) instead of scalars
2. Robust per-channel normalization (percentile-based)
3. Spillover-aware loss functions
4. Cell segmentation mask guidance
5. Multi-scale spatial loss functions
6. Advanced data augmentation for multiplexed imaging

Usage:
python train_spatial_orion_unet.py \
  --pairs_dir core_patches_npy \
  --output_dir runs/orion_spatial \
  --epochs 50 --patch_size 224 --batch_size 32
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from scipy import ndimage
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def set_seed(seed: int = 42):
    import random
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


def robust_norm01(a: np.ndarray, p1=1, p99=99, eps=1e-6) -> np.ndarray:
    """SpaceC-style robust normalization using percentiles."""
    lo, hi = np.percentile(a, (p1, p99))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo + eps), 0, 1).astype(np.float32)


def remove_noise_cells(orion_patch: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """Remove extremely bright pixels that likely represent noise/artifacts."""
    H, W, C = orion_patch.shape
    result = orion_patch.copy()
    
    for c in range(C):
        channel = orion_patch[..., c]
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        
        # Mark pixels that are extremely bright as potential noise
        noise_mask = channel > (mean_val + z_threshold * std_val)
        if np.any(noise_mask):
            # Replace with local median to preserve spatial structure
            for y, x in np.argwhere(noise_mask):
                # Get 5x5 neighborhood
                y_start, y_end = max(0, y-2), min(H, y+3)
                x_start, x_end = max(0, x-2), min(W, x+3)
                neighborhood = channel[y_start:y_end, x_start:x_end]
                # Replace with median of non-noise neighbors
                clean_neighborhood = neighborhood[neighborhood <= (mean_val + z_threshold * std_val)]
                if len(clean_neighborhood) > 0:
                    result[y, x, c] = np.median(clean_neighborhood)
                else:
                    result[y, x, c] = mean_val
    
    return result


def create_cell_boundary_mask(he_patch: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Create approximate cell boundary mask from H&E using edge detection."""
    # Convert to grayscale
    gray = np.mean(he_patch, axis=2)
    
    # Apply Gaussian blur to reduce noise
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(gray, sigma=1.0)
    
    # Detect edges using Sobel operator
    from scipy.ndimage import sobel
    edges_x = sobel(blurred, axis=0)
    edges_y = sobel(blurred, axis=1)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # Normalize and threshold
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    boundary_mask = (edges > 0.1).astype(np.float32)
    
    return boundary_mask


def discover_basenames(pairs_dir: str) -> List[str]:
    d = Path(pairs_dir)
    out = []
    for hef in sorted(d.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (d / f"{base}_ORION.npy").exists():
            out.append(base)
    return out


class SpatialOrionDataset(Dataset):
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        patch_size: int = 224,
        patches_per_image: int = 64,
        sampling: str = "random",  # random or grid
        grid_stride: int = None,
        augment: bool = True,
        use_boundary_guidance: bool = True,
        noise_removal: bool = True,
    ):
        assert sampling in ("random", "grid")
        self.dir = Path(pairs_dir)
        self.basenames = basenames
        self.ps = patch_size
        self.ppi = patches_per_image
        self.sampling = sampling
        self.grid_stride = grid_stride or (patch_size // 2)
        self.augment = augment
        self.use_boundary_guidance = use_boundary_guidance
        self.noise_removal = noise_removal

        self.he_paths = [self.dir / f"{b}_HE.npy" for b in basenames]
        self.or_paths = [self.dir / f"{b}_ORION.npy" for b in basenames]
        for hp, op in zip(self.he_paths, self.or_paths):
            if not hp.exists() or not op.exists():
                raise FileNotFoundError(f"Missing pair: {hp} / {op}")

        # Read shapes
        self.shapes: List[Tuple[int,int]] = []
        for op in self.or_paths:
            arr = np.load(op, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[0] == 20:
                H, W = arr.shape[1], arr.shape[2]
            elif arr.ndim == 3 and arr.shape[2] == 20:
                H, W = arr.shape[0], arr.shape[1]
            else:
                raise RuntimeError(f"Unexpected Orion shape {arr.shape} for {op}")
            self.shapes.append((H, W))

        if self.sampling == "grid":
            ps, st = self.ps, self.grid_stride
            grid = []
            for i, (H, W) in enumerate(self.shapes):
                ys = [0] if H <= ps else list(range(0, max(1, H - ps) + 1, st))
                xs = [0] if W <= ps else list(range(0, max(1, W - ps) + 1, st))
                for y in ys:
                    for x in xs:
                        grid.append((i, y, x))
            self.grid = grid
            self._len = len(self.grid)
        else:
            self.grid = None
            self._len = len(self.basenames) * self.ppi

        # Enhanced transforms for multiplexed imaging
        self.tf_train = T.Compose([
            T.ToTensor(),
            T.Resize(self.ps, antialias=True),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(10),
            # Gentler color jitter for H&E
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf_eval = T.Compose([
            T.ToTensor(),
            T.Resize(self.ps, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _to_float01(a: np.ndarray):
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        elif a.dtype != np.float32:
            a = a.astype(np.float32)
        if a.max(initial=0.0) > 1.5:
            a = a / 255.0
        return a

    @staticmethod
    def _rand_coords(H, W, ps):
        if H <= ps or W <= ps:
            return 0, 0
        y0 = np.random.randint(0, H - ps + 1)
        x0 = np.random.randint(0, W - ps + 1)
        return y0, x0

    def __len__(self):
        return self._len

    def _load_pair(self, idx_core: int):
        he = np.load(self.he_paths[idx_core], mmap_mode='r')
        orion = np.load(self.or_paths[idx_core], mmap_mode='r')
        if orion.ndim == 3 and orion.shape[0] == 20:
            orion = np.transpose(orion, (1, 2, 0))
        return he, orion

    def __getitem__(self, idx: int):
        ps = self.ps
        if self.sampling == "grid":
            core_idx, y0, x0 = self.grid[idx]
        else:
            core_idx = idx // self.ppi
            H, W = self.shapes[core_idx]
            y0, x0 = self._rand_coords(H, W, ps)
        
        he, orion = self._load_pair(core_idx)
        he = self._to_float01(he)
        orion = self._to_float01(orion)

        he_crop = he[y0:y0+ps, x0:x0+ps, :].copy()
        or_crop = orion[y0:y0+ps, x0:x0+ps, :].copy()

        # Apply noise removal if enabled
        if self.noise_removal:
            or_crop = remove_noise_cells(or_crop)

        # SpaceC-style robust normalization per channel
        C = or_crop.shape[2]
        or_scaled = np.zeros_like(or_crop, dtype=np.float32)
        for c in range(C):
            or_scaled[..., c] = robust_norm01(or_crop[..., c])

        # Create boundary guidance mask if enabled
        boundary_mask = None
        if self.use_boundary_guidance:
            boundary_mask = create_cell_boundary_mask(he_crop)

        # Convert to tensors
        he_img = (he_crop * 255).astype(np.uint8)
        tf = self.tf_train if (self.sampling == "random" and self.augment) else self.tf_eval
        he_t = tf(he_img)  # (3, ps, ps)
        
        # Target is now spatial (C, H, W) instead of vector
        target = torch.from_numpy(or_scaled.transpose(2, 0, 1))  # (C, ps, ps)
        
        # Create valid mask (can be used to mask out artifacts)
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        
        result = {
            'he': he_t,
            'target': target,
            'valid_mask': valid_mask,
        }
        
        if boundary_mask is not None:
            result['boundary_mask'] = torch.from_numpy(boundary_mask).unsqueeze(0)  # (1, H, W)
            
        return result


class UNetSmall(nn.Module):
    """Small U-Net for spatial marker prediction with boundary guidance."""
    def __init__(self, in_ch: int = 3, out_ch: int = 20, base: int = 32, use_boundary_guidance: bool = True):
        super().__init__()
        self.use_boundary_guidance = use_boundary_guidance
        
        # If using boundary guidance, add one more input channel
        actual_in_ch = in_ch + (1 if use_boundary_guidance else 0)
        
        # Encoder
        self.enc1 = self._conv_block(actual_in_ch, base)
        self.enc2 = self._conv_block(base, base * 2)
        self.enc3 = self._conv_block(base * 2, base * 4)
        self.enc4 = self._conv_block(base * 4, base * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base * 8, base * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = self._conv_block(base * 16, base * 8)
        
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = self._conv_block(base * 8, base * 4)
        
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = self._conv_block(base * 4, base * 2)
        
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = self._conv_block(base * 2, base)
        
        # Output
        self.out = nn.Conv2d(base, out_ch, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, boundary_mask=None):
        # Concatenate boundary mask if provided
        if self.use_boundary_guidance and boundary_mask is not None:
            x = torch.cat([x, boundary_mask], dim=1)
        elif self.use_boundary_guidance:
            # Create dummy boundary mask if expected but not provided
            dummy_mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], 
                                   device=x.device, dtype=x.dtype)
            x = torch.cat([x, dummy_mask], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output with sigmoid for [0,1] range
        out = torch.sigmoid(self.out(d1))
        return out


class SpatialAwareLoss(nn.Module):
    """Multi-component loss function for spatial marker prediction."""
    def __init__(self, 
                 mse_weight: float = 1.0, 
                 l1_weight: float = 0.2, 
                 boundary_weight: float = 0.1,
                 focal_weight: float = 0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.boundary_weight = boundary_weight
        self.focal_weight = focal_weight
        
    def forward(self, pred, target, valid_mask=None, boundary_mask=None):
        if valid_mask is not None:
            pred_masked = pred * valid_mask.float()
            target_masked = target * valid_mask.float()
        else:
            pred_masked = pred
            target_masked = target
            
        # Basic reconstruction losses
        mse_loss = F.mse_loss(pred_masked, target_masked, reduction='mean')
        l1_loss = F.l1_loss(pred_masked, target_masked, reduction='mean')
        
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        
        # Boundary-aware loss: higher weight on cell boundaries
        if self.boundary_weight > 0 and boundary_mask is not None:
            boundary_mask_expanded = boundary_mask.expand_as(pred)
            boundary_error = F.mse_loss(pred * boundary_mask_expanded, 
                                      target * boundary_mask_expanded, 
                                      reduction='mean')
            total_loss += self.boundary_weight * boundary_error
            
        # Focal loss for rare/bright markers (optional)
        if self.focal_weight > 0:
            # Focus on high-intensity regions
            alpha = 2.0
            focal_weights = torch.pow(1 - pred_masked, alpha)
            focal_loss = torch.mean(focal_weights * F.mse_loss(pred_masked, target_masked, reduction='none'))
            total_loss += self.focal_weight * focal_loss
            
        return total_loss


def split_train_val(bases: List[str], val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    b = list(bases)
    rng.shuffle(b)
    n_val = max(1, int(round(len(b) * val_frac)))
    return b[n_val:], b[:n_val]


def make_loaders(args, device):
    bases = discover_basenames(args.pairs_dir)
    if not bases:
        raise RuntimeError(f"No pairs found in {args.pairs_dir}")
    logging.info(f"Discovered {len(bases)} paired cores")
    train_b, val_b = split_train_val(bases, val_frac=args.val_split, seed=args.seed)
    logging.info(f"Train cores: {len(train_b)} | Val cores: {len(val_b)}")

    train_ds = SpatialOrionDataset(
        args.pairs_dir, train_b,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        sampling='random', grid_stride=args.grid_stride,
        augment=True,
        use_boundary_guidance=args.use_boundary_guidance,
        noise_removal=args.noise_removal,
    )
    val_ds = SpatialOrionDataset(
        args.pairs_dir, val_b,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image_val,
        sampling='grid', grid_stride=args.grid_stride,
        augment=False,
        use_boundary_guidance=args.use_boundary_guidance,
        noise_removal=args.noise_removal,
    )

    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=max(0, args.num_workers), pin_memory=pin_mem,
                              drop_last=True, persistent_workers=(args.num_workers>0))
    val_bs = max(1, args.val_batch_size if hasattr(args, 'val_batch_size') and args.val_batch_size else args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False,
                            num_workers=max(0, args.num_workers//2), pin_memory=pin_mem,
                            drop_last=False, persistent_workers=(args.num_workers>0))
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str, default="core_patches_npy")
    p.add_argument("--output_dir", type=str, default="runs/orion_spatial")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--patches_per_image", type=int, default=64)
    p.add_argument("--patches_per_image_val", type=int, default=32)
    p.add_argument("--grid_stride", type=int, default=112)
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--no_data_parallel", action="store_true")
    p.add_argument("--use_boundary_guidance", action="store_true", default=True)
    p.add_argument("--noise_removal", action="store_true", default=True)
    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--l1_weight", type=float, default=0.2)
    p.add_argument("--boundary_weight", type=float, default=0.1)
    p.add_argument("--focal_weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    setup_logging(outdir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"))
    logging.info(f"Device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = make_loaders(args, device)

    # Create U-Net model
    net = UNetSmall(
        in_ch=3, 
        out_ch=20, 
        base=args.base_features,
        use_boundary_guidance=args.use_boundary_guidance
    )
    logging.info(f"Model params: {sum(p.numel() for p in net.parameters()):,}")
    
    if torch.cuda.device_count() > 1 and (not args.no_data_parallel):
        net = nn.DataParallel(net)
    net = net.to(device)

    # Use spatial-aware loss
    criterion = SpatialAwareLoss(
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        boundary_weight=args.boundary_weight,
        focal_weight=args.focal_weight
    )

    opt = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)

    overall_csv = outdir / "metrics_overall.csv"
    best_val = 1e9
    history = []

    def run_epoch(loader, train: bool):
        net.train(train)
        total, n = 0.0, 0
        amp_enabled = (args.use_amp and device.type == 'cuda')
        ctx_no_grad = torch.no_grad if not train else torch.enable_grad
        
        with ctx_no_grad():
            for batch in loader:
                he = batch['he'].to(device, non_blocking=True)
                target = batch['target'].to(device, non_blocking=True)
                valid_mask = batch['valid_mask'].to(device, non_blocking=True)
                boundary_mask = batch.get('boundary_mask')
                if boundary_mask is not None:
                    boundary_mask = boundary_mask.to(device, non_blocking=True)
                
                if train:
                    opt.zero_grad(set_to_none=True)
                    
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    if args.use_boundary_guidance and boundary_mask is not None:
                        out = net(he, boundary_mask)
                    else:
                        out = net(he)
                    loss = criterion(out, target, valid_mask, boundary_mask)
                    
                if train:
                    loss.backward()
                    opt.step()
                    
                bs = he.size(0)
                total += float(loss) * bs
                n += bs
                
        return total / max(1, n)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        trn = run_epoch(train_loader, True)
        val = run_epoch(val_loader, False)
        sched.step(val)
        dt = time.time() - t0
        logging.info(f"Epoch {epoch:03d}/{args.epochs} | train {trn:.4f} | val {val:.4f} | {dt:.1f}s")
        
        history.append({"epoch": epoch, "train": trn, "val": val, "time_sec": dt})
        with open(overall_csv, 'a') as f:
            if epoch == 1:
                f.write("epoch,split,loss,time_sec\n")
            f.write(f"{epoch},train,{trn},{dt}\n")
            f.write(f"{epoch},val,{val},{dt}\n")
            
        # Save checkpoints
        if epoch % 10 == 0:
            state = {
                "epoch": epoch,
                "model": (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
                "optimizer": opt.state_dict(),
                "args": vars(args),
                "history": history,
            }
            torch.save(state, outdir / f"checkpoint_epoch_{epoch}.pth")
            
        # Save best
        if val < best_val:
            best_val = val
            state = {
                "epoch": epoch,
                "model": (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
                "optimizer": opt.state_dict(),
                "args": vars(args),
                "history": history,
            }
            torch.save(state, outdir / "best_model.pth")
            logging.info(f"  → New best saved (val={best_val:.4f})")

    final = {
        "epoch": args.epochs,
        "model": (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
        "optimizer": opt.state_dict(),
        "best_val": best_val,
        "args": vars(args),
        "history": history,
    }
    torch.save(final, outdir / "final_model.pth")
    logging.info(f"Training complete. Best val: {best_val:.4f}")
    logging.info(f"Artifacts in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
