#!/usr/bin/env python3
"""
Fast H&E → Orion training script.

Goals:
- Clear and minimal CLI
- Lightweight U-Net
- Fast dataloading with random sampling for train, grid for val
- AMP, channels_last, AdamW+Cosine schedule
- Best-checkpoint + small CSV metrics

Data layout (pairs_dir):
  core_###_HE.npy     -> float32/uint8 (H, W, 3)
  core_###_ORION.npy  -> float32/uint8 (H, W, 20) or (20, H, W)
"""

import os
import time
import json
import math
import random
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ------------------------ Utils ------------------------ #

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


@torch.no_grad()
def robust_norm(img: np.ndarray, p1=1, p99=99, eps=1e-6):
    lo, hi = np.percentile(img, (p1, p99))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo + eps), 0, 1).astype(np.float32)


# ------------------------ Dataset ------------------------ #

class OrionPairsDataset(Dataset):
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        patch_size: int = 256,
        patches_per_image: int = 8,
        sampling: str = "random",  # "random" or "grid"
        grid_stride: int = None,
        augment: bool = True,
    ):
        assert sampling in ("random", "grid")
        self.pairs_dir = Path(pairs_dir)
        self.basenames = basenames
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.sampling = sampling
        self.grid_stride = grid_stride or patch_size
        self.augment = augment

        self.he_paths = [self.pairs_dir / f"{b}_HE.npy" for b in basenames]
        self.ori_paths = [self.pairs_dir / f"{b}_ORION.npy" for b in basenames]
        for hp, op in zip(self.he_paths, self.ori_paths):
            if not hp.exists() or not op.exists():
                raise FileNotFoundError(f"Missing pair: {hp} / {op}")

        # Load shapes lazily via mmap
        self._shapes: List[Tuple[int,int]] = []
        for op in self.ori_paths:
            arr = np.load(op, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[0] == 20:
                H, W = arr.shape[1], arr.shape[2]
            elif arr.ndim == 3 and arr.shape[2] == 20:
                H, W = arr.shape[0], arr.shape[1]
            else:
                raise RuntimeError(f"Unexpected ORION shape {arr.shape} for {op}")
            self._shapes.append((H, W))

        if self.sampling == "random":
            self._len = len(self.basenames) * self.patches_per_image
            self._grid = None
        else:
            ps, st = self.patch_size, self.grid_stride
            grid = []
            for i, (H, W) in enumerate(self._shapes):
                ys = [0] if H <= ps else list(range(0, max(1, H - ps) + 1, st))
                xs = [0] if W <= ps else list(range(0, max(1, W - ps) + 1, st))
                for y in ys:
                    for x in xs:
                        grid.append((i, y, x))
            self._grid = grid
            self._len = len(grid)

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

    @staticmethod
    def _rand_coords(H, W, ps):
        if H <= ps or W <= ps:
            return 0, 0
        y0 = random.randint(0, H - ps)
        x0 = random.randint(0, W - ps)
        return y0, x0

    def __getitem__(self, idx: int):
        ps = self.patch_size
        if self.sampling == "random":
            core = idx // self.patches_per_image
            he = np.load(self.he_paths[core], mmap_mode='r')
            ori = np.load(self.ori_paths[core], mmap_mode='r')
            if ori.ndim == 3 and ori.shape[0] == 20:
                ori = np.transpose(ori, (1, 2, 0))
            he = self._pad_if_needed(he, ps, ps)
            ori = self._pad_if_needed(ori, ps, ps)
            H, W = he.shape[:2]
            y0, x0 = self._rand_coords(H, W, ps)
        else:
            core, y0, x0 = self._grid[idx]
            he = np.load(self.he_paths[core], mmap_mode='r')
            ori = np.load(self.ori_paths[core], mmap_mode='r')
            if ori.ndim == 3 and ori.shape[0] == 20:
                ori = np.transpose(ori, (1, 2, 0))
            he = self._pad_if_needed(he, ps, ps)
            ori = self._pad_if_needed(ori, ps, ps)

        he_patch = he[y0:y0+ps, x0:x0+ps, :]
        ori_patch = ori[y0:y0+ps, x0:x0+ps, :]

        # to float32 [0,1]
        he_patch = he_patch.astype(np.float32)
        ori_patch = ori_patch.astype(np.float32)
        if he_patch.max(initial=0.0) > 1.5:
            he_patch /= 255.0
        if ori_patch.max(initial=0.0) > 1.5:
            ori_patch /= 255.0

        # light aug
        if self.augment and self.sampling == "random":
            if random.random() < 0.5:
                he_patch = np.flip(he_patch, 0).copy(); ori_patch = np.flip(ori_patch, 0).copy()
            if random.random() < 0.5:
                he_patch = np.flip(he_patch, 1).copy(); ori_patch = np.flip(ori_patch, 1).copy()
            k = random.randint(0, 3)
            if k:
                he_patch = np.rot90(he_patch, k, axes=(0, 1)).copy()
                ori_patch = np.rot90(ori_patch, k, axes=(0, 1)).copy()

        he_t = torch.from_numpy(he_patch.transpose(2, 0, 1))
        ori_t = torch.from_numpy(ori_patch.transpose(2, 0, 1))
        return he_t, ori_t


def discover_basenames(pairs_dir: str) -> List[str]:
    p = Path(pairs_dir)
    bases = []
    for hef in sorted(p.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (p / f"{base}_ORION.npy").exists():
            bases.append(base)
    return bases


def simple_split(bases: List[str], val_frac=0.2, seed=42):
    rng = random.Random(seed)
    b = bases[:]
    rng.shuffle(b)
    n_val = int(round(len(b) * val_frac))
    return b[n_val:], b[:n_val]


# ------------------------ Model ------------------------ #

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class LightweightUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=20, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*4, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.c1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        xb = self.b(self.p3(x3))
        y3 = self.u3(xb)
        if y3.shape[2:] != x3.shape[2:]:
            y3 = F.interpolate(y3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        y3 = self.c3(torch.cat([y3, x3], dim=1))
        y2 = self.u2(y3)
        if y2.shape[2:] != x2.shape[2:]:
            y2 = F.interpolate(y2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        y2 = self.c2(torch.cat([y2, x2], dim=1))
        y1 = self.u1(y2)
        if y1.shape[2:] != x1.shape[2:]:
            y1 = F.interpolate(y1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        y1 = self.c1(torch.cat([y1, x1], dim=1))
        return torch.sigmoid(self.out(y1))


class CombinedLoss(nn.Module):
    def __init__(self, mse_w=1.0, l1_w=0.5, ssim_w=0.0):
        super().__init__()
        self.mse_w, self.l1_w, self.ssim_w = mse_w, l1_w, ssim_w

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        total = self.mse_w*mse + self.l1_w*mae
        return total, {"mse": float(mse), "mae": float(mae), "total": float(total)}


# ------------------------ Train/Val ------------------------ #

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_accum=1):
    model.train()
    running = {"loss":0.0, "mse":0.0, "mae":0.0}
    n = 0
    t0 = time.time()
    logging.info(f"Starting training epoch with {len(loader)} batches (grad accum: {grad_accum})...")
    for bi, (he, ori) in enumerate(loader):
        if bi == 0:
            logging.info("First batch loaded successfully!")
        he = he.to(device, non_blocking=True)
        ori = ori.to(device, non_blocking=True)

        if scaler is not None:
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                pred = model(he)
                loss, metrics = criterion(pred, ori)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
        else:
            pred = model(he)
            loss, metrics = criterion(pred, ori)
            loss = loss / grad_accum
            loss.backward()

        if (bi + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = he.size(0)
        n += bs
        running["loss"] += metrics["total"] * bs
        running["mse"]  += metrics["mse"]   * bs
        running["mae"]  += metrics["mae"]   * bs

        if bi % 50 == 0 and bi > 0:
            elapsed = time.time() - t0
            speed = bi / max(1e-6, elapsed)
            eta_min = (len(loader) - bi) / max(1e-6, speed) / 60
            logging.info(f"Batch {bi}/{len(loader)} | Speed: {speed:.2f} batch/sec | ETA: {eta_min:.1f}min")

    for k in running:
        running[k] /= max(1, n)
    return running


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running = {"loss":0.0, "mse":0.0, "mae":0.0}
    n = 0
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
    for k in running:
        running[k] /= max(1, n)
    return running


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
    logging.info(f"Discovered {len(bases)} paired cores")
    train_b, val_b = simple_split(bases, val_frac=args.val_split, seed=args.seed)
    if getattr(args, 'val_cores_limit', 0):
        val_b = val_b[: max(0, args.val_cores_limit)]
    logging.info(f"Train cores: {len(train_b)} | Val cores: {len(val_b)}")

    train_ds = OrionPairsDataset(
        args.pairs_dir, train_b,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        sampling="random", augment=True,
    )
    val_sampling = getattr(args, 'val_sampling', 'grid')
    val_ds = OrionPairsDataset(
        args.pairs_dir, val_b,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image_val,
        sampling=val_sampling, grid_stride=args.grid_stride, augment=False,
    )
    # Optionally cap validation size for speed
    if getattr(args, 'val_max_patches', 0) and len(val_ds) > args.val_max_patches:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, list(range(args.val_max_patches)))

    pin = (device.type == 'cuda')
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, persistent_workers=(args.num_workers > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size), shuffle=False,
        num_workers=max(1, args.num_workers//2), pin_memory=pin, persistent_workers=(args.num_workers > 0), drop_last=False,
    )
    logging.info(f"Dataset size: {len(train_ds)} train patches, {len(val_ds)} val samples; using {args.num_workers} workers (pin_memory={pin})")
    return train_loader, val_loader


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs without val improvement)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--patches_per_image", type=int, default=12)
    p.add_argument("--patches_per_image_val", type=int, default=8)
    p.add_argument("--grid_stride", type=int, default=128)
    p.add_argument("--val_sampling", type=str, default="grid", choices=["grid","random"], help="validation sampling mode")
    p.add_argument("--val_max_patches", type=int, default=0, help="cap number of validation patches (0=all)")
    p.add_argument("--val_cores_limit", type=int, default=0, help="limit number of validation cores (0=all)")
    p.add_argument("--val_every", type=int, default=1, help="run validation every N epochs")
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    setup_logging(outdir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"))
    logging.info(f"Device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = make_loaders(args, device)

    model = LightweightUNet(in_ch=3, out_ch=20, base=args.base_features)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Lightweight U-Net params: {total_params:,}")

    if torch.cuda.device_count() > 1:
        logging.info(f"Detected {torch.cuda.device_count()} GPUs → DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    criterion = CombinedLoss(mse_w=1.0, l1_w=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate*0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and device.type == 'cuda'))

    overall_csv = outdir / "metrics_overall.csv"
    best_val = float("inf")
    best_path = outdir / "best_model.pth"
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_accum=args.gradient_accumulation_steps)
        do_val = (epoch % args.val_every == 0)
        if do_val:
            va = validate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        lr = optimizer.param_groups[0]['lr']
        if do_val:
            logging.info(f"Epoch {epoch:03d}/{args.epochs} | train: loss={tr['loss']:.4f} mse={tr['mse']:.4f} mae={tr['mae']:.4f} | val: loss={va['loss']:.4f} mse={va['mse']:.4f} mae={va['mae']:.4f} | lr={lr:.2e} | {dt:.1f}s")
        else:
            logging.info(f"Epoch {epoch:03d}/{args.epochs} | train: loss={tr['loss']:.4f} mse={tr['mse']:.4f} mae={tr['mae']:.4f} | lr={lr:.2e} | {dt:.1f}s")
        append_csv(overall_csv, ["epoch","split","loss","mse","mae","lr","time_sec"], [epoch, "train", tr['loss'], tr['mse'], tr['mae'], lr, dt])
        if do_val:
            append_csv(overall_csv, ["epoch","split","loss","mse","mae","lr","time_sec"], [epoch, "val",   va['loss'], va['mse'], va['mae'], lr, dt])

        if do_val and va['loss'] < best_val:
            best_val = va['loss']
            state = {
                "epoch": epoch,
                "model": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": best_val,
                "args": vars(args),
            }
            torch.save(state, best_path)
            logging.info(f"  → New best saved (val_loss={best_val:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info(f"Early stopping: no val improvement for {args.patience} epochs")
                break

    logging.info("Training complete.")
    logging.info(f"Artifacts in: {outdir.resolve()}")


if __name__ == "__main__":
    main()


