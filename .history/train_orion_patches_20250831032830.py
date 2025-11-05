#!/usr/bin/env python3
"""
H&E → Orion segmentation-style training (simple, fast, morphology-aware).

Key ideas
---------
- Predict per-marker probability maps (20 channels) as segmentation masks.
- Train with BCE + soft Dice; emphasize sparse positives via positive-aware sampling.
- Threshold continuous Orion channels into pseudo-masks via percentile per-channel.
- Lightweight U-Net; 256 patches default; AMP-friendly; clear logs/visuals.

Data layout (pairs_dir)
-----------------------
  core_###_HE.npy     -> float32 (H, W, 3) in [0,1] or [0,255]
  core_###_ORION.npy  -> float32 (H, W, 20) or (20, H, W)

Usage examples
--------------
python train_orion_patches.py \
  --pairs_dir core_patches_npy \
  --output_dir runs/orion_seg \
  --epochs 40 --patch_size 256 --batch_size 8 --num_workers 8 --use_amp
"""

import os
import json
import math
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ------------------------ Utils & Logging ------------------------ #

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


def robust_norm(img: np.ndarray, p1=1, p99=99, eps=1e-6):
    lo, hi = np.percentile(img, (p1, p99))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo + eps), 0, 1).astype(np.float32)


# ------------------------ Dataset ------------------------ #

def discover_basenames(pairs_dir: str) -> List[str]:
    pairs_dir = Path(pairs_dir)
    bases = []
    for hef in sorted(pairs_dir.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (pairs_dir / f"{base}_ORION.npy").exists():
            bases.append(base)
    return bases


class OrionSegDataset(Dataset):
    """
    Patch sampler with positive-aware sampling and on-the-fly Orion binarization.

    - Sampling modes:
      * random: each __getitem__ draws a fresh random crop
      * grid: deterministic tiling (for validation)
    - Targets are binary masks per channel computed by percentile thresholds.
    """
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        patch_size: int = 256,
        patches_per_image: int = 64,
        sampling: str = "random",
        grid_stride: int = None,
        pos_frac: float = 0.6,
        pos_percentile: float = 99.0,
        min_positive_pixels: int = 32,
        augment: bool = True,
    ):
        assert sampling in ("random", "grid")
        self.dir = Path(pairs_dir)
        self.basenames = basenames
        self.patch = patch_size
        self.ppi = patches_per_image
        self.sampling = sampling
        self.grid_stride = grid_stride or (patch_size // 2)
        self.pos_frac = pos_frac
        self.pos_percentile = pos_percentile
        self.min_positive_pixels = min_positive_pixels
        self.augment = augment

        self.he_paths = [self.dir / f"{b}_HE.npy" for b in basenames]
        self.or_paths = [self.dir / f"{b}_ORION.npy" for b in basenames]
        for hp, op in zip(self.he_paths, self.or_paths):
            if not hp.exists() or not op.exists():
                raise FileNotFoundError(f"Missing pair: {hp} / {op}")

        # read shapes
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

        if self.sampling == "random":
            self._len = len(self.basenames) * self.ppi
            self.grid = None
        else:
            ps, st = self.patch, self.grid_stride
            grid = []
            for i, (H, W) in enumerate(self.shapes):
                ys = [0] if H <= ps else list(range(0, max(1, H - ps) + 1, st))
                xs = [0] if W <= ps else list(range(0, max(1, W - ps) + 1, st))
                for y in ys:
                    for x in xs:
                        grid.append((i, y, x))
            self.grid = grid
            self._len = len(grid)

        self.cache = {}

    def __len__(self):
        return self._len

    def _load_pair(self, idx_core: int):
        if idx_core in self.cache:
            return self.cache[idx_core]
        he = np.load(self.he_paths[idx_core], mmap_mode='r')
        orion = np.load(self.or_paths[idx_core], mmap_mode='r')
        if orion.ndim == 3 and orion.shape[0] == 20:
            orion = np.transpose(orion, (1, 2, 0))
        self.cache[idx_core] = (he, orion)
        return he, orion

    @staticmethod
    def _rand_coords(H, W, ps):
        if H <= ps or W <= ps:
            return 0, 0
        y0 = np.random.randint(0, H - ps + 1)
        x0 = np.random.randint(0, W - ps + 1)
        return y0, x0

    def _sample_coords_positive_aware(self, orion: np.ndarray) -> Tuple[int,int]:
        ps = self.patch
        H, W, C = orion.shape
        want_pos = np.random.rand() < self.pos_frac
        for _ in range(20):
            y0, x0 = self._rand_coords(H, W, ps)
            if not want_pos:
                return y0, x0
            crop = orion[y0:y0+ps, x0:x0+ps, :]
            # quick positive heuristic using top-k percentile per channel
            thr = np.percentile(crop.reshape(-1, C), self.pos_percentile, axis=0)
            pos = (crop >= thr).sum()
            if pos >= self.min_positive_pixels:
                return y0, x0
        return self._rand_coords(H, W, ps)

    @staticmethod
    def _to_float01(a: np.ndarray):
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        elif a.dtype != np.float32:
            a = a.astype(np.float32)
        if a.max(initial=0.0) > 1.5:
            a = a / 255.0
        return a

    def _binarize(self, crop_orion: np.ndarray) -> np.ndarray:
        # per-channel percentile threshold, stable for sparse markers
        H, W, C = crop_orion.shape
        flat = crop_orion.reshape(-1, C)
        thr = np.percentile(flat, self.pos_percentile, axis=0)
        mask = (crop_orion >= thr.reshape(1,1,C)).astype(np.float32)
        return mask

    def _augment(self, he, mask):
        # flips + 90° rotations; safe for masks
        if np.random.rand() < 0.5:
            he = np.flip(he, 0).copy(); mask = np.flip(mask, 0).copy()
        if np.random.rand() < 0.5:
            he = np.flip(he, 1).copy(); mask = np.flip(mask, 1).copy()
        k = np.random.randint(0, 4)
        if k:
            he = np.rot90(he, k, axes=(0,1)).copy()
            mask = np.rot90(mask, k, axes=(0,1)).copy()
        return he, mask

    def __getitem__(self, idx: int):
        ps = self.patch
        if self.sampling == "random":
            core_idx = idx // self.ppi
            he, orion = self._load_pair(core_idx)
            y0, x0 = self._sample_coords_positive_aware(orion)
        else:
            core_idx, y0, x0 = self.grid[idx]
            he, orion = self._load_pair(core_idx)

        he_crop = he[y0:y0+ps, x0:x0+ps, :]
        or_crop = orion[y0:y0+ps, x0:x0+ps, :]
        he_crop = self._to_float01(he_crop)
        or_crop = self._to_float01(or_crop)

        mask = self._binarize(or_crop)
        if self.augment and self.sampling == "random":
            he_crop, mask = self._augment(he_crop, mask)

        # ensure writable, contiguous arrays for safe torch conversion
        if (not he_crop.flags.writeable) or (not he_crop.flags['C_CONTIGUOUS']):
            he_crop = he_crop.copy(order='C')
        if (not mask.flags.writeable) or (not mask.flags['C_CONTIGUOUS']):
            mask = mask.copy(order='C')
        he_t = torch.from_numpy(he_crop.transpose(2,0,1))
        m_t = torch.from_numpy(mask.transpose(2,0,1))
        return he_t, m_t


def split_train_val(bases: List[str], val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    b = list(bases)
    rng.shuffle(b)
    n_val = max(1, int(round(len(b) * val_frac)))
    return b[n_val:], b[:n_val]


# ------------------------ Model ------------------------ #

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=20, base=32):
        super().__init__()
        ch = base
        self.enc1 = ConvBlock(in_ch, ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(ch, ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(ch*2, ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bot  = ConvBlock(ch*4, ch*8)

        self.up3 = nn.ConvTranspose2d(ch*8, ch*4, 2, stride=2)
        self.dec3 = ConvBlock(ch*8, ch*4)
        self.up2 = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.dec2 = ConvBlock(ch*4, ch*2)
        self.up1 = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch*2, ch)
        self.out = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bot(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        logits = self.out(d1)
        return logits


# ------------------------ Loss & Metrics ------------------------ #

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.7, dice_w=0.3, smooth=1.0):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # BCE on logits (AMP-safe)
        bce = self.bce(pred, target)
        # Dice on probabilities
        prob = torch.sigmoid(pred)
        dims = (0,2,3)
        intersection = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice_loss = 1 - dice.mean()
        total = self.bce_w * bce + self.dice_w * dice_loss
        return total, {"bce": float(bce), "dice_loss": float(dice_loss), "total": float(total)}


@torch.no_grad()
def dice_per_channel(pred: torch.Tensor, target: torch.Tensor, eps=1.0):
    dims = (0,2,3)
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2*inter + eps) / (union + eps)
    return dice.detach().cpu().numpy()


# ------------------------ Visualization ------------------------ #

@torch.no_grad()
def save_sample_png(model, loader, device, out_path: Path, gamma: float = 0.85):
    model.eval()
    for he, mask in loader:
        he = he.to(device)
        logits = model(he)
        pred = torch.sigmoid(logits)
        he_np = he[0].float().cpu().numpy().transpose(1,2,0)
        gt_np = mask[0].float().cpu().numpy()
        pr_np = pred[0].float().cpu().numpy()
        break

    he_disp = he_np.copy()
    for c in range(min(3, he_disp.shape[2])):
        he_disp[..., c] = robust_norm(he_disp[..., c])
    gt_max = robust_norm(gt_np.max(axis=0))
    pr_max = robust_norm(pr_np.max(axis=0))

    rows, cols = 21, 2
    fig, ax = plt.subplots(rows, cols, figsize=(8, 1.8*rows))
    ax[0,0].imshow(he_disp); ax[0,0].set_title("H&E"); ax[0,0].axis('off')
    ax[0,1].imshow(np.stack([gt_max**gamma]*3, -1)); ax[0,1].set_title("GT max"); ax[0,1].axis('off')
    for i in range(20):
        r = i+1
        ax[r,0].imshow(gt_np[i]**gamma, vmin=0, vmax=1, cmap='magma'); ax[r,0].set_title(f"Ch {i} GT"); ax[r,0].axis('off')
        ax[r,1].imshow(pr_np[i]**gamma, vmin=0, vmax=1, cmap='magma'); ax[r,1].set_title(f"Ch {i} Pred"); ax[r,1].axis('off')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close('all')


# ------------------------ Train / Validate ------------------------ #

def train_one_epoch(model, loader, crit, opt, device, scaler=None):
    model.train()
    running = {"loss":0., "bce":0., "dice_loss":0.}
    n = 0
    for he, mask in loader:
        he = he.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        bs = he.size(0)
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.float16):
                pred = model(he)
                loss, metrics = crit(pred, mask)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        else:
            if device.type == 'mps':
                with torch.amp.autocast('mps', enabled=True, dtype=torch.float16):
                    pred = model(he)
                    loss, metrics = crit(pred, mask)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            else:
                pred = model(he)
                loss, metrics = crit(pred, mask)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        n += bs
        for k in running:
            running[k] += float(metrics.get(k, 0.0)) * bs if k in metrics else float(loss) * bs
    for k in running:
        running[k] /= max(1, n)
    return running


@torch.no_grad()
def validate(model, loader, crit, device):
    model.eval()
    running = {"loss":0., "bce":0., "dice_loss":0.}
    n = 0
    dice_ch_sum = np.zeros(20, dtype=np.float64)
    n_batches = 0
    for he, mask in loader:
        he = he.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(he)
        pred = torch.sigmoid(logits)
        loss, metrics = crit(pred, mask)
        dch = dice_per_channel((torch.sigmoid(logits)>0.5).float(), mask)
        dice_ch_sum += dch
        n_batches += 1
        bs = he.size(0)
        n += bs
        for k in running:
            running[k] += float(metrics.get(k, 0.0)) * bs if k in metrics else float(loss) * bs
    for k in running:
        running[k] /= max(1, n)
    per_ch_dice = (dice_ch_sum / max(1, n_batches)).astype(float)
    return running, per_ch_dice


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
    train_b, val_b = split_train_val(bases, val_frac=args.val_split, seed=args.seed)
    logging.info(f"Train cores: {len(train_b)} | Val cores: {len(val_b)}")

    train_ds = OrionSegDataset(
        args.pairs_dir, train_b,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        sampling='random', grid_stride=args.grid_stride,
        pos_frac=args.pos_frac, pos_percentile=args.pos_percentile,
        min_positive_pixels=args.min_positive_pixels,
        augment=True,
    )
    val_ds = OrionSegDataset(
        args.pairs_dir, val_b,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image_val,
        sampling='grid', grid_stride=args.grid_stride,
        pos_frac=0.0, pos_percentile=args.pos_percentile,
        min_positive_pixels=args.min_positive_pixels,
        augment=False,
    )

    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=max(0, args.num_workers), pin_memory=pin_mem,
                              drop_last=True, persistent_workers=(args.num_workers>0))
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False,
                            num_workers=max(0, args.num_workers//2), pin_memory=pin_mem,
                            drop_last=False, persistent_workers=(args.num_workers>0))
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str, default="core_patches_npy")
    p.add_argument("--output_dir", type=str, default="runs/orion_seg")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--patches_per_image", type=int, default=64)
    p.add_argument("--patches_per_image_val", type=int, default=16)
    p.add_argument("--grid_stride", type=int, default=128)
    p.add_argument("--pos_frac", type=float, default=0.6)
    p.add_argument("--pos_percentile", type=float, default=99.0)
    p.add_argument("--min_positive_pixels", type=int, default=32)
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
    
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = make_loaders(args, device)

    net = UNetSmall(in_ch=3, out_ch=20, base=32)
    logging.info(f"Model params: {sum(p.numel() for p in net.parameters()):,}")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.to(device)
    if device.type == 'cuda':
        net = net.to(memory_format=torch.channels_last)

    crit = BCEDiceLoss(bce_w=0.7, dice_w=0.3)
    opt = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.learning_rate*0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and device.type == 'cuda'))

    overall_csv = outdir / "metrics_overall.csv"
    perch_csv = outdir / "val_per_channel_dice.csv"
    best_val = -1.0
    history = []

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        trn = train_one_epoch(net, train_loader, crit, opt, device, scaler=scaler)
        val_overall, val_dice_ch = validate(net, val_loader, crit, device)
        sched.step()
        dt = time.time() - t0
        lr = opt.param_groups[0]['lr']

        logging.info(
            f"Epoch {epoch:03d}/{args.epochs} | train: loss={trn['loss']:.4f} bce={trn['bce']:.4f} dice={trn['dice_loss']:.4f} | "
            f"val: loss={val_overall['loss']:.4f} bce={val_overall['bce']:.4f} dice={val_overall['dice_loss']:.4f} | "
            f"meanDice={val_dice_ch.mean():.4f} | lr={lr:.2e} | {dt:.1f}s"
        )
        history.append({"epoch": epoch, "train": trn, "val": val_overall, "lr": lr, "time_sec": dt, "mean_dice": float(val_dice_ch.mean())})

        append_csv(
            overall_csv,
            headers=["epoch","split","loss","bce","dice_loss","mean_dice","lr","time_sec"],
            values=[epoch,"train",trn["loss"],trn["bce"],trn["dice_loss"],"",lr,dt],
        )
        append_csv(
            overall_csv,
            headers=["epoch","split","loss","bce","dice_loss","mean_dice","lr","time_sec"],
            values=[epoch,"val",val_overall["loss"],val_overall["bce"],val_overall["dice_loss"],float(val_dice_ch.mean()),lr,dt],
        )
        for ch in range(20):
            append_csv(
                perch_csv,
                headers=["epoch","channel","dice"],
                values=[epoch, ch, float(val_dice_ch[ch])],
            )

        # save best (by mean dice)
        if float(val_dice_ch.mean()) > best_val:
            best_val = float(val_dice_ch.mean())
            state = {
                "epoch": epoch,
                "model": (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "mean_dice": best_val,
                "args": vars(args),
                "history": history,
            }
            torch.save(state, outdir / "best_model.pth")
            logging.info(f"  → New best saved (meanDice={best_val:.4f})")

        # periodic viz
        if epoch % 5 == 0 or epoch == args.epochs:
            try:
                save_sample_png(net, val_loader, device, outdir / f"predictions_epoch_{epoch}.png")
            except Exception as e:
                logging.warning(f"Could not save sample predictions: {e}")

    final = {
        "epoch": args.epochs,
        "model": (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "best_mean_dice": best_val,
        "args": vars(args),
        "history": history,
    }
    torch.save(final, outdir / "final_model.pth")
    logging.info(f"Training complete. Best mean Dice: {best_val:.4f}")
    logging.info(f"Artifacts in: {outdir.resolve()}")


if __name__ == "__main__":
    main()


