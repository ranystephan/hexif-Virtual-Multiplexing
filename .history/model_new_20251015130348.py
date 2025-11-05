#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H&E → ORION (20ch) — Distributed (DDP) training for multi-GPU.

Key features:
  • Global per-marker quantile scaling (TRAIN set only) + log1p space
  • ConvNeXt-Tiny encoder (timm) + light FPN/UNet decoder
  • Center-window (ROSIE-style) weighted objective (MSE) + coverage + optional MS-SSIM + TV
  • Stratified sampling toward signal-bearing patches
  • Proper multi-GPU via torch.distributed, DistributedDataParallel (DDP)
  • Rank-0-only logging, plotting, and checkpointing
  • Scaler (quantiles) fit on rank 0, then barrier + load on all ranks

Launch with torchrun (examples at bottom).

torchrun --nproc_per_node=4 model_new.py \
  --pairs_dir core_patches_npy \
  --output_dir runs/orion_center_ddp_cw12 \
  --epochs 80 \
  --batch_size 12 --val_batch_size 10 \
  --patch_size 224 --center_window 12 \
  --pos_frac 0.65 --pos_threshold 0.10 --resample_tries 10 \
  --lr 3e-4 --weight_decay 1e-4 \
  --quantile_low 1.0 --quantile_high 99.5 \
  --sanity_every 1 --slide_stride 160 \
  --num_workers 6

"""

import os
import json
import time
import argparse
import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T

try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except Exception:
    HAS_MSSSIM = False

try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# --------------------------- DDP utils ---------------------------

def ddp_setup(backend: str = "nccl"):
    """Initialize DDP from torchrun env vars."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # single-process fallback (useful for debugging)
        rank, world_size, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
    torch.distributed.init_process_group(backend=backend, timeout=timedelta(seconds=7200))
    return rank, world_size, local_rank


def is_main_process():
    return (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_reduce_mean(t: torch.Tensor):
    if not torch.distributed.is_initialized():
        return t
    rt = t.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


# --------------------------- Misc utils --------------------------

def set_seed(seed: int = 42, rank: int = 0):
    import random
    seed = seed + rank  # make per-rank deterministic but distinct
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path, rank: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ("train.log" if rank == 0 else f"train_rank{rank}.log")
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s | %(levelname)s | r{rank} | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()] if rank == 0
                 else [logging.FileHandler(log_file)],
    )
    if rank == 0:
        logging.info(f"Logs → {log_file}")


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


# ---------------------- Global Quantile Scaler -------------------

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

    def save(self, path: Path):
        if is_main_process():
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text()))

    def fit_from_train(self, pairs_dir: str, basenames: List[str], max_pixels: int = 50_000_000):
        logging.info(f"Fitting global quantiles on {len(basenames)} train cores...")
        samples = [[] for _ in range(self.C)]
        total_pixels = 0
        rng = np.random.RandomState(0)

        for b in basenames:
            p = Path(pairs_dir) / f"{b}_ORION.npy"
            arr = np.load(p, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[0] == self.C:
                arr = np.transpose(arr, (1, 2, 0))
            arr = _np_to_float01(arr)
            H, W, C = arr.shape
            assert C == self.C, f"Expected {self.C} channels, got {C}"
            n = H * W
            take = min(200_000, n)
            idx = rng.choice(n, size=take, replace=False)
            flat = arr.reshape(-1, C)
            sub = flat[idx]
            for c in range(C):
                samples[c].append(sub[:, c])
            total_pixels += take
            if total_pixels >= max_pixels:
                break

        for c in range(self.C):
            if len(samples[c]) == 0:
                self.qlo[c], self.qhi[c] = 0.0, 1.0
                continue
            v = np.concatenate(samples[c], axis=0)
            qlo = np.percentile(v, self.q_low)
            qhi = np.percentile(v, self.q_high)
            if qhi <= qlo:
                qhi = qlo + 1e-6
            self.qlo[c] = float(qlo)
            self.qhi[c] = float(qhi)

        logging.info(f"Quantiles (ch0): q{self.q_low}={self.qlo[0]:.4g}, q{self.q_high}={self.qhi[0]:.4g}")


# ---------------------------- Dataset ----------------------------

class HE2OrionDataset(Dataset):
    """
    Returns:
      he:  (3, ps, ps) float32 normalized
      tgt_log: (C, ps, ps) float32, global-scaled then log1p
      info: dict
    """
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        scaler: QuantileScaler,
        patch_size: int = 224,
        mode: str = "train",
        grid_stride: int = 112,
        augment: bool = True,
        center_window: int = 12,
        pos_frac: float = 0.6,
        pos_threshold: float = 0.10,
        resample_tries: int = 8,
        samples_per_core: int = 64,
    ):
        assert mode in ("train", "val")
        self.dir = Path(pairs_dir)
        self.basenames = basenames
        self.ps = patch_size
        self.mode = mode
        self.grid_stride = grid_stride
        self.augment = augment and (mode == "train")
        self.center_window = max(0, int(center_window))
        self.pos_frac = float(pos_frac)
        self.pos_threshold = float(pos_threshold)
        self.resample_tries = int(resample_tries)
        self.scaler = scaler
        self.C = scaler.C
        self.samples_per_core = samples_per_core

        self.he_paths = [self.dir / f"{b}_HE.npy" for b in basenames]
        self.or_paths = [self.dir / f"{b}_ORION.npy" for b in basenames]
        for hp, op in zip(self.he_paths, self.or_paths):
            if not hp.exists() or not op.exists():
                raise FileNotFoundError(f"Missing pair: {hp} / {op}")

        # shapes
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

        if self.mode == "val":
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
        else:
            self.grid = None
            self._len = len(self.basenames) * self.samples_per_core

        self.tf_train = T.Compose([
            T.ToPILImage(),
            T.Resize(self.ps, antialias=True),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.03),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
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

    @staticmethod
    def _rand_coords(H, W, ps):
        if H <= ps or W <= ps:
            return 0, 0
        y0 = np.random.randint(0, H - ps + 1)
        x0 = np.random.randint(0, W - ps + 1)
        return y0, x0

    def _is_positive_patch(self, or_scaled_log: np.ndarray) -> bool:
        thresh = self.pos_threshold
        C = or_scaled_log.shape[2]
        tot = or_scaled_log.shape[0] * or_scaled_log.shape[1]
        for c in range(C):
            frac = float((or_scaled_log[..., c] > thresh).sum()) / max(1, tot)
            if frac > 0.01:
                return True
        return False

    def __getitem__(self, idx: int):
        ps = self.ps
        if self.mode == "val":
            core_idx, y0, x0 = self.grid[idx]
            he, orion = self._load_pair(core_idx)
            he_img = (he[y0:y0+ps, x0:x0+ps, :]*255).astype(np.uint8)
            or_crop = orion[y0:y0+ps, x0:x0+ps, :].copy()
            or_log = self._scale_to_log(or_crop)
            he_t = self.tf_eval(he_img)
            info = {"y0": y0, "x0": x0, "core_idx": core_idx, "basename": self.basenames[core_idx]}
            return {"he": he_t, "tgt_log": torch.from_numpy(or_log.transpose(2,0,1)), "info": info}

        # train: stratified sampling
        core_idx = np.random.randint(0, len(self.basenames))
        he, orion = self._load_pair(core_idx)
        want_pos = (np.random.rand() < self.pos_frac)
        tries = self.resample_tries if want_pos else 1
        for _ in range(tries):
            y0, x0 = self._rand_coords(*self.shapes[core_idx], ps)
            he_img = (he[y0:y0+ps, x0:x0+ps, :]*255).astype(np.uint8)
            or_crop = orion[y0:y0+ps, x0:x0+ps, :].copy()
            or_log = self._scale_to_log(or_crop)
            if (not want_pos) or self._is_positive_patch(or_log):
                break

        he_t = self.tf_train(he_img) if self.augment else self.tf_eval(he_img)
        info = {"y0": y0, "x0": x0, "core_idx": core_idx, "basename": self.basenames[core_idx]}
        return {"he": he_t, "tgt_log": torch.from_numpy(or_log.transpose(2,0,1)), "info": info}


# ------------------------ Model (ConvNeXt UNet) ------------------

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


# ----------------------------- Loss ------------------------------

class OrionLoss(nn.Module):
    def __init__(self, center_window: int = 12,
                 w_center_mse: float = 1.0,
                 pos_boost: float = 3.0,
                 pos_tau: float = 0.10,
                 w_cov: float = 0.2,
                 w_msssim: float = 0.15,
                 w_tv: float = 1e-4):
        super().__init__()
        self.cw = int(center_window)
        self.w_center_mse = w_center_mse
        self.pos_boost = pos_boost
        self.pos_tau = pos_tau
        self.w_cov = w_cov
        self.w_msssim = w_msssim if HAS_MSSSIM else 0.0
        self.w_tv = w_tv

    def forward(self, pred_log, tgt_log):
        B, C, H, W = pred_log.shape
        if self.cw > 0:
            y1 = H//2 - self.cw//2
            x1 = W//2 - self.cw//2
            y2 = y1 + self.cw
            x2 = x1 + self.cw
            pc = pred_log[:,:,y1:y2, x1:x2]
            tc = tgt_log [:,:,y1:y2, x1:x2]
        else:
            pc, tc = pred_log, tgt_log

        w = 1.0 + self.pos_boost * (tc > self.pos_tau).float()
        center_mse = (w * (pc - tc).pow(2)).mean()
        loss = self.w_center_mse * center_mse

        if self.w_cov > 0:
            pred_lin = torch.expm1(pred_log)
            tgt_lin  = torch.expm1(tgt_log)
            cov = (pred_lin.sum(dim=(2,3)) - tgt_lin.sum(dim=(2,3))).pow(2).mean()
            loss = loss + self.w_cov * cov

        if self.w_msssim > 0:
            pred_blur = F.avg_pool2d(pred_log, 3, 1, 1)
            tgt_blur  = F.avg_pool2d(tgt_log,  3, 1, 1)
            def _norm01(x):
                x = x - x.amin(dim=(2,3), keepdim=True)
                x = x / (x.amax(dim=(2,3), keepdim=True) - 1e-6)
                return x
            p01 = _norm01(pred_blur)
            t01 = _norm01(tgt_blur)
            ssim = 1.0 - ms_ssim(p01, t01, data_range=1.0, size_average=True)
            loss = loss + self.w_msssim * ssim

        if self.w_tv > 0:
            tv = (pred_log[:,:,:,1:] - pred_log[:,:,:,:-1]).abs().mean() + \
                 (pred_log[:,:,1:,:] - pred_log[:,:,:-1,:]).abs().mean()
            loss = loss + self.w_tv * tv
        return loss


# ------------------------ Sliding Reconstruction -----------------

@torch.no_grad()
def slide_reconstruct(model: nn.Module, he_img: np.ndarray, tf_eval: T.Compose,
                      ps: int, stride: int = 16, device: torch.device = torch.device("cpu")):
    H, W, _ = he_img.shape
    out_accum = None
    weight = None
    for y in range(0, max(1, H - ps) + 1, stride):
        for x in range(0, max(1, W - ps) + 1, stride):
            he_crop = (he_img[y:y+ps, x:x+ps, :]*255).astype(np.uint8)
            he_t = tf_eval(he_crop).unsqueeze(0).to(device)
            pred_log = model(he_t).detach().cpu().numpy()[0]  # (C,ps,ps)
            if out_accum is None:
                C = pred_log.shape[0]
                out_accum = np.zeros((C, H, W), dtype=np.float32)
                weight = np.zeros((1, H, W), dtype=np.float32)
            y2 = min(H, y + ps); x2 = min(W, x + ps)
            ph = y2 - y; pw = x2 - x
            out_accum[:, y:y2, x:x2] += pred_log[:, :ph, :pw]
            weight[:, y:y2, x:x2] += 1.0
    out_log = out_accum / np.clip(weight, 1e-6, None)
    return out_log


# ----------------------------- Train/Eval ------------------------

def train_one_epoch(loader, model, criterion, opt, device, use_amp=True, grad_clip=1.0, epoch=1, sampler: Optional[DistributedSampler]=None):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type=='cuda')
    total, n = 0.0, 0

    for batch in loader:
        he = batch['he'].to(device, non_blocking=True)
        tgt_log = batch['tgt_log'].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda'):
            pred_log = model(he)
            loss = criterion(pred_log, tgt_log)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()

        bs = he.size(0)
        total += float(loss) * bs
        n += bs

    # average across processes
    t = torch.tensor([total, n], device=device, dtype=torch.float32)
    t = all_reduce_mean(t)
    total_m, n_m = t[0].item(), t[1].item()
    return total_m / max(1.0, n_m)


@torch.no_grad()
def validate_one_epoch(loader, model, criterion, device, use_amp=True, sampler: Optional[DistributedSampler]=None):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        he = batch['he'].to(device, non_blocking=True)
        tgt_log = batch['tgt_log'].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda'):
            pred_log = model(he)
            loss = criterion(pred_log, tgt_log)
        bs = he.size(0)
        total += float(loss) * bs
        n += bs

    t = torch.tensor([total, n], device=device, dtype=torch.float32)
    t = all_reduce_mean(t)
    total_m, n_m = t[0].item(), t[1].item()
    return total_m / max(1.0, n_m)


def save_sanity_png(outdir: Path, he_img: np.ndarray, pred_log: np.ndarray, tgt_log: np.ndarray,
                    epoch: int, tag: str = ""):
    if not HAS_MATPLOTLIB:
        logging.warning("matplotlib not available; skipping sanity plot.")
        return
    
    cmaps = ["viridis","magma","plasma","cividis","inferno","Greens"]
    pick = [0,1,2,3,4,5] if pred_log.shape[0] >= 6 else list(range(min(6, pred_log.shape[0])))

    def inv(x_log):
        lin = np.expm1(x_log)
        return np.clip(lin, 0, 1)

    H, W, _ = he_img.shape
    fig, axes = plt.subplots(nrows=len(pick), ncols=3, figsize=(9, 3*len(pick)))
    for i, c in enumerate(pick):
        axes[i,0].imshow(he_img); axes[i,0].set_title("H&E"); axes[i,0].axis('off')
        axes[i,1].imshow(inv(tgt_log[c]), cmap=cmaps[i%len(cmaps)]); axes[i,1].set_title(f"GT ch{c}"); axes[i,1].axis('off')
        axes[i,2].imshow(inv(pred_log[c]), cmap=cmaps[i%len(cmaps)]); axes[i,2].set_title(f"Pred ch{c}"); axes[i,2].axis('off')
    fig.suptitle(f"Epoch {epoch} {tag}", y=0.995)
    outpath = outdir / f"sanity_epoch_{epoch:03d}{'_'+tag if tag else ''}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved sanity check → {outpath}")


# ----------------------------- Main ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str, default="core_patches_npy")
    p.add_argument("--output_dir", type=str, default="runs/orion_center_ddp")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=12, help="per-GPU batch size")
    p.add_argument("--val_batch_size", type=int, default=10, help="per-GPU val batch size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--grid_stride", type=int, default=112)
    p.add_argument("--center_window", type=int, default=12)
    p.add_argument("--pos_frac", type=float, default=0.65)
    p.add_argument("--pos_threshold", type=float, default=0.10)
    p.add_argument("--resample_tries", type=int, default=10)
    p.add_argument("--samples_per_core", type=int, default=64)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quantile_low", type=float, default=1.0)
    p.add_argument("--quantile_high", type=float, default=99.5)
    p.add_argument("--scaler_path", type=str, default="")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--sanity_every", type=int, default=1)
    p.add_argument("--slide_stride", type=int, default=160)
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--no_ms_ssim", action="store_true", default=False)
    args = p.parse_args()

    rank, world_size, local_rank = ddp_setup(args.backend)
    set_seed(args.seed, rank)
    outdir = Path(args.output_dir)
    setup_logging(outdir, rank)

    torch.backends.cudnn.benchmark = True

    assert HAS_TIMM, "This script requires timm. pip install timm"
    if not HAS_MSSSIM and not args.no_ms_ssim and is_main_process():
        logging.warning("pytorch_msssim not found; MS-SSIM term will be disabled.")
    if not HAS_MATPLOTLIB and is_main_process():
        logging.warning("matplotlib not found; sanity check plots will be disabled.")

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if is_main_process():
        logging.info(f"DDP world_size={world_size}, local_rank={local_rank}, device={device}")

    # --------- Discover & split ----------
    bases = discover_basenames(args.pairs_dir)
    if not bases:
        if is_main_process(): raise RuntimeError(f"No pairs found in {args.pairs_dir}")
        else: return
    if is_main_process():
        logging.info(f"Discovered {len(bases)} paired cores")
    train_b, val_b = split_train_val(bases, val_frac=args.val_split, seed=args.seed)
    if is_main_process():
        logging.info(f"Train cores: {len(train_b)} | Val cores: {len(val_b)}")

    # --------- Scaler fit/load (rank-0) ----------
    scaler_file = Path(args.scaler_path) if args.scaler_path else (outdir / "orion_scaler.json")
    if scaler_file.exists():
        scaler = QuantileScaler.load(scaler_file)
        if is_main_process(): logging.info(f"Loaded scaler from {scaler_file}")
    else:
        scaler = QuantileScaler(q_low=args.quantile_low, q_high=args.quantile_high, C=20)
        if is_main_process():
            scaler.fit_from_train(args.pairs_dir, train_b)
            scaler.save(scaler_file)
            logging.info(f"Saved scaler → {scaler_file}")
        barrier()
        # ensure all ranks read the file
        scaler = QuantileScaler.load(scaler_file)

    # --------- Datasets & samplers ----------
    train_ds = HE2OrionDataset(
        args.pairs_dir, train_b, scaler,
        patch_size=args.patch_size, mode="train", grid_stride=args.grid_stride,
        augment=True, center_window=args.center_window,
        pos_frac=args.pos_frac, pos_threshold=args.pos_threshold, resample_tries=args.resample_tries,
        samples_per_core=args.samples_per_core
    )
    val_ds = HE2OrionDataset(
        args.pairs_dir, val_b, scaler,
        patch_size=args.patch_size, mode="val", grid_stride=args.grid_stride,
        augment=False, center_window=args.center_window, pos_frac=0.0
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=pin_mem,
                              drop_last=True, persistent_workers=(args.num_workers>0), prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=pin_mem,
                            drop_last=False, persistent_workers=(args.num_workers>0), prefetch_factor=2)

    # --------- Model, DDP, loss, optim ----------
    model = ConvNeXtUNet(out_ch=20, base_ch=192, softplus_beta=1.0).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if device.type=='cuda' else None,
        output_device=local_rank if device.type=='cuda' else None,
        find_unused_parameters=False
    )

    crit = OrionLoss(
        center_window=args.center_window,
        w_center_mse=1.0,
        pos_boost=3.0,
        pos_tau=args.pos_threshold,
        w_cov=0.2,
        w_msssim=(0.15 if (HAS_MSSSIM and not args.no_ms_ssim) else 0.0),
        w_tv=1e-4,
    )

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # Save config (rank 0)
    if is_main_process():
        with open(outdir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # --------- Training loop ----------
    best_val = 1e9
    history = []

    # Cache a full H&E/Orion for sanity plotting (rank 0)
    he0, or0, val_name0 = None, None, None
    if is_main_process() and len(val_ds.basenames) > 0:
        he0, or0 = val_ds._load_pair(0)
        val_name0 = val_ds.basenames[0]

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        trn = train_one_epoch(train_loader, model, crit, opt, device, use_amp=args.use_amp, grad_clip=1.0, epoch=epoch, sampler=train_sampler)
        val = validate_one_epoch(val_loader, model, crit, device, use_amp=args.use_amp, sampler=val_sampler)
        sched.step()
        dt = time.time() - t0

        if is_main_process():
            logging.info(f"Epoch {epoch:03d}/{args.epochs} | train {trn:.4f} | val {val:.4f} | {dt:.1f}s")
            history.append({"epoch": epoch, "train": trn, "val": val, "time_sec": dt})
            with open(outdir / "metrics.csv", "a") as f:
                if epoch == 1:
                    f.write("epoch,split,loss,time_sec\n")
                f.write(f"{epoch},train,{trn},{dt}\n")
                f.write(f"{epoch},val,{val},{dt}\n")

            # checkpointing
            if val < best_val:
                best_val = val
                state = {"epoch": epoch,
                         "model": model.module.state_dict(),
                         "optimizer": opt.state_dict(),
                         "history": history,
                         "scaler": scaler.to_dict()}
                torch.save(state, outdir / "best_model.pth")
                logging.info(f"  → New best saved (val={best_val:.4f})")

            if epoch % args.save_every == 0:
                state = {"epoch": epoch,
                         "model": model.module.state_dict(),
                         "optimizer": opt.state_dict(),
                         "history": history,
                         "scaler": scaler.to_dict()}
                torch.save(state, outdir / f"checkpoint_epoch_{epoch}.pth")

            # sanity plot
            if epoch % args.sanity_every == 0 and he0 is not None:
                model.eval()
                with torch.no_grad():
                    pred_log = slide_reconstruct(model.module, he0, val_ds.tf_eval,
                                                 ps=args.patch_size, stride=args.slide_stride, device=device)
                    tgt_log = val_ds._scale_to_log(or0).transpose(2,0,1)
                save_sanity_png(outdir, he0, pred_log, tgt_log, epoch, tag=val_name0)

    # final save
    if is_main_process():
        final = {"epoch": args.epochs,
                 "model": model.module.state_dict(),
                 "optimizer": opt.state_dict(),
                 "best_val": best_val,
                 "history": history,
                 "scaler": scaler.to_dict()}
        torch.save(final, outdir / "final_model.pth")
        logging.info(f"Training complete. Best val: {best_val:.4f}")
        logging.info(f"Artifacts in: {outdir.resolve()}")

    # clean up
    barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
