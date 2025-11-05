#!/usr/bin/env python3
"""
Rosie-style H&E → Orion regression training (ConvNeXt backbone).

- Predict 20 continuous marker channels from H&E patches using an ImageNet
  pretrained ConvNeXt-Small head (linear layer with 20 outputs).
- Supervision: L1 + masked MSE on continuous Orion targets (scaled to [0,1]).
- Data: core_###_HE.npy (H,W,3), core_###_ORION.npy (H,W,20 or 20,H,W)
  Random crops for train, grid crops for validation.

Usage:
python train_orion_patches.py \
  --pairs_dir core_patches_npy \
  --output_dir runs/orion_regress \
  --epochs 40 --patch_size 224 --batch_size 64 --num_workers 8
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm


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
    lo, hi = np.percentile(a, (p1, p99))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo + eps), 0, 1).astype(np.float32)


def discover_basenames(pairs_dir: str) -> List[str]:
    d = Path(pairs_dir)
    out = []
    for hef in sorted(d.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (d / f"{base}_ORION.npy").exists():
            out.append(base)
    return out


class OrionRegressDataset(Dataset):
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        patch_size: int = 224,
        patches_per_image: int = 128,
        sampling: str = "random",  # random or grid
        grid_stride: int = None,
        augment: bool = True,
    ):
        assert sampling in ("random", "grid")
        self.dir = Path(pairs_dir)
        self.basenames = basenames
        self.ps = patch_size
        self.ppi = patches_per_image
        self.sampling = sampling
        self.grid_stride = grid_stride or (patch_size // 2)
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

        # transforms (Rosie-like: ToTensor, Resize, jitter/norm)
        self.tf_train = T.Compose([
            T.ToTensor(),
            T.Resize(self.ps, antialias=True),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

        he_crop = he[y0:y0+ps, x0:x0+ps, :]
        or_crop = orion[y0:y0+ps, x0:x0+ps, :]

        # normalize Orion per-channel to [0,1] robustly
        C = or_crop.shape[2]
        or_scaled = np.zeros_like(or_crop, dtype=np.float32)
        for c in range(C):
            or_scaled[..., c] = robust_norm01(or_crop[..., c])

        # convert to tensors
        he_img = (he_crop*255).astype(np.uint8)
        tf = self.tf_train if (self.sampling == "random" and self.augment) else self.tf_eval
        he_t = tf(he_img)  # (3, ps, ps)
        target = torch.from_numpy(or_scaled.transpose(2,0,1))  # (C, ps, ps)
        # global pooling target to regression vector (Rosie predicts vector per patch)
        target_vec = target.view(C, -1).mean(dim=1)
        valid_mask = torch.ones_like(target_vec, dtype=torch.bool)
        return he_t, target_vec, valid_mask


class ConvNeXtHead(nn.Module):
    def __init__(self, num_outputs: int = 20):
        super().__init__()
        m = tvm.convnext_small(weights=tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_outputs)
        self.backbone = m
    def forward(self, x):
        return self.backbone(x)


def masked_mse_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, l1_w=0.2):
    mask = mask.bool()
    p = torch.masked_select(pred, mask)
    t = torch.masked_select(target, mask)
    mse = F.mse_loss(p, t, reduction='mean')
    l1 = F.l1_loss(p, t, reduction='mean')
    return mse + l1_w * l1


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

    train_ds = OrionRegressDataset(
        args.pairs_dir, train_b,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        sampling='random', grid_stride=args.grid_stride,
        augment=True,
    )
    val_ds = OrionRegressDataset(
        args.pairs_dir, val_b,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image_val,
        sampling='grid', grid_stride=args.grid_stride,
        augment=False,
    )

    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=max(0, args.num_workers), pin_memory=pin_mem,
                              drop_last=True, persistent_workers=(args.num_workers>0))
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size*2), shuffle=False,
                            num_workers=max(0, args.num_workers//2), pin_memory=pin_mem,
                            drop_last=False, persistent_workers=(args.num_workers>0))
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str, default="core_patches_npy")
    p.add_argument("--output_dir", type=str, default="runs/orion_regress")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--patches_per_image", type=int, default=128)
    p.add_argument("--patches_per_image_val", type=int, default=64)
    p.add_argument("--grid_stride", type=int, default=112)
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

    net = ConvNeXtHead(num_outputs=20)
    logging.info(f"Model params: {sum(p.numel() for p in net.parameters()):,}")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.to(device)

    opt = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)

    overall_csv = outdir / "metrics_overall.csv"
    best_val = 1e9
    history = []

    def run_epoch(loader, train: bool):
        net.train(train)
        total, n = 0.0, 0
        for he, vec, mask in loader:
            he = he.to(device, non_blocking=True)
            vec = vec.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if train:
                opt.zero_grad(set_to_none=True)
            out = net(he)
            loss = masked_mse_l1(out, vec, mask)
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
            if epoch == 1 and not overall_csv.exists():
                f.write("epoch,split,loss,time_sec\n")
            f.write(f"{epoch},train,{trn},{dt}\n")
            f.write(f"{epoch},val,{val},{dt}\n")
        # save best
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



