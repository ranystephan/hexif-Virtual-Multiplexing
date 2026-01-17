#!/usr/bin/env python3
"""
HEXIF Training Entry Point.
Launch with torchrun for multi-GPU support:
torchrun --nproc_per_node=NUM_GPUS train.py ...
"""
import argparse
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from hexif.utils import ddp_setup, set_seed, setup_logging, barrier, is_main_process
from hexif.data import QuantileScaler, HE2OrionDataset, load_channel_statistics
from hexif.model import SwinUNet
from hexif.loss import OrionLoss
from hexif.training import HexifTrainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str, default="core_patches_npy", help="Directory with _HE.npy and _ORION.npy files")
    p.add_argument("--output_dir", type=str, default="runs/training_run")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=12, help="per-GPU batch size")
    p.add_argument("--val_batch_size", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2"])
    p.add_argument("--scaler_path", type=str, default="", help="Path to existing scaler or where to save new one")
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true", help="Disable AMP")
    
    # Advanced params
    p.add_argument("--center_window", type=int, default=12)
    p.add_argument("--w_center", type=float, default=1.0)
    p.add_argument("--w_cov", type=float, default=0.2)
    p.add_argument("--w_msssim", type=float, default=0.15)
    p.add_argument("--w_tv", type=float, default=1e-4)
    p.add_argument("--w_presence", type=float, default=0.0)
    
    args = p.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank = ddp_setup(args.backend)
    set_seed(args.seed, rank)
    outdir = Path(args.output_dir)
    setup_logging(outdir, rank)
    
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    
    if is_main_process():
        logging.info(f"Starting training on {world_size} devices. Output: {outdir}")
        
    # Discover data
    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        if is_main_process(): raise FileNotFoundError(f"Pairs dir {pairs_dir} not found")
        else: return

    bases = []
    for f in sorted(pairs_dir.glob("*_HE.npy")):
        b = f.stem.replace("_HE", "")
        if (pairs_dir / f"{b}_ORION.npy").exists():
            bases.append(b)
            
    if not bases:
        if is_main_process(): raise RuntimeError("No valid pairs found.")
        else: return
        
    # Split
    # Simple deterministic split
    import random
    random.seed(args.seed)
    random.shuffle(bases)
    n_val = int(len(bases) * args.val_split)
    val_bases = bases[:n_val]
    train_bases = bases[n_val:]
    
    if is_main_process():
        logging.info(f"Data: {len(train_bases)} train, {len(val_bases)} val cores.")

    # Scaler
    scaler_file = Path(args.scaler_path) if args.scaler_path else (outdir / "orion_scaler.json")
    if scaler_file.exists():
        scaler = QuantileScaler.load(scaler_file)
        if is_main_process(): logging.info(f"Loaded scaler from {scaler_file}")
    else:
        scaler = QuantileScaler(C=20)
        if is_main_process():
            scaler.fit_from_train(str(pairs_dir), train_bases)
            scaler.save(scaler_file)
            logging.info(f"Created and saved scaler to {scaler_file}")
        barrier()
        scaler = QuantileScaler.load(scaler_file)
        
    # Datasets
    train_ds = HE2OrionDataset(str(pairs_dir), train_bases, scaler, patch_size=args.patch_size, mode="train",
                               center_window=args.center_window)
    val_ds = HE2OrionDataset(str(pairs_dir), val_bases, scaler, patch_size=args.patch_size, mode="val",
                             center_window=args.center_window)
                             
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, sampler=val_sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)
                            
    # Model
    model = SwinUNet(out_ch=20).to(device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None)
        
    # Loss
    crit = OrionLoss(center_window=args.center_window, loss_type=args.loss_type,
                     w_center=args.w_center, w_cov=args.w_cov, w_msssim=args.w_msssim,
                     w_tv=args.w_tv, w_presence=args.w_presence)
                     
    # Opt
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    
    # Trainer
    trainer = HexifTrainer(model, crit, opt, sched, train_loader, val_loader, val_ds,
                           device, scaler, outdir, use_amp=not args.no_amp)
                           
    trainer.train(args.epochs)
    
    if is_main_process():
        # Clean up DDP
        pass
        
    barrier()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
