#!/usr/bin/env python3
"""
Quick test script to verify training performance without all the complexity.
"""

import os
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Simple dataset for testing
class SimpleOrionDataset(Dataset):
    def __init__(self, pairs_dir, basenames, patch_size=128):
        self.pairs_dir = Path(pairs_dir)
        self.basenames = basenames[:10]  # Only use first 10 cores for testing
        self.patch_size = patch_size
        self.patches_per_core = 50  # Much smaller for testing
        
        # Verify files exist
        for b in self.basenames:
            he_path = self.pairs_dir / f"{b}_HE.npy"
            ori_path = self.pairs_dir / f"{b}_ORION.npy"
            if not he_path.exists() or not ori_path.exists():
                raise FileNotFoundError(f"Missing: {he_path} or {ori_path}")
        
        print(f"Test dataset: {len(self.basenames)} cores, {len(self)} total patches")
    
    def __len__(self):
        return len(self.basenames) * self.patches_per_core
    
    def __getitem__(self, idx):
        core_idx = idx // self.patches_per_core
        basename = self.basenames[core_idx]
        
        # Load files
        he = np.load(self.pairs_dir / f"{basename}_HE.npy", mmap_mode='r')
        ori = np.load(self.pairs_dir / f"{basename}_ORION.npy", mmap_mode='r')
        
        # Simple normalization
        if he.max() > 1.0:
            he = he / 255.0
        if ori.ndim == 3 and ori.shape[0] == 20:
            ori = np.transpose(ori, (1, 2, 0))
        if ori.max() > 1.0:
            ori = ori / 255.0
        
        # Random crop
        H, W = he.shape[:2]
        ps = self.patch_size
        if H > ps and W > ps:
            y = np.random.randint(0, H - ps)
            x = np.random.randint(0, W - ps)
        else:
            y, x = 0, 0
        
        he_patch = he[y:y+ps, x:x+ps, :3]
        ori_patch = ori[y:y+ps, x:x+ps, :20]
        
        # Pad if needed
        if he_patch.shape[0] < ps or he_patch.shape[1] < ps:
            he_patch = np.pad(he_patch, ((0, max(0, ps-he_patch.shape[0])), 
                                       (0, max(0, ps-he_patch.shape[1])), (0, 0)))
            ori_patch = np.pad(ori_patch, ((0, max(0, ps-ori_patch.shape[0])), 
                                        (0, max(0, ps-ori_patch.shape[1])), (0, 0)))
        
        return (torch.from_numpy(he_patch.transpose(2,0,1)).float(),
                torch.from_numpy(ori_patch.transpose(2,0,1)).float())

# Simple model
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 20, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

def discover_basenames(pairs_dir):
    pairs_dir = Path(pairs_dir)
    bases = []
    for hef in sorted(pairs_dir.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (pairs_dir / f"{base}_ORION.npy").exists():
            bases.append(base)
    return bases

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", default="core_patches_npy")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data
    basenames = discover_basenames(args.pairs_dir)
    print(f"Found {len(basenames)} cores")
    
    dataset = SimpleOrionDataset(args.pairs_dir, basenames)
    loader = DataLoader(dataset, batch_size=args.batch_size, 
                       num_workers=args.num_workers, shuffle=True)
    
    # Model
    model = SimpleUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Starting test with {len(loader)} batches...")
    
    # Test training loop
    model.train()
    start_time = time.time()
    
    for batch_idx, (he, ori) in enumerate(loader):
        he = he.to(device)
        ori = ori.to(device)
        
        optimizer.zero_grad()
        pred = model(he)
        loss = criterion(pred, ori)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}, "
                  f"Time: {elapsed:.1f}s, Speed: {batch_idx+1}/{elapsed:.1f} = {(batch_idx+1)/elapsed:.2f} batch/sec")
        
        if batch_idx >= 20:  # Test first 20 batches
            break
    
    total_time = time.time() - start_time
    print(f"Completed 20 batches in {total_time:.1f}s = {20/total_time:.2f} batch/sec")
    print("If this is fast, the issue is in the complex dataset code!")

if __name__ == "__main__":
    main()
