#!/usr/bin/env python3
"""
Fast training script using pre-extracted patches.
This should be 100x faster than the original approach.
"""

import os
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Simple dataset for pre-extracted patches
class FastPatchDataset(Dataset):
    def __init__(self, patch_files, augment=False):
        """
        Args:
            patch_files: List of HE patch file paths
            augment: Whether to apply augmentation
        """
        self.patch_files = patch_files
        self.augment = augment
        
        # Verify all files exist
        missing = []
        for pf in patch_files:
            ori_file = pf.parent / pf.name.replace("_HE", "_ORION")
            if not pf.exists() or not ori_file.exists():
                missing.append(pf)
        
        if missing:
            print(f"Warning: {len(missing)} missing files, removing from dataset")
            self.patch_files = [pf for pf in patch_files if pf not in missing]
        
        print(f"Dataset initialized with {len(self.patch_files)} patches")
    
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        he_file = self.patch_files[idx]
        ori_file = he_file.parent / he_file.name.replace("_HE", "_ORION")
        
        # Load pre-extracted patches (should be very fast!)
        he_patch = np.load(he_file).astype(np.float32)
        ori_patch = np.load(ori_file).astype(np.float32)
        
        # Simple augmentation
        if self.augment and np.random.random() > 0.5:
            # Random flip
            if np.random.random() > 0.5:
                he_patch = np.flip(he_patch, axis=0).copy()
                ori_patch = np.flip(ori_patch, axis=0).copy()
            if np.random.random() > 0.5:
                he_patch = np.flip(he_patch, axis=1).copy()
                ori_patch = np.flip(ori_patch, axis=1).copy()
        
        # Convert to tensors
        he_tensor = torch.from_numpy(he_patch.transpose(2, 0, 1))
        ori_tensor = torch.from_numpy(ori_patch.transpose(2, 0, 1))
        
        return he_tensor, ori_tensor

# Simple U-Net model
class FastUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=20):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)
        
        # Decoder
        self.dec3 = self._make_layer(128, 64)
        self.dec2 = self._make_layer(64, 32)
        self.dec1 = nn.Conv2d(32, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Simple forward pass (no skip connections for speed)
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        
        x = self.upsample(x)
        x = self.dec3(x)
        x = self.upsample(x)
        x = self.dec2(x)
        x = self.dec1(x)
        
        return torch.sigmoid(x)

def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "train.log"),
            logging.StreamHandler()
        ]
    )

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (he, ori) in enumerate(loader):
        he = he.to(device, non_blocking=True)
        ori = ori.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        pred = model(he)
        loss = criterion(pred, ori)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            logging.info(f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, "
                        f"Loss: {loss.item():.4f}, Speed: {speed:.2f} batch/sec")
    
    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - start_time
    logging.info(f"Epoch {epoch} completed: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")
    return avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for he, ori in loader:
            he = he.to(device, non_blocking=True)
            ori = ori.to(device, non_blocking=True)
            pred = model(he)
            loss = criterion(pred, ori)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", default="fast_patches")
    parser.add_argument("--output_dir", default="fast_training_output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    # Find all patch files
    patch_dir = Path(args.patch_dir)
    he_files = list(patch_dir.glob("*_HE.npy"))
    logging.info(f"Found {len(he_files)} patches")
    
    if len(he_files) == 0:
        raise ValueError(f"No patches found in {patch_dir}. Run create_patch_dataset.py first!")
    
    # Split data
    train_files, val_files = train_test_split(he_files, test_size=args.val_split, random_state=42)
    logging.info(f"Train patches: {len(train_files)}, Val patches: {len(val_files)}")
    
    # Create datasets
    train_dataset = FastPatchDataset(train_files, augment=True)
    val_dataset = FastPatchDataset(val_files, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model, loss, optimizer
    model = FastUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch}/{args.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            logging.info(f"New best model saved (val_loss={val_loss:.4f})")
    
    logging.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
