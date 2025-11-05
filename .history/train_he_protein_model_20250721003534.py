"""
H&E to Protein Expression Prediction Training Script

This script trains a deep learning model to predict multiplex protein expression
from H&E images using the registered training pairs from the VALIS pipeline.

Features:
- U-Net architecture optimized for H&E to protein prediction
- Data augmentation and preprocessing
- Training with validation split
- Model checkpointing and saving
- Comprehensive evaluation metrics
- Result visualization and monitoring
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=15, min_delta=0.0001, restore_best_weights=True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def setup_logging(output_dir: str, level=logging.INFO):
    """Setup comprehensive logging."""
    # Create logs directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_file = log_dir / "training.log"
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class HEProteinDataset(Dataset):
    """Dataset for H&E to protein prediction training pairs."""
    
    def __init__(self, pairs_dir: str, patch_pairs: List[Tuple[str, str]], 
                 transform=None, augment=True):
        """
        Args:
            pairs_dir: Directory containing training pairs
            patch_pairs: List of (HE_filename, Orion_filename) tuples
            transform: Optional transforms to apply
            augment: Whether to apply data augmentation
        """
        self.pairs_dir = Path(pairs_dir)
        self.patch_pairs = patch_pairs
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.patch_pairs)
    
    def __getitem__(self, idx):
        he_filename, orion_filename = self.patch_pairs[idx]
        
        # Load images
        he_path = self.pairs_dir / he_filename
        orion_path = self.pairs_dir / orion_filename
        
        he_img = np.load(he_path).astype(np.float32)
        orion_img = np.load(orion_path).astype(np.float32)
        
        # Normalize H&E to [0, 1]
        if he_img.max() > 1.0:
            he_img = he_img / 255.0
        
        # Normalize Orion to [0, 1]
        if orion_img.max() > 1.0:
            orion_img = orion_img / 255.0
        
        # Convert to tensors
        if he_img.ndim == 3:  # RGB H&E
            he_tensor = torch.from_numpy(he_img.transpose(2, 0, 1))  # HWC -> CHW
        else:  # Grayscale H&E
            he_tensor = torch.from_numpy(he_img).unsqueeze(0)  # Add channel dim
        
        orion_tensor = torch.from_numpy(orion_img).unsqueeze(0)  # Add channel dim
        
        # Apply augmentations
        if self.augment:
            he_tensor, orion_tensor = self._apply_augmentations(he_tensor, orion_tensor)
        
        # Apply additional transforms if provided
        if self.transform:
            he_tensor = self.transform(he_tensor)
            orion_tensor = self.transform(orion_tensor)
        
        return he_tensor, orion_tensor
    
    def _apply_augmentations(self, he_tensor, orion_tensor):
        """Apply synchronized augmentations to both images."""
        
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            he_tensor = TF.hflip(he_tensor)
            orion_tensor = TF.hflip(orion_tensor)
        
        # Random vertical flip
        if torch.rand(1) > 0.5:
            he_tensor = TF.vflip(he_tensor)
            orion_tensor = TF.vflip(orion_tensor)
        
        # Random rotation (90 degree increments to maintain tissue structure)
        if torch.rand(1) > 0.5:
            angle = torch.randint(0, 4, (1,)).item() * 90
            he_tensor = TF.rotate(he_tensor, angle)
            orion_tensor = TF.rotate(orion_tensor, angle)
        
        # Random brightness/contrast for H&E only (preserve protein quantification)
        if torch.rand(1) > 0.5:
            brightness_factor = 0.8 + torch.rand(1) * 0.4  # [0.8, 1.2]
            contrast_factor = 0.8 + torch.rand(1) * 0.4    # [0.8, 1.2]
            he_tensor = TF.adjust_brightness(he_tensor, brightness_factor.item())
            he_tensor = TF.adjust_contrast(he_tensor, contrast_factor.item())
        
        return he_tensor, orion_tensor


class UNetEncoder(nn.Module):
    """U-Net encoder block."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class UNetDecoder(nn.Module):
    """U-Net decoder block."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv1 = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class HEProteinUNet(nn.Module):
    """U-Net for H&E to protein prediction."""
    
    def __init__(self, in_channels=3, out_channels=1, base_features=64):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetEncoder(in_channels, base_features)
        self.enc2 = UNetEncoder(base_features, base_features * 2)
        self.enc3 = UNetEncoder(base_features * 2, base_features * 4)
        self.enc4 = UNetEncoder(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_features * 8, base_features * 16, 3, padding=1),
            nn.BatchNorm2d(base_features * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 16, base_features * 16, 3, padding=1),
            nn.BatchNorm2d(base_features * 16),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec4 = UNetDecoder(base_features * 16, base_features * 8, base_features * 8)
        self.dec3 = UNetDecoder(base_features * 8, base_features * 4, base_features * 4)
        self.dec2 = UNetDecoder(base_features * 4, base_features * 2, base_features * 2)
        self.dec1 = UNetDecoder(base_features * 2, base_features, base_features)
        
        # Final output
        self.final = nn.Conv2d(base_features, out_channels, 1)
        
    def forward(self, x):
        # Encoder
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Final output
        x = self.final(x)
        
        return torch.sigmoid(x)  # Output in [0, 1] range


class ProteinPredictionLoss(nn.Module):
    """Combined loss for protein prediction."""
    
    def __init__(self, mse_weight=1.0, mae_weight=0.5, ssim_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, pred, target):
        # Basic regression losses
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        
        # SSIM loss (structural similarity)
        ssim_loss = 1 - self._ssim(pred, target)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.mae_weight * mae_loss + 
                     self.ssim_weight * ssim_loss)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(), 
            'ssim': ssim_loss.item(),
            'total': total_loss.item()
        }
    
    def _ssim(self, pred, target, window_size=11):
        """Compute SSIM between predicted and target images."""
        # Simple SSIM implementation
        mu1 = F.avg_pool2d(pred, window_size, 1, padding=window_size//2)
        mu2 = F.avg_pool2d(target, window_size, 1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, 1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, 1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


def find_training_pairs(pairs_dir: str) -> List[Tuple[str, str]]:
    """Find all training pairs in the directory."""
    pairs_path = Path(pairs_dir)
    
    # Find all HE files
    he_files = list(pairs_path.glob("*_HE.npy"))
    
    pairs = []
    for he_file in he_files:
        orion_file = he_file.name.replace("_HE.npy", "_ORION.npy")
        orion_path = pairs_path / orion_file
        
        if orion_path.exists():
            pairs.append((he_file.name, orion_file))
    
    return pairs


def create_data_loaders(pairs_dir: str, batch_size: int = 16, val_split: float = 0.2,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Find all pairs
    pairs = find_training_pairs(pairs_dir)
    print(f"Found {len(pairs)} training pairs")
    
    if len(pairs) == 0:
        raise ValueError("No training pairs found!")
    
    # Split into train/validation
    train_pairs, val_pairs = train_test_split(pairs, test_size=val_split, random_state=42)
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    
    # Create datasets
    train_dataset = HEProteinDataset(pairs_dir, train_pairs, augment=True)
    val_dataset = HEProteinDataset(pairs_dir, val_pairs, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_metrics = {'mse': 0, 'mae': 0, 'ssim': 0}
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (he_img, protein_img) in enumerate(pbar):
        he_img = he_img.to(device)
        protein_img = protein_img.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_protein = model(he_img)
        loss, metrics = criterion(pred_protein, protein_img)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{metrics['mse']:.4f}",
            'mae': f"{metrics['mae']:.4f}"
        })
    
    # Average metrics
    avg_loss = total_loss / len(train_loader)
    avg_metrics = {key: val / len(train_loader) for key, val in total_metrics.items()}
    
    return avg_loss, avg_metrics


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_metrics = {'mse': 0, 'mae': 0, 'ssim': 0}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for he_img, protein_img in pbar:
            he_img = he_img.to(device)
            protein_img = protein_img.to(device)
            
            # Forward pass
            pred_protein = model(he_img)
            loss, metrics = criterion(pred_protein, protein_img)
            
            # Update metrics
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{metrics['mse']:.4f}",
                'mae': f"{metrics['mae']:.4f}"
            })
    
    # Average metrics
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {key: val / len(val_loader) for key, val in total_metrics.items()}
    
    return avg_loss, avg_metrics


def save_sample_predictions(model, val_loader, device, save_path, num_samples=4):
    """Save sample predictions for visualization."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, (he_img, protein_img) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            he_img = he_img.to(device)
            protein_img = protein_img.to(device)
            
            pred_protein = model(he_img)
            
            # Convert to numpy for visualization
            he_np = he_img[0].cpu().numpy()
            protein_np = protein_img[0, 0].cpu().numpy()
            pred_np = pred_protein[0, 0].cpu().numpy()
            
            # H&E image
            if he_np.shape[0] == 3:  # RGB
                he_vis = np.transpose(he_np, (1, 2, 0))
            else:  # Grayscale
                he_vis = he_np[0]
                he_vis = np.stack([he_vis] * 3, axis=-1)
            
            axes[i, 0].imshow(he_vis)
            axes[i, 0].set_title('H&E Input')
            axes[i, 0].axis('off')
            
            # Ground truth protein
            axes[i, 1].imshow(protein_np, cmap='hot')
            axes[i, 1].set_title('Ground Truth Protein')
            axes[i, 1].axis('off')
            
            # Predicted protein
            axes[i, 2].imshow(pred_np, cmap='hot')
            axes[i, 2].set_title('Predicted Protein')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train H&E to protein prediction model")
    parser.add_argument("--pairs_dir", type=str, default="output/registration_output/training_pairs",
                       help="Directory containing training pairs")
    parser.add_argument("--output_dir", type=str, default="protein_prediction_model",
                       help="Output directory for model and results")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save model every N epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                       help="Early stopping patience (epochs)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001,
                       help="Minimum change for early stopping")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("="*60)
    logger.info("H&E TO PROTEIN PREDICTION TRAINING")
    logger.info("="*60)
    logger.info(f"Training pairs directory: {args.pairs_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Early stopping patience: {args.early_stopping_patience}")
    
    # Save configuration
    config = vars(args)
    config['device'] = str(device)
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    logger.info("Training configuration saved")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        args.pairs_dir, 
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Initializing model...")
    model = HEProteinUNet(in_channels=3, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss and optimizer
    criterion = ProteinPredictionLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        restore_best_weights=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [],
        'train_mae': [], 'val_mae': [],
        'train_ssim': [], 'val_ssim': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    logger.info("="*60)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_metrics['mse'])
        history['val_mse'].append(val_metrics['mse'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['train_ssim'].append(train_metrics['ssim'])
        history['val_ssim'].append(val_metrics['ssim'])
        history['learning_rate'].append(new_lr)
        
        epoch_time = time.time() - start_time
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1:3d}/{args.epochs} ({epoch_time:.1f}s) | "
                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val MSE: {val_metrics['mse']:.4f} | Val MAE: {val_metrics['mae']:.4f} | "
                   f"LR: {new_lr:.2e}")
        
        # Log learning rate changes
        if new_lr != old_lr:
            logger.info(f"  → Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'args': vars(args)
            }, output_path / 'best_model.pth')
            logger.info(f"  → New best model saved (val_loss: {val_loss:.4f})")
        
        # Check early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            logger.info(f"Best validation loss: {early_stopping.best_loss:.4f}")
            logger.info(f"No improvement for {args.early_stopping_patience} epochs")
            break
        
        # Log early stopping progress
        if early_stopping.counter > 0:
            logger.info(f"  → Early stopping: {early_stopping.counter}/{args.early_stopping_patience} epochs without improvement")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_path / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"  → Checkpoint saved: {checkpoint_path}")
            
            # Save sample predictions
            pred_path = output_path / f'predictions_epoch_{epoch+1}.png'
            save_sample_predictions(model, val_loader, device, pred_path)
            logger.info(f"  → Sample predictions saved: {pred_path}")
        
        # Save training history periodically
        if (epoch + 1) % 5 == 0:
            history_df = pd.DataFrame(history)
            history_df.to_csv(output_path / 'training_history.csv', index=False)
    
    logger.info("="*60)
    logger.info("Training completed!")
    
    # Log final results
    final_epoch = len(history['train_loss'])
    logger.info(f"Total epochs trained: {final_epoch}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final learning rate: {history['learning_rate'][-1]:.2e}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, output_path / 'final_model.pth')
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path / 'training_history.csv', index=False)
    
    # Plot training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs_range, history['train_loss'], label='Train')
    ax1.plot(epochs_range, history['val_loss'], label='Validation')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history['train_mse'], label='Train')
    ax2.plot(epochs_range, history['val_mse'], label='Validation')
    ax2.set_title('Mean Squared Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epochs_range, history['train_mae'], label='Train')
    ax3.plot(epochs_range, history['val_mae'], label='Validation')
    ax3.set_title('Mean Absolute Error')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(epochs_range, history['train_ssim'], label='Train')
    ax4.plot(epochs_range, history['val_ssim'], label='Validation')
    ax4.set_title('SSIM Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('SSIM Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to: {output_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main() 