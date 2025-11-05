"""
Multi-Protein H&E to Orion Prediction Training Script

This script trains a state-of-the-art multi-protein prediction model that predicts
19 protein expressions simultaneously from H&E images using advanced U-Net architecture
with attention mechanisms, protein-specific heads, and biological constraints.

Features:
- Multi-channel data loading from original Orion files
- Advanced U-Net with attention and multi-scale processing
- Protein-specific prediction heads with learned scaling
- Comprehensive loss function (MSE + SSIM + Correlation + Deep Supervision)
- Progressive training with biological constraints
- Early stopping and comprehensive logging
- Multi-protein visualization and evaluation
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
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our custom modules
from multi_channel_loader import create_multi_channel_data_loaders, verify_multi_channel_data
from multi_protein_model import create_multi_protein_model, MultiProteinLoss

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# GPU information
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


class EarlyStopping:
    """Early stopping with best model restoration."""
    
    def __init__(self, patience=15, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
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
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "multi_protein_training.log"
    
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch with detailed metrics."""
    model.train()
    total_loss = 0
    total_metrics = {
        'mse': 0, 'mae': 0, 'ssim': 0, 
        'correlation': 0, 'deep_supervision': 0
    }
    batch_count = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, (he_img, protein_img) in enumerate(pbar):
        he_img = he_img.to(device, non_blocking=True)
        protein_img = protein_img.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_output = model(he_img)
        loss, metrics = criterion(pred_output, protein_img)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for key in total_metrics:
            if key in metrics:
                total_metrics[key] += metrics[key]
        batch_count += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{metrics.get('mse', 0):.4f}",
            'ssim': f"{metrics.get('ssim', 0):.4f}"
        })
        
        # Memory cleanup
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    # Average metrics
    avg_loss = total_loss / batch_count
    avg_metrics = {key: val / batch_count for key, val in total_metrics.items()}
    
    return avg_loss, avg_metrics


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch with detailed metrics."""
    model.eval()
    total_loss = 0
    total_metrics = {
        'mse': 0, 'mae': 0, 'ssim': 0,
        'correlation': 0, 'deep_supervision': 0
    }
    batch_count = 0
    
    # For per-protein analysis
    protein_mse = np.zeros(19)
    protein_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        for he_img, protein_img in pbar:
            he_img = he_img.to(device, non_blocking=True)
            protein_img = protein_img.to(device, non_blocking=True)
            
            # Forward pass
            pred_output = model(he_img)
            loss, metrics = criterion(pred_output, protein_img)
            
            # Extract main prediction for per-protein analysis
            if isinstance(pred_output, tuple):
                main_pred = pred_output[0]
            else:
                main_pred = pred_output
            
            # Per-protein MSE calculation
            for i in range(min(19, main_pred.shape[1])):
                protein_mse[i] += torch.nn.functional.mse_loss(
                    main_pred[:, i], protein_img[:, i]
                ).item()
            protein_samples += 1
            
            # Update metrics
            total_loss += loss.item()
            for key in total_metrics:
                if key in metrics:
                    total_metrics[key] += metrics[key]
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{metrics.get('mse', 0):.4f}",
                'ssim': f"{metrics.get('ssim', 0):.4f}"
            })
    
    # Average metrics
    avg_loss = total_loss / batch_count
    avg_metrics = {key: val / batch_count for key, val in total_metrics.items()}
    
    # Per-protein averages
    if protein_samples > 0:
        avg_metrics['per_protein_mse'] = protein_mse / protein_samples
    
    return avg_loss, avg_metrics


def save_multi_protein_predictions(model, val_loader, device, save_path, 
                                  protein_names=None, num_samples=4):
    """Save multi-protein prediction visualizations."""
    model.eval()
    
    if protein_names is None:
        protein_names = [f"Protein_{i+1}" for i in range(19)]
    
    # Create visualization with 5 columns × 4 rows for 19 proteins
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    axes = axes.flatten()
    
    with torch.no_grad():
        for batch_idx, (he_img, protein_img) in enumerate(val_loader):
            if batch_idx >= num_samples:
                break
                
            he_img = he_img.to(device)
            protein_img = protein_img.to(device)
            
            pred_output = model(he_img)
            if isinstance(pred_output, tuple):
                pred_proteins = pred_output[0]
            else:
                pred_proteins = pred_output
            
            # Use first sample in batch
            sample_idx = 0
            he_np = he_img[sample_idx].cpu().numpy()
            
            # Show H&E image first
            if batch_idx == 0:
                axes[0].imshow(np.transpose(he_np, (1, 2, 0)))
                axes[0].set_title('H&E Input')
                axes[0].axis('off')
            
            # Show all protein predictions
            for protein_idx in range(min(19, pred_proteins.shape[1])):
                ax_idx = protein_idx + 1 if batch_idx == 0 else protein_idx
                if ax_idx < len(axes):
                    pred_np = pred_proteins[sample_idx, protein_idx].cpu().numpy()
                    true_np = protein_img[sample_idx, protein_idx].cpu().numpy()
                    
                    # Create side-by-side comparison
                    combined = np.hstack([true_np, pred_np])
                    
                    axes[ax_idx].imshow(combined, cmap='hot')
                    axes[ax_idx].set_title(f'{protein_names[protein_idx]}\n(GT | Pred)')
                    axes[ax_idx].axis('off')
            
            break  # Only process first batch for visualization
    
    # Hide unused axes
    for i in range(len(protein_names) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_protein_performance_plot(per_protein_mse, protein_names, save_path):
    """Create per-protein performance visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # MSE per protein (bar plot)
    bars = ax1.bar(range(len(per_protein_mse)), per_protein_mse)
    ax1.set_xlabel('Protein Index')
    ax1.set_ylabel('MSE')
    ax1.set_title('Per-Protein MSE Performance')
    ax1.set_xticks(range(len(per_protein_mse)))
    
    # Color bars by performance (green = good, red = poor)
    max_mse = np.max(per_protein_mse)
    for i, bar in enumerate(bars):
        normalized_mse = per_protein_mse[i] / max_mse
        color = plt.cm.RdYlGn_r(normalized_mse)
        bar.set_color(color)
    
    # Performance distribution (histogram)
    ax2.hist(per_protein_mse, bins=10, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(per_protein_mse), color='red', linestyle='--', 
               label=f'Mean: {np.mean(per_protein_mse):.4f}')
    ax2.axvline(np.median(per_protein_mse), color='blue', linestyle='--', 
               label=f'Median: {np.median(per_protein_mse):.4f}')
    ax2.set_xlabel('MSE')
    ax2.set_ylabel('Count')
    ax2.set_title('MSE Distribution Across Proteins')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train multi-protein H&E to Orion prediction model")
    parser.add_argument("--pairs_dir", type=str, 
                       default="output/registration_output/training_pairs",
                       help="Directory containing H&E training pairs")
    parser.add_argument("--original_orion_dir", type=str, required=True,
                       help="Directory containing original multi-channel Orion files")
    parser.add_argument("--output_dir", type=str, default="multi_protein_model",
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
    parser.add_argument("--base_features", type=int, default=64,
                       help="Base number of features in U-Net")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save model every N epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                       help="Early stopping patience (epochs)")
    parser.add_argument("--orion_suffix", type=str, default="_Orion.tif",
                       help="Suffix for original Orion files")
    parser.add_argument("--verify_data", action="store_true",
                       help="Verify data loading before training")
    parser.add_argument("--progressive_training", action="store_true",
                       help="Use progressive training with increasing loss complexity")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("="*60)
    logger.info("MULTI-PROTEIN H&E TO ORION PREDICTION TRAINING")
    logger.info("="*60)
    logger.info(f"H&E pairs directory: {args.pairs_dir}")
    logger.info(f"Original Orion directory: {args.original_orion_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Base features: {args.base_features}")
    logger.info(f"Progressive training: {args.progressive_training}")
    
    # Save configuration
    config = vars(args)
    config['device'] = str(device)
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    logger.info("Training configuration saved")
    
    # Verify data loading if requested
    if args.verify_data:
        logger.info("Verifying multi-channel data loading...")
        success = verify_multi_channel_data(args.pairs_dir, args.original_orion_dir)
        if not success:
            logger.error("Data verification failed. Please check your paths and data.")
            return
        logger.info("Data verification successful!")
    
    # Create data loaders
    logger.info("Creating multi-channel data loaders...")
    try:
        train_loader, val_loader = create_multi_channel_data_loaders(
            pairs_dir=args.pairs_dir,
            original_orion_dir=args.original_orion_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            protein_channels=list(range(1, 20)),  # Channels 1-19
            orion_suffix=args.orion_suffix
        )
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        logger.error("Please verify that:")
        logger.error(f"  1. H&E pairs exist in: {args.pairs_dir}")
        logger.error(f"  2. Original Orion files exist in: {args.original_orion_dir}")
        logger.error(f"  3. Orion files have suffix: {args.orion_suffix}")
        return
    
    # Create model
    logger.info("Initializing multi-protein model...")
    protein_names = [f"Protein_{i+1}" for i in range(19)]
    model = create_multi_protein_model(
        num_proteins=19,
        base_features=args.base_features,
        protein_names=protein_names,
        use_deep_supervision=True
    ).to(device)
    
    # Loss function with progressive training
    if args.progressive_training:
        criterion = MultiProteinLoss(
            mse_weight=1.0, ssim_weight=0.0, 
            correlation_weight=0.0, deep_supervision_weight=0.1
        )
    else:
        criterion = MultiProteinLoss(
            mse_weight=1.0, ssim_weight=0.3,
            correlation_weight=0.1, deep_supervision_weight=0.2
        )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=0.0001,
        restore_best_weights=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [],
        'train_ssim': [], 'val_ssim': [],
        'train_correlation': [], 'val_correlation': [],
        'learning_rate': [], 'per_protein_mse': []
    }
    
    best_val_loss = float('inf')
    
    logger.info("Starting multi-protein training...")
    logger.info("="*60)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Progressive training: adjust loss weights
        if args.progressive_training:
            if epoch >= 30 and epoch < 60:
                criterion.ssim_weight = 0.3
                logger.info("  → Added SSIM loss (progressive training)")
            elif epoch >= 60:
                criterion.ssim_weight = 0.3
                criterion.correlation_weight = 0.1
                logger.info("  → Added correlation loss (progressive training)")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_metrics.get('mse', 0))
        history['val_mse'].append(val_metrics.get('mse', 0))
        history['train_ssim'].append(train_metrics.get('ssim', 0))
        history['val_ssim'].append(val_metrics.get('ssim', 0))
        history['train_correlation'].append(train_metrics.get('correlation', 0))
        history['val_correlation'].append(val_metrics.get('correlation', 0))
        history['learning_rate'].append(new_lr)
        
        if 'per_protein_mse' in val_metrics:
            history['per_protein_mse'].append(val_metrics['per_protein_mse'])
        
        epoch_time = time.time() - start_time
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1:3d}/{args.epochs} ({epoch_time:.1f}s) | "
                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val MSE: {val_metrics.get('mse', 0):.4f} | "
                   f"Val SSIM: {val_metrics.get('ssim', 0):.4f} | "
                   f"LR: {new_lr:.2e}")
        
        # Log learning rate changes
        if new_lr != old_lr:
            logger.info(f"  → Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Log per-protein performance
        if 'per_protein_mse' in val_metrics:
            per_protein_mse = val_metrics['per_protein_mse']
            logger.info(f"  → Per-protein MSE: mean={np.mean(per_protein_mse):.4f}, "
                       f"std={np.std(per_protein_mse):.4f}, "
                       f"min={np.min(per_protein_mse):.4f}, "
                       f"max={np.max(per_protein_mse):.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'args': vars(args),
                'protein_names': protein_names
            }, output_path / 'best_multi_protein_model.pth')
            logger.info(f"  → New best model saved (val_loss: {val_loss:.4f})")
        
        # Check early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            logger.info(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break
        
        # Log early stopping progress
        if early_stopping.counter > 0:
            logger.info(f"  → Early stopping: {early_stopping.counter}/{args.early_stopping_patience}")
        
        # Save checkpoint and visualizations
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_path / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'args': vars(args),
                'protein_names': protein_names
            }, checkpoint_path)
            logger.info(f"  → Checkpoint saved: {checkpoint_path}")
            
            # Save multi-protein predictions
            pred_path = output_path / f'multi_protein_predictions_epoch_{epoch+1}.png'
            save_multi_protein_predictions(
                model, val_loader, device, pred_path, protein_names
            )
            logger.info(f"  → Multi-protein predictions saved: {pred_path}")
            
            # Save per-protein performance
            if 'per_protein_mse' in val_metrics:
                perf_path = output_path / f'protein_performance_epoch_{epoch+1}.png'
                create_protein_performance_plot(
                    val_metrics['per_protein_mse'], protein_names, perf_path
                )
                logger.info(f"  → Protein performance plot saved: {perf_path}")
        
        # Save training history periodically
        if (epoch + 1) % 5 == 0:
            history_df = pd.DataFrame({k: v for k, v in history.items() 
                                     if k != 'per_protein_mse'})
            history_df.to_csv(output_path / 'training_history.csv', index=False)
    
    logger.info("="*60)
    logger.info("Multi-protein training completed!")
    
    # Log final results
    final_epoch = len(history['train_loss'])
    logger.info(f"Total epochs trained: {final_epoch}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final learning rate: {history['learning_rate'][-1]:.2e}")
    
    # Save final model
    final_model_path = output_path / 'final_multi_protein_model.pth'
    torch.save({
        'epoch': final_epoch - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'args': vars(args),
        'protein_names': protein_names,
        'best_val_loss': best_val_loss
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Generate comprehensive training curves
    logger.info("Generating comprehensive training curves...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs_range, history['train_loss'], label='Train', alpha=0.8)
    axes[0, 0].plot(epochs_range, history['val_loss'], label='Validation', alpha=0.8)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE curves
    axes[0, 1].plot(epochs_range, history['train_mse'], label='Train', alpha=0.8)
    axes[0, 1].plot(epochs_range, history['val_mse'], label='Validation', alpha=0.8)
    axes[0, 1].set_title('Mean Squared Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM curves
    axes[0, 2].plot(epochs_range, history['train_ssim'], label='Train', alpha=0.8)
    axes[0, 2].plot(epochs_range, history['val_ssim'], label='Validation', alpha=0.8)
    axes[0, 2].set_title('SSIM Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('SSIM Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Correlation curves
    axes[1, 0].plot(epochs_range, history['train_correlation'], label='Train', alpha=0.8)
    axes[1, 0].plot(epochs_range, history['val_correlation'], label='Validation', alpha=0.8)
    axes[1, 0].set_title('Protein Correlation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(epochs_range, history['learning_rate'], alpha=0.8, color='orange')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Best model marker
    axes[1, 2].plot(epochs_range, history['val_loss'], alpha=0.8, color='red')
    best_epoch = np.argmin(history['val_loss']) + 1
    axes[1, 2].scatter(best_epoch, best_val_loss, color='gold', s=100, zorder=5,
                      label=f'Best (Epoch {best_epoch})')
    axes[1, 2].set_title('Validation Loss with Best Model')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Validation Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = output_path / 'comprehensive_training_curves.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved: {curves_path}")
    
    # Final summary
    logger.info("="*60)
    logger.info("MULTI-PROTEIN TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f} (epoch {np.argmin(history['val_loss']) + 1})")
    logger.info(f"Model predicts: {len(protein_names)} proteins simultaneously")
    logger.info("Generated files:")
    logger.info(f"  - Best model: {output_path / 'best_multi_protein_model.pth'}")
    logger.info(f"  - Final model: {final_model_path}")
    logger.info(f"  - Training config: {output_path / 'training_config.json'}")
    logger.info(f"  - Training curves: {curves_path}")
    logger.info(f"  - Training logs: {output_path / 'logs' / 'multi_protein_training.log'}")
    logger.info("Multi-protein training completed successfully! 🚀")


if __name__ == "__main__":
    main() 