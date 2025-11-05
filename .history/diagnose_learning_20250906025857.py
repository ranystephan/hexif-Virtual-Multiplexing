#!/usr/bin/env python3
"""
Diagnostic script to identify why the model isn't learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from train_spatial_orion_unet import make_loaders, UNetSmall, SpatialAwareLoss

class Args:
    pairs_dir = "core_patches_npy"
    patch_size = 256
    batch_size = 16
    val_batch_size = 8
    patches_per_image = 8
    patches_per_image_val = 4
    grid_stride = 128
    base_features = 32
    use_boundary_guidance = True
    noise_removal = True
    num_workers = 4
    val_split = 0.1
    seed = 42
    mse_weight = 0.5
    l1_weight = 0.7
    boundary_weight = 0.4
    focal_weight = 0.3

def analyze_data_distribution():
    """Analyze the data distribution to understand the target values"""
    print("🔍 Analyzing data distribution...")
    
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        train_loader, val_loader = make_loaders(args, device)
        
        # Collect statistics from a few batches
        target_values = []
        input_values = []
        
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Analyze first 3 batches
                break
                
            he = batch['he']
            target = batch['target']
            
            target_values.append(target.detach().cpu().numpy())
            input_values.append(he.detach().cpu().numpy())
            
        target_values = np.concatenate(target_values, axis=0)
        input_values = np.concatenate(input_values, axis=0)
        
        print(f"Target shape: {target_values.shape}")
        print(f"Target range: [{target_values.min():.4f}, {target_values.max():.4f}]")
        print(f"Target mean: {target_values.mean():.4f}")
        print(f"Target std: {target_values.std():.4f}")
        print(f"Target sparsity (zeros): {(target_values == 0).mean():.2%}")
        
        # Check per-channel statistics
        print("\nPer-channel statistics:")
        for c in range(min(5, target_values.shape[1])):  # First 5 channels
            channel_data = target_values[:, c, :, :]
            print(f"  Channel {c}: mean={channel_data.mean():.4f}, std={channel_data.std():.4f}, max={channel_data.max():.4f}")
        
        return target_values, input_values
        
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")
        return None, None

def analyze_model_predictions():
    """Analyze what the model is actually predicting"""
    print("\n🔍 Analyzing model predictions...")
    
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNetSmall(in_ch=3, out_ch=20, base=32, use_boundary_guidance=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    try:
        train_loader, _ = make_loaders(args, device)
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 1:  # Just one batch
                    break
                    
                he = batch['he'].to(device)
                target = batch['target'].to(device)
                boundary_mask = batch.get('boundary_mask')
                if boundary_mask is not None:
                    boundary_mask = boundary_mask.to(device)
                
                # Get predictions
                if boundary_mask is not None:
                    pred = model(he, boundary_mask)
                else:
                    pred = model(he)
                
                pred_np = pred.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                
                print(f"Prediction shape: {pred_np.shape}")
                print(f"Prediction range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
                print(f"Prediction mean: {pred_np.mean():.4f}")
                print(f"Prediction std: {pred_np.std():.4f}")
                
                # Check if predictions are too uniform
                pred_var = np.var(pred_np, axis=(2, 3)).mean()  # Spatial variance
                print(f"Spatial variance (should be >0.001): {pred_var:.6f}")
                
                # Compare with targets
                mse = np.mean((pred_np - target_np)**2)
                mae = np.mean(np.abs(pred_np - target_np))
                print(f"MSE with targets: {mse:.4f}")
                print(f"MAE with targets: {mae:.4f}")
                
                return pred_np, target_np
                
    except Exception as e:
        print(f"❌ Model analysis failed: {e}")
        return None, None

def analyze_loss_function():
    """Analyze the loss function behavior"""
    print("\n🔍 Analyzing loss function...")
    
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function
    criterion = SpatialAwareLoss(
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        boundary_weight=args.boundary_weight,
        focal_weight=args.focal_weight
    )
    
    # Test with dummy data
    batch_size, channels, height, width = 4, 20, 256, 256
    
    # Create test cases
    pred_uniform = torch.full((batch_size, channels, height, width), 0.5, device=device)
    pred_random = torch.rand((batch_size, channels, height, width), device=device)
    target_sparse = torch.zeros((batch_size, channels, height, width), device=device)
    target_sparse[:, :5, 50:150, 50:150] = 1.0  # Some regions with signal
    
    valid_mask = torch.ones_like(target_sparse, dtype=torch.bool)
    
    loss_uniform = criterion(pred_uniform, target_sparse, valid_mask)
    loss_random = criterion(pred_random, target_sparse, valid_mask)
    loss_perfect = criterion(target_sparse, target_sparse, valid_mask)
    
    print(f"Loss with uniform predictions (0.5): {loss_uniform:.4f}")
    print(f"Loss with random predictions: {loss_random:.4f}")
    print(f"Loss with perfect predictions: {loss_perfect:.4f}")
    
    # Check gradient magnitudes
    pred_test = torch.rand((batch_size, channels, height, width), device=device, requires_grad=True)
    loss_test = criterion(pred_test, target_sparse, valid_mask)
    loss_test.backward()
    
    grad_magnitude = pred_test.grad.abs().mean().item()
    print(f"Gradient magnitude: {grad_magnitude:.6f}")
    
    if grad_magnitude < 1e-6:
        print("⚠️  Gradients are very small - possible vanishing gradient problem")
    elif grad_magnitude > 1.0:
        print("⚠️  Gradients are very large - possible exploding gradient problem")

def check_learning_rate():
    """Check if learning rate is appropriate"""
    print("\n🔍 Checking learning rate...")
    
    # Typical learning rate ranges for different loss scales
    current_loss = 0.25  # Approximate current loss
    
    print(f"Current loss scale: {current_loss}")
    
    if current_loss > 1.0:
        print("💡 Loss is high - try higher learning rate (1e-3 to 5e-3)")
    elif current_loss > 0.1:
        print("💡 Loss is moderate - current LR might be OK, try 1e-4 to 1e-3")
    else:
        print("💡 Loss is low - try lower learning rate (1e-5 to 1e-4)")

def main():
    print("🚀 Learning Diagnostic Tool")
    print("=" * 50)
    
    # Run diagnostics
    target_data, input_data = analyze_data_distribution()
    pred_data, target_comparison = analyze_model_predictions()
    analyze_loss_function()
    check_learning_rate()
    
    print("\n📊 Recommendations:")
    print("1. If targets are very sparse: Increase focal_weight or use weighted loss")
    print("2. If predictions are uniform: Check model architecture or initialization")
    print("3. If gradients are small: Increase learning rate or check loss scaling")
    print("4. If loss is high: Try different normalization or loss function")
    print("5. If no spatial variance: Model might be collapsing to mean prediction")

if __name__ == "__main__":
    main()
