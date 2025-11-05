#!/usr/bin/env python3
"""
Training Monitor - Real-time analysis of training progress and issues
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import json

def plot_training_metrics(csv_path: Path, output_dir: Path):
    """Plot training and validation metrics to monitor progress."""
    try:
        df = pd.read_csv(csv_path)
        
        # Separate training and validation data
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        
        if len(train_df) == 0 or len(val_df) == 0:
            print("❌ Insufficient data for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Loss curves
        axes[0, 0].plot(train_df['epoch'], train_df['loss'], 'b-', label='Training', marker='o', markersize=4)
        axes[0, 0].plot(val_df['epoch'], val_df['loss'], 'r-', label='Validation', marker='s', markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss stability (validation loss variation)
        if len(val_df) > 3:
            val_loss_rolling_std = val_df['loss'].rolling(window=3, min_periods=1).std()
            axes[0, 1].plot(val_df['epoch'], val_loss_rolling_std, 'g-', marker='d', markersize=4)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Validation Loss Rolling Std')
            axes[0, 1].set_title('Training Stability (Lower = More Stable)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add stability threshold line
            stability_threshold = 0.05
            axes[0, 1].axhline(y=stability_threshold, color='r', linestyle='--', 
                              label=f'Stability Threshold ({stability_threshold})')
            axes[0, 1].legend()
        
        # Plot 3: Learning rate (if available)
        axes[1, 0].plot(train_df['epoch'], train_df['time_sec'], 'm-', marker='x', markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Overfitting detection (training vs validation gap)
        if len(train_df) == len(val_df):
            gap = val_df['loss'].values - train_df['loss'].values
            epochs = train_df['epoch'].values
            axes[1, 1].plot(epochs, gap, 'orange', marker='v', markersize=4)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Validation - Training Loss')
            axes[1, 1].set_title('Overfitting Detection (Lower = Better)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add overfitting warning threshold
            overfitting_threshold = 0.1
            axes[1, 1].axhline(y=overfitting_threshold, color='r', linestyle='--', 
                              label=f'Warning Threshold ({overfitting_threshold})')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_dir / 'training_monitor.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training plots saved: {plot_path}")
        
        # Print current statistics
        latest_train = train_df.iloc[-1]['loss']
        latest_val = val_df.iloc[-1]['loss']
        latest_epoch = train_df.iloc[-1]['epoch']
        
        print(f"\n📊 Latest Statistics (Epoch {latest_epoch}):")
        print(f"  Training Loss: {latest_train:.4f}")
        print(f"  Validation Loss: {latest_val:.4f}")
        print(f"  Train-Val Gap: {latest_val - latest_train:.4f}")
        
        # Stability analysis
        if len(val_df) >= 5:
            recent_val_std = val_df['loss'].tail(5).std()
            print(f"  Recent Val Stability (std): {recent_val_std:.4f}")
            if recent_val_std > 0.05:
                print("  ⚠️  HIGH INSTABILITY - Consider reducing learning rate")
            else:
                print("  ✅ Training appears stable")
                
        # Convergence analysis
        if len(train_df) >= 5:
            early_train = train_df['loss'].head(3).mean()
            recent_train = train_df['loss'].tail(3).mean()
            improvement = early_train - recent_train
            print(f"  Training Improvement: {improvement:.4f}")
            if improvement < 0.05:
                print("  ⚠️  SLOW CONVERGENCE - Consider adjusting hyperparameters")
            else:
                print("  ✅ Good convergence rate")
        
    except Exception as e:
        print(f"❌ Error plotting metrics: {e}")

def analyze_config(config_path: Path):
    """Analyze training configuration for potential issues."""
    try:
        with open(config_path) as f:
            config = json.load(f)
            
        print("\n🔧 Configuration Analysis:")
        
        # Learning rate analysis
        lr = config.get('learning_rate', 0)
        print(f"  Learning Rate: {lr}")
        if lr > 3e-4:
            print("    ⚠️  Learning rate might be too high for stability")
        elif lr < 5e-5:
            print("    ⚠️  Learning rate might be too low for good convergence")
        else:
            print("    ✅ Learning rate seems reasonable")
            
        # Batch size analysis
        batch_size = config.get('batch_size', 0)
        print(f"  Batch Size: {batch_size}")
        if batch_size < 8:
            print("    ⚠️  Small batch size may cause unstable gradients")
        elif batch_size > 32:
            print("    ⚠️  Large batch size may slow convergence")
        else:
            print("    ✅ Batch size seems reasonable")
            
        # Loss weights analysis
        mse_w = config.get('mse_weight', 1.0)
        l1_w = config.get('l1_weight', 0.0)
        boundary_w = config.get('boundary_weight', 0.0)
        focal_w = config.get('focal_weight', 0.0)
        var_w = config.get('variance_weight', 0.0)
        
        total_weight = mse_w + l1_w + boundary_w + focal_w + var_w
        print(f"  Total Loss Weight: {total_weight:.2f}")
        print(f"  Loss Components: MSE={mse_w}, L1={l1_w}, Boundary={boundary_w}, Focal={focal_w}, Variance={var_w}")
        
        if total_weight > 2.0:
            print("    ⚠️  High total loss weight may cause training instability")
        if var_w > 0.2:
            print("    ⚠️  High variance weight may interfere with convergence")
        if focal_w > 0.2:
            print("    ⚠️  High focal weight may cause instability")
            
    except Exception as e:
        print(f"❌ Error analyzing config: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Training output directory to monitor')
    parser.add_argument('--watch', action='store_true', 
                       help='Watch for changes and update plots continuously')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds when watching')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    csv_path = output_dir / 'metrics_overall.csv'
    config_path = output_dir / 'config.json'
    
    if not output_dir.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        return
        
    print(f"🔍 Monitoring training in: {output_dir}")
    
    # Analyze configuration once
    if config_path.exists():
        analyze_config(config_path)
    
    if args.watch:
        print(f"\n🔄 Watching for updates every {args.interval} seconds...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                if csv_path.exists():
                    plot_training_metrics(csv_path, output_dir)
                else:
                    print(f"⏳ Waiting for metrics file: {csv_path}")
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
    else:
        # Single run
        if csv_path.exists():
            plot_training_metrics(csv_path, output_dir)
        else:
            print(f"❌ Metrics file not found: {csv_path}")

if __name__ == "__main__":
    main()
