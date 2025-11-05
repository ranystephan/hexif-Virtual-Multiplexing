#!/usr/bin/env python3
"""
Diagnostic script to identify training bottlenecks.
Run this to understand why training is slow.
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from train_spatial_orion_unet import make_loaders, UNetSmall

class Args:
    """Mock args for testing"""
    pairs_dir = "core_patches_npy"
    patch_size = 384
    batch_size = 32
    val_batch_size = 16
    patches_per_image = 16  # Reduced for testing
    patches_per_image_val = 8
    grid_stride = 192
    base_features = 32
    use_boundary_guidance = True
    noise_removal = True
    num_workers = 16
    val_split = 0.2
    seed = 42

def benchmark_data_loading():
    """Test data loading speed"""
    print("🔍 Benchmarking data loading...")
    
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        train_loader, val_loader = make_loaders(args, device)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Time data loading
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            if i >= 5:  # Test first 5 batches
                break
            print(f"Batch {i+1}: {batch['he'].shape}, {batch['target'].shape}")
        
        data_time = time.time() - start_time
        print(f"⏱️ Data loading time (5 batches): {data_time:.2f}s ({data_time/5:.2f}s per batch)")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    return True

def benchmark_model_forward():
    """Test model forward pass speed"""
    print("\n🔍 Benchmarking model forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    print(f"Device: {device}")
    print(f"GPUs available: {num_gpus}")
    
    # Create model
    model = UNetSmall(in_ch=3, out_ch=20, base=32, use_boundary_guidance=True)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {num_gpus} GPUs")
    
    model = model.to(device)
    model.eval()
    
    # Test different batch sizes
    batch_sizes = [4, 8, 16, 32]
    patch_size = 384
    
    for bs in batch_sizes:
        try:
            # Create dummy input
            he = torch.randn(bs, 3, patch_size, patch_size, device=device)
            boundary = torch.randn(bs, 1, patch_size, patch_size, device=device)
            
            # Warmup
            with torch.no_grad():
                _ = model(he, boundary)
            
            # Benchmark
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    output = model(he, boundary)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_time = time.time() - start_time
            
            print(f"Batch size {bs}: {forward_time/10:.3f}s per forward pass")
            
            # Check memory usage
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"  Memory used: {memory_used:.2f}GB")
                torch.cuda.reset_peak_memory_stats()
            
        except RuntimeError as e:
            print(f"Batch size {bs}: ❌ Failed - {e}")

def benchmark_gpu_utilization():
    """Check GPU utilization"""
    print("\n🔍 Checking GPU utilization...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / (1024**3)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
        
        print(f"GPU {i} ({props.name}):")
        print(f"  Total memory: {memory_total:.1f}GB")
        print(f"  Allocated: {memory_allocated:.1f}GB")
        print(f"  Cached: {memory_cached:.1f}GB")

def main():
    print("🚀 Training Speed Diagnostic Tool")
    print("=" * 50)
    
    # Check basic setup
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Run benchmarks
    benchmark_gpu_utilization()
    
    if benchmark_data_loading():
        benchmark_model_forward()
    
    print("\n📊 Recommendations:")
    print("1. If data loading is slow: Increase num_workers or reduce patch_size")
    print("2. If forward pass is slow: Reduce batch_size or model size")
    print("3. If GPU utilization is low: Check DataParallel setup")
    print("4. If memory is full: Reduce batch_size or patch_size")

if __name__ == "__main__":
    main()
