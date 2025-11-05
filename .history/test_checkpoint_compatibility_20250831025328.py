#!/usr/bin/env python3
"""
Test script to verify checkpoint compatibility between fast_train_orion.py and test_inference_epoch10.ipynb
"""

import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_checkpoint_compatibility():
    print("Testing checkpoint compatibility...")
    
    # Try importing models
    try:
        from fast_train_orion import LightweightUNet
        print("✓ Successfully imported LightweightUNet from fast_train_orion.py")
    except Exception as e:
        print(f"✗ Failed to import LightweightUNet: {e}")
        return False
    
    # Create a dummy checkpoint like fast_train_orion.py would save
    device = torch.device("cpu")
    model = LightweightUNet(in_channels=3, out_channels=20, base=32)
    
    # Create checkpoint structure
    checkpoint = {
        "epoch": 10,
        "model": model.state_dict(),
        "val_loss": 0.1234,
        "args": {
            "base_features": 32,
            "use_advanced_model": False,
        }
    }
    
    print("✓ Created dummy checkpoint structure")
    
    # Test model inference from keys (like the notebook does)
    state_dict = checkpoint['model']
    keys = list(state_dict.keys())
    
    # Check if we can identify LightweightUNet from keys
    if any(k.startswith(('d1.', 'c1.', 'u1.', 'u2.', 'u3.')) for k in keys):
        model_class = 'LightweightUNet'
        print("✓ Correctly identified LightweightUNet from state dict keys")
    else:
        print("✗ Failed to identify LightweightUNet from state dict keys")
        return False
    
    # Test loading weights
    try:
        test_model = LightweightUNet(in_channels=3, out_channels=20, base=32)
        test_model.load_state_dict(state_dict)
        print("✓ Successfully loaded state dict into LightweightUNet")
    except Exception as e:
        print(f"✗ Failed to load state dict: {e}")
        return False
    
    # Test inference
    try:
        test_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            output = test_model(dummy_input)
            expected_shape = (1, 20, 256, 256)
            if output.shape == expected_shape:
                print(f"✓ Inference successful, output shape: {output.shape}")
            else:
                print(f"✗ Wrong output shape: {output.shape}, expected: {expected_shape}")
                return False
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    print("\n🎉 All compatibility tests passed!")
    print("The test_inference_epoch10.ipynb notebook should work with fast_train_orion.py checkpoints.")
    return True

if __name__ == "__main__":
    success = test_checkpoint_compatibility()
    sys.exit(0 if success else 1)
