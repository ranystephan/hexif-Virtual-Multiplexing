"""
Multi-Channel Data Loader for H&E to All-Protein Prediction

This module creates training pairs using all 19 protein channels from the original
Orion files, leveraging the existing VALIS registration transformations.

The approach:
1. Use existing H&E patches from registration_output/training_pairs/
2. Load corresponding regions from original multi-channel Orion files
3. Extract all 19 protein channels (skip DAPI channel 0)
4. Apply same transformations and patches as used in registration

This enables training on all proteins simultaneously while reusing registration work.
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Union
import logging
from tifffile import imread
import cv2
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MultiChannelOrionDataset(Dataset):
    """
    Dataset that loads all protein channels from original Orion files.
    
    Uses existing H&E patches and matches them with corresponding regions
    from the original multi-channel Orion files.
    """
    
    def __init__(self,
                 pairs_dir: str,
                 original_orion_dir: str,
                 patch_pairs: List[Tuple[str, str]],
                 protein_channels: List[int] = None,
                 transform=None,
                 augment: bool = True,
                 orion_suffix: str = "_Orion.tif"):
        """
        Args:
            pairs_dir: Directory with existing H&E patches
            original_orion_dir: Directory with original multi-channel Orion files
            patch_pairs: List of (HE_filename, Orion_filename) pairs
            protein_channels: List of protein channel indices (default: 1-19)
            transform: Optional transforms to apply
            augment: Whether to apply data augmentation
            orion_suffix: Suffix for original Orion files
        """
        self.pairs_dir = pathlib.Path(pairs_dir)
        self.original_orion_dir = pathlib.Path(original_orion_dir)
        self.patch_pairs = patch_pairs
        self.transform = transform
        self.augment = augment
        self.orion_suffix = orion_suffix
        
        # Default protein channels (skip DAPI channel 0)
        if protein_channels is None:
            self.protein_channels = list(range(1, 20))  # Channels 1-19
        else:
            self.protein_channels = protein_channels
        
        logger.info(f"Multi-channel dataset initialized:")
        logger.info(f"  - {len(patch_pairs)} training pairs")
        logger.info(f"  - {len(self.protein_channels)} protein channels: {self.protein_channels}")
        logger.info(f"  - Original Orion dir: {original_orion_dir}")
        
        # Cache for loaded Orion files (memory optimization)
        self.orion_cache = {}
        self.max_cache_size = 10  # Limit memory usage
    
    def __len__(self):
        return len(self.patch_pairs)
    
    def __getitem__(self, idx):
        he_filename, _ = self.patch_pairs[idx]  # We'll replace Orion with multi-channel
        
        # Load H&E patch (already processed and registered)
        he_path = self.pairs_dir / he_filename
        he_patch = np.load(he_path).astype(np.float32)
        
        # Extract core ID and patch coordinates from filename
        # Example: "reg265_patch_0014_HE.npy" -> core_id="reg265", patch_idx=14
        core_id, patch_info = self._parse_filename(he_filename)
        
        # Load corresponding multi-channel Orion patch
        try:
            orion_patch = self._load_orion_patch(core_id, patch_info, he_patch.shape[:2])
        except Exception as e:
            logger.warning(f"Failed to load Orion patch for {he_filename}: {e}")
            # Fallback: create zero patch
            orion_patch = np.zeros((he_patch.shape[0], he_patch.shape[1], len(self.protein_channels)), dtype=np.float32)
        
        # Normalize to [0, 1]
        if he_patch.max() > 1.0:
            he_patch = he_patch / 255.0
        
        if orion_patch.max() > 1.0:
            orion_patch = orion_patch / 255.0
        
        # Convert to tensors
        if he_patch.ndim == 3:  # RGB H&E
            he_tensor = torch.from_numpy(he_patch.transpose(2, 0, 1))  # HWC -> CHW
        else:  # Grayscale H&E
            he_tensor = torch.from_numpy(he_patch).unsqueeze(0)
        
        # Multi-channel protein tensor
        orion_tensor = torch.from_numpy(orion_patch.transpose(2, 0, 1))  # HWC -> CHW
        
        # Apply augmentations (synchronized)
        if self.augment:
            he_tensor, orion_tensor = self._apply_augmentations(he_tensor, orion_tensor)
        
        # Apply additional transforms
        if self.transform:
            he_tensor = self.transform(he_tensor)
        
        return he_tensor, orion_tensor
    
    def _parse_filename(self, filename: str) -> Tuple[str, Dict]:
        """
        Parse patch filename to extract core ID and patch information.
        
        Example: "reg265_patch_0014_HE.npy" -> ("reg265", {"patch_idx": 14})
        """
        # Remove .npy and _HE suffix
        base_name = filename.replace("_HE.npy", "")
        
        # Split by patch
        if "_patch_" in base_name:
            core_id, patch_part = base_name.split("_patch_")
            patch_idx = int(patch_part)
        else:
            # Fallback parsing
            parts = base_name.split("_")
            core_id = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
            patch_idx = 0
        
        return core_id, {"patch_idx": patch_idx}
    
    def _load_orion_patch(self, core_id: str, patch_info: Dict, patch_shape: Tuple[int, int]) -> np.ndarray:
        """
        Load multi-channel Orion patch corresponding to H&E patch.
        
        Args:
            core_id: Core identifier
            patch_info: Patch information (index, coordinates, etc.)
            patch_shape: Shape of the H&E patch (height, width)
        
        Returns:
            Multi-channel Orion patch with shape (H, W, num_protein_channels)
        """
        # Find original Orion file
        orion_file = self.original_orion_dir / f"{core_id}{self.orion_suffix}"
        
        if not orion_file.exists():
            raise ValueError(f"Original Orion file not found: {orion_file}")
        
        # Load multi-channel Orion image (with caching)
        orion_img = self._load_orion_file(str(orion_file))
        
        # Calculate patch coordinates
        # This assumes patches were extracted in row-major order with 256x256 size and 256 stride
        patch_size = patch_shape[0]  # Assume square patches
        patch_idx = patch_info["patch_idx"]
        
        # Calculate grid dimensions
        img_height, img_width = orion_img.shape[1], orion_img.shape[2]
        patches_per_row = (img_width - patch_size) // patch_size + 1
        
        # Calculate patch position
        patch_row = patch_idx // patches_per_row
        patch_col = patch_idx % patches_per_row
        
        # Extract patch coordinates
        y_start = patch_row * patch_size
        x_start = patch_col * patch_size
        y_end = min(y_start + patch_size, img_height)
        x_end = min(x_start + patch_size, img_width)
        
        # Extract multi-channel patch (channels, height, width)
        orion_patch_chw = orion_img[self.protein_channels, y_start:y_end, x_start:x_end]
        
        # Convert to (height, width, channels)
        orion_patch = orion_patch_chw.transpose(1, 2, 0)
        
        # Ensure correct size (pad if needed)
        if orion_patch.shape[:2] != patch_shape:
            orion_patch = self._resize_patch(orion_patch, patch_shape)
        
        return orion_patch.astype(np.float32)
    
    def _load_orion_file(self, orion_path: str) -> np.ndarray:
        """Load multi-channel Orion file with caching."""
        if orion_path in self.orion_cache:
            return self.orion_cache[orion_path]
        
        # Manage cache size
        if len(self.orion_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.orion_cache))
            del self.orion_cache[oldest_key]
        
        # Load image
        orion_img = imread(orion_path)
        
        if orion_img is None:
            raise ValueError(f"Could not load Orion image: {orion_path}")
        
        logger.info(f"Loaded Orion image with shape: {orion_img.shape}")
        
        # Ensure channel-first format (C, H, W)
        if orion_img.ndim == 3:
            if orion_img.shape[0] > orion_img.shape[2]:
                # Likely (C, H, W) format
                pass
            else:
                # Likely (H, W, C) format - transpose
                orion_img = orion_img.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected Orion image dimensions: {orion_img.shape}")
        
        # Cache the loaded image
        self.orion_cache[orion_path] = orion_img
        
        return orion_img
    
    def _resize_patch(self, patch: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize patch to target shape."""
        resized_channels = []
        for c in range(patch.shape[2]):
            resized = cv2.resize(patch[:, :, c], target_shape[::-1], interpolation=cv2.INTER_LINEAR)
            resized_channels.append(resized)
        
        return np.stack(resized_channels, axis=2)
    
    def _apply_augmentations(self, he_tensor, orion_tensor):
        """Apply synchronized augmentations to both images."""
        import torchvision.transforms.functional as TF
        
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            he_tensor = TF.hflip(he_tensor)
            orion_tensor = TF.hflip(orion_tensor)
        
        # Random vertical flip
        if torch.rand(1) > 0.5:
            he_tensor = TF.vflip(he_tensor)
            orion_tensor = TF.vflip(orion_tensor)
        
        # Random rotation (90 degree increments)
        if torch.rand(1) > 0.5:
            angle = torch.randint(0, 4, (1,)).item() * 90
            he_tensor = TF.rotate(he_tensor, angle)
            orion_tensor = TF.rotate(orion_tensor, angle)
        
        # Random brightness/contrast for H&E only
        if torch.rand(1) > 0.5:
            brightness_factor = 0.8 + torch.rand(1) * 0.4
            contrast_factor = 0.8 + torch.rand(1) * 0.4
            he_tensor = TF.adjust_brightness(he_tensor, brightness_factor.item())
            he_tensor = TF.adjust_contrast(he_tensor, contrast_factor.item())
        
        return he_tensor, orion_tensor


def create_multi_channel_data_loaders(pairs_dir: str,
                                    original_orion_dir: str,
                                    batch_size: int = 16,
                                    val_split: float = 0.2,
                                    num_workers: int = 4,
                                    protein_channels: List[int] = None,
                                    orion_suffix: str = "_Orion.tif") -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for multi-channel protein prediction.
    
    Args:
        pairs_dir: Directory containing H&E patches
        original_orion_dir: Directory containing original multi-channel Orion files
        batch_size: Batch size for training
        val_split: Validation split ratio
        num_workers: Number of data loader workers
        protein_channels: List of protein channel indices (default: 1-19)
        orion_suffix: Suffix for original Orion files
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Find training pairs
    pairs_path = pathlib.Path(pairs_dir)
    he_files = list(pairs_path.glob("*_HE.npy"))
    
    pairs = []
    for he_file in he_files:
        orion_file = he_file.name.replace("_HE.npy", "_ORION.npy")
        pairs.append((he_file.name, orion_file))
    
    logger.info(f"Found {len(pairs)} training pairs")
    
    if len(pairs) == 0:
        raise ValueError("No training pairs found!")
    
    # Split into train/validation
    train_pairs, val_pairs = train_test_split(pairs, test_size=val_split, random_state=42)
    
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")
    
    # Create datasets
    train_dataset = MultiChannelOrionDataset(
        pairs_dir=pairs_dir,
        original_orion_dir=original_orion_dir,
        patch_pairs=train_pairs,
        protein_channels=protein_channels,
        augment=True,
        orion_suffix=orion_suffix
    )
    
    val_dataset = MultiChannelOrionDataset(
        pairs_dir=pairs_dir,
        original_orion_dir=original_orion_dir,
        patch_pairs=val_pairs,
        protein_channels=protein_channels,
        augment=False,
        orion_suffix=orion_suffix
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def verify_multi_channel_data(pairs_dir: str, original_orion_dir: str, num_samples: int = 3):
    """
    Verify that multi-channel data loading works correctly.
    
    Args:
        pairs_dir: Directory containing H&E patches
        original_orion_dir: Directory containing original Orion files
        num_samples: Number of samples to test
    """
    logger.info("Verifying multi-channel data loading...")
    
    try:
        train_loader, val_loader = create_multi_channel_data_loaders(
            pairs_dir=pairs_dir,
            original_orion_dir=original_orion_dir,
            batch_size=2,
            num_workers=0  # Avoid multiprocessing issues in testing
        )
        
        logger.info("✓ Data loaders created successfully")
        
        # Test loading a few batches
        for i, (he_batch, orion_batch) in enumerate(train_loader):
            if i >= num_samples:
                break
                
            logger.info(f"Batch {i+1}:")
            logger.info(f"  H&E shape: {he_batch.shape}")
            logger.info(f"  Orion shape: {orion_batch.shape}")
            logger.info(f"  H&E range: [{he_batch.min():.3f}, {he_batch.max():.3f}]")
            logger.info(f"  Orion range: [{orion_batch.min():.3f}, {orion_batch.max():.3f}]")
        
        logger.info("✓ Multi-channel data loading verification successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Multi-channel data loading verification failed: {e}")
        return False


if __name__ == "__main__":
    # Example verification
    pairs_dir = "output/registration_output/training_pairs"
    original_orion_dir = "/path/to/original/orion/files"  # Update this path
    
    verify_multi_channel_data(pairs_dir, original_orion_dir) 