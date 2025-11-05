"""
Registration Dataset for H&E to Multiplex Protein Prediction

This module provides PyTorch dataset classes for loading registered training pairs
and integrates with the existing ROSIE training pipeline.
"""

import os
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class RegistrationDataset(Dataset):
    """
    PyTorch dataset for registered H&E to multiplex protein training pairs.
    
    This dataset loads pre-registered image pairs and provides them in a format
    suitable for training deep learning models for in-silico staining.
    """
    
    def __init__(self, 
                 pairs_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None,
                 normalize_he: bool = True,
                 normalize_orion: bool = True,
                 he_channels: int = 3,
                 orion_channels: int = 1,
                 patch_size: int = 256):
        """
        Initialize the registration dataset.
        
        Args:
            pairs_dir: Directory containing training pairs (.npy files)
            transform: Transform to apply to H&E images
            target_transform: Transform to apply to Orion images
            normalize_he: Whether to normalize H&E images to [0, 1]
            normalize_orion: Whether to normalize Orion images to [0, 1]
            he_channels: Number of channels in H&E images
            orion_channels: Number of channels in Orion images
            patch_size: Size of image patches
        """
        self.pairs_dir = pathlib.Path(pairs_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_he = normalize_he
        self.normalize_orion = normalize_orion
        self.he_channels = he_channels
        self.orion_channels = orion_channels
        self.patch_size = patch_size
        
        # Find all training pairs
        self.pair_files = self._find_training_pairs()
        logger.info(f"Found {len(self.pair_files)} training pairs in {pairs_dir}")
        
        if len(self.pair_files) == 0:
            raise ValueError(f"No training pairs found in {pairs_dir}")
    
    def _find_training_pairs(self) -> List[Tuple[str, str]]:
        """Find all H&E and Orion training pair files."""
        he_files = list(self.pairs_dir.glob("*_HE.npy"))
        pair_files = []
        
        for he_file in he_files:
            # Extract base name
            base_name = he_file.stem.replace("_HE", "")
            orion_file = self.pairs_dir / f"{base_name}_ORION.npy"
            
            if orion_file.exists():
                pair_files.append((str(he_file), str(orion_file)))
            else:
                logger.warning(f"No Orion file found for {he_file}")
        
        return pair_files
    
    def __len__(self) -> int:
        return len(self.pair_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training pair.
        
        Args:
            idx: Index of the training pair
            
        Returns:
            Tuple of (he_image, orion_image) as torch tensors
        """
        he_path, orion_path = self.pair_files[idx]
        
        # Load images
        he_img = np.load(he_path)
        orion_img = np.load(orion_path)
        
        # Ensure correct shape and type
        he_img = self._preprocess_he(he_img)
        orion_img = self._preprocess_orion(orion_img)
        
        # Apply transforms
        if self.transform is not None:
            he_img = self.transform(he_img)
        
        if self.target_transform is not None:
            orion_img = self.target_transform(orion_img)
        
        return he_img, orion_img
    
    def _preprocess_he(self, he_img: np.ndarray) -> np.ndarray:
        """Preprocess H&E image."""
        # Ensure correct shape
        if he_img.ndim == 2:
            he_img = np.stack([he_img] * self.he_channels, axis=-1)
        elif he_img.ndim == 3 and he_img.shape[2] != self.he_channels:
            if he_img.shape[2] > self.he_channels:
                he_img = he_img[:, :, :self.he_channels]
            else:
                he_img = np.concatenate([he_img] * (self.he_channels // he_img.shape[2] + 1), axis=2)
                he_img = he_img[:, :, :self.he_channels]
        
        # Normalize to [0, 1]
        if self.normalize_he:
            if he_img.dtype == np.uint8:
                he_img = he_img.astype(np.float32) / 255.0
            else:
                he_img = he_img.astype(np.float32)
        
        return he_img
    
    def _preprocess_orion(self, orion_img: np.ndarray) -> np.ndarray:
        """Preprocess Orion image."""
        # Ensure correct shape
        if orion_img.ndim == 2:
            orion_img = orion_img[:, :, np.newaxis]
        elif orion_img.ndim == 3 and orion_img.shape[2] != self.orion_channels:
            if orion_img.shape[2] > self.orion_channels:
                orion_img = orion_img[:, :, :self.orion_channels]
            else:
                orion_img = np.concatenate([orion_img] * (self.orion_channels // orion_img.shape[2] + 1), axis=2)
                orion_img = orion_img[:, :, :self.orion_channels]
        
        # Normalize to [0, 1]
        if self.normalize_orion:
            if orion_img.dtype == np.uint8:
                orion_img = orion_img.astype(np.float32) / 255.0
            else:
                orion_img = orion_img.astype(np.float32)
        
        return orion_img


class MultiChannelRegistrationDataset(RegistrationDataset):
    """
    Extended dataset for multi-channel Orion images.
    
    This dataset handles cases where the Orion images contain multiple protein channels
    and allows for flexible channel selection and processing.
    """
    
    def __init__(self, 
                 pairs_dir: str,
                 orion_channels: List[int] = None,
                 channel_names: List[str] = None,
                 **kwargs):
        """
        Initialize multi-channel registration dataset.
        
        Args:
            pairs_dir: Directory containing training pairs
            orion_channels: List of channel indices to use (None for all)
            channel_names: Names of the channels for reference
            **kwargs: Additional arguments passed to RegistrationDataset
        """
        self.orion_channels = orion_channels
        self.channel_names = channel_names
        
        # Update orion_channels count
        if orion_channels is not None:
            kwargs['orion_channels'] = len(orion_channels)
        
        super().__init__(pairs_dir, **kwargs)
    
    def _preprocess_orion(self, orion_img: np.ndarray) -> np.ndarray:
        """Preprocess multi-channel Orion image."""
        # Select specific channels if specified
        if self.orion_channels is not None and orion_img.ndim == 3:
            orion_img = orion_img[:, :, self.orion_channels]
        
        return super()._preprocess_orion(orion_img)


def get_default_transforms(patch_size: int = 256, 
                          augment: bool = True,
                          normalize: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get default transforms for registration dataset.
    
    Args:
        patch_size: Size of image patches
        augment: Whether to apply data augmentation
        normalize: Whether to normalize images
        
    Returns:
        Tuple of (input_transform, target_transform)
    """
    input_transforms = []
    target_transforms = []
    
    # Convert to tensor
    input_transforms.append(transforms.ToTensor())
    target_transforms.append(transforms.ToTensor())
    
    if augment:
        # Data augmentation for input images
        input_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        ])
    
    if normalize:
        # Normalize to ImageNet stats for pre-trained models
        input_transforms.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    
    return transforms.Compose(input_transforms), transforms.Compose(target_transforms)


def create_data_loaders(pairs_dir: str,
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       num_workers: int = 4,
                       **dataset_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders for registration dataset.
    
    Args:
        pairs_dir: Directory containing training pairs
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = RegistrationDataset(pairs_dir, **dataset_kwargs)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader


def integrate_with_rosie(pairs_dir: str, 
                        rosie_config: Dict,
                        **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Integrate registration dataset with existing ROSIE training pipeline.
    
    Args:
        pairs_dir: Directory containing registered training pairs
        rosie_config: Configuration dictionary from ROSIE
        **kwargs: Additional arguments for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Extract relevant parameters from ROSIE config
    batch_size = rosie_config.get('BATCH_SIZE', 32)
    patch_size = rosie_config.get('PATCH_SIZE', 256)
    num_workers = rosie_config.get('NUM_WORKERS', 4)
    
    # Create transforms compatible with ROSIE
    input_transform, target_transform = get_default_transforms(
        patch_size=patch_size,
        augment=True,
        normalize=True
    )
    
    # Create data loaders
    return create_data_loaders(
        pairs_dir=pairs_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=input_transform,
        target_transform=target_transform,
        patch_size=patch_size,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'pairs_dir': './registration_output/training_pairs',
        'batch_size': 32,
        'patch_size': 256,
        'num_workers': 4
    }
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(**config)
        
        # Test loading a batch
        for batch_idx, (he_batch, orion_batch) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  H&E shape: {he_batch.shape}")
            print(f"  Orion shape: {orion_batch.shape}")
            print(f"  H&E range: [{he_batch.min():.3f}, {he_batch.max():.3f}]")
            print(f"  Orion range: [{orion_batch.min():.3f}, {orion_batch.max():.3f}]")
            
            if batch_idx >= 2:  # Just test first few batches
                break
        
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure you have run the registration pipeline first to generate training pairs.") 