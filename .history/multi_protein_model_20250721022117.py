"""
Multi-Channel Protein Prediction Model Architecture

This module implements state-of-the-art deep learning architectures for predicting
19 protein expressions from H&E images simultaneously. 

Key innovations:
- Multi-scale feature extraction with attention mechanisms
- Protein-specific prediction heads with biological constraints
- Advanced loss functions for multi-target learning
- Squeeze-and-excitation blocks for channel-wise recalibration
- Progressive supervision for better gradient flow

Architecture is inspired by ROSIE but optimized for multi-protein prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
import math
from collections import OrderedDict


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Channel attention
        self.channel_attention = SEBlock(channels, reduction)
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        x = self.channel_attention(x)
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)
        
        return x * spatial_weight


class ResidualBlock(nn.Module):
    """Enhanced residual block with attention and normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 use_attention: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()
        
        # Attention module
        if use_attention:
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        out = self.attention(out)
        
        return out


class MultiScaleEncoder(nn.Module):
    """Multi-scale encoder with parallel paths for different receptive fields."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Different kernel sizes for multi-scale features
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 5, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 7, padding=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolution for large receptive field
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch_3x3 = self.branch_3x3(x)
        branch_5x5 = self.branch_5x5(x)
        branch_7x7 = self.branch_7x7(x)
        branch_dilated = self.branch_dilated(x)
        
        # Concatenate all branches
        features = torch.cat([branch_3x3, branch_5x5, branch_7x7, branch_dilated], dim=1)
        
        return self.fusion(features)


class ProteinSpecificHead(nn.Module):
    """Protein-specific prediction head with biological constraints."""
    
    def __init__(self, in_channels: int, protein_name: str = None):
        super().__init__()
        self.protein_name = protein_name
        
        # Protein-specific feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()  # Protein expression in [0, 1]
        )
        
        # Learnable protein-specific scaling
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        features = self.feature_extractor(x)
        prediction = self.predictor(features)
        
        # Apply learnable scaling and bias
        prediction = prediction * self.scale + self.bias
        prediction = torch.clamp(prediction, 0, 1)  # Ensure valid range
        
        return prediction


class MultiProteinUNet(nn.Module):
    """
    Advanced U-Net for simultaneous prediction of 19 protein expressions.
    
    Features:
    - Multi-scale encoding with attention mechanisms
    - Protein-specific prediction heads
    - Progressive supervision
    - Skip connections with feature recalibration
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 num_proteins: int = 19,
                 base_features: int = 64,
                 protein_names: List[str] = None,
                 use_deep_supervision: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.num_proteins = num_proteins
        self.use_deep_supervision = use_deep_supervision
        
        if protein_names is None:
            protein_names = [f"Protein_{i+1}" for i in range(num_proteins)]
        self.protein_names = protein_names
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 7, padding=3),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        encoder_channels = [base_features, base_features*2, base_features*4, base_features*8, base_features*16]
        
        for i in range(4):
            # Multi-scale encoding
            self.encoder_blocks.append(
                nn.Sequential(
                    MultiScaleEncoder(encoder_channels[i], encoder_channels[i+1]),
                    ResidualBlock(encoder_channels[i+1], encoder_channels[i+1], 
                                use_attention=True, dropout_rate=dropout_rate)
                )
            )
            self.encoder_pools.append(nn.MaxPool2d(2))
        
        # Bottleneck with enhanced features
        self.bottleneck = nn.Sequential(
            MultiScaleEncoder(encoder_channels[4], encoder_channels[4]),
            ResidualBlock(encoder_channels[4], encoder_channels[4], use_attention=True),
            ResidualBlock(encoder_channels[4], encoder_channels[4], use_attention=True)
        )
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for i in range(4):
            # Upsample
            self.decoder_upsamples.append(
                nn.ConvTranspose2d(encoder_channels[4-i], encoder_channels[4-i], 2, stride=2)
            )
            
            # Decoder block with skip connection
            self.decoder_blocks.append(
                nn.Sequential(
                    ResidualBlock(encoder_channels[4-i] + encoder_channels[3-i], encoder_channels[3-i], 
                                use_attention=True, dropout_rate=dropout_rate),
                    ResidualBlock(encoder_channels[3-i], encoder_channels[3-i], use_attention=True)
                )
            )
        
        # Protein-specific prediction heads
        self.protein_heads = nn.ModuleDict()
        for i, protein_name in enumerate(self.protein_names):
            self.protein_heads[f"protein_{i}"] = ProteinSpecificHead(base_features, protein_name)
        
        # Deep supervision heads (for intermediate supervision)
        if self.use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleDict()
            supervision_channels = [base_features*2, base_features*4, base_features*8]
            for i, channels in enumerate(supervision_channels):
                self.deep_supervision_heads[f"level_{i}"] = nn.Sequential(
                    nn.Conv2d(channels, base_features, 3, padding=1),
                    nn.BatchNorm2d(base_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_features, num_proteins, 1),
                    nn.Sigmoid()
                )
        
        # Global feature aggregation for cross-protein interactions
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_features, num_proteins, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input processing
        x = self.input_conv(x)
        
        # Encoder path
        encoder_features = []
        for i, (encoder_block, pool) in enumerate(zip(self.encoder_blocks, self.encoder_pools)):
            x = encoder_block(x)
            encoder_features.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        deep_supervision_outputs = []
        for i, (upsample, decoder_block) in enumerate(zip(self.decoder_upsamples, self.decoder_blocks)):
            x = upsample(x)
            
            # Skip connection with feature recalibration
            skip_features = encoder_features[-(i+1)]
            
            # Ensure spatial dimensions match
            if x.shape[2:] != skip_features.shape[2:]:
                x = F.interpolate(x, size=skip_features.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip_features], dim=1)
            x = decoder_block(x)
            
            # Deep supervision
            if self.use_deep_supervision and i < 3:
                supervision_output = self.deep_supervision_heads[f"level_{i}"](x)
                if supervision_output.shape[2:] != (256, 256):  # Target size
                    supervision_output = F.interpolate(supervision_output, size=(256, 256), 
                                                     mode='bilinear', align_corners=False)
                deep_supervision_outputs.append(supervision_output)
        
        # Protein-specific predictions
        protein_predictions = []
        for i in range(self.num_proteins):
            pred = self.protein_heads[f"protein_{i}"](x)
            protein_predictions.append(pred)
        
        # Stack predictions
        main_output = torch.cat(protein_predictions, dim=1)
        
        # Global context for cross-protein modeling
        global_context = self.global_context(x)
        global_context = global_context.expand(-1, -1, main_output.shape[2], main_output.shape[3])
        
        # Combine with global context
        main_output = main_output * (1 + 0.1 * global_context)
        main_output = torch.clamp(main_output, 0, 1)
        
        if self.use_deep_supervision and self.training:
            return main_output, deep_supervision_outputs
        else:
            return main_output
    
    def predict_single_protein(self, x: torch.Tensor, protein_idx: int) -> torch.Tensor:
        """Predict a single protein (useful for inference)."""
        full_prediction = self.forward(x)
        if isinstance(full_prediction, tuple):
            full_prediction = full_prediction[0]
        
        return full_prediction[:, protein_idx:protein_idx+1, :, :]
    
    def get_protein_names(self) -> List[str]:
        """Get list of protein names."""
        return self.protein_names


class MultiProteinLoss(nn.Module):
    """
    Advanced loss function for multi-protein prediction.
    
    Combines:
    - Per-protein MSE loss
    - Structural similarity (SSIM) loss
    - Protein correlation consistency loss
    - Deep supervision loss
    """
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 ssim_weight: float = 0.3,
                 correlation_weight: float = 0.1,
                 deep_supervision_weight: float = 0.2,
                 protein_weights: Optional[List[float]] = None):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.correlation_weight = correlation_weight
        self.deep_supervision_weight = deep_supervision_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Per-protein weights for handling class imbalance
        self.protein_weights = protein_weights
    
    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Compute SSIM loss for structural similarity."""
        # Simple SSIM implementation for batch processing
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
        return 1 - ssim_map.mean()
    
    def _correlation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Encourage consistent protein correlations."""
        batch_size, num_proteins = pred.shape[:2]
        
        # Flatten spatial dimensions
        pred_flat = pred.view(batch_size, num_proteins, -1)
        target_flat = target.view(batch_size, num_proteins, -1)
        
        # Compute correlation matrices
        pred_corr = torch.corrcoef(pred_flat.view(-1, num_proteins).T)
        target_corr = torch.corrcoef(target_flat.view(-1, num_proteins).T)
        
        # L2 loss between correlation matrices
        return F.mse_loss(pred_corr, target_corr)
    
    def forward(self, pred_output, target, deep_supervision_targets=None):
        """
        Compute total loss.
        
        Args:
            pred_output: Model output (main prediction or tuple with deep supervision)
            target: Ground truth target
            deep_supervision_targets: Targets for deep supervision (optional)
        """
        # Handle deep supervision output
        if isinstance(pred_output, tuple):
            main_pred, deep_supervision_preds = pred_output
        else:
            main_pred = pred_output
            deep_supervision_preds = None
        
        losses = {}
        
        # Main prediction losses
        losses['mse'] = self.mse_loss(main_pred, target)
        losses['mae'] = self.mae_loss(main_pred, target)
        
        # SSIM loss (channel-wise)
        ssim_loss = 0
        for i in range(main_pred.shape[1]):
            ssim_loss += self._ssim_loss(main_pred[:, i:i+1], target[:, i:i+1])
        losses['ssim'] = ssim_loss / main_pred.shape[1]
        
        # Protein correlation consistency
        if self.correlation_weight > 0:
            losses['correlation'] = self._correlation_loss(main_pred, target)
        
        # Deep supervision loss
        if deep_supervision_preds is not None and self.deep_supervision_weight > 0:
            deep_loss = 0
            for deep_pred in deep_supervision_preds:
                deep_loss += self.mse_loss(deep_pred, target)
            losses['deep_supervision'] = deep_loss / len(deep_supervision_preds)
        
        # Weighted protein losses
        if self.protein_weights is not None:
            protein_losses = []
            for i, weight in enumerate(self.protein_weights):
                if i < main_pred.shape[1]:
                    protein_loss = self.mse_loss(main_pred[:, i:i+1], target[:, i:i+1])
                    protein_losses.append(weight * protein_loss)
            losses['weighted_protein'] = sum(protein_losses) / len(protein_losses)
        
        # Total loss
        total_loss = (
            self.mse_weight * losses['mse'] +
            self.ssim_weight * losses['ssim']
        )
        
        if 'correlation' in losses:
            total_loss += self.correlation_weight * losses['correlation']
        
        if 'deep_supervision' in losses:
            total_loss += self.deep_supervision_weight * losses['deep_supervision']
        
        if 'weighted_protein' in losses:
            total_loss += losses['weighted_protein']
        
        losses['total'] = total_loss
        
        return total_loss, losses


def create_multi_protein_model(num_proteins: int = 19,
                             base_features: int = 64,
                             protein_names: List[str] = None,
                             use_deep_supervision: bool = True) -> MultiProteinUNet:
    """
    Create a multi-protein prediction model.
    
    Args:
        num_proteins: Number of proteins to predict (default: 19)
        base_features: Base number of features (controls model size)
        protein_names: List of protein names (optional)
        use_deep_supervision: Whether to use deep supervision
    
    Returns:
        Configured MultiProteinUNet model
    """
    model = MultiProteinUNet(
        in_channels=3,
        num_proteins=num_proteins,
        base_features=base_features,
        protein_names=protein_names,
        use_deep_supervision=use_deep_supervision,
        dropout_rate=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Multi-Protein Model Created:")
    print(f"  - Proteins: {num_proteins}")
    print(f"  - Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  - Base features: {base_features}")
    print(f"  - Deep supervision: {use_deep_supervision}")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_multi_protein_model(num_proteins=19, base_features=64)
    
    # Test input
    test_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(test_input)
        if isinstance(output, tuple):
            main_output, deep_outputs = output
            print(f"Main output shape: {main_output.shape}")
            print(f"Deep supervision outputs: {len(deep_outputs)}")
        else:
            print(f"Output shape: {output.shape}")
    
    print("✓ Model test completed successfully") 