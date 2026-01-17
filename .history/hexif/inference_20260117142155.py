import torch
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union, List
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from .model import SwinUNet
from .data import QuantileScaler
from .utils import slide_reconstruct

class HexifPredictor:
    def __init__(self, model_path: str, scaler_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        self.tf_eval = T.Compose([
            T.ToPILImage(),
            T.ToTensor(), # Resizing handled by slide logic if needed, but slide_reconstruct takes patch size
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        
    def _load_model(self, path: str) -> SwinUNet:
        logging.info(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        model = SwinUNet(out_ch=20, base_ch=192).to(self.device)
        
        # Handle state dict keys (DDP adds 'module.' prefix)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        model.eval()
        return model

    def _load_scaler(self, path: str) -> QuantileScaler:
        logging.info(f"Loading scaler from {path}")
        return QuantileScaler.load(Path(path))

    def predict_image(self, image: np.ndarray, patch_size: int = 224, stride: int = 160) -> np.ndarray:
        """
        Runs inference on a single image (H, W, 3).
        Returns predicted log-transformed Orion channels (20, H, W).
        """
        # Ensure image is float32 [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        logging.info(f"Running inference on image shape {image.shape}...")
        
        # We need a custom transform wrapper because slide_reconstruct expects a transform that takes numpy/PIL
        # and returns a tensor. Our tf_eval does that.
        
        # But slide_reconstruct takes tf_eval and applies it to patches.
        # slide_reconstruct expects the transform to resize? No, it crops patches of size `ps`.
        # So we just need ToTensor and Normalize.
        
        # We need to ensure the transform handles the resizing if patch_size differs from input patch?
        # slide_reconstruct: crops `ps` x `ps` from image. 
        # So the model expects `ps` x `ps`.
        
        tf = T.Compose([
            T.ToPILImage(),
            T.Resize(patch_size, antialias=True), # Just in case, though slide_reconstruct crops exactly ps
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        pred_log = slide_reconstruct(self.model, image, tf, ps=patch_size, stride=stride, device=self.device)
        
        # Inverse transform log1p -> linear
        # But wait, the scaler also has quantiles.
        # The model output is in log1p space relative to the scaled data.
        # y_log = log1p( (x - qlo)/(qhi - qlo) )
        # x_scaled = expm1(y_log)
        # x = x_scaled * (qhi - qlo) + qlo
        
        pred_linear = self._inverse_transform(pred_log)
        return pred_linear

    def _inverse_transform(self, pred_log: np.ndarray) -> np.ndarray:
        """
        Converts (C, H, W) log-space predictions back to linear counts/intensity.
        """
        C, H, W = pred_log.shape
        out = np.zeros_like(pred_log)
        
        for c in range(C):
            # 1. Inverse log1p
            val = np.expm1(pred_log[c])
            val = np.clip(val, 0, None)
            
            # 2. Inverse quantile scaling
            # x_norm = (x - qlo) / (range)
            # x = x_norm * range + qlo
            
            qlo = self.scaler.qlo[c]
            qhi = self.scaler.qhi[c]
            rng = qhi - qlo + 1e-6
            
            val = val * rng + qlo
            out[c] = val
            
        return out

    def save_diagnostics(self, he_image: np.ndarray, pred_linear: np.ndarray, output_path: str):
        """
        Creates a diagnostic plot of the predictions.
        """
        # Select some key markers to display (or all if few)
        # Assume 20 markers. Let's show a grid.
        
        # Normalize HE for display
        if he_image.dtype != np.uint8:
            he_disp = (np.clip(he_image, 0, 1) * 255).astype(np.uint8)
        else:
            he_disp = he_image
            
        C = pred_linear.shape[0]
        rows = int(np.ceil((C + 1) / 5)) # +1 for H&E
        fig, axes = plt.subplots(rows, 5, figsize=(20, 4*rows))
        axes = axes.flatten()
        
        # Plot H&E
        axes[0].imshow(he_disp)
        axes[0].set_title("Input H&E")
        axes[0].axis("off")
        
        # Plot channels
        for c in range(C):
            ax = axes[c+1]
            # Robust normalization for display
            img = pred_linear[c]
            p01, p99 = np.percentile(img, (1, 99))
            img_norm = np.clip((img - p01) / (p99 - p01 + 1e-6), 0, 1)
            
            ax.imshow(img_norm, cmap="magma")
            ax.set_title(f"Channel {c}")
            ax.axis("off")
            
        # Hide empty axes
        for i in range(C+1, len(axes)):
            axes[i].axis("off")
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
