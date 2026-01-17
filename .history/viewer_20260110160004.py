#!/usr/bin/env python3
"""
HEXIF Core Viewer — Interactive visualization for H&E → ORION predictions.

Launch with:
    streamlit run viewer.py -- --pairs_dir core_patches_npy --checkpoint runs/nov5/focal_l1_plateau/best_model.pth

Features:
    • Side-by-side H&E, Ground Truth, and Prediction views
    • Channel selection with marker names
    • Colormap customization
    • Intensity range adjustment
    • Metrics display (correlation, MSE, etc.)
    • Caching for fast navigation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import streamlit as st

# Optional imports for model inference
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# ─────────────────────────────────────────────────────────────────────────────
# Marker names (adjust to match your panel)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MARKER_NAMES = [
    "Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4",
    "Ch 5", "Ch 6", "Ch 7", "Ch 8", "Ch 9",
    "Ch 10", "Ch 11", "Ch 12", "Ch 13", "Ch 14",
    "Ch 15", "Ch 16", "Ch 17", "Ch 18", "Ch 19",
]

# You can customize this to match your actual marker panel
MARKER_NAMES_20CH = [
    "DAPI", "CD3", "CD31", "CD45", "CD34",
    "FoxP3", "CD56", "CD8", "CD11c", "PARP1",
    "CD206", "CD7", "HLA-DR", "Ki67", "Podoplanin",
    "CD4", "CA9", "Vimentin", "PD1", "CD11b",
]

# Colormaps that work well for fluorescence data
COLORMAPS = [
    "viridis", "magma", "plasma", "inferno", "cividis",
    "Greens", "Blues", "Reds", "Purples", "Oranges",
    "hot", "turbo", "coolwarm", "gray", "bone"
]

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _np_to_float01(a: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] float32."""
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype in (np.uint16, np.int16):
        a = a.astype(np.float32)
        if a.max(initial=0.0) > 1.5:
            a = a / (np.percentile(a, 99.9) + 1e-6)
    elif a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.max(initial=0.0) > 1.5:
        a = a / 255.0
    return a


def discover_basenames(pairs_dir: str) -> List[str]:
    """Find all core_*_HE.npy files with matching ORION."""
    d = Path(pairs_dir)
    out = []
    for hef in sorted(d.glob("core_*_HE.npy")):
        base = hef.stem.replace("_HE", "")
        if (d / f"{base}_ORION.npy").exists():
            out.append(base)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Quantile Scaler (from model_nov4.py)
# ─────────────────────────────────────────────────────────────────────────────

class QuantileScaler:
    """Global per-channel quantile normalization."""
    
    def __init__(self, q_low=1.0, q_high=99.5, C=20):
        self.q_low = q_low
        self.q_high = q_high
        self.qlo = np.zeros(C, dtype=np.float32)
        self.qhi = np.ones(C, dtype=np.float32)
        self.C = C

    @classmethod
    def from_dict(cls, d: Dict):
        obj = cls(d.get("q_low", 1.0), d.get("q_high", 99.5), d.get("C", 20))
        obj.qlo = np.array(d["qlo"], dtype=np.float32)
        obj.qhi = np.array(d["qhi"], dtype=np.float32)
        return obj

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text()))

    def scale_to_log(self, orion: np.ndarray) -> np.ndarray:
        """Apply quantile scaling + log1p transform."""
        C = self.C
        out = np.zeros_like(orion, dtype=np.float32)
        for c in range(C):
            x = (orion[..., c] - self.qlo[c]) / (self.qhi[c] - self.qlo[c] + 1e-6)
            x = np.clip(x, 0, None)
            out[..., c] = np.log1p(x)
        return out

    def inverse_log(self, log_data: np.ndarray) -> np.ndarray:
        """Convert from log1p domain back to [0, ~1] linear."""
        return np.expm1(log_data)


# ─────────────────────────────────────────────────────────────────────────────
# Model Definition (from model_nov4.py)
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH and HAS_TIMM:
    import torch.nn as nn

    class SwinUNet(nn.Module):
        def __init__(self, out_ch: int = 20, base_ch: int = 192, softplus_beta: float = 1.0):
            super().__init__()
            self.enc = timm.create_model(
                'swin_tiny_patch4_window7_224', pretrained=False, features_only=True, out_indices=(0,1,2,3)
            )
            enc_chs = self.enc.feature_info.channels()
            self.lats = nn.ModuleList([nn.Conv2d(c, base_ch, 1) for c in enc_chs])
            self.smooth3 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
            self.smooth2 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
            self.smooth1 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
            self.smooth0 = nn.Sequential(nn.Conv2d(base_ch, base_ch//2, 3, padding=1), nn.ReLU(inplace=True))
            self.out = nn.Conv2d(base_ch//2, out_ch, 1)
            self.softplus = nn.Softplus(beta=softplus_beta)

        def forward(self, x):
            feats = self.enc(x)
            feats = [f.permute(0, 3, 1, 2) for f in feats]
            f3 = self.lats[3](feats[3])
            f2 = self._upsum(f3, self.lats[2](feats[2]))
            f2 = self.smooth3(f2)
            f1 = self._upsum(f2, self.lats[1](feats[1]))
            f1 = self.smooth2(f1)
            f0 = self._upsum(f1, self.lats[0](feats[0]))
            f0 = self.smooth1(f0)
            up = F.interpolate(f0, size=x.shape[-2:], mode='bilinear', align_corners=False)
            up = self.smooth0(up)
            y = self.out(up)
            y = self.softplus(y)
            return y

        @staticmethod
        def _up(x, size_hw):
            return F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)

        def _upsum(self, x_small, x_skip):
            x_up = self._up(x_small, x_skip.shape[-2:])
            return x_up + x_skip


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained model from checkpoint."""
    if not HAS_TORCH or not HAS_TIMM:
        return None, None
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SwinUNet(out_ch=20, base_ch=192, softplus_beta=1.0)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    scaler = None
    if "scaler" in ckpt:
        scaler = QuantileScaler.from_dict(ckpt["scaler"])
    
    return model, scaler


def get_eval_transform(patch_size: int = 224):
    """Get the evaluation transform matching training."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(patch_size, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def slide_reconstruct(model, he_img: np.ndarray, tf_eval, ps: int = 224, 
                      stride: int = 112, device: str = "cpu") -> np.ndarray:
    """Sliding window inference over full core."""
    H, W, _ = he_img.shape
    out_accum = None
    weight = None
    
    for y in range(0, max(1, H - ps) + 1, stride):
        for x in range(0, max(1, W - ps) + 1, stride):
            he_crop = (he_img[y:y+ps, x:x+ps, :] * 255).astype(np.uint8)
            he_t = tf_eval(he_crop).unsqueeze(0).to(device)
            pred_log = model(he_t).detach().cpu().numpy()[0]  # (C, ps, ps)
            
            if out_accum is None:
                C = pred_log.shape[0]
                out_accum = np.zeros((C, H, W), dtype=np.float32)
                weight = np.zeros((1, H, W), dtype=np.float32)
            
            y2 = min(H, y + ps)
            x2 = min(W, x + ps)
            ph = y2 - y
            pw = x2 - x
            out_accum[:, y:y2, x:x2] += pred_log[:, :ph, :pw]
            weight[:, y:y2, x:x2] += 1.0
    
    out_log = out_accum / np.clip(weight, 1e-6, None)
    return out_log


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading with Caching
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_core_data(pairs_dir: str, basename: str, n_channels: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Load H&E and ORION data for a single core."""
    d = Path(pairs_dir)
    he_path = d / f"{basename}_HE.npy"
    or_path = d / f"{basename}_ORION.npy"
    
    he = np.load(he_path)
    orion = np.load(or_path)
    
    # Normalize to channel-last format
    if orion.ndim == 3 and orion.shape[0] == n_channels:
        orion = np.transpose(orion, (1, 2, 0))
    
    he = _np_to_float01(he)
    orion = _np_to_float01(orion)
    
    return he, orion


@st.cache_data
def get_prediction_for_core(_model, _scaler, _tf_eval, he: np.ndarray, 
                             ps: int, stride: int, device: str,
                             cache_key: str) -> np.ndarray:
    """Get model prediction for a core (cached)."""
    pred_log = slide_reconstruct(_model, he, _tf_eval, ps=ps, stride=stride, device=device)
    return pred_log


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute comparison metrics between GT and prediction."""
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    # MSE
    mse = np.mean((gt_flat - pred_flat) ** 2)
    
    # MAE
    mae = np.mean(np.abs(gt_flat - pred_flat))
    
    # Pearson correlation
    if gt_flat.std() > 1e-8 and pred_flat.std() > 1e-8:
        corr = np.corrcoef(gt_flat, pred_flat)[0, 1]
    else:
        corr = 0.0
    
    # SSIM-like structural comparison (simplified)
    gt_mean, pred_mean = gt_flat.mean(), pred_flat.mean()
    gt_std, pred_std = gt_flat.std(), pred_flat.std()
    cov = np.mean((gt_flat - gt_mean) * (pred_flat - pred_mean))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * gt_mean * pred_mean + c1) * (2 * cov + c2)) / \
           ((gt_mean**2 + pred_mean**2 + c1) * (gt_std**2 + pred_std**2 + c2))
    
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "Pearson r": float(corr),
        "SSIM": float(ssim),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="HEXIF Core Viewer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
        <style>
        .block-container { padding-top: 1rem; }
        .stMetric { background-color: #1e1e1e; border-radius: 8px; padding: 10px; }
        h1 { color: #00d4aa; }
        .sidebar .sidebar-content { background-color: #0e1117; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔬 HEXIF Core Viewer")
    st.markdown("*Interactive visualization for H&E → ORION predictions*")
    
    # ─── Sidebar Configuration ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data directory
        pairs_dir = st.text_input(
            "📁 Pairs Directory",
            value="core_patches_npy",
            help="Directory containing core_*_HE.npy and core_*_ORION.npy files"
        )
        
        # Model checkpoint (optional)
        checkpoint_path = st.text_input(
            "🧠 Model Checkpoint (optional)",
            value="",
            help="Path to trained model checkpoint for predictions"
        )
        
        # Scaler file (if no checkpoint)
        scaler_path = st.text_input(
            "📊 Scaler JSON (if no checkpoint)",
            value="",
            help="Path to orion_scaler.json if not loading from checkpoint"
        )
        
        st.divider()
        
        # Inference settings
        st.subheader("🔧 Inference Settings")
        patch_size = st.slider("Patch Size", 128, 512, 224, step=32)
        stride = st.slider("Stride", 32, 224, 112, step=16)
        device = st.selectbox("Device", ["cpu", "cuda", "mps"])
        
        st.divider()
        
        # Visualization settings
        st.subheader("🎨 Visualization")
        colormap = st.selectbox("Colormap", COLORMAPS, index=0)
        use_marker_names = st.checkbox("Use Marker Names", value=True)
        show_metrics = st.checkbox("Show Metrics", value=True)
        show_histogram = st.checkbox("Show Histogram", value=False)
    
    # ─── Discover Cores ──────────────────────────────────────────────────────
    if not Path(pairs_dir).exists():
        st.error(f"❌ Directory not found: `{pairs_dir}`")
        st.info("Please provide a valid path to your data directory.")
        return
    
    basenames = discover_basenames(pairs_dir)
    if not basenames:
        st.error(f"❌ No paired cores found in `{pairs_dir}`")
        st.info("Looking for files matching pattern: `core_*_HE.npy` + `core_*_ORION.npy`")
        return
    
    st.success(f"✅ Found **{len(basenames)}** paired cores")
    
    # ─── Load Model (if checkpoint provided) ─────────────────────────────────
    model, scaler = None, None
    has_predictions = False
    
    if checkpoint_path and Path(checkpoint_path).exists():
        with st.spinner("Loading model..."):
            model, scaler = load_model(checkpoint_path, device)
            if model is not None:
                has_predictions = True
                st.sidebar.success("✅ Model loaded!")
            else:
                st.sidebar.warning("⚠️ Could not load model (missing torch/timm?)")
    
    # Load scaler separately if needed
    if scaler is None and scaler_path and Path(scaler_path).exists():
        scaler = QuantileScaler.load(Path(scaler_path))
    
    # Default scaler if none available
    if scaler is None:
        scaler = QuantileScaler(q_low=1.0, q_high=99.5, C=20)
        st.sidebar.info("ℹ️ Using default scaler (no quantile normalization)")
    
    # ─── Core Selection ──────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_core = st.selectbox(
            "🔎 Select Core",
            basenames,
            format_func=lambda x: f"{x} ({basenames.index(x) + 1}/{len(basenames)})"
        )
    
    with col2:
        # Channel selection
        marker_names = MARKER_NAMES_20CH if use_marker_names else DEFAULT_MARKER_NAMES
        n_channels = len(marker_names)
        
        channel_options = [f"Ch {i}: {marker_names[i]}" for i in range(n_channels)]
        selected_channel_str = st.selectbox("📍 Select Channel", channel_options)
        selected_channel = int(selected_channel_str.split(":")[0].replace("Ch ", ""))
    
    # ─── Load Data ───────────────────────────────────────────────────────────
    with st.spinner(f"Loading {selected_core}..."):
        he, orion = load_core_data(pairs_dir, selected_core, n_channels)
    
    # Apply scaler to get log-space GT
    gt_log = scaler.scale_to_log(orion)
    
    # Get prediction if model available
    pred_log = None
    if has_predictions and model is not None:
        tf_eval = get_eval_transform(patch_size)
        cache_key = f"{selected_core}_{patch_size}_{stride}"
        
        with st.spinner("Running inference..."):
            pred_log = slide_reconstruct(
                model, he, tf_eval, 
                ps=patch_size, stride=stride, device=device
            )
    
    # ─── Intensity Range Controls ────────────────────────────────────────────
    st.divider()
    
    # Compute range from data
    gt_ch = gt_log[..., selected_channel]
    gt_min, gt_max = float(gt_ch.min()), float(gt_ch.max())
    
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        vmin = st.slider("Intensity Min", 0.0, max(gt_max, 1.0), 0.0, step=0.01)
    with col_range2:
        vmax = st.slider("Intensity Max", 0.0, max(gt_max, 2.0), min(gt_max, 1.5), step=0.01)
    
    # ─── Display Images ──────────────────────────────────────────────────────
    st.divider()
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    # Determine number of columns based on whether we have predictions
    if has_predictions and pred_log is not None:
        cols = st.columns(3)
        titles = ["H&E", f"Ground Truth: {marker_names[selected_channel]}", f"Prediction: {marker_names[selected_channel]}"]
    else:
        cols = st.columns(2)
        titles = ["H&E", f"Ground Truth: {marker_names[selected_channel]}"]
    
    # H&E
    with cols[0]:
        st.markdown(f"### {titles[0]}")
        fig_he, ax_he = plt.subplots(figsize=(8, 8))
        ax_he.imshow(he)
        ax_he.axis('off')
        ax_he.set_title(f"H&E — {selected_core}", fontsize=10)
        st.pyplot(fig_he, use_container_width=True)
        plt.close(fig_he)
    
    # Ground Truth
    with cols[1]:
        st.markdown(f"### {titles[1]}")
        fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
        gt_display = scaler.inverse_log(gt_ch)  # Convert back to linear for display
        im_gt = ax_gt.imshow(gt_display, cmap=colormap, vmin=vmin, vmax=vmax)
        ax_gt.axis('off')
        ax_gt.set_title(f"GT — {marker_names[selected_channel]}", fontsize=10)
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        st.pyplot(fig_gt, use_container_width=True)
        plt.close(fig_gt)
    
    # Prediction
    if has_predictions and pred_log is not None:
        with cols[2]:
            st.markdown(f"### {titles[2]}")
            pred_ch = pred_log[selected_channel]  # (H, W)
            fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
            pred_display = scaler.inverse_log(pred_ch)
            im_pred = ax_pred.imshow(pred_display, cmap=colormap, vmin=vmin, vmax=vmax)
            ax_pred.axis('off')
            ax_pred.set_title(f"Pred — {marker_names[selected_channel]}", fontsize=10)
            plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            st.pyplot(fig_pred, use_container_width=True)
            plt.close(fig_pred)
    
    # ─── Metrics ─────────────────────────────────────────────────────────────
    if show_metrics and has_predictions and pred_log is not None:
        st.divider()
        st.subheader("📊 Metrics")
        
        gt_linear = scaler.inverse_log(gt_ch)
        pred_linear = scaler.inverse_log(pred_log[selected_channel])
        metrics = compute_metrics(gt_linear, pred_linear)
        
        metric_cols = st.columns(4)
        for i, (name, value) in enumerate(metrics.items()):
            with metric_cols[i]:
                st.metric(name, f"{value:.4f}")
    
    # ─── Histogram ───────────────────────────────────────────────────────────
    if show_histogram:
        st.divider()
        st.subheader("📈 Intensity Distribution")
        
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        
        gt_linear = scaler.inverse_log(gt_ch).flatten()
        ax_hist.hist(gt_linear, bins=100, alpha=0.7, label="Ground Truth", color="#00d4aa")
        
        if has_predictions and pred_log is not None:
            pred_linear = scaler.inverse_log(pred_log[selected_channel]).flatten()
            ax_hist.hist(pred_linear, bins=100, alpha=0.7, label="Prediction", color="#ff6b6b")
        
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title(f"Intensity Distribution — {marker_names[selected_channel]}")
        ax_hist.legend()
        ax_hist.set_xlim(0, max(vmax, 1.0))
        
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)
    
    # ─── Multi-Channel Overview ──────────────────────────────────────────────
    st.divider()
    with st.expander("🔬 Multi-Channel Overview (all channels)", expanded=False):
        n_cols = 5
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig_multi, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for ch in range(n_channels):
            ax = axes[ch]
            ch_data = scaler.inverse_log(gt_log[..., ch])
            ax.imshow(ch_data, cmap=colormap, vmin=0, vmax=0.5)
            ax.set_title(f"{marker_names[ch]}", fontsize=8)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        fig_multi.suptitle(f"All Channels — {selected_core}", fontsize=12)
        fig_multi.tight_layout()
        st.pyplot(fig_multi, use_container_width=True)
        plt.close(fig_multi)
    
    # ─── Navigation ──────────────────────────────────────────────────────────
    st.divider()
    nav_cols = st.columns([1, 2, 1])
    
    current_idx = basenames.index(selected_core)
    
    with nav_cols[0]:
        if st.button("⬅️ Previous Core", disabled=(current_idx == 0)):
            st.session_state.selected_core_idx = current_idx - 1
            st.rerun()
    
    with nav_cols[1]:
        st.markdown(f"<center>Core **{current_idx + 1}** of **{len(basenames)}**</center>", unsafe_allow_html=True)
    
    with nav_cols[2]:
        if st.button("Next Core ➡️", disabled=(current_idx == len(basenames) - 1)):
            st.session_state.selected_core_idx = current_idx + 1
            st.rerun()


if __name__ == "__main__":
    main()
