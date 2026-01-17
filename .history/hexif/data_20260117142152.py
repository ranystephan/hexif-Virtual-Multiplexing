import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset
import torchvision.transforms as T
from .utils import is_main_process

def _np_to_float01(a: np.ndarray) -> np.ndarray:
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

def load_channel_statistics(stats_path: Optional[str], C: int, primary_threshold: float,
                             weight_power: float = 1.0, weight_clip: float = 5.0,
                             sampling_temperature: float = 1.0,
                             speckle_topk: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """Load aggregated statistics (from data_explorer summary) if available."""

    if not stats_path:
        return None, None, []

    path = Path(stats_path)
    if not path.exists():
        logging.warning(f"Channel stats file {stats_path} not found. Proceeding without it.")
        return None, None, []

    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logging.error(f"Failed to read channel stats at {stats_path}: {exc}. Ignoring file.")
        return None, None, []

    aggregated = data.get("aggregated_channel_stats", {})
    if not aggregated:
        return None, None, []

    # Determine which coverage key matches our primary threshold.
    first_entry = next(iter(aggregated.values()))
    coverage_key = f"pos_fraction_{primary_threshold:.3f}_mean"
    coverage_candidates = [k for k in first_entry.keys() if k.startswith("pos_fraction_") and k.endswith("_mean")]
    if coverage_key not in first_entry:
        if coverage_candidates:
            fallback_key = coverage_candidates[0]
            logging.warning(f"Coverage key {coverage_key} not found in stats; falling back to {fallback_key}.")
            coverage_key = fallback_key
        else:
            logging.warning("No coverage metrics found in channel stats; defaulting to uniform weights.")
            coverage_key = None

    coverage = np.ones(C, dtype=np.float32)
    mean_intensity = np.ones(C, dtype=np.float32)
    speckle = np.zeros(C, dtype=np.float32)
    for ch in range(C):
        ch_data = aggregated.get(str(ch)) or aggregated.get(int(ch))
        if not ch_data:
            continue
        if coverage_key is not None:
            coverage[ch] = float(ch_data.get(coverage_key, coverage[ch]))
        mean_intensity[ch] = float(ch_data.get("mean_mean", mean_intensity[ch]))
        speckle[ch] = float(ch_data.get("speckle_score_mean", speckle[ch]))

    eps = 1e-4
    # Loss weights emphasise channels with low coverage.
    channel_weights = ((coverage + eps) ** (-weight_power)).astype(np.float32)
    if weight_clip > 0:
        channel_weights = np.clip(channel_weights, 1.0, weight_clip)

    if np.allclose(channel_weights, channel_weights[0]):
        channel_weights = None

    # Sampling probabilities favour rare channels (inverse coverage) but also consider intensity.
    inv_cov = 1.0 / (coverage + eps)
    raw_scores = inv_cov * (mean_intensity + eps)
    raw_scores = raw_scores ** (1.0 / max(1e-3, sampling_temperature))
    if not np.isfinite(raw_scores).any():
        sampling_probs = None
    else:
        raw_scores = np.clip(raw_scores, 1e-8, None)
        sampling_probs = raw_scores / raw_scores.sum()

    # Identify top speckle channels for optional boosted sampling.
    if speckle_topk > 0:
        topk_idx = np.argsort(-speckle)[:speckle_topk]
        speckle_priority = [int(i) for i in topk_idx]
    else:
        speckle_priority = []

    return channel_weights, sampling_probs, speckle_priority

class QuantileScaler:
    def __init__(self, q_low=1.0, q_high=99.5, C=20):
        self.q_low = q_low
        self.q_high = q_high
        self.qlo = np.zeros(C, dtype=np.float32)
        self.qhi = np.ones(C, dtype=np.float32)
        self.C = C

    def to_dict(self) -> Dict:
        return {"q_low": self.q_low, "q_high": self.q_high,
                "qlo": self.qlo.tolist(), "qhi": self.qhi.tolist(), "C": self.C}

    @classmethod
    def from_dict(cls, d: Dict):
        obj = cls(d.get("q_low", 1.0), d.get("q_high", 99.5), d.get("C", 20))
        obj.qlo = np.array(d["qlo"], dtype=np.float32)
        obj.qhi = np.array(d["qhi"], dtype=np.float32)
        return obj

    def save(self, path: Path):
        if is_main_process():
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text()))

    def fit_from_train(self, pairs_dir: str, basenames: List[str], max_pixels: int = 50_000_000):
        logging.info(f"Fitting global quantiles on {len(basenames)} train cores...")
        samples = [[] for _ in range(self.C)]
        total_pixels = 0
        rng = np.random.RandomState(0)

        for b in basenames:
            p = Path(pairs_dir) / f"{b}_ORION.npy"
            arr = np.load(p, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[0] == self.C:
                arr = np.transpose(arr, (1, 2, 0))
            arr = _np_to_float01(arr)
            H, W, C = arr.shape
            assert C == self.C, f"Expected {self.C} channels, got {C}"
            n = H * W
            take = min(200_000, n)
            idx = rng.choice(n, size=take, replace=False)
            flat = arr.reshape(-1, C)
            sub = flat[idx]
            for c in range(C):
                samples[c].append(sub[:, c])
            total_pixels += take
            if total_pixels >= max_pixels:
                break

        for c in range(self.C):
            if len(samples[c]) == 0:
                self.qlo[c], self.qhi[c] = 0.0, 1.0
                continue
            v = np.concatenate(samples[c], axis=0)
            qlo = np.percentile(v, self.q_low)
            qhi = np.percentile(v, self.q_high)
            min_range = 1e-3  # Or 1e-2. Tune this.
            if (qhi - qlo) < min_range:
                qhi = qlo + min_range
            self.qlo[c] = float(qlo)
            self.qhi[c] = float(qhi)

        logging.info(f"Quantiles (ch0): q{self.q_low}={self.qlo[0]:.4g}, q{self.q_high}={self.qhi[0]:.4g}")


class HE2OrionDataset(Dataset):
    """
    Returns:
      he:  (3, ps, ps) float32 normalized
      tgt_log: (C, ps, ps) float32, global-scaled then log1p
      info: dict
    """
    def __init__(
        self,
        pairs_dir: str,
        basenames: List[str],
        scaler: QuantileScaler,
        patch_size: int = 224,
        mode: str = "train",
        grid_stride: int = 112,
        augment: bool = True,
        center_window: int = 12,
        pos_frac: float = 0.6,
        pos_threshold: float = 0.10,
        resample_tries: int = 8,
        samples_per_core: int = 64,
        channel_sampling_weights: Optional[np.ndarray] = None,
        min_pos_fraction: float = 0.01,
        channel_min_pos_fraction: float = 0.002,
        speckle_boost_channels: Optional[List[int]] = None,
    ):
        assert mode in ("train", "val")
        self.dir = Path(pairs_dir)
        self.basenames = basenames
        self.ps = patch_size
        self.mode = mode
        self.grid_stride = grid_stride
        self.augment = augment and (mode == "train")
        self.center_window = max(0, int(center_window))
        self.pos_frac = float(pos_frac)
        self.pos_threshold = float(pos_threshold)
        self.resample_tries = int(resample_tries)
        self.scaler = scaler
        self.C = scaler.C
        self.samples_per_core = samples_per_core
        self.min_pos_fraction = float(min_pos_fraction)
        self.channel_min_pos_fraction = float(channel_min_pos_fraction)

        if channel_sampling_weights is not None:
            csw = np.asarray(channel_sampling_weights, dtype=np.float64)
            if csw.shape[0] != self.C:
                raise ValueError("channel_sampling_weights must have length equal to number of channels")
            if csw.sum() <= 0:
                self.channel_sampling_weights = None
            else:
                csw = np.clip(csw, 1e-8, None)
                self.channel_sampling_weights = (csw / csw.sum()).astype(np.float32)
        else:
            self.channel_sampling_weights = None

        self.speckle_boost_channels = set(speckle_boost_channels or [])

        self.he_paths = [self.dir / f"{b}_HE.npy" for b in basenames]
        self.or_paths = [self.dir / f"{b}_ORION.npy" for b in basenames]
        for hp, op in zip(self.he_paths, self.or_paths):
            if not hp.exists() or not op.exists():
                raise FileNotFoundError(f"Missing pair: {hp} / {op}")

        # shapes
        self.shapes: List[Tuple[int, int]] = []
        for op in self.or_paths:
            arr = np.load(op, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[0] == self.C:
                H, W = arr.shape[1], arr.shape[2]
            elif arr.ndim == 3 and arr.shape[2] == self.C:
                H, W = arr.shape[0], arr.shape[1]
            else:
                raise RuntimeError(f"Unexpected Orion shape {arr.shape} for {op}")
            self.shapes.append((H, W))

        if self.mode == "val":
            ps, st = self.ps, self.grid_stride
            grid = []
            for i, (H, W) in enumerate(self.shapes):
                ys = [0] if H <= ps else list(range(0, max(1, H - ps) + 1, st))
                xs = [0] if W <= ps else list(range(0, max(1, W - ps) + 1, st))
                for y in ys:
                    for x in xs:
                        grid.append((i, y, x))
            self.grid = grid
            self._len = len(grid)
        else:
            self.grid = None
            self._len = len(self.basenames) * self.samples_per_core

        self.tf_train = T.Compose([
            T.ToPILImage(),
            T.Resize(self.ps, antialias=True),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.03),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        self.tf_eval = T.Compose([
            T.ToPILImage(),
            T.Resize(self.ps, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return self._len

    def _load_pair(self, idx_core: int):
        he = np.load(self.he_paths[idx_core], mmap_mode='r')
        orion = np.load(self.or_paths[idx_core], mmap_mode='r')
        if orion.ndim == 3 and orion.shape[0] == self.C:
            orion = np.transpose(orion, (1, 2, 0))
        he = _np_to_float01(he)
        orion = _np_to_float01(orion)
        return he, orion

    def _scale_to_log(self, or_patch: np.ndarray) -> np.ndarray:
        C = self.C
        out = np.zeros_like(or_patch, dtype=np.float32)
        for c in range(C):
            x = (or_patch[..., c] - self.scaler.qlo[c]) / (self.scaler.qhi[c] - self.scaler.qlo[c] + 1e-6)
            x = np.clip(x, 0, None)
            out[..., c] = np.log1p(x)
        return out

    @staticmethod
    def _rand_coords(H, W, ps):
        if H <= ps or W <= ps:
            return 0, 0
        y0 = np.random.randint(0, H - ps + 1)
        x0 = np.random.randint(0, W - ps + 1)
        return y0, x0

    def _is_positive_patch(self, or_scaled_log: np.ndarray, channel: Optional[int] = None) -> bool:
        C = or_scaled_log.shape[2]
        tot = or_scaled_log.shape[0] * or_scaled_log.shape[1]
        if channel is not None and (channel < 0 or channel >= C):
            channel = None

        channels = [channel] if channel is not None else range(C)
        thresh = self.pos_threshold
        frac_thresh = self.channel_min_pos_fraction if channel is not None else self.min_pos_fraction

        for c in channels:
            frac = float((or_scaled_log[..., c] > thresh).sum()) / max(1, tot)
            if frac >= frac_thresh:
                return True
        return False

    def __getitem__(self, idx: int):
        ps = self.ps
        if self.mode == "val":
            core_idx, y0, x0 = self.grid[idx]
            he, orion = self._load_pair(core_idx)
            he_img = (he[y0:y0+ps, x0:x0+ps, :]*255).astype(np.uint8)
            or_crop = orion[y0:y0+ps, x0:x0+ps, :].copy()
            or_log = self._scale_to_log(or_crop)
            he_t = self.tf_eval(he_img)
            info = {"y0": y0, "x0": x0, "core_idx": core_idx, "basename": self.basenames[core_idx], "target_channel": -1}
            return {"he": he_t, "tgt_log": torch.from_numpy(or_log.transpose(2,0,1)), "info": info}

        # train: stratified sampling
        core_idx = np.random.randint(0, len(self.basenames))
        he, orion = self._load_pair(core_idx)
        want_pos = (np.random.rand() < self.pos_frac)
        target_channel: Optional[int] = None
        tries = self.resample_tries if want_pos else 1
        if want_pos and self.channel_sampling_weights is not None:
            target_channel = int(np.random.choice(self.C, p=self.channel_sampling_weights))
            if target_channel in self.speckle_boost_channels:
                tries = max(tries, self.resample_tries * 2)

        for _ in range(tries):
            y0, x0 = self._rand_coords(*self.shapes[core_idx], ps)
            he_img = (he[y0:y0+ps, x0:x0+ps, :]*255).astype(np.uint8)
            or_crop = orion[y0:y0+ps, x0:x0+ps, :].copy()
            or_log = self._scale_to_log(or_crop)
            if (not want_pos) or self._is_positive_patch(or_log, channel=target_channel):
                break

        he_t = self.tf_train(he_img) if self.augment else self.tf_eval(he_img)
        tc_val = int(target_channel) if target_channel is not None else -1
        info = {"y0": y0, "x0": x0, "core_idx": core_idx, "basename": self.basenames[core_idx], "target_channel": tc_val}
        return {"he": he_t, "tgt_log": torch.from_numpy(or_log.transpose(2,0,1)), "info": info}
