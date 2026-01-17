import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except Exception:
    HAS_MSSSIM = False

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class OrionLoss(nn.Module):
    def __init__(self, center_window: int = 12,
                 loss_type: str = 'l2',
                 w_center: float = 1.0,
                 pos_boost: float = 3.0,
                 pos_tau: float = 0.10,
                 w_cov: float = 0.2,
                 w_msssim: float = 0.15,
                 w_tv: float = 1e-4,
                 w_presence: float = 0.0,
                 presence_temperature: float = 0.15,
                 use_focal_presence: bool = False,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25):
        super().__init__()
        assert loss_type in ('l1', 'l2')
        self.cw = int(center_window)
        self.loss_type = loss_type
        self.w_center = w_center
        self.pos_boost = pos_boost
        self.pos_tau = pos_tau
        self.w_cov = w_cov
        self.w_msssim = w_msssim if HAS_MSSSIM else 0.0
        self.w_tv = w_tv
        self.w_presence = w_presence
        self.presence_temperature = max(1e-3, float(presence_temperature))
        self.use_focal_presence = use_focal_presence
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.channel_weight_map: Optional[torch.Tensor] = None
        self.channel_weight_vec: Optional[torch.Tensor] = None

    def set_channel_weights(self, weights: Optional[torch.Tensor]):
        if weights is None:
            self.channel_weight_map = None
            self.channel_weight_vec = None
            return
        if weights.ndim != 1:
            raise ValueError("channel weights must be 1D")
        self.channel_weight_map = weights.view(1, -1, 1, 1)
        self.channel_weight_vec = weights.view(1, -1)

    def forward(self, pred_log, tgt_log):
        B, C, H, W = pred_log.shape
        if self.cw > 0:
            y1 = H//2 - self.cw//2
            x1 = W//2 - self.cw//2
            y2 = y1 + self.cw
            x2 = x1 + self.cw
            pc = pred_log[:,:,y1:y2, x1:x2]
            tc = tgt_log [:,:,y1:y2, x1:x2]
        else:
            pc, tc = pred_log, tgt_log

        w = 1.0 + self.pos_boost * (tc > self.pos_tau).float()
        if self.channel_weight_map is not None:
            w = w * self.channel_weight_map

        if self.loss_type == 'l1':
            center_loss = (w * (pc - tc).abs()).mean()
        else: # l2
            center_loss = (w * (pc - tc).pow(2)).mean()
        loss = self.w_center * center_loss


        if self.w_msssim > 0:
            pred_blur = F.avg_pool2d(pred_log, 3, 1, 1)
            tgt_blur  = F.avg_pool2d(tgt_log,  3, 1, 1)
            def _norm01(x):
                x = x - x.amin(dim=(2,3), keepdim=True)
                x = x / (x.amax(dim=(2,3), keepdim=True) + 1e-6)
                return x
            p01 = _norm01(pred_blur)
            t01 = _norm01(tgt_blur)
            ssim = 1.0 - ms_ssim(p01, t01, data_range=1.0, size_average=True)
            loss = loss + self.w_msssim * ssim

        if self.w_tv > 0:
            tv = (pred_log[:,:,:,1:] - pred_log[:,:,:,:-1]).abs().mean() + \
                 (pred_log[:,:,1:,:] - pred_log[:,:,:-1,:]).abs().mean()
            loss = loss + self.w_tv * tv

        if self.w_cov > 0:
            pred_mean = pred_log.mean(dim=(2, 3))
            tgt_mean = tgt_log.mean(dim=(2, 3))
            cov_err = (pred_mean - tgt_mean).abs()
            if self.channel_weight_vec is not None:
                cov_err = cov_err * self.channel_weight_vec
            loss = loss + self.w_cov * cov_err.mean()

        if self.w_presence > 0:
            pred_max = pred_log.amax(dim=(2, 3))
            tgt_max = tgt_log.amax(dim=(2, 3))
            tgt_presence = (tgt_max > self.pos_tau).float()
            logits = (pred_max - self.pos_tau) / self.presence_temperature
            if self.use_focal_presence:
                presence_loss = sigmoid_focal_loss(logits, tgt_presence,
                                                   alpha=self.focal_alpha,
                                                   gamma=self.focal_gamma,
                                                   reduction='mean')
            else:
                presence_loss = F.binary_cross_entropy_with_logits(logits, tgt_presence, reduction='none')

            if self.channel_weight_vec is not None:
                presence_loss = presence_loss * self.channel_weight_vec
            loss = loss + self.w_presence * presence_loss.mean()
        return loss
