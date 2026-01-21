import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

HAS_TIMM = True

class SwinUNet(nn.Module):
    def __init__(self, out_ch: int = 20, base_ch: int = 192, softplus_beta: float = 1.0, presence_head: bool = False):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required. pip install timm")
            
        self.enc = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, features_only=True, out_indices=(0,1,2,3)
        )
        enc_chs = self.enc.feature_info.channels()
        self.lats = nn.ModuleList([nn.Conv2d(c, base_ch, 1) for c in enc_chs])
        self.smooth3 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.smooth1 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.smooth0 = nn.Sequential(nn.Conv2d(base_ch, base_ch//2, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(base_ch//2, out_ch, 1)
        self.softplus = nn.Softplus(beta=softplus_beta)
        self.presence_head = None
        self.has_presence_head = bool(presence_head)
        if presence_head:
            self.presence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base_ch//2, out_ch),
            )
        
    def forward(self, x, return_presence: bool = False):
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
        y = self.softplus(y)  # log1p domain is >=0
        if return_presence and self.presence_head is not None:
            presence_logits = self.presence_head(up)
            return y, presence_logits
        return y
        
    @staticmethod
    def _up(x, size_hw):
        return F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)
        
    def _upsum(self, x_small, x_skip):
        x_up = self._up(x_small, x_skip.shape[-2:])
        return x_up + x_skip
