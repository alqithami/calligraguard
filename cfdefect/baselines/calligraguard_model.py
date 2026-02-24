from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: This is a lightweight, fully-self-contained "CalligraGuard-Lite" inspector.
# It is designed to be runnable in environments where torchvision may be unavailable
# or mismatched. It supports:
#   - universal mode (image-only)
#   - referenced mode (image + reference image + diffs)
#   - optional SVG-V conditioning via FiLM from precomputed SVG renders.
#
# Outputs:
#   - mask logits (B,1,H,W)
#   - type logits (B,C)
#   - path logits (B,P) multi-label

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNetBackbone(nn.Module):
    """UNet that also returns a bottleneck feature for classification/attribution heads."""
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8)

        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.c1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))  # bottleneck

        y3 = self.u3(x4)
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.c3(y3)

        y2 = self.u2(y3)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.c2(y2)

        y1 = self.u1(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.c1(y1)

        mask_logits = self.out(y1)
        return mask_logits, x4

class SvgvEncoder(nn.Module):
    """Encodes a small grayscale SVG render (e.g., 64x64) into a conditioning vector."""
    def __init__(self, in_ch: int = 1, cdim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, cdim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc(h)

class FiLM(nn.Module):
    """Feature-wise linear modulation: y = (1+gamma)*x + beta."""
    def __init__(self, cdim: int, feat_ch: int):
        super().__init__()
        self.to_gamma = nn.Linear(cdim, feat_ch)
        self.to_beta = nn.Linear(cdim, feat_ch)

    def forward(self, feat: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(c).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(c).unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * feat + beta

@dataclass
class CalligraGuardOutput:
    mask_logits: torch.Tensor
    type_logits: torch.Tensor
    path_logits: torch.Tensor
    score: torch.Tensor

class CalligraGuardLite(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_types: int,
                 max_path_id: int,
                 base: int = 32,
                 use_svgv: bool = True,
                 svgv_cdim: int = 128):
        super().__init__()
        self.num_types = int(num_types)
        self.max_path_id = int(max_path_id)
        self.use_svgv = bool(use_svgv)

        self.backbone = UNetBackbone(in_ch=in_ch, base=base)

        bottleneck_ch = base * 8
        self.svgv_enc = SvgvEncoder(in_ch=1, cdim=svgv_cdim) if self.use_svgv else None
        self.film = FiLM(cdim=svgv_cdim, feat_ch=bottleneck_ch) if self.use_svgv else None

        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_types),
        )
        self.path_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.max_path_id),
        )

    def forward(self,
                x: torch.Tensor,
                svgv: Optional[torch.Tensor] = None) -> CalligraGuardOutput:
        mask_logits, feat = self.backbone(x)

        if self.use_svgv:
            if svgv is None:
                raise ValueError("svgv conditioning enabled but svgv is None")
            c = self.svgv_enc(svgv)
            feat = self.film(feat, c)

        type_logits = self.type_head(feat)
        path_logits = self.path_head(feat)

        # detection score: max sigmoid(mask) (shape B,)
        score = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values
        return CalligraGuardOutput(mask_logits=mask_logits,
                                  type_logits=type_logits,
                                  path_logits=path_logits,
                                  score=score)

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary Dice loss (expects logits and {0,1} target)."""
    prob = torch.sigmoid(logits)
    prob = prob.flatten(1)
    tgt = target.flatten(1)
    inter = (prob * tgt).sum(dim=1)
    den = prob.sum(dim=1) + tgt.sum(dim=1)
    dice = (2*inter + eps) / (den + eps)
    return 1.0 - dice.mean()
