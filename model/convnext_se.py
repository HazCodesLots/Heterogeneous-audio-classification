"""
ConvNeXt-SplitBand-GAPGMP — adapted for DCASE 2026 Task 1
(Heterogeneous Audio Classification with the Broad Sound Taxonomy)

Changes from ConvNeXt-UST champion:
  - Single-label classification (CrossEntropyLoss) instead of multi-label BCE
  - 23 BST second-level classes with 5-top-level hierarchy
  - Hierarchical F-score (hF, λ=0.75) as primary metric
  - Audio: 32 kHz sample rate, up to 30 s clips
  - Mel: fmax=16000 to capture full BST sound range
  - BSD10k/BSD35k CSV format dataset loader
  - Optional confidence-score weighting during training
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x.div(keep_prob) * rand


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, max(dim // reduction, 4))
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(max(dim // reduction, 4), dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3))
        y = self.fc2(self.act(self.fc1(y)))
        return x * self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma   = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.se = SEBlock(dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = self.se(x)
        return shortcut + self.drop_path(x)


class _Stem(nn.Module):
    """Patchify stem: Conv2d + channel-last LayerNorm, self-contained."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                                       # (B, C, H, W)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # channel-last norm
        return x


class ConvNeXt2D(nn.Module):
    def __init__(self, input_channels=1, depths=(2, 2, 6, 2),
                 dims=(64, 128, 256, 512), drop_path_rate=0.2,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.stem = _Stem(input_channels, dims[0])
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i, d in enumerate(depths):
            blocks = nn.Sequential(*[
                ConvNeXtBlock(dims[i], dp_rates[cur + j], layer_scale_init_value)
                for j in range(d)
            ])
            self.stages.append(blocks)
            cur += d
            if i < len(depths) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.LayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                ))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                ds = self.downsamples[i]
                x = ds[1](ds[0](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class SplitBandPooling(nn.Module):
    """
    Split feature map at mid-frequency, apply GAP+GMP to each half,
    concatenate → project.  [gap_low | gmp_low | gap_high | gmp_high] → (B, 4C) → proj → (B, output_dim)
    """
    def __init__(self, input_dim=512, output_dim=512, dropout=0.1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        mid = x.shape[2] // 2
        low, high = x[:, :, :mid, :], x[:, :, mid:, :]
        B = x.size(0)
        fused = torch.cat([
            self.gap(low).view(B, -1),
            self.gmp(low).view(B, -1),
            self.gap(high).view(B, -1),
            self.gmp(high).view(B, -1),
        ], dim=1)
        return self.proj(fused)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=23, dropout_rate=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class ConvNeXtBST(nn.Module):
    """
    ConvNeXt-SplitBand-GAPGMP for DCASE 2026 Task 1.
    Outputs raw logits over 23 BST second-level classes (single-label).
    """
    def __init__(self,
                 convnext_params: dict,
                 pooling_params: dict,
                 mlp_params: dict):
        super().__init__()
        self.backbone = ConvNeXt2D(**convnext_params)
        self.pool     = SplitBandPooling(**pooling_params)
        self.head     = MLPClassifier(**mlp_params)

    def forward(self, x):
        return self.head(self.pool(self.backbone(x)))


# ---------------------------------------------------------------------------
# Default config factory
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 23, drop_path_rate: float = 0.2) -> ConvNeXtBST:
    return ConvNeXtBST(
        convnext_params=dict(
            input_channels=1,
            depths=(2, 2, 6, 2),
            dims=(64, 128, 256, 512),
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=1e-6,
        ),
        pooling_params=dict(input_dim=512, output_dim=512, dropout=0.1),
        mlp_params=dict(input_dim=512, num_classes=num_classes, dropout_rate=0.3),
    )
