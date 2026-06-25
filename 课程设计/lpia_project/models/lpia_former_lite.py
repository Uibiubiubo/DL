from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RetinexGammaPrior(nn.Module):
    def __init__(self, gamma_min: float = 0.35, gamma_max: float = 1.15):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        gamma_raw, illumination_raw = raw[:, 0:1], raw[:, 1:2]
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(gamma_raw)
        learned_illumination = torch.sigmoid(illumination_raw)
        luminance = rgb_to_luminance(x)
        illumination = 0.5 * learned_illumination + 0.5 * luminance
        prior = torch.clamp(x, 1e-6, 1.0) ** gamma
        dark_mask = 1.0 - torch.clamp(illumination, 0.0, 1.0)
        return prior, gamma, illumination, dark_mask


class WindowAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * 2
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.illumination_gate = nn.Sequential(
            nn.Conv2d(1, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        window = min(self.window_size, h, w)
        pad_h = (window - h % window) % window
        pad_w = (window - w % window) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            illumination = F.pad(illumination, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, hp, wp = x.shape
        gate = self.illumination_gate(F.interpolate(illumination, size=(hp, wp), mode="bilinear", align_corners=False))
        x = x * (1.0 + gate)

        windows = x.unfold(2, window, window).unfold(3, window, window)
        windows = windows.permute(0, 2, 3, 4, 5, 1).contiguous()
        windows = windows.view(-1, window * window, c)

        attended = self.norm1(windows)
        attended, _ = self.attn(attended, attended, attended, need_weights=False)
        windows = windows + attended
        windows = windows + self.ffn(self.norm2(windows))

        windows = windows.view(b, hp // window, wp // window, window, window, c)
        x = windows.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]


class LPIAFormerLite(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        gamma_min: float = 0.35,
        gamma_max: float = 1.15,
        window_size: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()
        b = base_channels
        self.prior = RetinexGammaPrior(gamma_min=gamma_min, gamma_max=gamma_max)
        self.enc1 = ConvBlock(8, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b * 4, b * 8)
        self.former = WindowAttentionBlock(b * 8, num_heads=num_heads, window_size=window_size)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Sequential(nn.Conv2d(b, 3, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        prior, gamma, illumination, dark_mask = self.prior(x)
        model_input = torch.cat([x, prior, illumination, dark_mask], dim=1)

        e1 = self.enc1(model_input)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        bottleneck_illumination = F.interpolate(
            illumination,
            size=bottleneck.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        bottleneck = self.former(bottleneck, bottleneck_illumination)

        d3 = self.dec3(torch.cat([self._match(self.up3(bottleneck), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        output = self.out(d1)

        if return_aux:
            return output, {
                "prior": prior,
                "gamma_map": gamma,
                "illumination": illumination,
                "dark_mask": dark_mask,
            }
        return output

    @staticmethod
    def _match(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
