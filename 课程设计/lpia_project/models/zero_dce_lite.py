from __future__ import annotations

import torch
from torch import nn


class ZeroDCELite(nn.Module):
    """Zero-DCE-style curve estimation baseline.

    The model predicts per-pixel RGB curve parameters and applies the iterative
    enhancement rule from the Zero-DCE family: x <- x + a * x * (1 - x).
    """

    def __init__(self, channels: int = 32, num_iterations: int = 8):
        super().__init__()
        self.num_iterations = num_iterations
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3 * num_iterations, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        curve_maps = self.features(x)
        enhanced = x
        curves = torch.chunk(curve_maps, self.num_iterations, dim=1)
        for curve in curves:
            enhanced = enhanced + curve * enhanced * (1.0 - enhanced)
            enhanced = torch.clamp(enhanced, 0.0, 1.0)

        if return_aux:
            mean_curve = curve_maps.mean(dim=1, keepdim=True)
            illumination = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            return enhanced, {
                "prior": enhanced,
                "gamma_map": mean_curve,
                "illumination": illumination,
                "dark_mask": 1.0 - illumination,
                "curve_maps": curve_maps,
            }
        return enhanced
