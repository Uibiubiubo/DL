from __future__ import annotations

from typing import Any, Callable

from lpia_project.models import LPIAFormerLite, ZeroDCELite


def register_project_models(namespace: dict[str, Any]) -> None:
    original_build_model: Callable[..., Any] = namespace["build_model"]
    cfg = namespace["CFG"]

    def build_model(variant: str = "lpia_lightunet", base_channels: int = cfg.base_channels):
        if variant.lower() == "lpia_former_lite":
            return LPIAFormerLite(
                base_channels=base_channels,
                gamma_min=0.35,
                gamma_max=1.15,
                window_size=8,
                num_heads=4,
            )
        if variant.lower() == "zero_dce_lite":
            return ZeroDCELite(channels=32, num_iterations=8)
        return original_build_model(variant=variant, base_channels=base_channels)

    namespace["build_model"] = build_model
    namespace["EXPERIMENTS"]["lpia_former_lite"] = (
        "Retinex/Gamma 双先验 + 光照引导窗口注意力 + 轻量 CNN 解码器"
    )
    namespace["EXPERIMENTS"]["zero_dce_lite"] = (
        "Zero-DCE 风格像素级曲线估计轻量深度基线"
    )
