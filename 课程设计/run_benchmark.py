import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lpia_project.notebook_runtime import load_notebook_namespace


DEFAULT_VARIANTS = [
    "plain_unet",
    "attention_unet",
    "fixed_gamma_unet",
    "learnable_gamma_unet",
    "lpia_lightunet",
    "lpia_former_lite",
    "zero_dce_lite",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model complexity and inference speed.")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/model_complexity.csv"))
    parser.add_argument("--prefer-checkpoint", action="store_true")
    return parser.parse_args()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def estimate_flops(model: nn.Module, sample: torch.Tensor) -> int:
    # Course-report friendly estimate: count Conv2d and Linear operations.
    hooks = []
    flops = 0

    def add_conv_flops(module: nn.Conv2d, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal flops
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_h = output.shape[2]
        out_w = output.shape[3]
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels // module.groups
        macs = batch_size * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w
        if module.bias is not None:
            macs += batch_size * out_channels * out_h * out_w
        flops += 2 * macs

    def add_linear_flops(module: nn.Linear, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal flops
        batch_size = output.shape[0] if output.ndim > 1 else 1
        macs = batch_size * module.in_features * module.out_features
        if module.bias is not None:
            macs += batch_size * module.out_features
        flops += 2 * macs

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(add_conv_flops))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(add_linear_flops))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()
    return int(flops)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        synchronize(device)

        times_ms: list[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(sample)
                end.record()
                synchronize(device)
                times_ms.append(float(start.elapsed_time(end)))
            else:
                import time

                start_time = time.perf_counter()
                model(sample)
                times_ms.append((time.perf_counter() - start_time) * 1000.0)

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    sorted_times = sorted(times_ms)
    mean_ms = sum(times_ms) / len(times_ms)
    variance = sum((value - mean_ms) ** 2 for value in times_ms) / len(times_ms)
    median_ms = sorted_times[len(sorted_times) // 2]
    return {
        "time_mean_ms": mean_ms,
        "time_std_ms": variance**0.5,
        "time_median_ms": median_ms,
        "peak_memory_mb": peak_memory_mb,
    }


def load_variant_model(ns: dict[str, Any], variant: str, checkpoint_dir: Path, prefer_checkpoint: bool) -> nn.Module:
    checkpoint = checkpoint_dir / f"{variant}_best.pt"
    if prefer_checkpoint and checkpoint.exists():
        return ns["load_checkpoint"](checkpoint)
    return ns["build_model"](variant).to(ns["DEVICE"])


def main() -> None:
    args = parse_args()
    ns = load_notebook_namespace(display=lambda value: None)
    device = ns["DEVICE"]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    sample = torch.rand(1, 3, args.height, args.width, device=device)
    rows = []

    for variant in variants:
        print(f"Benchmarking {variant} on {device} ({args.height}x{args.width})", flush=True)
        model = load_variant_model(ns, variant, args.checkpoint_dir, args.prefer_checkpoint)
        model = model.to(device)
        params = count_parameters(model)
        size_mb = model_size_mb(model)
        flops = estimate_flops(model, sample)
        speed = benchmark_inference(model, sample, device, args.warmup, args.repeats)
        rows.append(
            {
                "variant": variant,
                "input_h": args.height,
                "input_w": args.width,
                "device": str(device),
                "params": params,
                "params_m": params / 1e6,
                "model_size_mb": size_mb,
                "flops": flops,
                "gflops": flops / 1e9,
                **speed,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmark: {args.output}", flush=True)
    for row in rows:
        print(
            f"{row['variant']}: params={row['params_m']:.3f}M, "
            f"size={row['model_size_mb']:.2f}MB, gflops={row['gflops']:.2f}, "
            f"time={row['time_mean_ms']:.2f}±{row['time_std_ms']:.2f}ms, "
            f"peak_mem={row['peak_memory_mb']:.2f}MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
