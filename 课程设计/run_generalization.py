import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from lpia_project.notebook_runtime import load_notebook_namespace


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-reference generalization comparison.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/generalization/low"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/results/generalization"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("outputs/generalization_eval.csv"))
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--lpia-checkpoint", type=Path, default=Path("outputs/checkpoints/lpia_lightunet_strict50_best.pt"))
    parser.add_argument(
        "--former-checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/lpia_former_lite_strict50_best.pt"),
    )
    parser.add_argument(
        "--zero-dce-checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/zero_dce_lite_strict50_best.pt"),
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Generalization folder not found: {folder}")
    images = [path for path in sorted(folder.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")
    return images


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image)).save(path)


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    width = max(1, int(image.width * height / image.height))
    return image.resize((width, height), Image.BICUBIC)


def save_comparison_sheet(path: Path, panels: list[tuple[str, np.ndarray]], panel_height: int = 240) -> None:
    label_height = 28
    margin = 8
    font = ImageFont.load_default()
    resized: list[tuple[str, Image.Image]] = []
    for label, image in panels:
        resized.append((label, resize_to_height(Image.fromarray(to_uint8(image)), panel_height)))

    total_width = sum(image.width for _, image in resized) + margin * (len(resized) + 1)
    sheet = Image.new("RGB", (total_width, panel_height + label_height + margin * 2), "white")
    draw = ImageDraw.Draw(sheet)
    x = margin
    for label, image in resized:
        draw.text((x + 4, margin + 7), label, fill=(20, 20, 20), font=font)
        sheet.paste(image, (x, margin + label_height))
        x += image.width + margin
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def predict_model(ns: dict, model, image: np.ndarray) -> np.ndarray:
    return ns["predict_model_np"](model, image, ns["DEVICE"])


def method_records(
    ns: dict,
    image_name: str,
    input_image: np.ndarray,
    outputs: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    records = []
    for method, output in outputs.items():
        metrics = ns["brightness_metrics"](input_image, output)
        record: dict[str, float | str] = {
            "image": image_name,
            "method": method,
        }
        record.update(metrics)
        records.append(record)
    return records


def load_model_if_exists(ns: dict, checkpoint: Path, variant: str):
    if checkpoint.exists():
        return ns["load_checkpoint"](checkpoint)
    print(f"Checkpoint not found for {variant}, using untrained model: {checkpoint}", flush=True)
    return ns["build_model"](variant).to(ns["DEVICE"])


def main() -> None:
    args = parse_args()
    ns = load_notebook_namespace(display=lambda value: None)
    image_paths = list_images(args.input_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    models = {
        "Zero-DCE-Lite": load_model_if_exists(ns, args.zero_dce_checkpoint, "zero_dce_lite"),
        "LPIA-LightU-Net": load_model_if_exists(ns, args.lpia_checkpoint, "lpia_lightunet"),
        "LPIA-Former-Lite": load_model_if_exists(ns, args.former_checkpoint, "lpia_former_lite"),
    }
    for model in models.values():
        model.eval()

    traditional: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "Gamma(0.6)": ns["TRADITIONAL_METHODS"]["Gamma(0.6)"],
        "Retinex": ns["TRADITIONAL_METHODS"]["Retinex"],
    }

    records = []
    for image_path in image_paths:
        low = ns["read_rgb"](image_path)
        outputs: dict[str, np.ndarray] = {"Low": low}
        for name, fn in traditional.items():
            outputs[name] = fn(low)
        for name, model in models.items():
            with torch.no_grad():
                outputs[name] = predict_model(ns, model, low)

        image_dir = args.output_dir / image_path.stem
        for method, output in outputs.items():
            filename = method.lower().replace("(", "").replace(")", "").replace(".", "_").replace(" ", "_")
            save_rgb(image_dir / f"{image_path.stem}_{filename}.png", output)

        save_comparison_sheet(
            image_dir / f"{image_path.stem}_comparison_sheet.png",
            list(outputs.items()),
        )
        records.extend(method_records(ns, image_path.name, low, {k: v for k, v in outputs.items() if k != "Low"}))
        print(f"Saved generalization comparison: {image_dir}", flush=True)

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved metrics: {args.metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
