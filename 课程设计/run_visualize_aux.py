import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from lpia_project.data.split_manifest import load_pairs_from_split
from lpia_project.notebook_runtime import load_notebook_namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LPIA auxiliary visualizations.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/lpia_lightunet_best.pt"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/lol_v1_seed42.json"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/results/aux_visualizations"))
    parser.add_argument("--max-images", type=int, default=5)
    return parser.parse_args()


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def heatmap_gray(value: np.ndarray) -> np.ndarray:
    value = np.squeeze(value).astype(np.float32)
    min_value = float(value.min())
    max_value = float(value.max())
    normalized = (value - min_value) / max(max_value - min_value, 1e-6)

    # Lightweight blue-green-yellow-red colormap without extra dependencies.
    anchors = np.array(
        [
            [32, 45, 112],
            [38, 139, 210],
            [42, 161, 152],
            [181, 137, 0],
            [220, 50, 47],
        ],
        dtype=np.float32,
    )
    scaled = normalized * (len(anchors) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.clip(left + 1, 0, len(anchors) - 1)
    alpha = (scaled - left)[..., None]
    return ((1.0 - alpha) * anchors[left] + alpha * anchors[right]).astype(np.uint8)


def luminance(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def save_luminance_histogram(path: Path, low: np.ndarray, enhanced: np.ndarray, high: np.ndarray) -> None:
    width, height = 720, 360
    margin_left, margin_right, margin_top, margin_bottom = 54, 24, 28, 48
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bins = np.linspace(0.0, 1.0, 65)
    series = [
        ("Low", low, (60, 90, 190)),
        ("Enhanced", enhanced, (38, 150, 120)),
        ("Reference", high, (205, 90, 65)),
    ]
    histograms = [(name, np.histogram(values.ravel(), bins=bins)[0], color) for name, values, color in series]
    max_count = max(int(hist.max()) for _, hist, _ in histograms)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    axis_color = (45, 45, 45)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_height), fill=axis_color)
    draw.line((margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height), fill=axis_color)
    draw.text((margin_left, height - 28), "Luminance", fill=axis_color, font=font)
    draw.text((12, margin_top), "Count", fill=axis_color, font=font)

    for name, hist, color in histograms:
        points = []
        for index, count in enumerate(hist):
            x = margin_left + int(index / (len(hist) - 1) * plot_width)
            y = margin_top + plot_height - int((count / max(max_count, 1)) * plot_height)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)

    legend_x = margin_left + plot_width - 190
    for offset, (name, _, color) in enumerate(histograms):
        y = margin_top + offset * 20
        draw.rectangle((legend_x, y + 3, legend_x + 12, y + 15), fill=color)
        draw.text((legend_x + 18, y), name, fill=axis_color, font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_sheet(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    images = [Image.fromarray(panel) for _, panel in panels]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 26
    sheet = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, ((label, _), image) in enumerate(zip(panels, images)):
        x = index * width
        sheet.paste(image.resize((width, height)), (x, label_height))
        draw.text((x + 8, 7), label, fill=(20, 20, 20), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    args = parse_args()
    ns = load_notebook_namespace(display=lambda value: None)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.split_file.exists():
        raise FileNotFoundError(f"Split file not found: {args.split_file}")

    pairs = load_pairs_from_split(args.split_file, ns["CFG"].data_root, args.split)
    pairs = pairs[: args.max_images]
    model = ns["load_checkpoint"](args.checkpoint)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for low_path, high_path in pairs:
        low = ns["read_rgb"](low_path)
        high = ns["read_rgb"](high_path)
        tensor = ns["np_to_tensor"](low).unsqueeze(0).to(ns["DEVICE"])

        with torch.no_grad():
            pred, aux = model(tensor, return_aux=True)

        enhanced = ns["tensor_to_np"](pred[0])
        illumination = aux["illumination"][0, 0].detach().cpu().numpy()
        gamma_map = aux.get("gamma_map")
        gamma_np = gamma_map[0, 0].detach().cpu().numpy() if gamma_map is not None else illumination
        dark_mask = 1.0 - np.clip(illumination, 0.0, 1.0)
        error = np.abs(enhanced - high).mean(axis=2)

        stem = low_path.stem
        panels = [
            ("Low", to_uint8(low)),
            ("Enhanced", to_uint8(enhanced)),
            ("Reference", to_uint8(high)),
            ("Gamma", heatmap_gray(gamma_np)),
            ("Illumination", heatmap_gray(illumination)),
            ("Dark Mask", heatmap_gray(dark_mask)),
            ("Error", heatmap_gray(error)),
        ]

        image_dir = args.output_dir / stem
        image_dir.mkdir(parents=True, exist_ok=True)
        for label, panel in panels:
            filename = label.lower().replace(" ", "_")
            Image.fromarray(panel).save(image_dir / f"{stem}_{filename}.png")

        histogram = np.stack(
            [
                luminance(low),
                luminance(enhanced),
                luminance(high),
            ],
            axis=0,
        )
        np.save(image_dir / f"{stem}_luminance_values.npy", histogram)
        save_luminance_histogram(
            image_dir / f"{stem}_luminance_histogram.png",
            histogram[0],
            histogram[1],
            histogram[2],
        )
        save_sheet(image_dir / f"{stem}_aux_sheet.png", panels)
        print(f"Saved auxiliary visualization: {image_dir}", flush=True)


if __name__ == "__main__":
    main()
