from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lpia_project.notebook_runtime import load_notebook_namespace
from run_visualize_aux import heatmap_gray


CHECKPOINTS = {
    "LPIA-LightU-Net": Path("outputs/checkpoints/lpia_lightunet_strict50_best.pt"),
    "LPIA-Former-Lite": Path("outputs/checkpoints/lpia_former_lite_strict50_best.pt"),
    "Zero-DCE-Lite": Path("outputs/checkpoints/zero_dce_lite_strict50_best.pt"),
}


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def predict_with_aux(ns: dict, model, image: np.ndarray):
    tensor = ns["np_to_tensor"](image).unsqueeze(0).to(ns["DEVICE"])
    with torch.no_grad():
        output, aux = model(tensor, return_aux=True)
    enhanced = ns["tensor_to_np"](output[0])
    aux_images = {}
    for key in ["gamma_map", "illumination", "dark_mask"]:
        value = aux.get(key)
        if value is not None:
            aux_images[key] = heatmap_gray(value[0, 0].detach().cpu().numpy())
    return enhanced, aux_images


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Please install gradio first: pip install gradio") from exc

    ns = load_notebook_namespace(display=lambda value: None)
    models = {}
    statuses = []
    for name, checkpoint in CHECKPOINTS.items():
        if checkpoint.exists():
            models[name] = ns["load_checkpoint"](checkpoint)
            models[name].eval()
            statuses.append(f"{name}: loaded {checkpoint.name}")
        else:
            models[name] = ns["build_model"](
                {
                    "LPIA-LightU-Net": "lpia_lightunet",
                    "LPIA-Former-Lite": "lpia_former_lite",
                    "Zero-DCE-Lite": "zero_dce_lite",
                }[name]
            ).to(ns["DEVICE"])
            models[name].eval()
            statuses.append(f"{name}: checkpoint missing, using untrained model")

    methods = ["Gamma(0.6)", "CLAHE", "Retinex", *models.keys()]

    def enhance(image_uint8: np.ndarray, method: str):
        if image_uint8 is None:
            return None, None, None, None, pd.DataFrame([{"message": "Please upload an image first."}])

        image = image_uint8.astype(np.float32) / 255.0
        aux_images = {}
        start = time.perf_counter()
        if method == "Gamma(0.6)":
            enhanced = ns["gamma_correction"](image, gamma=0.6)
        elif method == "CLAHE":
            enhanced = ns["clahe_enhance"](image)
        elif method == "Retinex":
            enhanced = ns["retinex_enhance"](image)
        else:
            enhanced, aux_images = predict_with_aux(ns, models[method], image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        metrics = ns["brightness_metrics"](image, enhanced)
        metrics["time_ms"] = elapsed_ms
        metrics_df = pd.DataFrame([metrics]).T.reset_index()
        metrics_df.columns = ["metric", "value"]
        return (
            to_uint8(enhanced),
            aux_images.get("gamma_map"),
            aux_images.get("illumination"),
            aux_images.get("dark_mask"),
            metrics_df,
        )

    def compare(image_uint8: np.ndarray):
        if image_uint8 is None:
            return [], pd.DataFrame([{"message": "Please upload an image first."}])
        gallery = []
        rows = []
        for method in methods:
            try:
                enhanced, _, _, _, metrics = enhance(image_uint8, method)
                gallery.append((enhanced, method))
                row = {"method": method}
                row.update({item["metric"]: item["value"] for item in metrics.to_dict("records")})
                rows.append(row)
            except Exception as exc:
                rows.append({"method": method, "error": str(exc)})
        return gallery, pd.DataFrame(rows)

    with gr.Blocks(title="Low-Light Enhancement System") as demo:
        gr.Markdown("# Low-Light Enhancement System")
        gr.Markdown("<br>".join(statuses))
        with gr.Row():
            input_image = gr.Image(label="Low-light image", type="numpy")
            output_image = gr.Image(label="Enhanced result", type="numpy")
        with gr.Row():
            method_radio = gr.Radio(methods, value="LPIA-Former-Lite", label="Method")
            run_button = gr.Button("Run Single Method", variant="primary")
            compare_button = gr.Button("Compare Methods")
        with gr.Row():
            gamma_map = gr.Image(label="Gamma / Curve Map", type="numpy")
            illumination_map = gr.Image(label="Illumination Map", type="numpy")
            dark_mask = gr.Image(label="Dark Mask", type="numpy")
        metrics_table = gr.Dataframe(label="Inference and brightness metrics")
        gallery = gr.Gallery(label="Multi-method comparison", columns=3, height="auto")

        run_button.click(
            enhance,
            inputs=[input_image, method_radio],
            outputs=[output_image, gamma_map, illumination_map, dark_mask, metrics_table],
        )
        compare_button.click(compare, inputs=input_image, outputs=[gallery, metrics_table])
    return demo


def main():
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
