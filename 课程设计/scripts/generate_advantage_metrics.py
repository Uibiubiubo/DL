from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "results" / "advantage_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


METHOD_FILE_MAP = {
    "Gamma(0.6)": "gamma0_6",
    "Retinex": "retinex",
    "Zero-DCE-Lite": "zero-dce-lite",
    "LPIA-LightU-Net": "lpia-lightu-net",
    "LPIA-Former-Lite": "lpia-former-lite",
}


MODEL_LABELS = {
    "plain_unet": "Plain U-Net",
    "attention_unet": "Attention U-Net",
    "fixed_gamma_unet": "Fixed Gamma U-Net",
    "learnable_gamma_unet": "Learnable Gamma U-Net",
    "lpia_lightunet": "LPIA-LightU-Net",
    "lpia_former_lite": "LPIA-Former-Lite",
    "zero_dce_lite": "Zero-DCE-Lite",
}


COLORS = {
    "Plain U-Net": "#6B7280",
    "Attention U-Net": "#3B82F6",
    "Fixed Gamma U-Net": "#10B981",
    "Learnable Gamma U-Net": "#F59E0B",
    "LPIA-LightU-Net": "#EF4444",
    "LPIA-Former-Lite": "#8B5CF6",
    "Zero-DCE-Lite": "#14B8A6",
    "Gamma(0.6)": "#3B82F6",
    "Retinex": "#F97316",
}


def setup_matplotlib() -> None:
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]


def luminance(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def normalize_high(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span <= 1e-12:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - series.min()) / span


def normalize_low(series: pd.Series) -> pd.Series:
    return 1.0 - normalize_high(series)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ablation = pd.read_csv(OUTPUT_DIR / "strict_ablation_50epoch_with_zero_dce_summary.csv")
    benchmark = pd.read_csv(OUTPUT_DIR / "benchmarks" / "model_complexity_256_with_zero_dce.csv")
    test_eval = pd.read_csv(OUTPUT_DIR / "strict_ablation_50epoch_test_eval.csv")
    zero_eval = pd.read_csv(OUTPUT_DIR / "strict_ablation_50epoch_zero_dce_lite_test_eval.csv")
    test_eval = pd.concat([test_eval, zero_eval], ignore_index=True)
    generalization = pd.read_csv(OUTPUT_DIR / "generalization_eval.csv")
    return ablation, benchmark, test_eval, generalization


def paired_advantage_summary(ablation: pd.DataFrame, benchmark: pd.DataFrame, test_eval: pd.DataFrame) -> pd.DataFrame:
    data = pd.merge(ablation, benchmark, on="variant", how="inner")
    stability = (
        test_eval.groupby("variant", as_index=False)
        .agg(
            psnr_std=("psnr", "std"),
            ssim_std=("ssim", "std"),
            psnr_good_rate=("psnr", lambda s: float((s >= 20.0).mean())),
            over_exposed_std=("over_exposed_ratio", "std"),
        )
    )
    data = pd.merge(data, stability, on="variant", how="left")
    data["model"] = data["variant"].map(MODEL_LABELS)
    best_psnr = data["test_psnr"].max()
    best_ssim = data["test_ssim"].max()
    data["psnr_gap_to_best"] = best_psnr - data["test_psnr"]
    data["ssim_gap_to_best"] = best_ssim - data["test_ssim"]
    data["quality_retention"] = 0.5 * (data["test_psnr"] / best_psnr) + 0.5 * (data["test_ssim"] / best_ssim)
    data["psnr_per_mparam"] = data["test_psnr"] / data["params_m"]
    data["ssim_per_mparam"] = data["test_ssim"] / data["params_m"]
    data["fps_est"] = 1000.0 / data["time_mean_ms"].clip(lower=1e-6)

    data["quality_norm"] = 0.5 * normalize_high(data["test_psnr"]) + 0.5 * normalize_high(data["test_ssim"])
    data["lightweight_norm"] = 0.5 * normalize_low(data["params_m"]) + 0.5 * normalize_low(data["model_size_mb"])
    data["speed_norm"] = normalize_low(data["time_mean_ms"])
    data["stability_norm"] = 0.5 * normalize_low(data["psnr_std"]) + 0.5 * normalize_low(data["over_exposed_std"].fillna(0))
    data["edge_balanced_score"] = (
        0.40 * data["quality_norm"]
        + 0.30 * data["lightweight_norm"]
        + 0.15 * data["speed_norm"]
        + 0.15 * data["stability_norm"]
    )
    return data.sort_values("edge_balanced_score", ascending=False)


def generalization_exposure_records() -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    root = OUTPUT_DIR / "results" / "generalization"
    for image_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        low_path = image_dir / f"{image_dir.name}_low.png"
        if not low_path.exists():
            continue
        low = read_rgb(low_path)
        low_y = luminance(low)
        for method, slug in METHOD_FILE_MAP.items():
            output_path = image_dir / f"{image_dir.name}_{slug}.png"
            if not output_path.exists():
                continue
            out = read_rgb(output_path)
            y = luminance(out)
            color_shift = float(np.mean(np.abs(out.mean(axis=(0, 1)) - low.mean(axis=(0, 1)))))
            records.append(
                {
                    "image": image_dir.name,
                    "method": method,
                    "under_exposed_ratio": float((y < 0.05).mean()),
                    "over_exposed_ratio_from_image": float((y > 0.95).mean()),
                    "well_exposed_ratio": float(((y >= 0.10) & (y <= 0.90)).mean()),
                    "mean_y": float(y.mean()),
                    "mean_y_gain_from_image": float(y.mean() - low_y.mean()),
                    "color_shift_from_input": color_shift,
                }
            )
    return pd.DataFrame(records)


def generalization_advantage_summary(generalization: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    gen = (
        generalization.groupby("method", as_index=False)[["mean_y_gain", "dark_y_gain", "over_exposed_ratio"]]
        .mean()
    )
    exp = (
        exposure.groupby("method", as_index=False)[
            ["under_exposed_ratio", "over_exposed_ratio_from_image", "well_exposed_ratio", "color_shift_from_input"]
        ]
        .mean()
    )
    data = pd.merge(gen, exp, on="method", how="inner")
    data["dark_gain_over_overexposure"] = data["dark_y_gain"] / (data["over_exposed_ratio"] + 0.01)
    data["usable_exposure_score"] = (
        0.45 * normalize_high(data["dark_y_gain"])
        + 0.25 * normalize_high(data["well_exposed_ratio"])
        + 0.20 * normalize_low(data["over_exposed_ratio"])
        + 0.10 * normalize_low(data["color_shift_from_input"])
    )
    return data.sort_values("usable_exposure_score", ascending=False)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, bbox_inches="tight")
    plt.close(fig)


def plot_edge_score(summary: pd.DataFrame) -> None:
    data = summary.copy()
    labels = data["model"].tolist()
    colors = [COLORS.get(label, "#64748B") for label in labels]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(data)), data["edge_balanced_score"], color=colors, edgecolor="#111827", linewidth=0.5)
    ax.set_title("端侧轻量化综合评分（质量 + 轻量 + 速度 + 稳定性）")
    ax.set_ylabel("归一化综合评分")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(data["edge_balanced_score"]):
        ax.text(i, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "01_edge_balanced_score.png")


def plot_quality_retention_vs_size(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in summary.iterrows():
        label = row["model"]
        ax.scatter(
            row["model_size_mb"],
            row["quality_retention"],
            s=80 + row["fps_est"] * 1.4,
            color=COLORS.get(label, "#64748B"),
            edgecolor="#111827",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.annotate(label, (row["model_size_mb"], row["quality_retention"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_title("质量保持率 vs 模型体积")
    ax.set_xlabel("模型大小 / MB")
    ax.set_ylabel("质量保持率（相对最佳 PSNR/SSIM）")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.04, "理想方向：更靠左且更靠上；气泡越大表示 FPS 越高", transform=ax.transAxes, fontsize=9, color="#166534")
    save(fig, "02_quality_retention_vs_size.png")


def high_quality_lightweight_summary(summary: pd.DataFrame) -> pd.DataFrame:
    best_psnr = summary["test_psnr"].max()
    best_ssim = summary["test_ssim"].max()
    high_quality = summary[
        (summary["test_psnr"] >= best_psnr - 0.25)
        & (summary["test_ssim"] >= best_ssim - 0.005)
    ].copy()
    high_quality["psnr_gap_to_best"] = best_psnr - high_quality["test_psnr"]
    high_quality["ssim_gap_to_best"] = best_ssim - high_quality["test_ssim"]
    high_quality["size_reduction_vs_lpia_lightunet"] = np.nan
    ref = summary.loc[summary["model"] == "LPIA-LightU-Net", "model_size_mb"]
    if not ref.empty:
        ref_size = float(ref.iloc[0])
        high_quality["size_reduction_vs_lpia_lightunet"] = 1.0 - high_quality["model_size_mb"] / ref_size
    return high_quality.sort_values(["model_size_mb", "psnr_gap_to_best"])


def plot_high_quality_lightweight(summary: pd.DataFrame) -> pd.DataFrame:
    data = high_quality_lightweight_summary(summary)
    labels = data["model"].tolist()
    colors = [COLORS.get(label, "#64748B") for label in labels]
    x = np.arange(len(data))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    width = 0.38

    bars1 = ax1.bar(x - width / 2, data["test_psnr"], width, label="测试集 PSNR", color=colors, edgecolor="#111827", linewidth=0.5)
    bars2 = ax2.bar(x + width / 2, data["model_size_mb"], width, label="模型大小/MB", color="#94A3B8", edgecolor="#111827", linewidth=0.5)

    ax1.set_title("高质量模型中的轻量化优势")
    ax1.set_ylabel("测试集 PSNR / dB")
    ax2.set_ylabel("模型大小 / MB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_ylim(data["test_psnr"].min() - 0.2, data["test_psnr"].max() + 0.2)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#334155")

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")
    ax1.text(
        0.02,
        0.04,
        "筛选条件：PSNR 距最优不超过 0.25 dB，SSIM 距最优不超过 0.005",
        transform=ax1.transAxes,
        fontsize=9,
        color="#374151",
    )
    save(fig, "03_high_quality_lightweight_models.png")
    return data


def plot_generalization_score(summary: pd.DataFrame) -> None:
    data = summary.copy()
    colors = [COLORS.get(method, "#64748B") for method in data["method"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(data)), data["usable_exposure_score"], color=colors, edgecolor="#111827", linewidth=0.5)
    ax.set_title("泛化可用曝光综合评分（暗区提升 + 合理曝光 + 低过曝 + 低偏色）")
    ax.set_ylabel("归一化综合评分")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["method"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(data["usable_exposure_score"]):
        ax.text(i, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "04_generalization_usable_exposure_score.png")


def plot_exposure_stack(summary: pd.DataFrame) -> None:
    data = summary.sort_values("well_exposed_ratio", ascending=False)
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, data["under_exposed_ratio"], label="欠曝比例", color="#1D4ED8")
    ax.bar(x, data["well_exposed_ratio"], bottom=data["under_exposed_ratio"], label="合理曝光比例", color="#22C55E")
    bottom = data["under_exposed_ratio"] + data["well_exposed_ratio"]
    ax.bar(x, data["over_exposed_ratio_from_image"], bottom=bottom, label="过曝比例", color="#EF4444")
    ax.set_title("泛化输出曝光分布")
    ax.set_ylabel("像素比例")
    ax.set_xticks(x)
    ax.set_xticklabels(data["method"], rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "05_generalization_exposure_distribution.png")


def plot_color_shift(summary: pd.DataFrame) -> None:
    data = summary.sort_values("color_shift_from_input")
    colors = [COLORS.get(method, "#64748B") for method in data["method"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(data)), data["color_shift_from_input"], color=colors, edgecolor="#111827", linewidth=0.5)
    ax.set_title("泛化输出平均颜色偏移（越低越稳定）")
    ax.set_ylabel("RGB 均值绝对偏移")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["method"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(data["color_shift_from_input"]):
        ax.text(i, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "06_generalization_color_shift.png")


def main() -> None:
    setup_matplotlib()
    ablation, benchmark, test_eval, generalization = load_tables()

    paired_summary = paired_advantage_summary(ablation, benchmark, test_eval)
    exposure = generalization_exposure_records()
    generalization_summary = generalization_advantage_summary(generalization, exposure)

    save_table(paired_summary, OUTPUT_DIR / "advantage_paired_summary.csv")
    save_table(exposure, OUTPUT_DIR / "advantage_generalization_image_metrics.csv")
    save_table(generalization_summary, OUTPUT_DIR / "advantage_generalization_summary.csv")

    plot_edge_score(paired_summary)
    plot_quality_retention_vs_size(paired_summary)
    high_quality = plot_high_quality_lightweight(paired_summary)
    save_table(high_quality, OUTPUT_DIR / "advantage_high_quality_lightweight_summary.csv")
    plot_generalization_score(generalization_summary)
    plot_exposure_stack(generalization_summary)
    plot_color_shift(generalization_summary)

    print("已生成优势指标表：")
    print(OUTPUT_DIR / "advantage_paired_summary.csv")
    print(OUTPUT_DIR / "advantage_high_quality_lightweight_summary.csv")
    print(OUTPUT_DIR / "advantage_generalization_image_metrics.csv")
    print(OUTPUT_DIR / "advantage_generalization_summary.csv")
    print("已生成优势分析图：")
    for path in sorted(CHART_DIR.glob("*.png")):
        print(path.name)
    print("\n端侧轻量化综合评分：")
    print(paired_summary[["model", "test_psnr", "test_ssim", "params_m", "model_size_mb", "edge_balanced_score"]].to_string(index=False))
    print("\n泛化可用曝光综合评分：")
    print(generalization_summary[["method", "dark_y_gain", "over_exposed_ratio", "well_exposed_ratio", "color_shift_from_input", "usable_exposure_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
