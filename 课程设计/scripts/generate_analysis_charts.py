from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "analysis_charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_LABELS = {
    "plain_unet": "Plain U-Net",
    "attention_unet": "Attention U-Net",
    "fixed_gamma_unet": "Fixed Gamma U-Net",
    "learnable_gamma_unet": "Learnable Gamma U-Net",
    "lpia_lightunet": "LPIA-LightU-Net",
    "lpia_former_lite": "LPIA-Former-Lite",
    "zero_dce_lite": "Zero-DCE-Lite",
}

METHOD_COLORS = {
    "Plain U-Net": "#6B7280",
    "Attention U-Net": "#3B82F6",
    "Fixed Gamma U-Net": "#10B981",
    "Learnable Gamma U-Net": "#F59E0B",
    "LPIA-LightU-Net": "#EF4444",
    "LPIA-Former-Lite": "#8B5CF6",
    "Zero-DCE-Lite": "#14B8A6",
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
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 13


def model_name(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def color_for(label: str) -> str:
    return METHOD_COLORS.get(label, "#64748B")


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def load_main_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ablation = pd.read_csv(PROJECT_ROOT / "outputs" / "strict_ablation_50epoch_with_zero_dce_summary.csv")
    benchmark = pd.read_csv(PROJECT_ROOT / "outputs" / "benchmarks" / "model_complexity_256_with_zero_dce.csv")
    generalization = pd.read_csv(PROJECT_ROOT / "outputs" / "generalization_eval.csv")
    ablation["模型名称"] = ablation["variant"].map(model_name)
    benchmark["模型名称"] = benchmark["variant"].map(model_name)
    return ablation, benchmark, generalization


def plot_metric_bars(ablation: pd.DataFrame) -> None:
    data = ablation.sort_values("test_psnr", ascending=False)
    labels = data["模型名称"].tolist()
    colors = [color_for(label) for label in labels]
    x = np.arange(len(data))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(x, data["test_psnr"], color=colors, edgecolor="#111827", linewidth=0.5)
    axes[0].set_title("测试集 PSNR 对比")
    axes[0].set_ylabel("PSNR / dB")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(data["test_psnr"]):
        axes[0].text(i, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, data["test_ssim"], color=colors, edgecolor="#111827", linewidth=0.5)
    axes[1].set_title("测试集 SSIM 对比")
    axes[1].set_ylabel("SSIM")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(max(0.5, data["test_ssim"].min() - 0.04), min(1.0, data["test_ssim"].max() + 0.04))
    for i, v in enumerate(data["test_ssim"]):
        axes[1].text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("不同模型在 LOL 测试集上的客观指标对比", fontsize=15, fontweight="bold")
    save(fig, "01_test_psnr_ssim_bar.png")


def plot_ablation_gain(ablation: pd.DataFrame) -> None:
    base = float(ablation.loc[ablation["variant"] == "plain_unet", "test_psnr"].iloc[0])
    data = ablation.copy()
    data["PSNR提升"] = data["test_psnr"] - base
    data = data.sort_values("PSNR提升", ascending=False)
    labels = data["模型名称"].tolist()
    colors = ["#22C55E" if v >= 0 else "#EF4444" for v in data["PSNR提升"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(data)), data["PSNR提升"], color=colors, edgecolor="#111827", linewidth=0.5)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title("相对 Plain U-Net 的 PSNR 提升")
    ax.set_ylabel("PSNR 提升 / dB")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(data["PSNR提升"]):
        va = "bottom" if v >= 0 else "top"
        offset = 0.03 if v >= 0 else -0.03
        ax.text(i, v + offset, f"{v:+.2f}", ha="center", va=va, fontsize=8)
    save(fig, "02_ablation_psnr_gain.png")


def plot_complexity_tradeoff(ablation: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    data = pd.merge(ablation, benchmark, on=["variant", "模型名称"], how="inner")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in data.iterrows():
        label = row["模型名称"]
        size = 90 + row["time_mean_ms"] * 22
        ax.scatter(
            row["params_m"],
            row["test_psnr"],
            s=size,
            color=color_for(label),
            edgecolor="#111827",
            linewidth=0.7,
            alpha=0.88,
        )
        ax.annotate(label, (row["params_m"], row["test_psnr"]), xytext=(6, 4), textcoords="offset points", fontsize=8)

    ax.set_title("效果-轻量化权衡：参数量 vs 测试集 PSNR")
    ax.set_xlabel("参数量 / M")
    ax.set_ylabel("测试集 PSNR / dB")
    ax.grid(alpha=0.25)
    note = "气泡越大表示平均推理时间越长"
    ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=9, color="#374151")
    save(fig, "03_psnr_params_tradeoff.png")


def plot_speed_quality_tradeoff(ablation: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    data = pd.merge(ablation, benchmark, on=["variant", "模型名称"], how="inner")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in data.iterrows():
        label = row["模型名称"]
        ax.scatter(
            row["time_mean_ms"],
            row["test_ssim"],
            s=80 + row["params_m"] * 70,
            color=color_for(label),
            edgecolor="#111827",
            linewidth=0.7,
            alpha=0.88,
        )
        ax.annotate(label, (row["time_mean_ms"], row["test_ssim"]), xytext=(6, 4), textcoords="offset points", fontsize=8)

    ax.set_title("速度-结构质量权衡：推理时间 vs 测试集 SSIM")
    ax.set_xlabel("平均推理时间 / ms")
    ax.set_ylabel("测试集 SSIM")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.03, "气泡越大表示参数量越大", transform=ax.transAxes, fontsize=9, color="#374151")
    save(fig, "04_speed_ssim_tradeoff.png")


def plot_generalization_tradeoff(generalization: pd.DataFrame) -> None:
    summary = (
        generalization.groupby("method", as_index=False)[["mean_y_gain", "dark_y_gain", "over_exposed_ratio"]]
        .mean()
        .sort_values("dark_y_gain", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#14B8A6", "#F97316", "#8B5CF6", "#EF4444", "#3B82F6"]
    for color, (_, row) in zip(colors, summary.iterrows()):
        ax.scatter(row["dark_y_gain"], row["over_exposed_ratio"], s=180, color=color, edgecolor="#111827", linewidth=0.8)
        ax.annotate(row["method"], (row["dark_y_gain"], row["over_exposed_ratio"]), xytext=(8, 5), textcoords="offset points", fontsize=9)

    ax.set_title("泛化场景权衡：暗区提升 vs 过曝风险")
    ax.set_xlabel("暗区亮度提升")
    ax.set_ylabel("过曝比例")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.94, "理想方向：更靠右且更靠下", transform=ax.transAxes, fontsize=10, color="#166534")
    save(fig, "05_generalization_gain_overexposure.png")


def plot_training_curves() -> None:
    history_files = sorted((PROJECT_ROOT / "outputs").glob("*_strict50_history.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for file in history_files:
        variant = file.name.replace("_strict50_history.csv", "")
        label = model_name(variant)
        hist = pd.read_csv(file)
        color = color_for(label)
        if "val_psnr" in hist:
            axes[0].plot(hist["epoch"], hist["val_psnr"], label=label, color=color, linewidth=1.8)
        if "val_loss" in hist:
            axes[1].plot(hist["epoch"], hist["val_loss"], label=label, color=color, linewidth=1.8)

    axes[0].set_title("验证集 PSNR 训练曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val PSNR / dB")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("验证集 Loss 训练曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=7)
    save(fig, "06_training_curves.png")


def normalize_high(values: pd.Series) -> pd.Series:
    span = values.max() - values.min()
    if span == 0:
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - values.min()) / span


def normalize_low(values: pd.Series) -> pd.Series:
    return 1 - normalize_high(values)


def plot_radar(ablation: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    data = pd.merge(ablation, benchmark, on=["variant", "模型名称"], how="inner")
    selected = data[data["variant"].isin(["fixed_gamma_unet", "lpia_lightunet", "lpia_former_lite", "zero_dce_lite"])].copy()
    selected["PSNR"] = normalize_high(selected["test_psnr"])
    selected["SSIM"] = normalize_high(selected["test_ssim"])
    selected["轻量参数"] = normalize_low(selected["params_m"])
    selected["速度"] = normalize_low(selected["time_mean_ms"])
    selected["过曝控制"] = normalize_low(selected["test_over_exposed_ratio"])

    metrics = ["PSNR", "SSIM", "轻量参数", "速度", "过曝控制"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for _, row in selected.iterrows():
        label = row["模型名称"]
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, label=label, color=color_for(label), linewidth=2)
        ax.fill(angles, values, color=color_for(label), alpha=0.12)

    ax.set_title("代表模型综合能力雷达图（归一化）", pad=18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8)
    save(fig, "07_model_radar_normalized.png")


def main() -> None:
    setup_matplotlib()
    ablation, benchmark, generalization = load_main_tables()
    plot_metric_bars(ablation)
    plot_ablation_gain(ablation)
    plot_complexity_tradeoff(ablation, benchmark)
    plot_speed_quality_tradeoff(ablation, benchmark)
    plot_generalization_tradeoff(generalization)
    plot_training_curves()
    plot_radar(ablation, benchmark)
    print(f"已生成分析图目录：{OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
