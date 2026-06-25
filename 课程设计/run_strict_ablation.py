import os
from pathlib import Path

from lpia_project.data.split_manifest import load_pairs_from_split, split_summary
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


def _parse_variants() -> list[str]:
    raw = os.environ.get("LPIA_ABLATION_VARIANTS")
    if not raw:
        return DEFAULT_VARIANTS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _output_tag(epochs: int, variants: list[str]) -> str:
    raw = os.environ.get("LPIA_ABLATION_TAG")
    if raw:
        return raw
    if variants == DEFAULT_VARIANTS:
        return f"{epochs}epoch"
    return f"{epochs}epoch_{'_'.join(variants)}"


def main():
    ns = load_notebook_namespace(display=lambda value: print(value, flush=True))

    split_file = Path(os.environ.get("LPIA_SPLIT_FILE", "splits/lol_v1_seed42.json"))
    if not split_file.exists():
        raise FileNotFoundError(
            f"Strict ablation requires a split file. Generate it first: {split_file}"
        )

    ns["train_pairs"] = load_pairs_from_split(split_file, ns["CFG"].data_root, "train")
    ns["test_pairs"] = load_pairs_from_split(split_file, ns["CFG"].data_root, "val")
    final_test_pairs = load_pairs_from_split(split_file, ns["CFG"].data_root, "test")
    print(f"Using split file: {split_file} -> {split_summary(split_file)}", flush=True)
    print("Training uses split='train'; checkpoint selection uses split='val'.", flush=True)

    epochs = int(os.environ.get("LPIA_ABLATION_EPOCHS", ns["CFG"].epochs))
    variants = _parse_variants()
    output_tag = _output_tag(epochs, variants)
    records = []
    test_frames = []

    for variant in variants:
        print("=" * 80, flush=True)
        print(f"Strict ablation: {variant} ({epochs} epochs)", flush=True)
        ckpt_path = ns["CFG"].checkpoint_dir / f"{variant}_strict{epochs}_best.pt"
        _, hist = ns["fit_model"](variant=variant, epochs=epochs, checkpoint_path=ckpt_path)

        hist_path = ns["CFG"].output_dir / f"{variant}_strict{epochs}_history.csv"
        hist.to_csv(hist_path, index=False)
        best = hist.sort_values("val_psnr", ascending=False).iloc[0].to_dict()

        best_model = ns["load_checkpoint"](ckpt_path)
        test_df = ns["evaluate_deep_model"](
            best_model,
            final_test_pairs,
            method_name=variant,
        )
        test_df.insert(1, "variant", variant)
        test_path = ns["CFG"].output_dir / f"{variant}_strict{epochs}_test_eval.csv"
        test_df.to_csv(test_path, index=False)
        test_frames.append(test_df)
        test_mean = test_df[["psnr", "ssim", "time_ms", "mean_y_gain", "dark_y_gain", "over_exposed_ratio"]].mean()

        best.update(
            {
                "variant": variant,
                "description": ns["EXPERIMENTS"][variant],
                "epochs": epochs,
                "checkpoint": str(ckpt_path),
                "history": str(hist_path),
                "test_eval": str(test_path),
                "test_psnr": test_mean["psnr"],
                "test_ssim": test_mean["ssim"],
                "test_time_ms": test_mean["time_ms"],
                "test_mean_y_gain": test_mean["mean_y_gain"],
                "test_dark_y_gain": test_mean["dark_y_gain"],
                "test_over_exposed_ratio": test_mean["over_exposed_ratio"],
            }
        )
        records.append(best)

    ablation_df = ns["pd"].DataFrame(records)
    summary_path = ns["CFG"].output_dir / f"strict_ablation_{output_tag}_summary.csv"
    ablation_df.to_csv(summary_path, index=False)

    if test_frames:
        all_test_df = ns["pd"].concat(test_frames, ignore_index=True)
        all_test_path = ns["CFG"].output_dir / f"strict_ablation_{output_tag}_test_eval.csv"
        all_test_df.to_csv(all_test_path, index=False)
        print("Saved test details", all_test_path, flush=True)

    columns = [
        "variant",
        "epochs",
        "val_psnr",
        "val_ssim",
        "val_loss",
        "test_psnr",
        "test_ssim",
        "test_time_ms",
    ]
    print("Saved summary", summary_path, flush=True)
    print(ablation_df[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
