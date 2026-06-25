import os
from pathlib import Path

from lpia_project.data.split_manifest import load_pairs_from_split, split_summary
from lpia_project.notebook_runtime import load_notebook_namespace


def main():
    ns = load_notebook_namespace(display=lambda value: print(value, flush=True))
    split_file = Path(os.environ.get("LPIA_SPLIT_FILE", "splits/lol_v1_seed42.json"))
    if split_file.exists():
        ns["train_pairs"] = load_pairs_from_split(split_file, ns["CFG"].data_root, "train")
        ns["test_pairs"] = load_pairs_from_split(split_file, ns["CFG"].data_root, "val")
        print(f"Using split file for ablation validation: {split_file} -> {split_summary(split_file)}", flush=True)
    else:
        print(f"Split file not found, using notebook default pairs: {split_file}", flush=True)

    epochs = 10
    variants = [
        "plain_unet",
        "attention_unet",
        "fixed_gamma_unet",
        "learnable_gamma_unet",
        "lpia_lightunet",
        "lpia_former_lite",
        "zero_dce_lite",
    ]
    records = []
    for variant in variants:
        print("=" * 80, flush=True)
        print(f"Quick ablation: {variant} ({epochs} epochs)", flush=True)
        ckpt_path = ns["CFG"].checkpoint_dir / f"{variant}_quick{epochs}_best.pt"
        _, hist = ns["fit_model"](variant=variant, epochs=epochs, checkpoint_path=ckpt_path)
        hist_path = ns["CFG"].output_dir / f"{variant}_quick{epochs}_history.csv"
        hist.to_csv(hist_path, index=False)
        best = hist.sort_values("val_psnr", ascending=False).iloc[0].to_dict()
        best.update(
            {
                "variant": variant,
                "description": ns["EXPERIMENTS"][variant],
                "epochs": epochs,
                "checkpoint": str(ckpt_path),
                "history": str(hist_path),
            }
        )
        records.append(best)

    ablation_df = ns["pd"].DataFrame(records)
    out_path = ns["CFG"].output_dir / "ablation_results.csv"
    ablation_df.to_csv(out_path, index=False)
    print("Saved", out_path, flush=True)
    print(ablation_df[["variant", "epochs", "val_psnr", "val_ssim", "val_loss", "train_loss"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
