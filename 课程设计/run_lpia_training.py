import os
from pathlib import Path

from lpia_project.data.split_manifest import load_pairs_from_split, split_summary
from lpia_project.notebook_runtime import load_notebook_namespace


def main():
    ns = load_notebook_namespace(display=lambda value: print(value, flush=True))
    final_test_pairs = ns["test_pairs"]
    split_file = Path(os.environ.get("LPIA_SPLIT_FILE", "splits/lol_v1_seed42.json"))
    if split_file.exists():
        ns["train_pairs"] = load_pairs_from_split(split_file, ns["CFG"].data_root, "train")
        ns["test_pairs"] = load_pairs_from_split(split_file, ns["CFG"].data_root, "val")
        final_test_pairs = load_pairs_from_split(split_file, ns["CFG"].data_root, "test")
        print(f"Using split file: {split_file} -> {split_summary(split_file)}", flush=True)
        print("Validation uses split='val'; final evaluation uses split='test'.", flush=True)
    else:
        print(f"Split file not found, using notebook default pairs: {split_file}", flush=True)

    epochs = int(os.environ.get("LPIA_EPOCHS", ns["CFG"].epochs))
    print(f"Running LPIA-LightU-Net training for {epochs} epochs", flush=True)
    model, history = ns["fit_model"](
        variant="lpia_lightunet",
        epochs=epochs,
        checkpoint_path=ns["CFG"].best_ckpt,
    )
    print("Training finished", flush=True)
    print(history.tail().to_string(index=False), flush=True)

    print("Running final evaluation", flush=True)
    traditional_df = ns["evaluate_traditional_methods"](final_test_pairs)
    traditional_df.to_csv(ns["CFG"].output_dir / "traditional_eval.csv", index=False)
    deep_df = ns["evaluate_deep_model"](model, final_test_pairs)
    deep_df.to_csv(ns["CFG"].output_dir / "deep_eval.csv", index=False)
    all_eval_df = ns["pd"].concat([traditional_df, deep_df], ignore_index=True)
    all_eval_df.to_csv(ns["CFG"].output_dir / "all_eval.csv", index=False)
    print(all_eval_df.groupby("method")[["psnr", "ssim", "time_ms", "mean_y_gain", "dark_y_gain", "over_exposed_ratio"]].mean().to_string(), flush=True)
    print("All done", flush=True)


if __name__ == "__main__":
    main()
