from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairRecord:
    image_id: str
    low: str
    high: str


@dataclass(frozen=True)
class LolSplit:
    dataset: str
    seed: int
    train_subset: str
    test_subset: str
    train_count: int
    val_count: int
    test_count: int
    train: list[PairRecord]
    val: list[PairRecord]
    test: list[PairRecord]


def _image_files(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    files: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files[path.stem] = path
    return files


def find_pairs(data_root: Path, subset: str) -> list[PairRecord]:
    low_dir = data_root / subset / "low"
    high_dir = data_root / subset / "high"
    low_files = _image_files(low_dir)
    high_files = _image_files(high_dir)
    common_ids = sorted(set(low_files) & set(high_files), key=_natural_key)

    missing_high = sorted(set(low_files) - set(high_files), key=_natural_key)
    missing_low = sorted(set(high_files) - set(low_files), key=_natural_key)
    if missing_high or missing_low:
        raise ValueError(
            "Low/high image ids do not match. "
            f"missing_high={missing_high[:5]}, missing_low={missing_low[:5]}"
        )

    return [
        PairRecord(
            image_id=image_id,
            low=_relative_posix(low_files[image_id], data_root),
            high=_relative_posix(high_files[image_id], data_root),
        )
        for image_id in common_ids
    ]


def build_split(
    data_root: Path,
    train_subset: str = "our485",
    test_subset: str = "eval15",
    val_ratio: float = 0.1031,
    seed: int = 42,
) -> LolSplit:
    train_val_pairs = find_pairs(data_root, train_subset)
    test_pairs = find_pairs(data_root, test_subset)
    shuffled = list(train_val_pairs)
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, round(len(shuffled) * val_ratio))
    val_pairs = sorted(shuffled[:val_count], key=lambda record: _natural_key(record.image_id))
    train_pairs = sorted(shuffled[val_count:], key=lambda record: _natural_key(record.image_id))

    return LolSplit(
        dataset="LOL-v1",
        seed=seed,
        train_subset=train_subset,
        test_subset=test_subset,
        train_count=len(train_pairs),
        val_count=len(val_pairs),
        test_count=len(test_pairs),
        train=train_pairs,
        val=val_pairs,
        test=test_pairs,
    )


def save_split(split: LolSplit, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(split)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_split_csv(split: LolSplit, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "image_id", "low", "high"])
        writer.writeheader()
        for split_name, records in _iter_split_records(split):
            for record in records:
                writer.writerow({"split": split_name, **asdict(record)})


def _iter_split_records(split: LolSplit) -> Iterable[tuple[str, list[PairRecord]]]:
    yield "train", split.train
    yield "val", split.val
    yield "test", split.test


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _natural_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible LOL-v1 train/val/test split.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--train-subset", default="our485")
    parser.add_argument("--test-subset", default="eval15")
    parser.add_argument("--val-ratio", type=float, default=0.1031)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("splits/lol_v1_seed42.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("splits/lol_v1_seed42.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = build_split(
        data_root=args.data_root,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    save_split(split, args.output)
    save_split_csv(split, args.csv_output)
    print(
        f"Saved split to {args.output} "
        f"(train={split.train_count}, val={split.val_count}, test={split.test_count})"
    )


if __name__ == "__main__":
    main()
