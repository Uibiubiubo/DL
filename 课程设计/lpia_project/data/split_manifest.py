from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


SplitName = Literal["train", "val", "test"]


def load_pairs_from_split(
    split_path: str | Path,
    data_root: str | Path,
    split_name: SplitName,
) -> list[tuple[Path, Path]]:
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    root = Path(data_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if split_name not in payload:
        raise KeyError(f"Split '{split_name}' not found in {path}")

    pairs: list[tuple[Path, Path]] = []
    for record in payload[split_name]:
        pairs.append((root / record["low"], root / record["high"]))
    return pairs


def split_summary(split_path: str | Path) -> dict[str, int]:
    path = Path(split_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train": len(payload.get("train", [])),
        "val": len(payload.get("val", [])),
        "test": len(payload.get("test", [])),
    }
