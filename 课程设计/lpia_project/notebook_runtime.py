from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import nbformat


DisplayFn = Callable[[Any], None]


def load_notebook_namespace(
    notebook_path: str | Path = "LPIA_LightUNet_low_light_enhancement.ipynb",
    display: DisplayFn | None = None,
) -> dict[str, Any]:
    """Execute code cells from the project notebook into a reusable namespace."""
    path = Path(notebook_path)
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "display": display or (lambda value: None),
    }
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        exec(compile(cell.source, f"{path}:cell{index}", "exec"), namespace)
    from lpia_project.model_registry import register_project_models

    register_project_models(namespace)
    return namespace
