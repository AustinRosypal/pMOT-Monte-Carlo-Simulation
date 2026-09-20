from __future__ import annotations

import json
from pathlib import Path

import pytest


NOTEBOOKS = (
    "trajectory_explorer.ipynb",
    "trajectory_animation.ipynb",
    "capture_loading_explorer.ipynb",
)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_clean_valid_and_uses_physical_package(name: str) -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks" / "mot_multilevel" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "ipywidgets" in source
    assert "pmot.mot_multilevel" in source
    assert "mot_error" not in source
    compile("\n".join(line for line in source.splitlines() if not line.startswith("%")), str(path), "exec")

