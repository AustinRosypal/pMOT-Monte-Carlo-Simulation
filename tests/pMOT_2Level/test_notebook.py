"""Structural checks for the effective two-level pMOT trajectory notebook."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT / "notebooks" / "pMOT_2Level" / "trajectory_simulation.ipynb"
)


def test_notebook_is_clean_and_x_only_cell_follows_full_controls() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(NOTEBOOK_PATH), "exec")

    ids = [cell["id"] for cell in notebook["cells"]]
    assert ids.index("x-only-pmot-diagnostic") == ids.index("interactive-controls") + 1

    source = "".join(
        notebook["cells"][ids.index("x-only-pmot-diagnostic")]["source"]
    )
    compile(source, str(NOTEBOOK_PATH), "exec")
    assert "if beam.axis_name == 'horizontal_x'" in source
    assert "len(cooling) != 6 or len(trapping) != 2" in source
    assert "path_powers_w=(1.0e-3 * trapping_power_mw, 0.0, 0.0)" in source
    assert "Run x-only trajectory" in source
    assert "plot_trajectory_3d" in source
    assert "plot_cartesian_position_force_history" in source
    assert "cartesian_position_force_history.png" in source
    assert "animate_trajectory" in source
