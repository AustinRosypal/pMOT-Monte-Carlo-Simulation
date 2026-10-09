"""Structural and smoke checks for the interactive pMOT trajectory notebook."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pmot.mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from pmot.pmot.trajectory_plotting import plot_pmot_trajectory_diagnostics
from pmot.pmot.vector_only_trajectories import PMOTBeamHelicities
from pmot.pmot.vector_only_trajectories import build_vector_only_apparatus
from pmot.pmot.vector_only_trajectories import build_vector_only_trajectory_context
from pmot.pmot.vector_only_trajectories import inward_launch_state
from pmot.pmot.vector_only_trajectories import simulate_vector_only_pmot_trajectory
from pmot.pmot.vector_only_trajectories import vector_only_trajectory_dataframe
from pmot.pmot.vector_only_trajectory_diagnostics import (
    plot_vector_only_field_axis_polarization,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    vector_only_component_field_dataframe,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    vector_only_polarization_dataframe,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "pmot" / "trajectory_sampling.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_notebook_is_clean_valid_and_uses_the_pmot_kernel() -> None:
    notebook = _notebook()

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["display_name"] == (
        "Python 3 (pMOT Monte Carlo)"
    )
    assert "widgets" not in notebook["metadata"]
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"{NOTEBOOK_PATH}#cell-{index}", "exec")


def test_notebook_trajectory_helper_runs_short_local_axis_case(
    tmp_path,
    monkeypatch,
) -> None:
    notebook = _notebook()
    helper_source = next(
        cell["source"]
        for cell in notebook["cells"]
        if cell.get("id") == "trajectory-helpers"
    )
    namespace = {
        "Path": Path,
        "datetime": datetime,
        "perf_counter": perf_counter,
        "asdict": asdict,
        "replace": replace,
        "json": json,
        "np": np,
        "pd": pd,
        "plt": plt,
        "display": lambda *args, **kwargs: None,
        "OUTPUT_ROOT": tmp_path,
        "RateEquationTrajectoryConfig": RateEquationTrajectoryConfig,
        "plot_pmot_trajectory_diagnostics": plot_pmot_trajectory_diagnostics,
        "plot_vector_only_field_axis_polarization": (
            plot_vector_only_field_axis_polarization
        ),
        "vector_only_component_field_dataframe": (
            vector_only_component_field_dataframe
        ),
        "vector_only_polarization_dataframe": vector_only_polarization_dataframe,
        "PMOTBeamHelicities": PMOTBeamHelicities,
        "build_vector_only_apparatus": build_vector_only_apparatus,
        "build_vector_only_trajectory_context": build_vector_only_trajectory_context,
        "inward_launch_state": inward_launch_state,
        "simulate_vector_only_pmot_trajectory": simulate_vector_only_pmot_trajectory,
        "vector_only_trajectory_dataframe": vector_only_trajectory_dataframe,
    }
    exec(helper_source, namespace)
    monkeypatch.setattr(plt, "show", lambda: None)

    class _DisplayableManifest:
        @property
        def style(self):
            return self

        def format(self, *args, **kwargs):
            return self

    namespace["beam_manifest"] = lambda context: _DisplayableManifest()
    result = namespace["run_pmot_case"](
        radial_distance_mm=15.0,
        speed_m_per_s=0.0,
        duration_ms=0.005,
        time_step_us=2.5,
        plot_extent_mm=16.0,
        save_outputs=False,
    )

    assert result["output_directory"] is None
    sample_count = len(result["frame"])
    assert 3 <= sample_count <= 4
    assert len(result["component_field_frame"]) == sample_count
    assert result["polarization_frame"].shape == (sample_count, 55)
    assert np.asarray(
        result["record"].rate_equation.magnetic_fields_t
    ) == pytest.approx(np.zeros((sample_count, 3)), abs=0.0)
    plt.close("all")
