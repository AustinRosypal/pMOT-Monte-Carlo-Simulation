from __future__ import annotations

import json
from pathlib import Path

import pytest


NOTEBOOKS = (
    "trajectory_explorer.ipynb",
    "trajectory_animation.ipynb",
    "capture_loading_explorer.ipynb",
)

WIDGET_NOTEBOOKS = (
    "trajectory_explorer.ipynb",
    "trajectory_animation.ipynb",
)


def _code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_clean_valid_and_uses_physical_package(name: str) -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks" / "mot_multilevel" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = _code_source(notebook)
    assert "pmot.mot_multilevel" in source
    assert "mot_error" not in source
    compile("\n".join(line for line in source.splitlines() if not line.startswith("%")), str(path), "exec")


@pytest.mark.parametrize("name", WIDGET_NOTEBOOKS)
def test_interactive_explorer_notebook_uses_widgets(name: str) -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks" / "mot_multilevel" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert "ipywidgets" in _code_source(notebook)


def test_capture_loading_notebook_is_configurable_and_headless() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks" / "mot_multilevel" / "capture_loading_explorer.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = _code_source(notebook)
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "ipywidgets" not in source
    assert "PHYSICS = {" in source
    assert "CAPTURE = {" in source
    assert "RESULT = run_capture_loading_study(" in source
    assert "jupyter nbconvert" in markdown

    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell_source = "".join(cell["source"])
        if "RESULT = run_capture_loading_study(" in cell_source:
            break
        executable = "\n".join(
            line for line in cell_source.splitlines() if not line.startswith("%")
        )
        exec(compile(executable, str(path), "exec"), namespace)
    search = namespace["SEARCH_CONFIG"]
    assert search.disc_count == namespace["CAPTURE"]["direction_disc_count"]
    assert search.points_per_disc == namespace["CAPTURE"]["points_per_disc"]
    assert namespace["RUN_NAME"] == namespace["CAPTURE"]["run_name"]


def test_configured_trajectory_lab_uses_portable_kernel_and_full_animation() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks" / "mot_multilevel" / "configured_trajectory_lab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = _code_source(notebook)
    assert "to_html5_video()" in source
    assert "fallback_frames = min(requested_frames, 40)" in source
    assert "'specified_disc'" in source
    assert "'launch_polar_angle_deg'" in source
    assert "'launch_azimuthal_angle_deg'" in source
    assert "'impact_parameter_mm'" in source
    assert "'impact_azimuth_deg'" in source
    assert "build_incident_disc_from_angles" in source
    assert "pmot-env" not in json.dumps(notebook)
    compile(
        "\n".join(line for line in source.splitlines() if not line.startswith("%")),
        str(path),
        "exec",
    )

    namespace: dict[str, object] = {}
    for cell_index in (2, 4, 6, 8):
        cell_source = "".join(notebook["cells"][cell_index]["source"])
        executable = "\n".join(
            line for line in cell_source.splitlines() if not line.startswith("%")
        )
        exec(compile(executable, str(path), "exec"), namespace)

    simulation = dict(namespace["SIMULATION"])
    simulation.update({
        "launch_mode": "specified_disc",
        "launch_radius_mm": 15.0,
        "launch_speed_m_per_s": 3.0,
        "launch_polar_angle_deg": 90.0,
        "launch_azimuthal_angle_deg": 0.0,
        "impact_parameter_mm": 2.0,
        "impact_azimuth_deg": 0.0,
    })
    *_, states, metadata = namespace["build_notebook_configuration"](
        namespace["PHYSICS"], simulation
    )
    assert len(states) == 1
    assert states[0].position_m == pytest.approx((0.015, 0.002, 0.0), abs=1.0e-12)
    assert states[0].velocity_m_per_s == pytest.approx((-3.0, 0.0, 0.0), abs=1.0e-12)
    assert metadata[0]["mode"] == "specified_disc"
    assert metadata[0]["impact_radius_m"] == pytest.approx(0.002)


def test_mot_with_pmot_fields_notebook_uses_physical_solver_and_surrogate_field() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "notebooks"
        / "mot_multilevel"
        / "mot_with_pmot_fields_trajectory_lab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    source = _code_source(notebook)
    assert "pmot.mot_multilevel" in source
    assert "surrogate_effective_field_t" in source
    assert "inside_surrogate_cell" in source
    assert "magnetic_field_function=effective_field" in source
    assert "spatial_domain_function=inside_field_domain" in source
    assert "coil_config=None" in source
    assert "create_trajectory_animation" in source
    assert "to_html5_video()" in source
    assert "fallback_frames = min(requested_frames, 40)" in source
    assert "axis.set_yscale('symlog'" in source
    assert "'field_plot_symlog_linthresh_g'" in source
    assert "'manual'" in source
    assert "'specified_disc'" in source
    assert "'random_full_sphere'" in source
    assert "simulate_provisional_pmot_trajectory" not in source
    assert "simulate_vector_only_pmot_trajectory" not in source
    assert "mot_error" not in source
    assert "pmot-env" not in json.dumps(notebook)
    compile(
        "\n".join(line for line in source.splitlines() if not line.startswith("%")),
        str(path),
        "exec",
    )

