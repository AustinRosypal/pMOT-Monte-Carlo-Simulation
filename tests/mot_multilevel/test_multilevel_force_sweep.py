"""Focused checks for the multilevel detuning-force sweep."""

from __future__ import annotations

import json
from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np
import pytest

import pmot.mot_multilevel.force_sweep as force_sweep
from pmot.configuration import default_simulation_config
from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.force_sweep import (
    EVALUATION_COUNT,
    ForceSweepNumerics,
    _parabolic_minimum,
    build_argument_parser,
    build_force_sweep_configuration,
    detuning_n_grid,
    evaluate_force_detuning_point,
    run_force_detuning_sweep,
)


def _smoke_numerics() -> ForceSweepNumerics:
    return ForceSweepNumerics(
        position_step_m=0.2e-3,
        velocity_extent_m_per_s=8.0,
        velocity_step_m_per_s=1.0,
        restoring_relative_tolerance=0.1,
        turnaround_absolute_tolerance_m_per_s=0.2,
    )


def test_detuning_grid_has_25_finite_red_points_and_default_detuning() -> None:
    values = detuning_n_grid()
    assert values.shape == (25,)
    assert np.all(np.isfinite(values))
    assert np.all(values < 0.0)
    assert np.all(np.diff(values) < 0.0)
    assert values[0] == pytest.approx(-0.5)
    assert values[-1] == pytest.approx(-10.0)
    assert np.allclose(np.diff(values), -9.5 / 24.0)


def test_force_configuration_syncs_angular_and_ordinary_detuning() -> None:
    config, apparatus, beams = build_force_sweep_configuration(-1.25)
    expected_rad_per_s = -1.25 * config.natural_linewidth_rad_per_s
    expected_hz = expected_rad_per_s / (2.0 * pi)
    assert config.repumper_enabled
    assert config.cooling_detuning_rad_per_s == expected_rad_per_s
    assert apparatus.cooling.detuning_hz == expected_hz
    cooling_beams = [beam for beam in beams if beam.family == "cooling"]
    repump_beams = [beam for beam in beams if beam.family == "repump"]
    assert len(cooling_beams) == 6
    assert len(repump_beams) == 6
    assert all(beam.detuning_hz == expected_hz for beam in cooling_beams)

    base_config = default_multilevel_mot_config()
    assert config == replace(
        base_config,
        cooling_detuning_rad_per_s=expected_rad_per_s,
        repumper_enabled=True,
    )
    base_apparatus = default_simulation_config()
    assert apparatus == replace(
        base_apparatus,
        cooling=replace(base_apparatus.cooling, detuning_hz=expected_hz),
    )


def test_parabolic_turnaround_is_an_interior_force_minimum() -> None:
    velocities = np.linspace(0.0, 5.0, 11)
    forces = (velocities - 2.3) ** 2 - 7.0
    velocity, force, interior = _parabolic_minimum(velocities, forces)
    assert interior
    assert velocity == pytest.approx(2.3, abs=1.0e-12)
    assert force == pytest.approx(-7.0, abs=1.0e-12)

    boundary_velocity, _, boundary_interior = _parabolic_minimum(
        velocities,
        -velocities,
    )
    assert boundary_velocity == 5.0
    assert not boundary_interior


def test_real_rate_equation_smoke_has_restoring_slopes_and_turnarounds() -> None:
    row = evaluate_force_detuning_point(0, -1.0, _smoke_numerics())
    assert row["deterministic_evaluation_count"] == EVALUATION_COUNT == 1
    assert row["model_state_count"] == 24  # current 23-state cooling spec plus repump F'=0
    assert row["detuning_rad_per_s"] == pytest.approx(
        -row["linewidth_rad_per_s"]
    )
    for axis in "xyz":
        assert row[f"restoring_slope_{axis}_n_per_m"] < 0.0
        assert 0.0 < row[f"turnaround_velocity_{axis}_m_per_s"] < 8.0
        assert row[f"turnaround_force_{axis}_n"] < 0.0
        assert row[f"turnaround_{axis}_interior"]
        assert row[f"restoring_slope_{axis}_relative_change"] >= 0.0
        assert row[f"restoring_slope_{axis}_numerical_uncertainty_n_per_m"] >= 0.0
        assert row[f"turnaround_velocity_{axis}_absolute_change_m_per_s"] >= 0.0
        assert row[f"turnaround_velocity_{axis}_numerical_uncertainty_m_per_s"] >= 0.0
    assert row["restoring_slope_x_n_per_m"] == pytest.approx(
        row["restoring_slope_y_n_per_m"],
        rel=1.0e-12,
    )
    assert row["turnaround_velocity_x_m_per_s"] == pytest.approx(
        row["turnaround_velocity_y_m_per_s"],
        rel=1.0e-12,
    )
    assert row["all_converged"]


def test_strong_detuning_turnaround_is_interior_and_converged() -> None:
    row = evaluate_force_detuning_point(0, -10.0, ForceSweepNumerics())
    for axis in "xyz":
        assert 0.0 < row[f"turnaround_velocity_{axis}_m_per_s"] < 50.0
        assert row[f"turnaround_{axis}_interior"]
        assert row[f"turnaround_{axis}_converged"]


def test_checkpoint_resume_and_plotting_smoke(tmp_path, monkeypatch) -> None:
    output = tmp_path / "statistics"
    figures = tmp_path / "figures"
    first = run_force_detuning_sweep(
        detuning_n_values=(-1.0,),
        numerics=_smoke_numerics(),
        output_directory=output,
        figure_directory=figures,
        resume=True,
    )
    assert first["status"] == "completed"
    assert first["completed_point_count"] == 1
    for path in first["outputs"].values():
        assert Path(path).is_file()

    metadata = json.loads((output / "force_vs_detuning_metadata.json").read_text())
    assert metadata["repumper_enabled"]
    assert not metadata["gravity_included_in_force"]
    assert "not a force zero crossing" in metadata["turnaround_definition"]
    assert metadata["model_state_count"] == 24
    assert metadata["cooling_only_specification_state_count"] == 23
    assert metadata["deterministic_evaluation_count_per_point"] == 1
    assert metadata["total_deterministic_evaluations"] == 1
    assert metadata["resume_signature"]["deterministic_evaluation_count_per_point"] == 1
    assert metadata["statistical_uncertainty_applicable"] is False
    assert metadata["resume_signature"]["multilevel_config"]["repumper_enabled"]
    assert "coil_config" in metadata["resume_signature"]

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("resume should not recompute a completed detuning")

    monkeypatch.setattr(force_sweep, "evaluate_force_detuning_point", fail_if_recomputed)
    resumed = run_force_detuning_sweep(
        detuning_n_values=(-1.0,),
        numerics=_smoke_numerics(),
        output_directory=output,
        figure_directory=figures,
        resume=True,
    )
    assert resumed["status"] == "completed"
    assert resumed["completed_point_count"] == 1


def test_plots_annotate_all_axis_overlap(tmp_path, monkeypatch) -> None:
    from matplotlib.axes import Axes

    force_sweep.plt.switch_backend("Agg")
    row: dict[str, object] = {
        "detuning_n": -1.0,
        "linewidth_rad_per_s": 2.0 * pi * 6.07e6,
    }
    for axis in "xyz":
        row[f"restoring_slope_{axis}_n_per_m"] = -2.0e-19
        row[f"restoring_slope_{axis}_converged"] = True
        row[f"turnaround_velocity_{axis}_m_per_s"] = 5.0
        row[f"turnaround_{axis}_converged"] = True

    annotations: list[str] = []
    original_text = Axes.text

    def capture_text(self, *args, **kwargs):
        if len(args) >= 3:
            annotations.append(str(args[2]))
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture_text)
    force_sweep.plot_restoring_slopes_vs_detuning(
        [row], tmp_path / "restoring.png"
    )
    force_sweep.plot_damping_turnarounds_vs_detuning(
        [row], tmp_path / "turnaround.png"
    )
    assert sum("x=y=z" in annotation for annotation in annotations) == 2


def test_cli_accepts_explicit_no_resume() -> None:
    args = build_argument_parser().parse_args(["--no-resume"])
    assert not args.resume
