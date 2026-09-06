"""Focused checks for the deterministic two-level detuning-force sweep."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pmot.mot_simple.simple_force_sweep as force_sweep
from pmot.configuration import default_mot_apparatus_config
from pmot.magnetic_fields import (
    anti_helmholtz_axial_gradient_t_per_m,
    default_anti_helmholtz_config,
)
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_simple.simple_force_sweep import (
    EVALUATION_COUNT,
    SimpleForceSweepNumerics,
    _parabolic_minimum,
    build_argument_parser,
    build_simple_force_sweep_configuration,
    detuning_n_grid,
    evaluate_simple_force_detuning_point,
    run_simple_force_detuning_sweep,
)


def _smoke_numerics() -> SimpleForceSweepNumerics:
    return SimpleForceSweepNumerics(
        position_step_m=0.2e-3,
        velocity_extent_m_per_s=30.0,
        velocity_step_m_per_s=0.5,
        restoring_relative_tolerance=0.1,
        turnaround_absolute_tolerance_m_per_s=0.2,
    )


def test_exact_refined_detuning_grid() -> None:
    values = detuning_n_grid()
    assert values.shape == (111,)
    assert values[0] == pytest.approx(-0.5)
    assert values[-1] == pytest.approx(-6.0)
    assert np.allclose(np.diff(values), -0.05, rtol=0.0, atol=1.0e-13)
    assert np.all(np.isfinite(values))
    assert np.all(values < 0.0)


def test_configuration_is_cooling_only_and_syncs_ordinary_hz() -> None:
    simple, apparatus, beams = build_simple_force_sweep_configuration(-1.25)
    expected_hz = -1.25 * simple.linewidth_hz
    assert simple.cooling_detuning_hz == pytest.approx(expected_hz)
    assert apparatus.cooling.detuning_hz == pytest.approx(expected_hz)
    assert not simple.include_gravity
    assert len(beams) == 6
    assert all(beam.detuning_hz == pytest.approx(expected_hz) for beam in beams)
    assert all(beam.intensity_beam.power_w == pytest.approx(27.0e-3) for beam in beams)
    assert all(2.0 * beam.intensity_beam.beam_radius_m == pytest.approx(12.7e-3) for beam in beams)

    base_simple = default_simple_mot_config()
    assert simple == replace(
        base_simple,
        cooling_detuning_hz=expected_hz,
        include_gravity=False,
    )
    base_apparatus = default_mot_apparatus_config()
    assert apparatus == replace(
        base_apparatus,
        cooling=replace(
            base_apparatus.cooling,
            detuning_hz=expected_hz,
            power_w_per_beam=27.0e-3,
        ),
    )

    coil = default_anti_helmholtz_config()
    gradient = anti_helmholtz_axial_gradient_t_per_m(
        coil.radius_m,
        coil.turns_per_coil,
        coil.current_a,
    )
    assert gradient == pytest.approx(0.1)  # 10 G/cm = 0.1 T/m


def test_parabolic_turnaround_refines_an_interior_minimum() -> None:
    speeds = np.linspace(0.0, 5.0, 11)
    forces = (speeds - 2.3) ** 2 - 7.0
    speed, force, interior = _parabolic_minimum(speeds, forces)
    assert interior
    assert speed == pytest.approx(2.3, abs=1.0e-12)
    assert force == pytest.approx(-7.0, abs=1.0e-12)

    boundary_speed, _, boundary_interior = _parabolic_minimum(speeds, -speeds)
    assert boundary_speed == 5.0
    assert not boundary_interior


def test_real_simple_force_point_is_restoring_damping_and_converged() -> None:
    row = evaluate_simple_force_detuning_point(0, -1.0, SimpleForceSweepNumerics())
    assert row["deterministic_evaluation_count"] == EVALUATION_COUNT == 1
    assert row["detuning_hz"] == pytest.approx(-row["linewidth_hz"])
    assert row["cooling_power_w_per_beam"] == pytest.approx(27.0e-3)
    assert row["cooling_beam_diameter_m"] == pytest.approx(12.7e-3)
    for axis in "xyz":
        assert row[f"restoring_slope_{axis}_n_per_m"] < 0.0
        assert 0.0 < row[f"turnaround_velocity_{axis}_m_per_s"] < 50.0
        assert row[f"turnaround_force_{axis}_n"] < 0.0
        assert row[f"turnaround_{axis}_interior"]
        assert row[f"restoring_slope_{axis}_converged"]
        assert row[f"turnaround_{axis}_converged"]
        assert row[f"restoring_slope_{axis}_numerical_uncertainty_n_per_m"] >= 0.0
        assert row[f"turnaround_velocity_{axis}_numerical_uncertainty_m_per_s"] >= 0.0
    assert row["restoring_slope_x_n_per_m"] == pytest.approx(
        row["restoring_slope_y_n_per_m"],
        rel=1.0e-12,
    )
    assert row["restoring_slope_z_n_per_m"] == pytest.approx(
        2.0 * row["restoring_slope_x_n_per_m"],
        rel=5.0e-4,
    )
    assert row["turnaround_velocity_x_m_per_s"] == pytest.approx(
        row["turnaround_velocity_z_m_per_s"],
        rel=1.0e-12,
    )
    assert row["all_converged"]


def test_checkpoint_resume_metadata_and_plots(tmp_path, monkeypatch) -> None:
    output = tmp_path / "statistics"
    figures = tmp_path / "figures"
    first = run_simple_force_detuning_sweep(
        detuning_n_values=(-1.0,),
        numerics=_smoke_numerics(),
        output_directory=output,
        figure_directory=figures,
        resume=True,
    )
    assert first["status"] == "completed"
    assert first["completed_point_count"] == 1
    assert all(Path(path).is_file() for path in first["outputs"].values())

    metadata = json.loads((output / "force_vs_detuning_metadata.json").read_text())
    assert metadata["frequency_unit"] == "ordinary Hz"
    assert metadata["cooling_power_mw_per_beam"] == pytest.approx(27.0)
    assert metadata["cooling_component_count"] == 6
    assert not metadata["repumper_enabled"]
    assert metadata["repump_component_count"] == 0
    assert metadata["beam_diameter_mm"] == pytest.approx(12.7)
    assert metadata["quadrupole_axial_gradient_g_per_cm"] == pytest.approx(10.0)
    assert not metadata["gravity_included_in_force"]
    assert metadata["all_convergence_checks_passed"]
    assert "not a force zero crossing" in metadata["turnaround_definition"]
    assert metadata["statistical_uncertainty_applicable"] is False
    assert metadata["resume_signature"]["repump_component_count"] == 0

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("resume should not recompute completed points")

    monkeypatch.setattr(
        force_sweep,
        "evaluate_simple_force_detuning_point",
        fail_if_recomputed,
    )
    resumed = run_simple_force_detuning_sweep(
        detuning_n_values=(-1.0,),
        numerics=_smoke_numerics(),
        output_directory=output,
        figure_directory=figures,
        resume=True,
    )
    assert resumed["status"] == "completed"
    assert resumed["completed_point_count"] == 1


def test_plots_explicitly_annotate_coincident_axes(tmp_path, monkeypatch) -> None:
    from matplotlib.axes import Axes

    force_sweep.plt.switch_backend("Agg")
    row: dict[str, object] = {"detuning_n": -1.0}
    for axis in "xyz":
        row[f"restoring_slope_{axis}_n_per_m"] = -2.0e-20
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
    force_sweep.plot_simple_restoring_slopes_vs_detuning(
        [row], tmp_path / "restoring.png"
    )
    force_sweep.plot_simple_damping_turnarounds_vs_detuning(
        [row], tmp_path / "turnaround.png"
    )
    assert any("x=y" in annotation for annotation in annotations)
    assert any("x=y=z" in annotation for annotation in annotations)
    assert sum("not statistical error bars" in annotation for annotation in annotations) == 2


def test_cli_accepts_explicit_no_resume() -> None:
    args = build_argument_parser().parse_args(["--no-resume"])
    assert not args.resume
