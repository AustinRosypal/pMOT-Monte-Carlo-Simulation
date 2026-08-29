from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pmot.state import AtomState
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_simple.sampling import build_argument_parser
from pmot.mot_simple.sampling import CaptureSearchConfig
from pmot.mot_simple.sampling import DiscSample
from pmot.mot_simple.sampling import PointSample
from pmot.mot_simple.sampling import classify_trajectory
from pmot.mot_simple.sampling import plot_disc_plane_view
from pmot.mot_simple.sampling import sample_disc_points
from pmot.mot_simple.sampling import search_config_from_args
from pmot.mot_simple.sampling import summarize_capture_velocity_samples


def _point(position_m: tuple[float, float, float] = (15.0e-3, 0.0, 0.0)) -> PointSample:
    return PointSample(
        disc_index=0,
        point_index=0,
        theta_rad=0.0,
        phi_rad=0.0,
        theta_prime_rad=0.0,
        s_m=0.0,
        radial_distance_m=15.0e-3,
        initial_position_m=position_m,
        incident_unit_vector=(-1.0, 0.0, 0.0),
        launch_axis_unit_vector=(-1.0, 0.0, 0.0),
    )


def test_two_core_entries_with_intervening_exit_are_trapped(monkeypatch) -> None:
    positions = iter(
        [
            (1.0e-3, 0.0, 0.0),
            (3.0e-3, 0.0, 0.0),
            (1.0e-3, 0.0, 0.0),
        ]
    )

    def fake_rk4_step(beams, atom_state, time_step_s, coil_config, simple_config):
        return AtomState(next(positions), atom_state.velocity_m_per_s), (0.0, 0.0, 0.0), [], (0.0, 0.0, 0.0)

    monkeypatch.setattr("pmot.mot_simple.sampling.rk4_step", fake_rk4_step)
    config = replace(
        CaptureSearchConfig(),
        time_step_s=1.0,
        max_simulation_time_s=4.0,
        required_core_entries=2,
        bounded_core_residence_s=10.0,
    )
    result = classify_trajectory([], _point(), 1.0, object(), default_simple_mot_config(), config)
    assert result.trapped
    assert result.termination_reason == "two_core_entries"
    assert result.core_entry_count == 2


def test_continuous_five_ms_core_residence_is_trapped(monkeypatch) -> None:
    calls: list[float] = []

    def fake_rk4_step(beams, atom_state, time_step_s, coil_config, simple_config):
        calls.append(time_step_s)
        return atom_state, (0.0, 0.0, 0.0), [], (0.0, 0.0, 0.0)

    monkeypatch.setattr("pmot.mot_simple.sampling.rk4_step", fake_rk4_step)
    config = replace(
        CaptureSearchConfig(),
        time_step_s=1.0e-3,
        max_simulation_time_s=6.0e-3,
    )
    result = classify_trajectory(
        [],
        _point((1.0e-3, 0.0, 0.0)),
        0.0,
        object(),
        default_simple_mot_config(),
        config,
    )
    assert result.trapped
    assert result.termination_reason == "bounded_core_residence"
    assert result.core_entry_count == 1
    assert result.elapsed_time_s == pytest.approx(5.0e-3)
    assert len(calls) == 5


def test_core_exit_resets_continuous_residence_clock(monkeypatch) -> None:
    positions = iter(
        [
            (1.0e-3, 0.0, 0.0),
            (1.0e-3, 0.0, 0.0),
            (3.0e-3, 0.0, 0.0),
            (1.0e-3, 0.0, 0.0),
            (1.0e-3, 0.0, 0.0),
            (1.0e-3, 0.0, 0.0),
            (1.0e-3, 0.0, 0.0),
        ]
    )

    def fake_rk4_step(beams, atom_state, time_step_s, coil_config, simple_config):
        return AtomState(next(positions), atom_state.velocity_m_per_s), (0.0, 0.0, 0.0), [], (0.0, 0.0, 0.0)

    monkeypatch.setattr("pmot.mot_simple.sampling.rk4_step", fake_rk4_step)
    config = replace(
        CaptureSearchConfig(),
        time_step_s=1.0e-3,
        max_simulation_time_s=7.0e-3,
        required_core_entries=99,
    )
    result = classify_trajectory([], _point(), 0.0, object(), default_simple_mot_config(), config)
    assert not result.trapped
    assert result.termination_reason == "timeout"
    assert result.core_entry_count == 2
    assert result.elapsed_time_s == pytest.approx(7.0e-3)


def test_disc_points_are_independent_uniform_area_draws_without_forced_endpoints() -> None:
    disc = DiscSample(
        disc_index=3,
        theta_rad=0.0,
        phi_rad=0.0,
        outward_unit_vector=(0.0, 0.0, 1.0),
        incident_unit_vector=(0.0, 0.0, -1.0),
        center_position_m=(0.0, 0.0, 15.0e-3),
        basis_u=(1.0, 0.0, 0.0),
        basis_v=(0.0, 1.0, 0.0),
    )
    radius_m = 12.0e-3
    points = sample_disc_points(disc, 10_000, radius_m, False, np.random.default_rng(739))
    radii = np.asarray([point.s_m for point in points])
    area_fractions = (radii / radius_m) ** 2

    assert len(points) == 10_000
    assert np.all(radii > 0.0)
    assert np.all(radii < radius_m)
    assert float(np.mean(area_fractions)) == pytest.approx(0.5, abs=0.01)
    histogram, _ = np.histogram(area_fractions, bins=np.linspace(0.0, 1.0, 11))
    assert np.all(np.abs(histogram - 1_000) < 120)

    independent_points = sample_disc_points(
        disc,
        32,
        radius_m,
        False,
        np.random.default_rng(739),
    )
    legacy_center_points = sample_disc_points(
        disc,
        32,
        radius_m,
        True,
        np.random.default_rng(739),
    )
    assert independent_points[0].s_m > 0.0
    assert legacy_center_points[0].s_m == 0.0
    assert all(point.s_m < radius_m for point in legacy_center_points)


def test_core_residence_cli_value_is_persisted_in_summary() -> None:
    args = build_argument_parser().parse_args(["--required-core-residence-ms", "7.5"])
    config = search_config_from_args(args)
    summary = summarize_capture_velocity_samples([], config)
    assert config.bounded_core_residence_s == pytest.approx(7.5e-3)
    assert summary["search_config"]["bounded_core_residence_s"] == pytest.approx(7.5e-3)


def test_disc_plane_view_uses_expected_output_name(tmp_path) -> None:
    disc = DiscSample(
        disc_index=7,
        theta_rad=0.0,
        phi_rad=0.0,
        outward_unit_vector=(0.0, 0.0, 1.0),
        incident_unit_vector=(0.0, 0.0, -1.0),
        center_position_m=(0.0, 0.0, 15.0e-3),
        basis_u=(1.0, 0.0, 0.0),
        basis_v=(0.0, 1.0, 0.0),
    )
    point = PointSample(
        disc_index=7,
        point_index=0,
        theta_rad=0.0,
        phi_rad=0.0,
        theta_prime_rad=0.0,
        s_m=0.0,
        radial_distance_m=15.0e-3,
        initial_position_m=disc.center_position_m,
        incident_unit_vector=disc.incident_unit_vector,
        launch_axis_unit_vector=disc.incident_unit_vector,
    )
    path = plot_disc_plane_view(disc, [point], [], tmp_path)
    assert path == tmp_path / "disc_0007_plane_view.png"
    assert path.is_file()
