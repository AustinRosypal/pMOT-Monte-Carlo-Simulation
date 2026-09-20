"""Geometry and record-contract tests for pMOT trajectory plots."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pmot.pmot.configuration import build_pmot_cooling_and_repump_beams
from pmot.pmot.configuration import default_pmot_apparatus_config
from pmot.pmot.trajectory_plotting import draw_pmot_beam_volumes
from pmot.pmot.trajectory_plotting import gaussian_trapping_surface_mesh_mm
from pmot.pmot.trajectory_plotting import plot_pmot_trajectory_diagnostics
from pmot.pmot.trajectory_plotting import pmot_beam_legend_handles
from pmot.pmot.trapping_beams import beams_for_trapping_axis
from pmot.pmot.trapping_beams import build_trapping_beams


@pytest.fixture
def optical_beams():
    apparatus = default_pmot_apparatus_config()
    mot_beams = build_pmot_cooling_and_repump_beams(apparatus)
    trapping_beams = build_trapping_beams(apparatus.trapping_laser)
    return mot_beams, trapping_beams


@pytest.mark.parametrize("sense, expected_waist_x_mm", (("incident", -10.0), ("retro", 10.0)))
def test_gaussian_surface_has_true_waist_and_rayleigh_divergence(
    optical_beams,
    sense: str,
    expected_waist_x_mm: float,
) -> None:
    _, trapping_beams = optical_beams
    pair = beams_for_trapping_axis(trapping_beams, "horizontal_x")
    beam = next(item for item in pair if item.propagation_sense == sense)
    axial_extent_m = 20.0e-3
    axial_samples = 81
    x_mm, y_mm, z_mm = gaussian_trapping_surface_mesh_mm(
        beam,
        axial_extent_m=axial_extent_m,
        axial_samples=axial_samples,
        angular_samples=17,
    )

    assert x_mm.shape == y_mm.shape == z_mm.shape == (axial_samples, 17)
    coordinates_m = np.linspace(-axial_extent_m, axial_extent_m, axial_samples)
    direction = np.asarray(beam.direction, dtype=float)
    waist_coordinate_m = float(np.dot(beam.waist_position_m, direction))
    waist_index = int(np.argmin(np.abs(coordinates_m - waist_coordinate_m)))
    points_at_waist_m = 1.0e-3 * np.column_stack(
        (x_mm[waist_index], y_mm[waist_index], z_mm[waist_index])
    )
    radii_at_waist_m = np.linalg.norm(
        points_at_waist_m - np.asarray(beam.waist_position_m),
        axis=1,
    )
    np.testing.assert_allclose(
        radii_at_waist_m,
        beam.waist_radius_m,
        rtol=2.0e-12,
        atol=1.0e-15,
    )
    assert 1.0e3 * beam.waist_position_m[0] == pytest.approx(expected_waist_x_mm)

    origin_index = int(np.argmin(np.abs(coordinates_m)))
    points_at_origin_m = 1.0e-3 * np.column_stack(
        (x_mm[origin_index], y_mm[origin_index], z_mm[origin_index])
    )
    origin_radii_m = np.linalg.norm(points_at_origin_m, axis=1)
    expected_origin_radius_m = beam.waist_radius_m * np.sqrt(
        1.0 + (10.0e-3 / beam.rayleigh_range_m) ** 2
    )
    np.testing.assert_allclose(
        origin_radii_m,
        expected_origin_radius_m,
        rtol=2.0e-12,
        atol=1.0e-15,
    )


def test_true_scale_renderer_draws_three_cooling_paths_and_six_focused_components(
    optical_beams,
) -> None:
    mot_beams, trapping_beams = optical_beams
    cooling = [beam for beam in mot_beams if beam.family == "cooling"]
    assert len({beam.beam_radius_m for beam in cooling}) == 1
    assert cooling[0].beam_radius_m == pytest.approx(6.35e-3)

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    artists = draw_pmot_beam_volumes(
        axis,
        mot_beams,
        trapping_beams,
        axial_extent_m=18.0e-3,
        cooling_axial_samples=5,
        trapping_axial_samples=9,
        angular_samples=8,
    )
    assert len(artists.cooling_surfaces) == 3
    assert len(artists.trapping_surfaces) == 6
    assert len(artists.waist_markers) == 6
    assert len(artists.direction_arrows) == 6
    plt.close(figure)


def test_geometry_legend_identifies_shared_cooling_repump_volume() -> None:
    labels = [artist.get_label() for artist in pmot_beam_legend_handles()]
    assert labels[0] == "780 nm cooling/repump volume (12.7 mm diameter)"


def test_combined_trajectory_plot_accepts_nested_rate_record_and_saves(
    optical_beams,
    tmp_path,
) -> None:
    mot_beams, trapping_beams = optical_beams
    sample_count = 3
    base = SimpleNamespace(
        times_s=[0.0, 1.0e-6, 2.0e-6],
        positions_m=[(15.0e-3, 0.0, 0.0), (14.98e-3, 0.0, 0.0), (14.96e-3, 0.0, 0.0)],
        velocities_m_per_s=[(-17.0, 0.0, 0.0)] * sample_count,
        forces_n=[(1.0e-21, 0.0, 0.0)] * sample_count,
        total_scattering_rates_per_s=[2.0e6] * sample_count,
        beam_scattering_rates_per_s=[tuple([2.0e6 / len(mot_beams)] * len(mot_beams))] * sample_count,
    )
    record = SimpleNamespace(rate_equation=base)
    path = tmp_path / "trajectory.png"
    figure, axes = plot_pmot_trajectory_diagnostics(
        record,
        mot_beams,
        trapping_beams,
        path=path,
        axial_extent_m=16.0e-3,
    )

    assert path.is_file()
    assert set(axes) == {"trajectory", "position", "velocity", "force", "scattering"}
    assert len(axes["trajectory"].lines) >= 1
    plt.close(figure)


def test_geometry_renderer_rejects_incomplete_beam_sets(optical_beams) -> None:
    mot_beams, trapping_beams = optical_beams
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    with pytest.raises(ValueError, match="six trapping components"):
        draw_pmot_beam_volumes(axis, mot_beams, trapping_beams[:-1])
    with pytest.raises(ValueError, match="three Cartesian axes"):
        draw_pmot_beam_volumes(
            axis,
            [beam for beam in mot_beams if beam.axis_name != "vertical_z"],
            trapping_beams,
        )
    plt.close(figure)
