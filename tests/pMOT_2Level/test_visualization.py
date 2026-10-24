from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pmot.pMOT_2Level import PMOT2LevelTrajectoryRecord
from pmot.pMOT_2Level import add_beam_geometry
from pmot.pMOT_2Level import animate_trajectory
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import plot_force_velocity_history
from pmot.pMOT_2Level import plot_trajectory_3d


def _record() -> PMOT2LevelTrajectoryRecord:
    return PMOT2LevelTrajectoryRecord(
        times_s=(0.0, 1.0e-3, 2.0e-3),
        positions_m=((1.0e-3, 0.0, 0.0), (0.5e-3, 0.1e-3, 0.0), (0.0, 0.0, 0.0)),
        velocities_m_per_s=((-1.0, 0.0, 0.0), (-0.5, 0.1, 0.0), (0.0, 0.0, 0.0)),
        forces_n=((1.0e-21, 0.0, 0.0), (0.5e-21, -0.1e-21, 0.0), (0.0, 0.0, 0.0)),
        quantization_axes=((1.0, 0.0, 0.0),) * 3,
        axis_defined_by_vector_shift=(True, True, False),
        total_ac_stark_shifts_hz_for_sigma_plus=(0.0, 1.0, 0.0),
        total_scattering_rates_per_s=(1.0, 2.0, 1.0),
    )


def test_beam_geometry_is_rendered_in_plot_millimetres():
    class CapturingAxis:
        def __init__(self):
            self.surfaces = []
            self.quivers = []

        def plot_surface(self, x, y, z, **_):
            self.surfaces.append((np.asarray(x), np.asarray(y), np.asarray(z)))

        def plot(self, *_args, **_kwargs):
            return None

        def quiver(self, *args, **_kwargs):
            self.quivers.append(args)
            return None

    axis = CapturingAxis()
    x_trapping = tuple(
        beam
        for beam in build_pmot_2level_trapping_beams()
        if beam.axis_name == "horizontal_x"
    )
    add_beam_geometry(
        axis,
        build_pmot_2level_cooling_beams(),
        x_trapping,
        extent_m=20.0e-3,
    )
    plotted_coordinates = np.concatenate(
        [coordinate.ravel() for surface in axis.surfaces for coordinate in surface]
    )
    assert np.max(np.abs(plotted_coordinates)) >= 19.9
    assert len(axis.surfaces) == 5  # three cooling paths plus two x trapping beams
    assert len(axis.quivers) == 8  # six cooling plus two x trapping directions


def test_static_notebook_figures_include_expected_axes_and_units():
    record = _record()
    cooling = build_pmot_2level_cooling_beams()
    trapping = build_pmot_2level_trapping_beams()
    figure_3d, axis_3d = plot_trajectory_3d(
        record,
        cooling,
        trapping,
        extent_mm=16.0,
    )
    figure_history, (force_axis, velocity_axis) = plot_force_velocity_history(record)
    assert axis_3d.get_xlabel() == "x [mm]"
    assert force_axis.get_ylabel() == "radiation force [zN]"
    assert velocity_axis.get_ylabel() == "velocity [m/s]"
    assert len(figure_3d.axes) == 2  # spatial axis plus speed colorbar
    assert len(figure_history.axes) == 2
    plt.close(figure_3d)
    plt.close(figure_history)


def test_animation_updates_location_and_speed_marker():
    animation = animate_trajectory(
        _record(),
        build_pmot_2level_cooling_beams(),
        build_pmot_2level_trapping_beams(),
        extent_mm=16.0,
        maximum_frames=3,
        interval_ms=20,
    )
    artists = animation._func(2)
    assert len(artists) == 4
    speed_x, speed_y = artists[2].get_data()
    assert np.isclose(speed_x[0], 2.0)
    assert np.isclose(speed_y[0], 0.0)
    animation._draw_was_started = True
    plt.close(animation._fig)
