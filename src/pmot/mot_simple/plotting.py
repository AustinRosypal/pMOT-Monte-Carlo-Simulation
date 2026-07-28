"""Plotting helpers for the simplified two-level MOT model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..mot.magnetic_fields import plane_sample
from .simulation import SimpleMOTBeam
from .simulation import SimpleMOTTrajectoryRecord


def _prepare_output(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_magnetic_component_grid(
    coil_config,
    extent_m: float,
    samples_per_axis: int,
    path: Path | None = None,
):
    """Plot a 3x3 grid of Bx, By, Bz over xy, xz, yz planes."""

    plane_names = ("xy", "xz", "yz")
    component_names = ("bx_g", "by_g", "bz_g")
    plane_component_map = {
        ("xy", "bx_g"): "component_1_g",
        ("xy", "by_g"): "component_2_g",
        ("xy", "bz_g"): "bz_g",
        ("xz", "bx_g"): "component_1_g",
        ("xz", "by_g"): None,
        ("xz", "bz_g"): "component_2_g",
        ("yz", "bx_g"): None,
        ("yz", "by_g"): "component_1_g",
        ("yz", "bz_g"): "component_2_g",
    }
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 12.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    for row_index, component_name in enumerate(component_names):
        for column_index, plane_name in enumerate(plane_names):
            sample = plane_sample(plane_name, extent_m=extent_m, config=coil_config, samples_per_axis=samples_per_axis)
            axis = axes[row_index, column_index]
            axis.set_facecolor("#fbfaf6")
            extent = (
                1e3 * sample["axis_values_m"][0],
                1e3 * sample["axis_values_m"][-1],
                1e3 * sample["axis_values_m"][0],
                1e3 * sample["axis_values_m"][-1],
            )
            sample_key = plane_component_map[(plane_name, component_name)]
            if sample_key is None:
                values = np.zeros((len(sample["axis_values_m"]), len(sample["axis_values_m"])), dtype=float)
            else:
                values = np.array(sample[sample_key], dtype=float)
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                aspect="equal",
            )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            axis.set_title(f"{component_name.replace('_g', '').upper()} in {plane_name.upper()}")
            axis.set_xlabel(f"{plane_name[0]} [mm]")
            axis.set_ylabel(f"{plane_name[1]} [mm]")
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axes


def plot_simple_mot_geometry(
    beams: list[SimpleMOTBeam],
    initial_position_m,
    path: Path | None = None,
):
    """Plot the six-beam simplified MOT geometry."""

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")
    color_by_sign = {+1.0: "#b91c1c", -1.0: "#1d4ed8"}
    style_by_axis = {
        "horizontal_x": "-",
        "horizontal_y": "--",
        "vertical_z": "-.",
    }
    for beam in beams:
        direction = np.asarray(beam.direction, dtype=float)
        start = -60.0 * direction
        stop = 60.0 * direction
        axis.plot(
            [start[0], stop[0]],
            [start[1], stop[1]],
            [start[2], stop[2]],
            color=color_by_sign[beam.polarization_sign],
            linestyle=style_by_axis[beam.axis_name],
            linewidth=2.0,
            alpha=0.9,
        )
    axis.scatter([0.0], [0.0], [0.0], color="#111827", s=45, label="trap center")
    axis.scatter(
        [1e3 * initial_position_m[0]],
        [1e3 * initial_position_m[1]],
        [1e3 * initial_position_m[2]],
        color="#0f766e",
        s=55,
        label="initial atom",
    )
    axis.set_title("Simplified Two-Level MOT Geometry")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    axis.set_xlim(-65.0, 65.0)
    axis.set_ylim(-65.0, 65.0)
    axis.set_zlim(-65.0, 65.0)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.legend(loc="best")
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_simple_mot_diagnostics(
    trajectory: SimpleMOTTrajectoryRecord,
    path: Path | None = None,
):
    """Plot the deterministic two-level MOT diagnostics."""

    times_ms = 1e3 * np.asarray(trajectory.times_s, dtype=float)
    positions_mm = 1e3 * np.asarray(trajectory.positions_m, dtype=float)
    velocities = np.asarray(trajectory.velocities_m_per_s, dtype=float)
    forces = np.asarray(trajectory.forces_n, dtype=float)

    figure = plt.figure(figsize=(14, 11), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    grid = figure.add_gridspec(3, 2)

    ax_pos = figure.add_subplot(grid[0, 0])
    ax_vel = figure.add_subplot(grid[0, 1])
    ax_rates = figure.add_subplot(grid[1, 0])
    ax_force = figure.add_subplot(grid[1, 1])
    ax_traj = figure.add_subplot(grid[2, :], projection="3d")
    for axis in (ax_pos, ax_vel, ax_rates, ax_force):
        axis.set_facecolor("#fbfaf6")
    ax_traj.set_facecolor("#fbfaf6")

    ax_pos.plot(times_ms, positions_mm[:, 0], label="x", color="#b91c1c")
    ax_pos.plot(times_ms, positions_mm[:, 1], label="y", color="#1d4ed8")
    ax_pos.plot(times_ms, positions_mm[:, 2], label="z", color="#15803d")
    ax_pos.set_title("Position Components")
    ax_pos.set_xlabel("Time [ms]")
    ax_pos.set_ylabel("Position [mm]")
    ax_pos.grid(True, alpha=0.25)
    ax_pos.legend(loc="best")

    ax_vel.plot(times_ms, velocities[:, 0], label=r"$v_x$", color="#b91c1c")
    ax_vel.plot(times_ms, velocities[:, 1], label=r"$v_y$", color="#1d4ed8")
    ax_vel.plot(times_ms, velocities[:, 2], label=r"$v_z$", color="#15803d")
    ax_vel.set_title("Velocity Components")
    ax_vel.set_xlabel("Time [ms]")
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.grid(True, alpha=0.25)
    ax_vel.legend(loc="best")

    for label, values in trajectory.beam_scattering_rates_per_s.items():
        ax_rates.plot(times_ms, values, linewidth=1.4, label=label)
    ax_rates.plot(times_ms, trajectory.total_scattering_rates_per_s, color="#111827", linewidth=2.0, linestyle="--", label="total")
    ax_rates.set_title("Beam Scattering Rates")
    ax_rates.set_xlabel("Time [ms]")
    ax_rates.set_ylabel(r"Rate [s$^{-1}$]")
    ax_rates.grid(True, alpha=0.25)
    ax_rates.legend(loc="best", fontsize=8)

    ax_force.plot(times_ms, forces[:, 0], label=r"$F_x$", color="#b91c1c")
    ax_force.plot(times_ms, forces[:, 1], label=r"$F_y$", color="#1d4ed8")
    ax_force.plot(times_ms, forces[:, 2], label=r"$F_z$", color="#15803d")
    ax_force.set_title("Mean Force Components")
    ax_force.set_xlabel("Time [ms]")
    ax_force.set_ylabel("Force [N]")
    ax_force.grid(True, alpha=0.25)
    ax_force.legend(loc="best")

    ax_traj.plot(positions_mm[:, 0], positions_mm[:, 1], positions_mm[:, 2], color="#0f766e", linewidth=2.0)
    ax_traj.scatter([positions_mm[0, 0]], [positions_mm[0, 1]], [positions_mm[0, 2]], color="#b91c1c", s=45, label="start")
    ax_traj.scatter([positions_mm[-1, 0]], [positions_mm[-1, 1]], [positions_mm[-1, 2]], color="#111827", s=45, label="end")
    ax_traj.set_title("3D Atomic Trajectory")
    ax_traj.set_xlabel("x [mm]")
    ax_traj.set_ylabel("y [mm]")
    ax_traj.set_zlabel("z [mm]")
    max_extent = max(1.0, float(np.max(np.abs(positions_mm))))
    ax_traj.set_xlim(-1.1 * max_extent, 1.1 * max_extent)
    ax_traj.set_ylim(-1.1 * max_extent, 1.1 * max_extent)
    ax_traj.set_zlim(-1.1 * max_extent, 1.1 * max_extent)
    ax_traj.set_box_aspect((1.0, 1.0, 1.0))
    ax_traj.legend(loc="best")

    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure
