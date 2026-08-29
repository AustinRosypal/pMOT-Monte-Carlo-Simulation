"""Plotting helpers for the simplified two-level MOT model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..beam_plotting import beam_surface_mesh_mm
from ..magnetic_field_plotting import plot_magnetic_component_grid
from .simulation import SimpleMOTBeam
from .simulation import SimpleMOTTrajectoryRecord
from ..state import AtomState
from .simulation import mean_force_n


def _prepare_output(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def draw_simple_mot_beam_volumes(
    axis,
    beams: list[SimpleMOTBeam],
    length_m: float = 120.0e-3,
) -> None:
    axis_color = {
        "horizontal_x": "#f9a8d4",
        "horizontal_y": "#93c5fd",
        "vertical_z": "#86efac",
    }
    drawn_axes: set[str] = set()
    for beam in beams:
        if beam.axis_name in drawn_axes:
            continue
        drawn_axes.add(beam.axis_name)
        x_surface_mm, y_surface_mm, z_surface_mm = beam_surface_mesh_mm(
            direction=beam.direction,
            radius_m=beam.intensity_beam.beam_radius_m,
            length_m=length_m,
        )
        axis.plot_surface(
            x_surface_mm,
            y_surface_mm,
            z_surface_mm,
            color=axis_color[beam.axis_name],
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=0.22,
        )


def plot_simple_mot_geometry(
    beams: list[SimpleMOTBeam],
    initial_position_m,
    path: Path | None = None,
):
    """Plot the six-beam simplified MOT geometry with beam diameters to scale."""

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")
    draw_simple_mot_beam_volumes(axis, beams)
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


def plot_beam_polarization_diagram(
    beams: list[SimpleMOTBeam],
    path: Path | None = None,
):
    """Show the propagation direction and propagation-frame helicity of all beams."""

    figure = plt.figure(figsize=(13, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(121, projection="3d")
    table_axis = figure.add_subplot(122)
    axis.set_facecolor("#fbfaf6")
    table_axis.set_facecolor("#fbfaf6")

    color_by_polarization = {"sigma+": "#b91c1c", "sigma-": "#1d4ed8", "pi": "#15803d"}
    rows: list[list[str]] = []
    for beam in beams:
        direction = np.asarray(beam.direction, dtype=float)
        start = -48.0 * direction
        vector = 42.0 * direction
        color = color_by_polarization[beam.circular_polarization]
        polarization_label = {"sigma+": r"$\sigma^+$", "sigma-": r"$\sigma^-$", "pi": r"$\pi$"}[
            beam.circular_polarization
        ]
        axis.quiver(*start, *vector, color=color, linewidth=2.5, arrow_length_ratio=0.16)
        label_position = start + 0.48 * vector
        axis.text(
            *label_position,
            f"{polarization_label}\n"
            rf"$\hat{{k}}$=({direction[0]:+.0f},{direction[1]:+.0f},{direction[2]:+.0f})",
            color=color,
            ha="center",
            va="center",
            fontsize=9,
        )
        rows.append(
            [
                beam.axis_name.replace("horizontal_", "").replace("vertical_", ""),
                beam.propagation_sense,
                f"({direction[0]:+.0f}, {direction[1]:+.0f}, {direction[2]:+.0f})",
                beam.circular_polarization,
                f"{beam.polarization_sign:+.0f}",
            ]
        )

    axis.scatter([0.0], [0.0], [0.0], color="#111827", s=50)
    axis.set_title("Six Cooling Beams")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    axis.set_xlim(-55.0, 55.0)
    axis.set_ylim(-55.0, 55.0)
    axis.set_zlim(-55.0, 55.0)
    axis.set_box_aspect((1.0, 1.0, 1.0))

    table_axis.axis("off")
    table_axis.set_title("Beam Convention", pad=14)
    table = table_axis.table(
        cellText=rows,
        colLabels=["axis", "path", r"$\hat{k}$", "helicity", r"effective $\xi$"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    table_axis.text(
        0.5,
        0.12,
        r"$\sigma^\pm$ is defined while looking along each beam's propagation direction."
        "\nThe effective two-level Zeeman sign is axis-dependent; it is not a multilevel selection rule.",
        transform=table_axis.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure


def plot_simple_mot_force_curves(
    beams: list[SimpleMOTBeam],
    coil_config,
    simple_config,
    position_extent_m: float = 8.0e-3,
    velocity_extent_m_per_s: float = 15.0,
    sample_count: int = 161,
    path: Path | None = None,
):
    """Plot restoring-force and damping-force curves along all three axes.

    These panels show radiation-pressure force only. Gravity enters trajectory
    acceleration separately and is intentionally excluded from the force-law
    symmetry plots.
    """

    positions = np.linspace(-position_extent_m, position_extent_m, sample_count)
    velocities = np.linspace(-velocity_extent_m_per_s, velocity_extent_m_per_s, sample_count)
    axes_spec = (("x", 0), ("y", 1), ("z", 2))
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    for row, (axis_name, component) in enumerate(axes_spec):
        position_forces = []
        velocity_forces = []
        for coordinate in positions:
            position = [0.0, 0.0, 0.0]
            position[component] = float(coordinate)
            force, _, _ = mean_force_n(beams, AtomState(tuple(position), (0.0, 0.0, 0.0)), coil_config, simple_config)
            position_forces.append(force[component])
        for speed in velocities:
            velocity = [0.0, 0.0, 0.0]
            velocity[component] = float(speed)
            force, _, _ = mean_force_n(beams, AtomState((0.0, 0.0, 0.0), tuple(velocity)), coil_config, simple_config)
            velocity_forces.append(force[component])

        position_axis, velocity_axis = axes[row]
        position_axis.plot(1e3 * positions, position_forces, color="#7c3aed")
        velocity_axis.plot(velocities, velocity_forces, color="#0f766e")
        for panel in (position_axis, velocity_axis):
            panel.axhline(0.0, color="#64748b", linewidth=0.8)
            panel.axvline(0.0, color="#64748b", linewidth=0.8)
            panel.grid(True, alpha=0.25)
            panel.set_facecolor("#fbfaf6")
        position_axis.set_title(rf"Restoring curve: $F_{axis_name}({axis_name})$ at $v=0$")
        position_axis.set_xlabel(f"{axis_name} [mm]")
        position_axis.set_ylabel(f"$F_{axis_name}$ [N]")
        velocity_axis.set_title(rf"Damping curve: $F_{axis_name}(v_{axis_name})$ at $r=0$")
        velocity_axis.set_xlabel(rf"$v_{axis_name}$ [m/s]")
        velocity_axis.set_ylabel(f"$F_{axis_name}$ [N]")

    figure.suptitle("Simplified Two-Level MOT Force Curves (radiation pressure)", fontsize=15)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure


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
