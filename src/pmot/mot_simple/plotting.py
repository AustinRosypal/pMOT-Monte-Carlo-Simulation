"""Plotting helpers for the simplified two-level MOT model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np

from ..mot.magnetic_fields import anti_helmholtz_field_t
from .simulation import SimpleMOTBeam
from .simulation import SimpleMOTTrajectoryRecord


def _prepare_output(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(direction, trial))) > 0.95:
        trial = np.asarray([0.0, 1.0, 0.0], dtype=float)
    basis_1 = np.cross(direction, trial)
    basis_1 = basis_1 / np.linalg.norm(basis_1)
    basis_2 = np.cross(direction, basis_1)
    basis_2 = basis_2 / np.linalg.norm(basis_2)
    return basis_1, basis_2


def _beam_surface_mesh_mm(
    direction: tuple[float, float, float],
    radius_m: float,
    length_m: float,
    axial_samples: int = 25,
    angular_samples: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction_array = np.asarray(direction, dtype=float)
    direction_array = direction_array / np.linalg.norm(direction_array)
    basis_1, basis_2 = _orthonormal_basis(direction_array)
    axial_values_m = np.linspace(-0.5 * length_m, 0.5 * length_m, axial_samples)
    angular_values = np.linspace(0.0, 2.0 * np.pi, angular_samples)
    axial_grid_m, angular_grid = np.meshgrid(axial_values_m, angular_values, indexing="ij")
    radius_grid_m = (
        np.cos(angular_grid)[..., None] * basis_1[None, None, :]
        + np.sin(angular_grid)[..., None] * basis_2[None, None, :]
    ) * radius_m
    centerline_grid_m = axial_grid_m[..., None] * direction_array[None, None, :]
    surface_grid_mm = 1e3 * (centerline_grid_m + radius_grid_m)
    return surface_grid_mm[..., 0], surface_grid_mm[..., 1], surface_grid_mm[..., 2]


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
        x_surface_mm, y_surface_mm, z_surface_mm = _beam_surface_mesh_mm(
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


def plot_magnetic_component_grid(
    coil_config,
    extent_m: float,
    samples_per_axis: int,
    path: Path | None = None,
):
    """Plot a 3x3 grid of 3D magnetic-component surfaces over xy, xz, yz planes.

    The plane coordinates span the two horizontal axes. The selected field
    component is shown redundantly as the vertical coordinate and the color.
    """

    axis_values_m = np.linspace(-extent_m, extent_m, samples_per_axis)
    plane_names = ("xy", "xz", "yz")
    component_names = ("Bx", "By", "Bz")
    figure = plt.figure(figsize=(15.5, 13.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")

    def field_components_for_plane(plane_name: str):
        first_mesh, second_mesh = np.meshgrid(axis_values_m, axis_values_m)
        if plane_name == "xy":
            bx_t, by_t, bz_t = anti_helmholtz_field_t(first_mesh, second_mesh, 0.0, coil_config)
            return first_mesh, second_mesh, bx_t, by_t, bz_t
        if plane_name == "xz":
            bx_t, by_t, bz_t = anti_helmholtz_field_t(first_mesh, 0.0, second_mesh, coil_config)
            return first_mesh, second_mesh, bx_t, by_t, bz_t
        bx_t, by_t, bz_t = anti_helmholtz_field_t(0.0, first_mesh, second_mesh, coil_config)
        return first_mesh, second_mesh, bx_t, by_t, bz_t

    for row_index, component_name in enumerate(component_names):
        for column_index, plane_name in enumerate(plane_names):
            axis = figure.add_subplot(3, 3, row_index * 3 + column_index + 1, projection="3d")
            axis.set_facecolor("#fbfaf6")

            first_mesh_m, second_mesh_m, bx_t, by_t, bz_t = field_components_for_plane(plane_name)
            bx_g = 1.0e4 * np.asarray(bx_t, dtype=float)
            by_g = 1.0e4 * np.asarray(by_t, dtype=float)
            bz_g = 1.0e4 * np.asarray(bz_t, dtype=float)
            component_values_g = {
                "Bx": bx_g,
                "By": by_g,
                "Bz": bz_g,
            }[component_name]

            if plane_name == "xy":
                horizontal_1 = 1e3 * np.asarray(first_mesh_m, dtype=float)
                horizontal_2 = 1e3 * np.asarray(second_mesh_m, dtype=float)
                label_1 = "x [mm]"
                label_2 = "y [mm]"
            elif plane_name == "xz":
                horizontal_1 = 1e3 * np.asarray(first_mesh_m, dtype=float)
                horizontal_2 = 1e3 * np.asarray(second_mesh_m, dtype=float)
                label_1 = "x [mm]"
                label_2 = "z [mm]"
            else:
                horizontal_1 = 1e3 * np.asarray(first_mesh_m, dtype=float)
                horizontal_2 = 1e3 * np.asarray(second_mesh_m, dtype=float)
                label_1 = "y [mm]"
                label_2 = "z [mm]"

            max_abs_g = max(1.0e-12, float(np.max(np.abs(component_values_g))))
            normalizer = colors.TwoSlopeNorm(vmin=-max_abs_g, vcenter=0.0, vmax=max_abs_g)
            facecolors = cm.coolwarm(normalizer(component_values_g))
            axis.plot_surface(
                horizontal_1,
                horizontal_2,
                component_values_g,
                facecolors=facecolors,
                linewidth=0.0,
                antialiased=True,
                shade=False,
                alpha=0.96,
            )
            colorbar_mappable = cm.ScalarMappable(norm=normalizer, cmap="coolwarm")
            colorbar_mappable.set_array(component_values_g)
            figure.colorbar(colorbar_mappable, ax=axis, fraction=0.046, pad=0.04, label=f"{component_name} [G]")

            axis.set_title(f"{component_name} over {plane_name.upper()} plane")
            axis.set_xlabel(label_1)
            axis.set_ylabel(label_2)
            axis.set_zlabel(f"{component_name} [G]")
            max_extent_mm = 1e3 * extent_m
            axis.set_xlim(-max_extent_mm, max_extent_mm)
            axis.set_ylim(-max_extent_mm, max_extent_mm)
            axis.set_zlim(-1.05 * max_abs_g, 1.05 * max_abs_g)
            axis.set_box_aspect((1.0, 1.0, 0.8))
            axis.view_init(elev=24, azim=-58)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure


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
