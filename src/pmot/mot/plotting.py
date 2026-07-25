"""Plotting helpers for MOT magnetic-field validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_field_lineout(
    distances_mm: list[float],
    components_g: dict[str, list[float]],
    title: str,
    path: Path | None = None,
):
    """Plot magnetic-field components along a line scan."""

    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    color_map = {
        "bx_g": "#b91c1c",
        "by_g": "#1d4ed8",
        "bz_g": "#15803d",
        "bmag_g": "#111827",
    }
    label_map = {
        "bx_g": r"$B_x$ [G]",
        "by_g": r"$B_y$ [G]",
        "bz_g": r"$B_z$ [G]",
        "bmag_g": r"$|B|$ [G]",
    }
    for key, values in components_g.items():
        axis.plot(
            distances_mm,
            values,
            color=color_map.get(key, "#475569"),
            linewidth=2.0 if key != "bmag_g" else 1.4,
            linestyle="-" if key != "bmag_g" else "--",
            label=label_map.get(key, key),
        )
    axis.set_title(title)
    axis.set_xlabel("Distance [mm]")
    axis.set_ylabel("Field [G]")
    axis.grid(True, alpha=0.28)
    axis.legend(loc="best", frameon=True)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_field_heatmap(
    axis_values_m: list[float],
    grid_g: list[list[float]],
    plane: str,
    title: str,
    colorbar_label: str,
    path: Path | None = None,
):
    """Plot one magnetic-field scalar heatmap."""

    figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    extent = (
        1e3 * axis_values_m[0],
        1e3 * axis_values_m[-1],
        1e3 * axis_values_m[0],
        1e3 * axis_values_m[-1],
    )
    image = axis.imshow(
        np.array(grid_g),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="coolwarm",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    axis.set_title(title)
    axis.set_xlabel(f"{plane[0]} [mm]")
    axis.set_ylabel(f"{plane[1]} [mm]")
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_plane_quiver(
    axis_values_m: list[float],
    component_1_g: list[list[float]],
    component_2_g: list[list[float]],
    plane: str,
    title: str,
    stride: int = 8,
    path: Path | None = None,
):
    """Plot a 2D vector field on one Cartesian plane."""

    coordinates_mm = 1e3 * np.array(axis_values_m)
    coord_1, coord_2 = np.meshgrid(coordinates_mm, coordinates_mm)
    vector_1 = np.array(component_1_g)
    vector_2 = np.array(component_2_g)

    figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.quiver(
        coord_1[::stride, ::stride],
        coord_2[::stride, ::stride],
        vector_1[::stride, ::stride],
        vector_2[::stride, ::stride],
        color="#0f766e",
        pivot="mid",
        scale=None,
    )
    axis.set_title(title)
    axis.set_xlabel(f"{plane[0]} [mm]")
    axis.set_ylabel(f"{plane[1]} [mm]")
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.2)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_vector_cloud_3d(
    vector_cloud: list[tuple[float, float, float, float, float, float]],
    title: str,
    path: Path | None = None,
):
    """Plot a coarse 3D magnetic vector field."""

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")
    data = np.array(vector_cloud, dtype=float)
    if data.size == 0:
        raise ValueError("vector_cloud must be non-empty")
    extents_mm = 1e3 * np.abs(data[:, :3])
    max_extent_mm = max(1.0, float(np.max(extents_mm)))
    magnitudes_t = np.sqrt(np.sum(data[:, 3:] ** 2, axis=1))
    max_magnitude_t = max(1.0e-18, float(np.max(magnitudes_t)))
    visual_scale = 0.22 * max_extent_mm / max_magnitude_t
    axis.quiver(
        1e3 * data[:, 0],
        1e3 * data[:, 1],
        1e3 * data[:, 2],
        visual_scale * data[:, 3],
        visual_scale * data[:, 4],
        visual_scale * data[:, 5],
        color="#0f766e",
        linewidth=1.0,
        normalize=False,
    )
    axis.set_title(title)
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xlim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis.set_ylim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis.set_zlim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_mot_diagnostics_grid(
    times_s: list[float],
    positions_m: list[tuple[float, float, float]],
    velocities_m_per_s: list[tuple[float, float, float]],
    axis_scattering_rates_per_s: dict[str, list[float]],
    total_scattering_rates_per_s: list[float],
    title: str,
    path: Path | None = None,
):
    """Plot a compact MOT single-atom diagnostics grid."""

    times_ms = 1e3 * np.asarray(times_s, dtype=float)
    positions_mm = 1e3 * np.asarray(positions_m, dtype=float)
    velocities = np.asarray(velocities_m_per_s, dtype=float)
    speed = np.sqrt(np.sum(velocities**2, axis=1))

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    for axis in axes.flat:
        axis.set_facecolor("#fbfaf6")

    axes[0, 0].plot(times_ms, velocities[:, 0], color="#b91c1c", linewidth=2.0, label=r"$v_x$")
    axes[0, 0].plot(times_ms, velocities[:, 1], color="#1d4ed8", linewidth=2.0, label=r"$v_y$")
    axes[0, 0].plot(times_ms, velocities[:, 2], color="#15803d", linewidth=2.0, label=r"$v_z$")
    axes[0, 0].plot(times_ms, speed, color="#111827", linewidth=1.6, linestyle="--", label=r"$|v|$")
    axes[0, 0].set_title("Velocity Evolution")
    axes[0, 0].set_xlabel("Time [ms]")
    axes[0, 0].set_ylabel("Velocity [m/s]")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", frameon=True)

    axes[0, 1].plot(times_ms, positions_mm[:, 0], color="#b91c1c", linewidth=2.0, label="x")
    axes[0, 1].plot(times_ms, positions_mm[:, 1], color="#1d4ed8", linewidth=2.0, label="y")
    axes[0, 1].plot(times_ms, positions_mm[:, 2], color="#15803d", linewidth=2.0, label="z")
    axes[0, 1].plot(
        times_ms,
        np.sqrt(np.sum(positions_mm**2, axis=1)),
        color="#111827",
        linewidth=1.6,
        linestyle="--",
        label="radius",
    )
    axes[0, 1].set_title("Position Evolution")
    axes[0, 1].set_xlabel("Time [ms]")
    axes[0, 1].set_ylabel("Position [mm]")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best", frameon=True)

    color_map = {
        "horizontal_x": "#b91c1c",
        "horizontal_y": "#1d4ed8",
        "vertical_z": "#15803d",
    }
    label_map = {
        "horizontal_x": "x-axis beams",
        "horizontal_y": "y-axis beams",
        "vertical_z": "z-axis beams",
    }
    for axis_name, values in axis_scattering_rates_per_s.items():
        axes[1, 0].plot(
            times_ms,
            values,
            linewidth=2.0,
            color=color_map.get(axis_name, "#475569"),
            label=label_map.get(axis_name, axis_name),
        )
    axes[1, 0].plot(
        times_ms,
        total_scattering_rates_per_s,
        color="#111827",
        linewidth=1.6,
        linestyle="--",
        label="total",
    )
    axes[1, 0].set_title("Scattering Rates by Beam Axis")
    axes[1, 0].set_xlabel("Time [ms]")
    axes[1, 0].set_ylabel(r"Rate [s$^{-1}$]")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(loc="best", frameon=True)

    axis3d = figure.add_subplot(2, 2, 4, projection="3d")
    axis3d.set_facecolor("#fbfaf6")
    axis3d.plot(
        positions_mm[:, 0],
        positions_mm[:, 1],
        positions_mm[:, 2],
        color="#0f766e",
        linewidth=2.2,
    )
    axis3d.scatter(
        [positions_mm[0, 0]],
        [positions_mm[0, 1]],
        [positions_mm[0, 2]],
        color="#b91c1c",
        s=40,
        label="start",
    )
    axis3d.scatter(
        [positions_mm[-1, 0]],
        [positions_mm[-1, 1]],
        [positions_mm[-1, 2]],
        color="#111827",
        s=40,
        label="end",
    )
    axis3d.set_title("3D Trajectory")
    axis3d.set_xlabel("x [mm]")
    axis3d.set_ylabel("y [mm]")
    axis3d.set_zlabel("z [mm]")
    axis3d.legend(loc="best")
    max_extent_mm = max(1.0, float(np.max(np.abs(positions_mm))))
    axis3d.set_xlim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis3d.set_ylim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis3d.set_zlim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis3d.set_box_aspect((1.0, 1.0, 1.0))

    figure.suptitle(title, fontsize=14)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axes


def plot_many_atom_snapshot(
    positions_m: np.ndarray,
    speeds_m_per_s: np.ndarray,
    title: str,
    path: Path | None = None,
):
    """Plot one many-atom MOT snapshot colored by speed."""

    positions_mm = 1e3 * np.asarray(positions_m, dtype=float)
    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#fbfaf6")
    scatter = axis.scatter(
        positions_mm[:, 0],
        positions_mm[:, 1],
        positions_mm[:, 2],
        c=np.asarray(speeds_m_per_s, dtype=float),
        cmap="viridis",
        s=24,
        alpha=0.9,
    )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.08)
    colorbar.set_label("Speed [m/s]")
    axis.set_title(title)
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    max_extent_mm = max(1.0, float(np.max(np.abs(positions_mm))))
    axis.set_xlim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis.set_ylim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis.set_zlim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis
