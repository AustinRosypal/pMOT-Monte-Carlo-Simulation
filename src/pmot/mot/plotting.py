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
    axis.quiver(
        1e3 * data[:, 0],
        1e3 * data[:, 1],
        1e3 * data[:, 2],
        data[:, 3],
        data[:, 4],
        data[:, 5],
        color="#0f766e",
        linewidth=1.0,
        normalize=False,
        length=8.0,
    )
    axis.set_title(title)
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    axis.set_box_aspect((1.0, 1.0, 1.0))
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis

