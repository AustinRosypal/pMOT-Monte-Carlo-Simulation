"""Plotting helpers for reusable anti-Helmholtz field validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_field_lineout(distances_mm, components_g, title: str, path: Path | None = None):
    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    colors = {"bx_g": "#b91c1c", "by_g": "#1d4ed8", "bz_g": "#15803d", "bmag_g": "#111827"}
    labels = {"bx_g": r"$B_x$ [G]", "by_g": r"$B_y$ [G]", "bz_g": r"$B_z$ [G]", "bmag_g": r"$|B|$ [G]"}
    for key, values in components_g.items():
        axis.plot(distances_mm, values, color=colors.get(key), label=labels.get(key, key))
    axis.set(title=title, xlabel="Distance [mm]", ylabel="Field [G]")
    axis.grid(True, alpha=0.28)
    axis.legend(loc="best")
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_field_heatmap(axis_values_m, grid_g, plane: str, title: str, colorbar_label: str, path: Path | None = None):
    figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    extent = tuple(1e3 * value for value in (axis_values_m[0], axis_values_m[-1], axis_values_m[0], axis_values_m[-1]))
    image = axis.imshow(np.asarray(grid_g), origin="lower", extent=extent, aspect="equal", cmap="coolwarm")
    figure.colorbar(image, ax=axis, label=colorbar_label)
    axis.set(title=title, xlabel=f"{plane[0]} [mm]", ylabel=f"{plane[1]} [mm]")
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_plane_quiver(axis_values_m, component_1_g, component_2_g, plane: str, title: str, stride: int = 8, path: Path | None = None):
    coordinates_mm = 1e3 * np.asarray(axis_values_m)
    coordinate_1, coordinate_2 = np.meshgrid(coordinates_mm, coordinates_mm)
    vector_1 = np.asarray(component_1_g)
    vector_2 = np.asarray(component_2_g)
    figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    axis.quiver(
        coordinate_1[::stride, ::stride],
        coordinate_2[::stride, ::stride],
        vector_1[::stride, ::stride],
        vector_2[::stride, ::stride],
        color="#0f766e",
        pivot="mid",
    )
    axis.set(title=title, xlabel=f"{plane[0]} [mm]", ylabel=f"{plane[1]} [mm]")
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.2)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_vector_cloud_3d(vector_cloud, title: str, path: Path | None = None):
    data = np.asarray(vector_cloud, dtype=float)
    if data.size == 0:
        raise ValueError("vector_cloud must be non-empty")
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    max_extent_mm = max(1.0, float(np.max(1e3 * np.abs(data[:, :3]))))
    max_magnitude_t = max(1.0e-18, float(np.max(np.linalg.norm(data[:, 3:], axis=1))))
    visual_scale = 0.22 * max_extent_mm / max_magnitude_t
    axis.quiver(1e3 * data[:, 0], 1e3 * data[:, 1], 1e3 * data[:, 2], *(visual_scale * data[:, 3:]).T, color="#0f766e")
    axis.set(title=title, xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
    axis.set_xlim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis.set_ylim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis.set_zlim(-1.1 * max_extent_mm, 1.1 * max_extent_mm)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis
