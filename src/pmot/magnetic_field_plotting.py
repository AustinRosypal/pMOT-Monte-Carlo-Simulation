"""Shared plotting helpers for anti-Helmholtz field validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

from .magnetic_fields import anti_helmholtz_field_t


def _prepare_output(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_magnetic_component_grid(
    coil_config,
    extent_m: float,
    samples_per_axis: int,
    path: Path | None = None,
):
    """Plot all Cartesian field components over the xy, xz, and yz planes."""

    axis_values_m = np.linspace(-extent_m, extent_m, samples_per_axis)
    plane_names = ("xy", "xz", "yz")
    component_names = ("Bx", "By", "Bz")
    figure = plt.figure(figsize=(15.5, 13.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")

    def field_components_for_plane(plane_name: str):
        first_mesh, second_mesh = np.meshgrid(axis_values_m, axis_values_m)
        if plane_name == "xy":
            values = anti_helmholtz_field_t(
                first_mesh, second_mesh, 0.0, coil_config
            )
        elif plane_name == "xz":
            values = anti_helmholtz_field_t(
                first_mesh, 0.0, second_mesh, coil_config
            )
        else:
            values = anti_helmholtz_field_t(
                0.0, first_mesh, second_mesh, coil_config
            )
        return first_mesh, second_mesh, *values

    for row_index, component_name in enumerate(component_names):
        for column_index, plane_name in enumerate(plane_names):
            axis = figure.add_subplot(
                3,
                3,
                row_index * 3 + column_index + 1,
                projection="3d",
            )
            axis.set_facecolor("#fbfaf6")
            first_mesh_m, second_mesh_m, bx_t, by_t, bz_t = (
                field_components_for_plane(plane_name)
            )
            component_values_g = {
                "Bx": 1.0e4 * np.asarray(bx_t, dtype=float),
                "By": 1.0e4 * np.asarray(by_t, dtype=float),
                "Bz": 1.0e4 * np.asarray(bz_t, dtype=float),
            }[component_name]

            if plane_name == "xy":
                label_1, label_2 = "x [mm]", "y [mm]"
            elif plane_name == "xz":
                label_1, label_2 = "x [mm]", "z [mm]"
            else:
                label_1, label_2 = "y [mm]", "z [mm]"
            horizontal_1 = 1e3 * np.asarray(first_mesh_m, dtype=float)
            horizontal_2 = 1e3 * np.asarray(second_mesh_m, dtype=float)

            max_abs_g = max(1.0e-12, float(np.max(np.abs(component_values_g))))
            normalizer = colors.TwoSlopeNorm(
                vmin=-max_abs_g, vcenter=0.0, vmax=max_abs_g
            )
            axis.plot_surface(
                horizontal_1,
                horizontal_2,
                component_values_g,
                facecolors=cm.coolwarm(normalizer(component_values_g)),
                linewidth=0.0,
                antialiased=True,
                shade=False,
                alpha=0.96,
            )
            colorbar_mappable = cm.ScalarMappable(norm=normalizer, cmap="coolwarm")
            colorbar_mappable.set_array(component_values_g)
            figure.colorbar(
                colorbar_mappable,
                ax=axis,
                fraction=0.046,
                pad=0.04,
                label=f"{component_name} [G]",
            )

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
