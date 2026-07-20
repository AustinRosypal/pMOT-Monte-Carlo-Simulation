"""Notebook-oriented plotting helpers for the pMOT field studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import numpy as np


def _prepare_output(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_polarizability_curves(dataframe, path: Path | None = None):
    """Plot positive and negative scalar, vector, and tensor coefficients."""

    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")

    series_map = {
        "scalar_mhz_per_intensity": ("#1d4ed8", r"Scalar $\alpha^{(0)}$"),
        "vector_mhz_per_intensity": ("#b91c1c", r"Vector $\alpha^{(1)}$"),
        "tensor_mhz_per_intensity": ("#15803d", r"Tensor $\alpha^{(2)}$"),
    }
    for column, (color, label) in series_map.items():
        values = dataframe[column].to_numpy()
        positive = np.where(values > 0.0, values, np.nan)
        negative = np.where(values < 0.0, np.abs(values), np.nan)
        axis.plot(dataframe["frequency_thz"], positive, color=color, linewidth=2.1, label=f"+ {label}")
        axis.plot(dataframe["frequency_thz"], negative, color=color, linewidth=2.1, linestyle="--", label=f"- {label}")

    axis.set_xlabel(r"Optical Frequency $\nu$ (THz)")
    axis.set_ylabel(r"Effective Polarizability (MHz/[mW/(100$\mu$m)$^2$])")
    axis.set_title("Differential Polarizability Conversion")
    axis.set_yscale("log")
    axis.set_ylim(bottom=1e-6)
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend(loc="best", frameon=True)
    axis.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axis.ticklabel_format(style="plain", axis="x")

    secx = axis.secondary_xaxis(
        "top",
        functions=(
            lambda frequency_thz: 299792458.0 / (frequency_thz * 1e12) * 1e9,
            lambda wavelength_nm: 299792458.0 / (wavelength_nm * 1e-9) / 1e12,
        ),
    )
    secx.set_xlabel(r"Wavelength $\lambda$ (nm)")
    secx.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_intensity_lineout(
    distances_mm: list[float],
    intensities_w_per_m2: list[float],
    title: str,
    path: Path | None = None,
):
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.plot(distances_mm, intensities_w_per_m2, color="#0f766e", linewidth=2.2)
    axis.set_title(title)
    axis.set_xlabel("Distance along line [mm]")
    axis.set_ylabel("Total intensity [W/m$^2$]")
    axis.grid(True, alpha=0.3)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_beam_crossing_zoom(
    x_mm: list[float],
    intensities_w_per_m2: list[float],
    x_window_mm: float = 0.75,
    path: Path | None = None,
):
    figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.plot(x_mm, intensities_w_per_m2, color="#7c3aed", linewidth=2.2)
    axis.set_xlim(-x_window_mm, x_window_mm)
    axis.set_title("Beam-Crossing Intensity Near the Trap Center")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("Total intensity [W/m$^2$]")
    axis.grid(True, alpha=0.3)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_scalar_field_slice(
    axis_1_m: list[float],
    axis_2_m: list[float],
    grid: list[list[float]],
    plane: str,
    title: str,
    colorbar_label: str,
    path: Path | None = None,
    log_scale: bool = False,
):
    figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")

    extent = (
        1e3 * axis_1_m[0],
        1e3 * axis_1_m[-1],
        1e3 * axis_2_m[0],
        1e3 * axis_2_m[-1],
    )
    data = np.array(grid)
    if log_scale:
        positive = data[data > 0.0]
        vmin = float(positive.min()) if positive.size else 1.0
        image = axis.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=float(data.max()) if data.size else vmin),
        )
    else:
        image = axis.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
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


def plot_intensity_cloud_3d(
    cloud_points: list[tuple[float, float, float, float]],
    title: str = "3D Trapping-Field Intensity Cloud",
    path: Path | None = None,
):
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")

    xyz = np.array([(1e3 * x, 1e3 * y, 1e3 * z) for x, y, z, _ in cloud_points])
    values = np.array([value for _, _, _, value in cloud_points])
    if values.size == 0:
        raise ValueError("cloud_points must be non-empty")
    scatter = axis.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=values,
        s=5,
        cmap="plasma",
        alpha=0.55,
    )
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.75, pad=0.08)
    colorbar.set_label("Total intensity [W/m$^2$]")
    axis.set_title(title)
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_apparatus_geometry_3d(beams, path: Path | None = None):
    """Plot beam axes and waist positions for the current apparatus."""

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")

    colors = {
        "oblique_x": "#1d4ed8",
        "oblique_y": "#b45309",
        "normal_z": "#15803d",
    }
    for beam in beams:
        color = colors.get(beam.axis_name, "#334155")
        start = np.array(beam.waist_position_m) - 0.03 * np.array(beam.direction)
        stop = np.array(beam.waist_position_m) + 0.03 * np.array(beam.direction)
        axis.plot(
            [1e3 * start[0], 1e3 * stop[0]],
            [1e3 * start[1], 1e3 * stop[1]],
            [1e3 * start[2], 1e3 * stop[2]],
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )
        axis.scatter(
            [1e3 * beam.waist_position_m[0]],
            [1e3 * beam.waist_position_m[1]],
            [1e3 * beam.waist_position_m[2]],
            color=color,
            s=28,
        )

    axis.set_title("pMOT Beam Geometry")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis
