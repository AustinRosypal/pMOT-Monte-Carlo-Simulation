"""Notebook-oriented plotting helpers for the pMOT field studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
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
    axis.set_ylabel("Intensity [W/m$^2$]")
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
    axis.set_title("Beam-Line Intensity Near the Origin")
    axis.set_xlabel("Coordinate [mm]")
    axis.set_ylabel("Intensity [W/m$^2$]")
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


def plot_scalar_field_surface(
    axis_1_m: list[float],
    axis_2_m: list[float],
    grid: list[list[float]],
    plane: str,
    title: str,
    z_label: str,
    path: Path | None = None,
):
    """Render a scalar field slice as a 3D surface plot."""

    figure = plt.figure(figsize=(9.5, 7.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")
    x_mesh, y_mesh = np.meshgrid(1e3 * np.array(axis_1_m), 1e3 * np.array(axis_2_m))
    z_values = np.array(grid)
    surface = axis.plot_surface(
        x_mesh,
        y_mesh,
        z_values,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        rcount=min(150, len(axis_2_m)),
        ccount=min(150, len(axis_1_m)),
    )
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.72, pad=0.08)
    colorbar.set_label(z_label)
    axis.set_title(title)
    axis.set_xlabel(f"{plane[0]} [mm]")
    axis.set_ylabel(f"{plane[1]} [mm]")
    axis.set_zlabel(z_label)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_intensity_cloud_3d(
    cloud_points: list[tuple[float, float, float, float]],
    title: str = "3D MOT-Field Intensity Cloud",
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
    colorbar.set_label("Intensity [W/m$^2$]")
    axis.set_title(title)
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_intensity_cloud_3d_by_polarization(
    cloud_points_by_polarization: dict[str, list[tuple[float, float, float, float]]],
    title: str = "3D MOT-Field Intensity Cloud By Polarization",
    path: Path | None = None,
):
    """Plot 3D cloud samples with handedness-resolved styling."""

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")

    polarization_styles = {
        "right": {"color": "#d62828", "marker": "o", "label": "Right-handed circular"},
        "left": {"color": "#1d4ed8", "marker": "^", "label": "Left-handed circular"},
    }
    for circular_polarization, cloud_points in cloud_points_by_polarization.items():
        if not cloud_points:
            continue
        xyz = np.array([(1e3 * x, 1e3 * y, 1e3 * z) for x, y, z, _ in cloud_points])
        values = np.array([value for _, _, _, value in cloud_points])
        style = polarization_styles.get(
            circular_polarization,
            {"color": "#475569", "marker": "o", "label": circular_polarization},
        )
        normalized = values / values.max() if values.max() > 0.0 else np.ones_like(values)
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=10.0 + 18.0 * normalized,
            c=style["color"],
            marker=style["marker"],
            alpha=0.28,
            label=style["label"],
        )

    axis.set_title(title)
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    axis.legend(loc="upper right", frameon=True)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_velocity_history(
    times_s: list[float],
    velocities_m_per_s,
    title: str,
    path: Path | None = None,
):
    """Plot the three Cartesian velocity components and speed versus time."""

    data = np.array(velocities_m_per_s)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("velocities_m_per_s must be an array-like of shape (n, 3)")
    speeds = np.sqrt(np.sum(data**2, axis=1))

    figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.plot(1e3 * np.array(times_s), data[:, 0], color="#b91c1c", linewidth=2.0, label=r"$v_x$")
    axis.plot(1e3 * np.array(times_s), data[:, 1], color="#1d4ed8", linewidth=2.0, label=r"$v_y$")
    axis.plot(1e3 * np.array(times_s), data[:, 2], color="#15803d", linewidth=2.0, label=r"$v_z$")
    axis.plot(1e3 * np.array(times_s), speeds, color="#111827", linewidth=1.6, linestyle="--", label="speed")
    axis.set_title(title)
    axis.set_xlabel("Time [ms]")
    axis.set_ylabel("Velocity [m/s]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", frameon=True)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_recoil_direction_vectors(
    absorption_vectors,
    emission_vectors,
    title: str,
    path: Path | None = None,
):
    """Plot handed 3D recoil-direction summaries for absorption and emission."""

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")

    def _draw_vectors(vectors, color: str, label: str):
        if len(vectors) == 0:
            return
        array = np.array(vectors, dtype=float)
        axis.quiver(
            np.zeros(len(array)),
            np.zeros(len(array)),
            np.zeros(len(array)),
            array[:, 0],
            array[:, 1],
            array[:, 2],
            color=color,
            linewidth=1.8,
            alpha=0.8,
            label=label,
        )

    _draw_vectors(absorption_vectors, "#b91c1c", "Absorption")
    _draw_vectors(emission_vectors, "#1d4ed8", "Spontaneous emission")

    limit = 1.05
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.legend(loc="upper right", frameon=True)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis


def plot_apparatus_geometry_3d(beams, path: Path | None = None):
    """Plot beam axes and reference positions for the current apparatus.

    Color encodes circular polarization. Line width encodes optical family.
    Line style encodes apparatus axis.
    """

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")

    polarization_colors = {
        "right": "#d62828",
        "left": "#1d4ed8",
    }
    axis_line_styles = {
        "horizontal_x": "-",
        "horizontal_y": "--",
        "vertical_z": "-.",
    }
    family_widths = {"cooling": 2.4, "repump": 1.2}
    family_alphas = {"cooling": 0.95, "repump": 0.65}
    for beam in beams:
        color = polarization_colors.get(beam.circular_polarization, "#334155")
        line_style = axis_line_styles.get(beam.axis_name, "-")
        reference = np.array(beam.reference_position_m)
        start = reference - 0.03 * np.array(beam.direction)
        stop = reference + 0.03 * np.array(beam.direction)
        axis.plot(
            [1e3 * start[0], 1e3 * stop[0]],
            [1e3 * start[1], 1e3 * stop[1]],
            [1e3 * start[2], 1e3 * stop[2]],
            color=color,
            linewidth=family_widths.get(beam.family, 1.5),
            alpha=family_alphas.get(beam.family, 0.9),
            linestyle=line_style,
        )
        marker = "o" if beam.family == "cooling" else "^"
        axis.scatter(
            [1e3 * beam.reference_position_m[0]],
            [1e3 * beam.reference_position_m[1]],
            [1e3 * beam.reference_position_m[2]],
            color=color,
            s=32 if beam.family == "cooling" else 24,
            marker=marker,
            alpha=family_alphas.get(beam.family, 0.9),
        )
        arrow_origin = reference + 0.01 * np.array(beam.direction)
        arrow_vector = 0.01 * np.array(beam.direction)
        axis.quiver(
            1e3 * arrow_origin[0],
            1e3 * arrow_origin[1],
            1e3 * arrow_origin[2],
            1e3 * arrow_vector[0],
            1e3 * arrow_vector[1],
            1e3 * arrow_vector[2],
            color=color,
            linewidth=1.1,
            alpha=family_alphas.get(beam.family, 0.9),
            arrow_length_ratio=0.28,
        )

    axis.set_title("Cooling And Repump Beam Geometry")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    legend_items = [
        Line2D([0], [0], color=polarization_colors["right"], lw=2.0, label="Right-handed circular"),
        Line2D([0], [0], color=polarization_colors["left"], lw=2.0, label="Left-handed circular"),
        Line2D([0], [0], color="#475569", lw=2.4, label="Cooling beam family"),
        Line2D([0], [0], color="#475569", lw=1.2, label="Repump beam family"),
        Line2D([0], [0], color="#475569", lw=2.0, linestyle=axis_line_styles["horizontal_x"], label="Horizontal x axis"),
        Line2D([0], [0], color="#475569", lw=2.0, linestyle=axis_line_styles["horizontal_y"], label="Horizontal y axis"),
        Line2D([0], [0], color="#475569", lw=2.0, linestyle=axis_line_styles["vertical_z"], label="Vertical z axis"),
    ]
    axis.legend(handles=legend_items, loc="upper right", frameon=True)
    _prepare_output(path)
    if path is not None:
        figure.savefig(path, dpi=180)
    return figure, axis
