"""Generate reproducible geometry-stage validation outputs for the pMOT.

The figures in this module validate beam placement and intensity envelopes only.
They do not calculate an AC Stark Hamiltonian, an effective magnetic field, or
an atomic force.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..beams import axis_direction_from_name
from .configuration import build_pmot_cooling_and_repump_beams
from .configuration import default_pmot_apparatus_config
from .configuration import describe_pmot_configuration
from .configuration import pmot_paths
from .polarizability import differential_shift_coefficients_for_wavelength
from .trapping_beams import DEFAULT_TRAPPING_WAVELENGTH_M
from .trapping_beams import TrappingBeam
from .trapping_beams import beams_for_trapping_axis
from .trapping_beams import build_trapping_beams
from .trapping_beams import total_trapping_intensity_response_m_inv2
from .trapping_beams import trapping_beam_intensity_response_m_inv2
from .trapping_beams import vector_intensity_response_m_inv2


AXIS_LABELS = {
    "horizontal_x": "x",
    "horizontal_y": "y",
    "vertical_z": "z",
}
PLANE_COMPONENTS = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def _wavelength_tag(wavelength_m: float) -> str:
    return f"{1e9 * wavelength_m:.6f}".replace(".", "p") + "nm"


def _axis_positions(axis_name: str, coordinates_m: np.ndarray) -> np.ndarray:
    direction = np.asarray(axis_direction_from_name(axis_name), dtype=float)
    return coordinates_m[:, np.newaxis] * direction


def sample_axis_lineout(
    beams: list[TrappingBeam],
    axis_name: str,
    *,
    extent_m: float = 15.0e-3,
    sample_count: int = 6001,
) -> dict[str, np.ndarray]:
    """Sample incident, retro, scalar, and vector responses on one beam axis."""

    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")
    coordinates_m = np.linspace(-extent_m, extent_m, sample_count)
    positions_m = _axis_positions(axis_name, coordinates_m)
    pair = beams_for_trapping_axis(beams, axis_name)
    incident = next(beam for beam in pair if beam.propagation_sense == "incident")
    retro = next(beam for beam in pair if beam.propagation_sense == "retro")
    incident_response = trapping_beam_intensity_response_m_inv2(incident, positions_m)
    retro_response = trapping_beam_intensity_response_m_inv2(retro, positions_m)
    vector_response = vector_intensity_response_m_inv2(pair, positions_m)
    axis_direction = np.asarray(axis_direction_from_name(axis_name), dtype=float)
    return {
        "coordinate_m": coordinates_m,
        "incident_response_m_inv2": incident_response,
        "retro_response_m_inv2": retro_response,
        "total_response_m_inv2": incident_response + retro_response,
        "signed_axis_response_m_inv2": vector_response @ axis_direction,
    }


def _plane_positions(
    plane: str,
    coordinates_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plane not in PLANE_COMPONENTS:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    horizontal, vertical = np.meshgrid(coordinates_m, coordinates_m)
    points = np.zeros((horizontal.size, 3), dtype=float)
    component_1, component_2 = PLANE_COMPONENTS[plane]
    points[:, component_1] = horizontal.ravel()
    points[:, component_2] = vertical.ravel()
    return horizontal, vertical, points


def sample_intensity_plane(
    beams: list[TrappingBeam],
    plane: str,
    *,
    extent_m: float = 15.0e-3,
    sample_count: int = 601,
) -> dict[str, np.ndarray]:
    """Sample total intensity and the vector-intensity proxy in a trap plane."""

    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")
    coordinates_m = np.linspace(-extent_m, extent_m, sample_count)
    horizontal, vertical, points = _plane_positions(plane, coordinates_m)
    scalar_response = total_trapping_intensity_response_m_inv2(beams, points)
    vector_response = vector_intensity_response_m_inv2(beams, points)
    return {
        "axis_1_m": coordinates_m,
        "axis_2_m": coordinates_m,
        "total_response_m_inv2": scalar_response.reshape(horizontal.shape),
        "vector_response_m_inv2": vector_response.reshape((*horizontal.shape, 3)),
    }


def _save_lineout_csv(path: Path, samples: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(samples)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in zip(*(samples[column] for column in columns)):
            writer.writerow([f"{float(value):.12e}" for value in row])


def plot_axis_intensity_lineout(
    samples: dict[str, np.ndarray],
    *,
    axis_symbol: str,
    wavelength_nm: float,
    focus_offset_mm: float,
    path: Path,
) -> Path:
    coordinates_mm = 1e3 * samples["coordinate_m"]
    figure, (intensity_axis, vector_axis) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.4),
        gridspec_kw={"height_ratios": (1.45, 1.0)},
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#fbfaf6")
    for axis in (intensity_axis, vector_axis):
        axis.set_facecolor("#fbfaf6")
        axis.grid(True, alpha=0.28)

    intensity_axis.semilogy(
        coordinates_mm,
        samples["incident_response_m_inv2"],
        color="#2563eb",
        linewidth=1.8,
        label=f"incident; waist at −{focus_offset_mm:g} mm",
    )
    intensity_axis.semilogy(
        coordinates_mm,
        samples["retro_response_m_inv2"],
        color="#ea580c",
        linewidth=1.8,
        linestyle="--",
        label=f"retro; waist at +{focus_offset_mm:g} mm",
    )
    intensity_axis.semilogy(
        coordinates_mm,
        samples["total_response_m_inv2"],
        color="#111827",
        linewidth=1.45,
        alpha=0.82,
        label="unsigned incident+retro pair sum",
    )
    intensity_axis.set(
        ylabel="Intensity response per path watt [m$^{-2}$]",
        title=(
            f"{wavelength_nm:.6f} nm trapping-light intensity on the "
            f"{axis_symbol}-axis"
        ),
    )
    intensity_axis.legend(loc="upper center", ncols=3, frameon=True, fontsize=8.5)

    vector_axis.plot(
        coordinates_mm,
        samples["signed_axis_response_m_inv2"],
        color="#7c3aed",
        linewidth=2.0,
    )
    central_mask = np.abs(coordinates_mm) <= 2.0
    central_response = samples["signed_axis_response_m_inv2"][central_mask]
    central_limit = 1.08 * float(np.max(np.abs(central_response)))
    center_index = int(np.argmin(np.abs(coordinates_mm)))
    slope_m_inv3 = float(
        np.gradient(
            samples["signed_axis_response_m_inv2"],
            samples["coordinate_m"],
        )[center_index]
    )
    vector_axis.axhline(0.0, color="#374151", linewidth=1.0)
    vector_axis.scatter([0.0], [0.0], color="#7c3aed", s=28, zorder=4)
    vector_axis.set_xlim(-2.0, 2.0)
    vector_axis.set_ylim(-central_limit, central_limit)
    vector_axis.set(
        xlabel=f"{axis_symbol} [mm]",
        ylabel=(
            r"$I_{\mathrm{spin},%s}/P_{\rm path}$ [m$^{-2}$]"
            % axis_symbol
        ),
        title=(
            r"Signed optical-spin intensity factor near the origin "
            r"(zero at center; not $B_{\mathrm{eff}}$)"
        ),
    )
    vector_axis.text(
        0.03,
        0.07,
        (
            r"$\mathbf{I}_{\rm spin}=\sum_j s_j I_j\hat{\mathbf{k}}_j$; "
            rf"origin slope = {slope_m_inv3:.3e} m$^{{-3}}$"
        ),
        transform=vector_axis.transAxes,
        fontsize=9,
        color="#4c1d95",
    )
    for axis in (intensity_axis, vector_axis):
        axis.axvline(-focus_offset_mm, color="#2563eb", linestyle=":", linewidth=1.0)
        axis.axvline(focus_offset_mm, color="#ea580c", linestyle=":", linewidth=1.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def _log_normalized_response(response: np.ndarray) -> np.ndarray:
    peak = float(np.max(response))
    if peak <= 0.0:
        raise ValueError("intensity response must contain a positive value")
    return np.log10(np.maximum(response / peak, 1.0e-9))


def _draw_plane_panel(
    axis,
    plane: str,
    samples: dict[str, np.ndarray],
    focus_offset_mm: float,
):
    values = _log_normalized_response(samples["total_response_m_inv2"])
    coordinates_mm = 1e3 * samples["axis_1_m"]
    image = axis.imshow(
        values,
        origin="lower",
        extent=(
            coordinates_mm[0],
            coordinates_mm[-1],
            coordinates_mm[0],
            coordinates_mm[-1],
        ),
        aspect="equal",
        cmap="magma",
        vmin=-8.0,
        vmax=0.0,
        interpolation="nearest",
    )
    contours = axis.contour(
        coordinates_mm,
        coordinates_mm,
        values,
        levels=(-6.0, -4.0, -2.0),
        colors=("#e5e7eb",),
        linewidths=(0.55, 0.75, 0.95),
        alpha=0.78,
    )
    axis.clabel(contours, inline=True, fontsize=7, fmt=lambda level: f"10$^{{{level:.0f}}}$")
    axis.scatter(
        [-focus_offset_mm, focus_offset_mm, 0.0, 0.0],
        [0.0, 0.0, -focus_offset_mm, focus_offset_mm],
        marker="x",
        color="#f9fafb",
        s=34,
        linewidth=1.2,
        label="in-plane waist centers",
    )
    axis.scatter([0.0], [0.0], marker="+", color="#22d3ee", s=55, linewidth=1.5)
    axis.set(
        xlabel=f"{plane[0]} [mm]",
        ylabel=f"{plane[1]} [mm]",
        title=f"{plane.upper()} plane",
    )
    return image


def plot_intensity_planes(
    plane_samples: dict[str, dict[str, np.ndarray]],
    *,
    wavelength_nm: float,
    focus_offset_mm: float,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    image = None
    for axis, plane in zip(axes, ("xy", "xz", "yz")):
        axis.set_facecolor("#fbfaf6")
        image = _draw_plane_panel(axis, plane, plane_samples[plane], focus_offset_mm)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.018)
    colorbar.set_label(r"$\log_{10}(I/I_{\mathrm{plane,max}})$")
    figure.suptitle(
        f"{wavelength_nm:.6f} nm trapping-light intensity planes; "
        "three symmetric Cartesian round trips",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_individual_intensity_plane(
    plane: str,
    samples: dict[str, np.ndarray],
    *,
    wavelength_nm: float,
    focus_offset_mm: float,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    image = _draw_plane_panel(axis, plane, samples, focus_offset_mm)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}(I/I_{\mathrm{plane,max}})$")
    figure.suptitle(f"{wavelength_nm:.6f} nm trapping-light intensity", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_representative_axis_envelopes(
    beams: list[TrappingBeam],
    *,
    wavelength_nm: float,
    path: Path,
) -> Path:
    pair = beams_for_trapping_axis(beams, "horizontal_x")
    incident = next(beam for beam in pair if beam.propagation_sense == "incident")
    retro = next(beam for beam in pair if beam.propagation_sense == "retro")
    coordinate_m = np.linspace(-15.0e-3, 15.0e-3, 3001)

    def radius_mm(beam: TrappingBeam) -> np.ndarray:
        waist_axis_m = float(np.dot(beam.waist_position_m, beam.direction))
        coordinate_along_k_m = coordinate_m * float(beam.direction[0])
        axial_m = coordinate_along_k_m - waist_axis_m
        return 1e3 * beam.waist_radius_m * np.sqrt(
            1.0 + (axial_m / beam.rayleigh_range_m) ** 2
        )

    incident_radius_mm = radius_mm(incident)
    retro_radius_mm = radius_mm(retro)
    coordinate_mm = 1e3 * coordinate_m
    figure, axis = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.fill_between(
        coordinate_mm,
        -6.35,
        6.35,
        color="#16a34a",
        alpha=0.12,
        label="12.7 mm cooling/repump diameter",
    )
    axis.plot(coordinate_mm, incident_radius_mm, color="#2563eb", linewidth=2.0, label="incident envelope")
    axis.plot(coordinate_mm, -incident_radius_mm, color="#2563eb", linewidth=2.0)
    axis.plot(coordinate_mm, retro_radius_mm, color="#ea580c", linewidth=2.0, linestyle="--", label="retro envelope")
    axis.plot(coordinate_mm, -retro_radius_mm, color="#ea580c", linewidth=2.0, linestyle="--")
    axis.axvline(-10.0, color="#2563eb", linestyle=":", linewidth=1.1)
    axis.axvline(10.0, color="#ea580c", linestyle=":", linewidth=1.1)
    axis.scatter([-10.0, 10.0], [0.0, 0.0], color=("#2563eb", "#ea580c"), zorder=4)
    axis.axvline(0.0, color="#374151", linewidth=0.9, alpha=0.7)
    axis.set(
        xlabel="Position along representative Cartesian path [mm]",
        ylabel="Beam radius / envelope [mm]",
        title=(
            f"Reconstructed {wavelength_nm:.6f} nm round-trip geometry: "
            "waist centers separated by 20 mm"
        ),
        xlim=(-15.0, 15.0),
        ylim=(-7.0, 7.0),
    )
    axis.grid(True, alpha=0.28)
    axis.legend(loc="upper center", ncols=3, frameon=True, fontsize=8.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_apparatus_geometry_3d(
    mot_beams,
    trapping_beams: list[TrappingBeam],
    *,
    wavelength_nm: float,
    path: Path,
) -> Path:
    figure = plt.figure(figsize=(9.4, 8.0), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#fbfaf6")
    extent_m = 18.0e-3

    for axis_name in ("horizontal_x", "horizontal_y", "vertical_z"):
        direction = np.asarray(axis_direction_from_name(axis_name), dtype=float)
        endpoints = np.vstack((-extent_m * direction, extent_m * direction))
        axis.plot(
            1e3 * endpoints[:, 0],
            1e3 * endpoints[:, 1],
            1e3 * endpoints[:, 2],
            color="#16a34a",
            linewidth=8.0,
            alpha=0.18,
        )

    for beam in mot_beams:
        if beam.family != "repump" or beam.propagation_sense != "incident":
            continue
        direction = np.asarray(beam.direction, dtype=float)
        endpoints = np.vstack((-extent_m * direction, extent_m * direction))
        axis.plot(
            1e3 * endpoints[:, 0],
            1e3 * endpoints[:, 1],
            1e3 * endpoints[:, 2],
            color="#0891b2",
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
        )

    for beam in trapping_beams:
        direction = np.asarray(beam.direction, dtype=float)
        waist = np.asarray(beam.waist_position_m, dtype=float)
        start = waist - 8.0e-3 * direction
        stop = waist + 18.0e-3 * direction
        color = "#2563eb" if beam.propagation_sense == "incident" else "#ea580c"
        style = "-" if beam.propagation_sense == "incident" else "--"
        axis.plot(
            [1e3 * start[0], 1e3 * stop[0]],
            [1e3 * start[1], 1e3 * stop[1]],
            [1e3 * start[2], 1e3 * stop[2]],
            color=color,
            linestyle=style,
            linewidth=2.1,
            alpha=0.88,
        )
        axis.scatter(
            [1e3 * waist[0]],
            [1e3 * waist[1]],
            [1e3 * waist[2]],
            color=color,
            marker="x",
            s=42,
        )

    axis.scatter([0.0], [0.0], [0.0], color="#111827", marker="+", s=75, label="trap origin")
    axis.set(
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="z [mm]",
        title=(
            f"No-coil pMOT optical geometry ({wavelength_nm:.6f} nm trapping laser)"
        ),
        xlim=(-18.0, 18.0),
        ylim=(-18.0, 18.0),
        zlim=(-18.0, 18.0),
    )
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.legend(
        handles=[
            Line2D([0], [0], color="#16a34a", linewidth=7, alpha=0.25, label="6 cooling components"),
            Line2D([0], [0], color="#0891b2", linestyle="--", label="6 repump components"),
            Line2D([0], [0], color="#2563eb", linewidth=2, label="3 trapping incident components"),
            Line2D([0], [0], color="#ea580c", linewidth=2, linestyle="--", label="3 trapping retro components"),
            Line2D([0], [0], color="#111827", marker="+", linestyle="None", label="origin; external B = 0"),
        ],
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def run_geometry_validation(
    *,
    root: Path | None = None,
    figure_directory: Path | None = None,
    data_directory: Path | None = None,
    wavelength_m: float = DEFAULT_TRAPPING_WAVELENGTH_M,
    incident_helicity: str = "sigma+",
    retro_helicity: str | None = None,
    lineout_samples: int = 6001,
    plane_samples: int = 601,
) -> dict[str, object]:
    """Generate all first-stage pMOT geometry verification outputs."""

    apparatus = default_pmot_apparatus_config(
        trapping_wavelength_m=wavelength_m,
        incident_trapping_helicity=incident_helicity,
        retro_trapping_helicity=retro_helicity,
    )
    paths = pmot_paths(root)
    tag = f"geometry_validation_{_wavelength_tag(wavelength_m)}"
    figures = figure_directory or paths["outputs_figures"] / tag
    data = data_directory or paths["outputs_fields"] / tag
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    trapping_beams = build_trapping_beams(apparatus.trapping_laser)
    mot_beams = build_pmot_cooling_and_repump_beams(apparatus)
    wavelength_nm = 1e9 * wavelength_m
    focus_offset_mm = 1e3 * apparatus.trapping_laser.focus_offset_m

    output_plots: list[Path] = []
    lineout_files: list[Path] = []
    for axis_name, axis_symbol in AXIS_LABELS.items():
        samples = sample_axis_lineout(
            trapping_beams,
            axis_name,
            sample_count=lineout_samples,
        )
        lineout_path = data / f"trapping_intensity_{axis_symbol}_axis.csv"
        _save_lineout_csv(lineout_path, samples)
        lineout_files.append(lineout_path)
        output_plots.append(
            plot_axis_intensity_lineout(
                samples,
                axis_symbol=axis_symbol,
                wavelength_nm=wavelength_nm,
                focus_offset_mm=focus_offset_mm,
                path=figures / f"trapping_intensity_{axis_symbol}_axis.png",
            )
        )

    plane_data = {
        plane: sample_intensity_plane(
            trapping_beams,
            plane,
            sample_count=plane_samples,
        )
        for plane in ("xy", "xz", "yz")
    }
    for plane, samples in plane_data.items():
        np.savez_compressed(data / f"trapping_intensity_{plane}_plane.npz", **samples)
        output_plots.append(
            plot_individual_intensity_plane(
                plane,
                samples,
                wavelength_nm=wavelength_nm,
                focus_offset_mm=focus_offset_mm,
                path=figures / f"trapping_intensity_{plane}_plane.png",
            )
        )
    output_plots.append(
        plot_intensity_planes(
            plane_data,
            wavelength_nm=wavelength_nm,
            focus_offset_mm=focus_offset_mm,
            path=figures / "trapping_intensity_planes.png",
        )
    )
    output_plots.append(
        plot_representative_axis_envelopes(
            trapping_beams,
            wavelength_nm=wavelength_nm,
            path=figures / "representative_round_trip_envelopes.png",
        )
    )
    output_plots.append(
        plot_apparatus_geometry_3d(
            mot_beams,
            trapping_beams,
            wavelength_nm=wavelength_nm,
            path=figures / "pmot_optical_geometry_3d.png",
        )
    )

    coefficients = differential_shift_coefficients_for_wavelength(wavelength_nm)
    scalar_tensor_residual = (
        coefficients.scalar_mhz_per_intensity
        + coefficients.tensor_mhz_per_intensity
    )

    metadata = {
        "status": "geometry_validated_physics_not_implemented",
        "configuration": describe_pmot_configuration(apparatus),
        "modeling_statement": (
            "One trapping-laser frequency is routed into three Cartesian round-trip paths, "
            "giving six traveling components. Each path uses ideal Gaussian envelopes with "
            "waist centers at -10 mm and +10 mm. Incident and retro intensities are added "
            "incoherently (standing-wave averaged). Results are normalized per watt incident "
            "on each path because total trapping power and its path split are not yet specified."
        ),
        "physics_boundary": (
            "No AC Stark Hamiltonian, effective magnetic field, conservative force, "
            "trap-light scattering, or trajectory dynamics is calculated in this stage."
        ),
        "differential_polarizability_check": {
            "units": "MHz per [mW/(100 um)^2] using the repository conversion",
            "scalar": coefficients.scalar_mhz_per_intensity,
            "vector": coefficients.vector_mhz_per_intensity,
            "tensor": coefficients.tensor_mhz_per_intensity,
            "scalar_plus_tensor_residual": scalar_tensor_residual,
            "absolute_residual_over_absolute_vector": (
                abs(scalar_tensor_residual)
                / abs(coefficients.vector_mhz_per_intensity)
            ),
            "qualification": (
                "This verifies the stored differential-coefficient cancellation only; "
                "the full hyperfine-state angular factors remain for the physics stage."
            ),
        },
        "lineout_sample_count": lineout_samples,
        "plane_sample_count_per_axis": plane_samples,
        "figure_files": [str(path.resolve()) for path in output_plots],
        "lineout_files": [str(path.resolve()) for path in lineout_files],
        "plane_data_files": [
            str((data / f"trapping_intensity_{plane}_plane.npz").resolve())
            for plane in ("xy", "xz", "yz")
        ],
    }
    metadata_path = data / "geometry_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "figures_directory": figures,
        "data_directory": data,
        "plots": output_plots,
        "metadata": metadata_path,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the first-stage pMOT beam geometry")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=1e9 * DEFAULT_TRAPPING_WAVELENGTH_M,
    )
    parser.add_argument(
        "--incident-helicity",
        choices=("sigma+", "sigma-", "pi"),
        default="sigma+",
    )
    parser.add_argument(
        "--retro-helicity",
        choices=("sigma+", "sigma-", "pi"),
        default=None,
    )
    parser.add_argument("--lineout-samples", type=int, default=6001)
    parser.add_argument("--plane-samples", type=int, default=601)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_geometry_validation(
        root=args.root,
        figure_directory=args.figures_dir,
        data_directory=args.data_dir,
        wavelength_m=1e-9 * args.wavelength_nm,
        incident_helicity=args.incident_helicity,
        retro_helicity=args.retro_helicity,
        lineout_samples=args.lineout_samples,
        plane_samples=args.plane_samples,
    )
    print(f"Figures: {result['figures_directory']}", flush=True)
    print(f"Data: {result['data_directory']}", flush=True)
    for plot in result["plots"]:
        print(f"Saved: {plot}", flush=True)
    print(f"Metadata: {result['metadata']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
