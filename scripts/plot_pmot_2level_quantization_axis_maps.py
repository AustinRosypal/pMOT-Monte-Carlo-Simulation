"""Plot 2D maps of the effective pMOT quantization-axis geometry.

The dense color field is the signed out-of-plane angle of the vector-AC-Stark
axis.  Sparse arrows show its in-plane projection; dots point along the
positive omitted Cartesian coordinate and crosses point along the negative
coordinate.  The six focused 1529-nm 1/e^2 envelopes are drawn on every slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

from pmot.mot_strange.trapping_beams import helicity_sign
from pmot.mot_strange.trapping_beams import trapping_beam_intensity_response_m_inv2
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


DEFAULT_OUTPUT_ROOT = Path(
    "outputs/diagnostics/pMOT_2Level/quantization_axis_plane_maps_20261016"
)
DEFAULT_OFFSETS_MM = (-12.0, -1.0, 0.0, 1.0, 12.0)
DEFAULT_PATH_POWER_MW = 38.294486
DEFAULT_EXTENT_MM = 15.0
DEFAULT_GRID_POINTS = 241
DEFAULT_QUIVER_STRIDE = 12
RELATIVE_VECTOR_ZERO_TOLERANCE = 1.0e-12

PLANE_SPECS = {
    "xy": {"u": 0, "v": 1, "normal": 2, "labels": ("x", "y", "z")},
    "xz": {"u": 0, "v": 2, "normal": 1, "labels": ("x", "z", "y")},
    "yz": {"u": 1, "v": 2, "normal": 0, "labels": ("y", "z", "x")},
}

INCIDENT_COLOR = "#f28e2b"
RETRO_COLOR = "#8e5ea2"
IN_PLANE_COLOR = "#202124"
UNDEFINED_COLOR = "#666666"


def _positions_for_plane(
    plane: str,
    offset_mm: float,
    coordinates_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = PLANE_SPECS[plane]
    u_mm, v_mm = np.meshgrid(coordinates_mm, coordinates_mm, indexing="xy")
    positions_mm = np.zeros((u_mm.size, 3), dtype=float)
    positions_mm[:, spec["u"]] = u_mm.ravel()
    positions_mm[:, spec["v"]] = v_mm.ravel()
    positions_mm[:, spec["normal"]] = offset_mm
    return 1.0e-3 * positions_mm, u_mm, v_mm


def _beam_intensities(
    beams,
    positions_m: np.ndarray,
    path_power_w: float,
) -> np.ndarray:
    return np.stack(
        [
            path_power_w
            * trapping_beam_intensity_response_m_inv2(beam, positions_m)
            for beam in beams
        ],
        axis=1,
    )


def _vector_axis_map(
    beams,
    positions_m: np.ndarray,
    path_power_w: float,
    vector_polarizability_si: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intensities = _beam_intensities(beams, positions_m, path_power_w)
    directions = np.asarray([beam.direction for beam in beams], dtype=float)
    signs = np.asarray([helicity_sign(beam.helicity) for beam in beams], dtype=float)
    component_vectors = (
        -vector_polarizability_si
        * (signs[np.newaxis, :] * intensities)[:, :, np.newaxis]
        * directions[np.newaxis, :, :]
    )
    net = np.sum(component_vectors, axis=1)
    magnitude = np.linalg.norm(net, axis=1)
    scale = np.sum(np.linalg.norm(component_vectors, axis=2), axis=1)
    defined = (scale > 0.0) & (
        magnitude > RELATIVE_VECTOR_ZERO_TOLERANCE * scale
    )
    axes = np.full_like(net, np.nan)
    axes[defined] = net[defined] / magnitude[defined, np.newaxis]
    return axes, defined, intensities


def _beam_envelope_ratio(beam, positions_m: np.ndarray) -> np.ndarray:
    waist = np.asarray(beam.waist_position_m, dtype=float)
    direction = np.asarray(beam.direction, dtype=float)
    relative = positions_m - waist
    axial_m = relative @ direction
    radial = relative - axial_m[:, np.newaxis] * direction
    radial_squared_m2 = np.einsum("ij,ij->i", radial, radial)
    local_waist_m = beam.waist_radius_m * np.sqrt(
        1.0 + (axial_m / beam.rayleigh_range_m) ** 2
    )
    return np.sqrt(radial_squared_m2) / local_waist_m


def _draw_beam_envelopes(
    axis,
    beams,
    positions_m: np.ndarray,
    u_mm: np.ndarray,
    v_mm: np.ndarray,
) -> None:
    for beam in beams:
        ratio = _beam_envelope_ratio(beam, positions_m).reshape(u_mm.shape)
        color = INCIDENT_COLOR if beam.propagation_sense == "incident" else RETRO_COLOR
        linestyle = "-" if beam.propagation_sense == "incident" else "--"
        axis.contourf(
            u_mm,
            v_mm,
            ratio,
            levels=(-1.0, 1.0),
            colors=(color,),
            alpha=0.035,
        )
        axis.contour(
            u_mm,
            v_mm,
            ratio,
            levels=(1.0,),
            colors=(color,),
            linewidths=0.8,
            linestyles=linestyle,
            alpha=0.78,
        )


def _plot_map_panel(
    axis,
    *,
    plane: str,
    offset_mm: float,
    beams,
    coordinates_mm: np.ndarray,
    path_power_w: float,
    vector_polarizability_si: float,
    quiver_stride: int,
    color_norm: Normalize,
    cmap,
) -> dict[str, np.ndarray]:
    positions_m, u_mm, v_mm = _positions_for_plane(
        plane,
        offset_mm,
        coordinates_mm,
    )
    axes_flat, defined_flat, intensities = _vector_axis_map(
        beams,
        positions_m,
        path_power_w,
        vector_polarizability_si,
    )
    shape = u_mm.shape
    axes = axes_flat.reshape((*shape, 3))
    defined = defined_flat.reshape(shape)
    spec = PLANE_SPECS[plane]
    normal_component = axes[:, :, spec["normal"]]
    signed_angle_deg = np.degrees(np.arcsin(np.clip(normal_component, -1.0, 1.0)))
    signed_angle_deg[~defined] = np.nan

    angle_image = axis.pcolormesh(
        u_mm,
        v_mm,
        signed_angle_deg,
        cmap=cmap,
        norm=color_norm,
        shading="nearest",
        alpha=0.50,
        rasterized=True,
    )
    _draw_beam_envelopes(axis, beams, positions_m, u_mm, v_mm)

    sample = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))
    sample_defined = defined[sample]
    sample_u = u_mm[sample]
    sample_v = v_mm[sample]
    in_plane_u = axes[:, :, spec["u"]][sample]
    in_plane_v = axes[:, :, spec["v"]][sample]
    quiver = axis.quiver(
        sample_u[sample_defined],
        sample_v[sample_defined],
        0.95 * in_plane_u[sample_defined],
        0.95 * in_plane_v[sample_defined],
        color=IN_PLANE_COLOR,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0042,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
        alpha=0.86,
        zorder=5,
    )

    sample_angle = signed_angle_deg[sample]
    positive = sample_defined & (sample_angle > 2.0)
    negative = sample_defined & (sample_angle < -2.0)
    axis.scatter(
        sample_u[positive],
        sample_v[positive],
        c=sample_angle[positive],
        cmap=cmap,
        norm=color_norm,
        marker="o",
        s=22.0,
        edgecolors=IN_PLANE_COLOR,
        linewidths=0.35,
        zorder=7,
    )
    axis.scatter(
        sample_u[negative],
        sample_v[negative],
        c=sample_angle[negative],
        cmap=cmap,
        norm=color_norm,
        marker="x",
        s=25.0,
        linewidths=1.0,
        zorder=7,
    )
    undefined = ~sample_defined
    axis.scatter(
        sample_u[undefined],
        sample_v[undefined],
        marker="s",
        facecolors="none",
        edgecolors=UNDEFINED_COLOR,
        s=18.0,
        linewidths=0.75,
        zorder=8,
    )

    u_label, v_label, normal_label = spec["labels"]
    axis.axhline(0.0, color="0.3", linewidth=0.45, alpha=0.35)
    axis.axvline(0.0, color="0.3", linewidth=0.45, alpha=0.35)
    axis.set(
        xlabel=f"{u_label} [mm]",
        ylabel=f"{v_label} [mm]",
        title=f"{normal_label} = {offset_mm:+g} mm",
        aspect="equal",
        xlim=(float(coordinates_mm[0]), float(coordinates_mm[-1])),
        ylim=(float(coordinates_mm[0]), float(coordinates_mm[-1])),
    )
    axis.grid(alpha=0.10, linewidth=0.45)
    return {
        "axis": axes,
        "defined": defined,
        "signed_out_of_plane_angle_deg": signed_angle_deg,
        "component_intensities_w_per_m2": intensities.reshape((*shape, len(beams))),
        "angle_image": angle_image,
        "quiver": quiver,
    }


def _figure_legend(normal_label: str) -> list[Line2D]:
    return [
        Line2D(
            (0,),
            (0,),
            color=IN_PLANE_COLOR,
            marker=r"$\rightarrow$",
            linestyle="none",
            markersize=10,
            label="in-plane axis projection",
        ),
        Line2D(
            (0,),
            (0,),
            color=IN_PLANE_COLOR,
            marker="o",
            linestyle="none",
            markersize=5,
            label=f"dot: toward +{normal_label}",
        ),
        Line2D(
            (0,),
            (0,),
            color=IN_PLANE_COLOR,
            marker="x",
            linestyle="none",
            markersize=6,
            label=f"cross: toward -{normal_label}",
        ),
        Line2D(
            (0,),
            (0,),
            color=UNDEFINED_COLOR,
            marker="s",
            markerfacecolor="none",
            linestyle="none",
            markersize=5,
            label="vector shift zero: axis undefined",
        ),
        Line2D(
            (0,),
            (0,),
            color=INCIDENT_COLOR,
            linewidth=1.3,
            linestyle="-",
            label="incident 1529-nm 1/e² envelope",
        ),
        Line2D(
            (0,),
            (0,),
            color=RETRO_COLOR,
            linewidth=1.3,
            linestyle="--",
            label="retro 1529-nm 1/e² envelope",
        ),
    ]


def _plot_plane_family(
    plane: str,
    offsets_mm: tuple[float, ...],
    *,
    beams,
    coordinates_mm: np.ndarray,
    path_power_w: float,
    vector_polarizability_si: float,
    quiver_stride: int,
    output_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    cmap = plt.get_cmap("coolwarm")
    color_norm = Normalize(vmin=-90.0, vmax=90.0)
    figure, axes = plt.subplots(
        1,
        len(offsets_mm),
        figsize=(4.15 * len(offsets_mm), 5.2),
        constrained_layout=True,
        squeeze=False,
    )
    maps = {}
    image = None
    for panel, offset_mm in zip(axes[0], offsets_mm):
        maps[f"{offset_mm:+g}mm"] = _plot_map_panel(
            panel,
            plane=plane,
            offset_mm=offset_mm,
            beams=beams,
            coordinates_mm=coordinates_mm,
            path_power_w=path_power_w,
            vector_polarizability_si=vector_polarizability_si,
            quiver_stride=quiver_stride,
            color_norm=color_norm,
            cmap=cmap,
        )
        image = maps[f"{offset_mm:+g}mm"]["angle_image"]
    normal_label = PLANE_SPECS[plane]["labels"][2]
    figure.suptitle(
        f"pMOT quantization-axis maps in {plane.upper()} planes "
        f"(dot = +{normal_label}, cross = -{normal_label})",
        fontsize=14,
    )
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
    colorbar.set_label(
        f"signed angle toward omitted {normal_label} axis [deg]\n"
        "0° = in plane; ±90° = normal to plane"
    )
    figure.legend(
        handles=_figure_legend(normal_label),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--extent-mm", type=float, default=DEFAULT_EXTENT_MM)
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    parser.add_argument("--quiver-stride", type=int, default=DEFAULT_QUIVER_STRIDE)
    parser.add_argument("--path-power-mw", type=float, default=DEFAULT_PATH_POWER_MW)
    parser.add_argument(
        "--offsets-mm",
        type=float,
        nargs="+",
        default=DEFAULT_OFFSETS_MM,
    )
    arguments = parser.parse_args()
    if arguments.extent_mm <= 0.0:
        raise ValueError("extent-mm must be positive")
    if arguments.grid_points < 21:
        raise ValueError("grid-points must be at least 21")
    if arguments.quiver_stride < 1:
        raise ValueError("quiver-stride must be positive")
    if arguments.path_power_mw <= 0.0:
        raise ValueError("path-power-mw must be positive")

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    offsets_mm = tuple(float(value) for value in arguments.offsets_mm)
    coordinates_mm = np.linspace(
        -arguments.extent_mm,
        arguments.extent_mm,
        arguments.grid_points,
    )
    beams = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    wavelength_nm = 1.0e9 * beams[0].wavelength_m
    vector_polarizability_si = float(
        np.interp(wavelength_nm, table.wavelengths_nm, table.vector_si)
    )

    archive = {
        "coordinate_mm": coordinates_mm,
        "offset_mm": np.asarray(offsets_mm),
    }
    output_paths = {}
    for plane in PLANE_SPECS:
        output_path = output_root / f"quantization_axis_{plane}_planes.png"
        maps = _plot_plane_family(
            plane,
            offsets_mm,
            beams=beams,
            coordinates_mm=coordinates_mm,
            path_power_w=1.0e-3 * arguments.path_power_mw,
            vector_polarizability_si=vector_polarizability_si,
            quiver_stride=arguments.quiver_stride,
            output_path=output_path,
        )
        output_paths[plane] = str(output_path)
        for offset_key, values in maps.items():
            prefix = f"{plane}_{offset_key.replace('+', 'p').replace('-', 'm')}"
            archive[f"{prefix}_axis"] = values["axis"]
            archive[f"{prefix}_defined"] = values["defined"]
            archive[f"{prefix}_angle_deg"] = values[
                "signed_out_of_plane_angle_deg"
            ]
    np.savez_compressed(output_root / "quantization_axis_maps.npz", **archive)

    manifest = {
        "status": "PROVISIONAL_EFFECTIVE_TWO_LEVEL_PMOT_GEOMETRY",
        "meaning": (
            "Stationary-atom vector-AC-Stark quantization axis from the default "
            "six trapping components. It is undefined where the vector sum is zero."
        ),
        "trapping_wavelength_nm": wavelength_nm,
        "vector_polarizability_si": vector_polarizability_si,
        "path_power_mw": arguments.path_power_mw,
        "incident_helicities_xyz": [
            beam.helicity for beam in beams if beam.propagation_sense == "incident"
        ],
        "retro_helicities_xyz": [
            beam.helicity for beam in beams if beam.propagation_sense == "retro"
        ],
        "extent_mm": arguments.extent_mm,
        "grid_points_per_axis": arguments.grid_points,
        "quiver_stride": arguments.quiver_stride,
        "offsets_mm": offsets_mm,
        "signed_angle_definition": (
            "arcsin(n_normal), where normal is the positive omitted Cartesian axis"
        ),
        "symbol_definition": {
            "dot": "positive omitted-axis component",
            "cross": "negative omitted-axis component",
            "open_square": "vector shift zero; physical quantization axis undefined",
        },
        "beam_overlay": "true 1529-nm Gaussian 1/e^2 envelopes",
        "figures": output_paths,
        "data_archive": str(output_root / "quantization_axis_maps.npz"),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
