"""Plot multilevel-MOT quantization axes from the anti-Helmholtz field.

The plotting conventions intentionally match the pMOT axis-map diagnostic:
the dense color field gives signed out-of-plane angle, arrows show the
in-plane projection, dots point along the positive omitted Cartesian axis,
and crosses point along its negative.  The six overlapping counterpropagating
cooling components are shown as three 1/e^2 cylindrical envelopes.
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

from pmot.magnetic_fields import anti_helmholtz_axial_gradient_t_per_m
from pmot.magnetic_fields import anti_helmholtz_field_t
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.simulation import build_multilevel_cooling_beams


DEFAULT_OUTPUT_ROOT = Path(
    "outputs/diagnostics/mot_multilevel/quantization_axis_plane_maps_20261016"
)
DEFAULT_OFFSETS_MM = (-12.0, -1.0, 0.0, 1.0, 12.0)
DEFAULT_EXTENT_MM = 15.0
DEFAULT_GRID_POINTS = 241
DEFAULT_QUIVER_STRIDE = 12

PLANE_SPECS = {
    "xy": {"u": 0, "v": 1, "normal": 2, "labels": ("x", "y", "z")},
    "xz": {"u": 0, "v": 2, "normal": 1, "labels": ("x", "z", "y")},
    "yz": {"u": 1, "v": 2, "normal": 0, "labels": ("y", "z", "x")},
}

COOLING_COLOR = "#4c78a8"
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


def _magnetic_axis_map(
    positions_m: np.ndarray,
    coil_config,
    field_epsilon_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bx, by, bz = anti_helmholtz_field_t(
        positions_m[:, 0],
        positions_m[:, 1],
        positions_m[:, 2],
        coil_config,
    )
    field_t = np.column_stack((bx, by, bz))
    magnitude_t = np.linalg.norm(field_t, axis=1)
    defined = np.all(np.isfinite(field_t), axis=1) & (
        magnitude_t > field_epsilon_t
    )
    axes = np.full_like(field_t, np.nan)
    axes[defined] = field_t[defined] / magnitude_t[defined, np.newaxis]
    return axes, defined, field_t, magnitude_t


def _draw_cooling_envelopes(
    axis,
    cooling_beams,
    positions_m: np.ndarray,
    u_mm: np.ndarray,
    v_mm: np.ndarray,
) -> None:
    seen_axes: set[str] = set()
    for beam in cooling_beams:
        if beam.axis_name in seen_axes:
            continue
        seen_axes.add(beam.axis_name)
        direction = np.asarray(beam.direction, dtype=float)
        reference = np.asarray(beam.reference_position_m, dtype=float)
        relative = positions_m - reference
        axial_m = relative @ direction
        radial = relative - axial_m[:, np.newaxis] * direction
        radial_m = np.linalg.norm(radial, axis=1)
        local_waist_m = beam.gaussian_beam().waist_radius_m * np.sqrt(
            1.0 + (axial_m / beam.gaussian_beam().rayleigh_range_m) ** 2
        )
        ratio = (radial_m / local_waist_m).reshape(u_mm.shape)
        axis.contourf(
            u_mm,
            v_mm,
            ratio,
            levels=(-1.0, 1.0),
            colors=(COOLING_COLOR,),
            alpha=0.045,
        )
        axis.contour(
            u_mm,
            v_mm,
            ratio,
            levels=(1.0,),
            colors=(COOLING_COLOR,),
            linewidths=0.9,
            linestyles="--",
            alpha=0.82,
        )


def _plot_map_panel(
    axis,
    *,
    plane: str,
    offset_mm: float,
    coordinates_mm: np.ndarray,
    coil_config,
    field_epsilon_t: float,
    cooling_beams,
    quiver_stride: int,
    color_norm: Normalize,
    cmap,
) -> dict[str, np.ndarray]:
    positions_m, u_mm, v_mm = _positions_for_plane(
        plane,
        offset_mm,
        coordinates_mm,
    )
    axes_flat, defined_flat, field_flat, magnitude_flat = _magnetic_axis_map(
        positions_m,
        coil_config,
        field_epsilon_t,
    )
    shape = u_mm.shape
    axes = axes_flat.reshape((*shape, 3))
    defined = defined_flat.reshape(shape)
    field_t = field_flat.reshape((*shape, 3))
    magnitude_t = magnitude_flat.reshape(shape)
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
    _draw_cooling_envelopes(axis, cooling_beams, positions_m, u_mm, v_mm)

    sample = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))
    sample_defined = defined[sample]
    sample_u = u_mm[sample]
    sample_v = v_mm[sample]
    in_plane_u = axes[:, :, spec["u"]][sample]
    in_plane_v = axes[:, :, spec["v"]][sample]
    axis.quiver(
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
        "magnetic_field_t": field_t,
        "magnetic_field_magnitude_t": magnitude_t,
        "angle_image": angle_image,
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
            label="in-plane B-axis projection",
        ),
        Line2D(
            (0,),
            (0,),
            color=IN_PLANE_COLOR,
            marker="o",
            linestyle="none",
            markersize=5,
            label=f"dot: B toward +{normal_label}",
        ),
        Line2D(
            (0,),
            (0,),
            color=IN_PLANE_COLOR,
            marker="x",
            linestyle="none",
            markersize=6,
            label=f"cross: B toward -{normal_label}",
        ),
        Line2D(
            (0,),
            (0,),
            color=UNDEFINED_COLOR,
            marker="s",
            markerfacecolor="none",
            linestyle="none",
            markersize=5,
            label="B below axis threshold: axis undefined",
        ),
        Line2D(
            (0,),
            (0,),
            color=COOLING_COLOR,
            linewidth=1.3,
            linestyle="--",
            label="six cooling components: 1/e² envelopes",
        ),
    ]


def _plot_plane_family(
    plane: str,
    offsets_mm: tuple[float, ...],
    *,
    coordinates_mm: np.ndarray,
    coil_config,
    field_epsilon_t: float,
    cooling_beams,
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
            coordinates_mm=coordinates_mm,
            coil_config=coil_config,
            field_epsilon_t=field_epsilon_t,
            cooling_beams=cooling_beams,
            quiver_stride=quiver_stride,
            color_norm=color_norm,
            cmap=cmap,
        )
        image = maps[f"{offset_mm:+g}mm"]["angle_image"]
    normal_label = PLANE_SPECS[plane]["labels"][2]
    figure.suptitle(
        f"Multilevel MOT magnetic-field quantization axes in {plane.upper()} planes "
        f"(dot = +{normal_label}, cross = -{normal_label})",
        fontsize=14,
    )
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
    colorbar.set_label(
        f"signed B-axis angle toward omitted {normal_label} axis [deg]\n"
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

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    offsets_mm = tuple(float(value) for value in arguments.offsets_mm)
    coordinates_mm = np.linspace(
        -arguments.extent_mm,
        arguments.extent_mm,
        arguments.grid_points,
    )
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=10.0)
    multilevel_config = default_multilevel_mot_config()
    cooling_beams = build_multilevel_cooling_beams(config=multilevel_config)
    gradient_t_per_m = anti_helmholtz_axial_gradient_t_per_m(
        coil.radius_m,
        coil.turns_per_coil,
        coil.current_a,
    )

    archive = {
        "coordinate_mm": coordinates_mm,
        "offset_mm": np.asarray(offsets_mm),
    }
    output_paths = {}
    for plane in PLANE_SPECS:
        output_path = output_root / f"magnetic_quantization_axis_{plane}_planes.png"
        maps = _plot_plane_family(
            plane,
            offsets_mm,
            coordinates_mm=coordinates_mm,
            coil_config=coil,
            field_epsilon_t=multilevel_config.magnetic_field_epsilon_t,
            cooling_beams=cooling_beams,
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
            archive[f"{prefix}_field_t"] = values["magnetic_field_t"]
            archive[f"{prefix}_field_magnitude_t"] = values[
                "magnetic_field_magnitude_t"
            ]
    np.savez_compressed(output_root / "magnetic_quantization_axis_maps.npz", **archive)

    manifest = {
        "status": "MULTILEVEL_MOT_MAGNETIC_QUANTIZATION_AXIS_GEOMETRY",
        "meaning": (
            "Quantization axis n=B/|B| from the exact shared anti-Helmholtz "
            "field used by the multilevel MOT."
        ),
        "coil_radius_mm": 1.0e3 * coil.radius_m,
        "coil_center_separation_mm": 1.0e3 * coil.center_separation_m,
        "coil_turns_per_coil": coil.turns_per_coil,
        "coil_current_a": coil.current_a,
        "axial_gradient_g_per_cm": 100.0 * gradient_t_per_m,
        "magnetic_field_axis_threshold_t": multilevel_config.magnetic_field_epsilon_t,
        "cooling_power_mw_per_component": 1.0e3 * multilevel_config.cooling_power_w_per_beam,
        "cooling_beam_diameter_mm": 2.0e3 * cooling_beams[0].beam_radius_m,
        "cooling_component_count": len(cooling_beams),
        "extent_mm": arguments.extent_mm,
        "grid_points_per_axis": arguments.grid_points,
        "quiver_stride": arguments.quiver_stride,
        "offsets_mm": offsets_mm,
        "signed_angle_definition": (
            "arcsin(n_normal), where normal is the positive omitted Cartesian axis"
        ),
        "symbol_definition": {
            "dot": "positive omitted-axis magnetic-field component",
            "cross": "negative omitted-axis magnetic-field component",
            "open_square": "|B| at or below multilevel quantization-axis threshold",
        },
        "beam_overlay": (
            "three overlapping counterpropagating cooling-path 1/e^2 envelopes; "
            "six traveling components total"
        ),
        "comparison_note": (
            "Extent, offsets, grid resolution, quiver stride, signed-angle scale, "
            "and symbol conventions match the pMOT axis-map diagnostic."
        ),
        "figures": output_paths,
        "data_archive": str(output_root / "magnetic_quantization_axis_maps.npz"),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
