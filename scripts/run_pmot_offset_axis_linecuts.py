"""Plot centered surrogate-field line cuts at transverse offsets.

The polar-offset figures hold the true transverse radius fixed while varying
its azimuth.  The Cartesian-example figures reproduce offsets such as
``(u, v) = (a, a)`` and ``(a, -a)``; here ``(u, v)`` denotes the two
coordinates transverse to the plotted axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pmot.pmot.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
    surrogate_effective_field_t,
)


AXIS_NAMES = ("x", "y", "z")
TRANSVERSE_NAMES = (("y", "z"), ("x", "z"), ("x", "y"))
TRANSVERSE_INDICES = ((1, 2), (0, 2), (0, 1))
GAUSS_PER_TESLA = 1.0e4


def _line_points(
    axis_index: int,
    axial_coordinate_m: np.ndarray,
    transverse_u_m: float,
    transverse_v_m: float,
) -> np.ndarray:
    points = np.zeros((axial_coordinate_m.size, 3), dtype=float)
    points[:, axis_index] = axial_coordinate_m
    u_index, v_index = TRANSVERSE_INDICES[axis_index]
    points[:, u_index] = transverse_u_m
    points[:, v_index] = transverse_v_m
    return points


def _component_line_g(
    axis_index: int,
    axial_coordinate_m: np.ndarray,
    transverse_u_m: float,
    transverse_v_m: float,
) -> np.ndarray:
    points = _line_points(
        axis_index,
        axial_coordinate_m,
        transverse_u_m,
        transverse_v_m,
    )
    return GAUSS_PER_TESLA * surrogate_effective_field_t(points)[:, axis_index]


def _nearest_zero_mm(coordinate_m: np.ndarray, values_g: np.ndarray) -> float | None:
    exact = np.flatnonzero(values_g == 0.0)
    if exact.size:
        return float(1.0e3 * coordinate_m[exact[np.argmin(np.abs(coordinate_m[exact]))]])
    crossings = np.flatnonzero(values_g[:-1] * values_g[1:] < 0.0)
    if not crossings.size:
        return None
    candidates = []
    for index in crossings:
        x0, x1 = coordinate_m[index : index + 2]
        y0, y1 = values_g[index : index + 2]
        root = x0 - y0 * (x1 - x0) / (y1 - y0)
        candidates.append(root)
    return float(1.0e3 * min(candidates, key=abs))


def _format_axes(
    axis: plt.Axes,
    *,
    row: int,
    column: int,
    radius_or_amplitude_mm: float,
    central: bool,
) -> None:
    axis.axhline(0.0, color="0.65", linewidth=0.7, zorder=0)
    axis.axvline(0.0, color="0.65", linewidth=0.7, zorder=0)
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.set_title(
        rf"$B_{{{AXIS_NAMES[row]}}}$ along {AXIS_NAMES[row]}; "
        + rf"$r_\perp={radius_or_amplitude_mm:g}$ mm",
        fontsize=10,
    )
    if not central:
        axis.set_yscale("asinh", linear_width=1.0)
    if row == 2:
        axis.set_xlabel(f"{AXIS_NAMES[row]} [mm]")
    if column == 0:
        axis.set_ylabel(rf"$B_{{{AXIS_NAMES[row]}}}$ [G]")


def _plot_polar_grid(
    output_path: Path,
    coordinate_m: np.ndarray,
    radii_m: np.ndarray,
    angles_rad: np.ndarray,
    polar_lines_g: np.ndarray,
    on_axis_lines_g: np.ndarray,
    *,
    central: bool,
) -> None:
    column_count = len(radii_m)
    figure, axes = plt.subplots(
        3,
        column_count,
        figsize=(3.9 * column_count + 1.5, 11),
        sharex=True,
        squeeze=False,
    )
    colors = plt.get_cmap("twilight_shifted")(np.linspace(0.0, 1.0, len(angles_rad), endpoint=False))
    for axis_index in range(3):
        for radius_index, radius_m in enumerate(radii_m):
            panel = axes[axis_index, radius_index]
            panel.plot(
                1.0e3 * coordinate_m,
                on_axis_lines_g[axis_index],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="on axis" if axis_index == 0 and radius_index == 0 else None,
            )
            for angle_index, angle_rad in enumerate(angles_rad):
                panel.plot(
                    1.0e3 * coordinate_m,
                    polar_lines_g[axis_index, radius_index, angle_index],
                    color=colors[angle_index],
                    linewidth=1.0,
                    alpha=0.9,
                    label=(
                        rf"$\phi={np.rad2deg(angle_rad):g}^\circ$"
                        if axis_index == 0 and radius_index == 0
                        else None
                    ),
                )
            _format_axes(
                panel,
                row=axis_index,
                column=radius_index,
                radius_or_amplitude_mm=1.0e3 * radius_m,
                central=central,
            )
    figure.suptitle(
        "Effective-field component cuts at fixed true transverse radius\n"
        + ("central linear view" if central else "full-cell view; smooth asinh field scale"),
        fontsize=15,
        y=0.94,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=9,
        fontsize=9,
    )
    figure.subplots_adjust(top=0.84, bottom=0.08, hspace=0.22, wspace=0.20)
    figure.text(
        0.5,
        0.012,
        "All azimuthal curves at a given radius overlap: each Gaussian path is cylindrically symmetric about its axis.",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_cartesian_grid(
    output_path: Path,
    coordinate_m: np.ndarray,
    amplitudes_m: np.ndarray,
    cartesian_lines_g: np.ndarray,
    on_axis_lines_g: np.ndarray,
    *,
    central: bool,
) -> None:
    labels = (r"$(u,v)=(a,0)$", r"$(0,a)$", r"$(a,a)$", r"$(a,-a)$")
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    column_count = len(amplitudes_m)
    figure, axes = plt.subplots(
        3,
        column_count,
        figsize=(3.9 * column_count + 1.5, 11),
        sharex=True,
        squeeze=False,
    )
    for axis_index in range(3):
        u_name, v_name = TRANSVERSE_NAMES[axis_index]
        for amplitude_index, amplitude_m in enumerate(amplitudes_m):
            panel = axes[axis_index, amplitude_index]
            panel.plot(
                1.0e3 * coordinate_m,
                on_axis_lines_g[axis_index],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="on axis" if axis_index == 0 and amplitude_index == 0 else None,
            )
            for pattern_index, label in enumerate(labels):
                panel.plot(
                    1.0e3 * coordinate_m,
                    cartesian_lines_g[axis_index, amplitude_index, pattern_index],
                    color=colors[pattern_index],
                    linewidth=1.15,
                    label=label if axis_index == 0 and amplitude_index == 0 else None,
                )
            panel.set_title(
                rf"$B_{{{AXIS_NAMES[axis_index]}}}$ along {AXIS_NAMES[axis_index]}; "
                rf"$a={1e3*amplitude_m:g}$ mm",
                fontsize=10,
            )
            if not central:
                panel.set_yscale("asinh", linear_width=1.0)
            panel.axhline(0.0, color="0.65", linewidth=0.7, zorder=0)
            panel.axvline(0.0, color="0.65", linewidth=0.7, zorder=0)
            panel.grid(alpha=0.22, linewidth=0.6)
            if axis_index == 2:
                panel.set_xlabel(f"{AXIS_NAMES[axis_index]} [mm]")
            if amplitude_index == 0:
                panel.set_ylabel(rf"$B_{{{AXIS_NAMES[axis_index]}}}$ [G]")
            panel.text(
                0.03,
                0.96,
                rf"$(u,v)=({u_name},{v_name})$",
                transform=panel.transAxes,
                va="top",
                fontsize=8,
                color="0.25",
            )
    figure.suptitle(
        "Effective-field component cuts at requested Cartesian transverse offsets\n"
        + ("central linear view" if central else "full-cell view; smooth asinh field scale"),
        fontsize=15,
        y=0.94,
    )
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=5,
        fontsize=10,
    )
    figure.subplots_adjust(top=0.84, bottom=0.08, hspace=0.22, wspace=0.20)
    figure.text(
        0.5,
        0.012,
        r"Diagonal offsets have $r_\perp=\sqrt{2}a$; $(a,a)$ and $(a,-a)$ overlap by cylindrical symmetry.",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    campaign_root = (
        project_root
        / "outputs"
        / "diagnostics"
        / "pmot"
        / "MOT testing with a strange defined magnetic field"
        / "centered equal-power field 20261002"
    )
    figure_output = campaign_root / "figures" / "effective_field" / "offset_axis_linecuts"
    data_output = campaign_root / "data" / "effective_field" / "offset_axis_linecuts"
    fine_figure_output = figure_output / "fine_offsets_0_to_1mm"
    fine_data_output = data_output / "fine_offsets_0_to_1mm"
    figure_output.mkdir(parents=True, exist_ok=True)
    data_output.mkdir(parents=True, exist_ok=True)
    fine_figure_output.mkdir(parents=True, exist_ok=True)
    fine_data_output.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    full_coordinate_m = np.linspace(-cfg.cell_half_length_m, cfg.cell_half_length_m, 1601)
    central_coordinate_m = np.linspace(-3.0e-3, 3.0e-3, 1201)
    radii_m = 1.0e-3 * np.arange(1.0, 6.0)
    angles_rad = np.deg2rad(np.arange(0.0, 360.0, 45.0))
    amplitudes_m = radii_m.copy()
    cartesian_patterns = np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)))

    def calculate(
        coordinate_m: np.ndarray,
        target_radii_m: np.ndarray,
        target_amplitudes_m: np.ndarray,
    ):
        on_axis = np.empty((3, coordinate_m.size))
        polar = np.empty((3, target_radii_m.size, angles_rad.size, coordinate_m.size))
        cartesian = np.empty(
            (3, target_amplitudes_m.size, len(cartesian_patterns), coordinate_m.size)
        )
        for axis_index in range(3):
            on_axis[axis_index] = _component_line_g(axis_index, coordinate_m, 0.0, 0.0)
            for radius_index, radius_m in enumerate(target_radii_m):
                for angle_index, angle_rad in enumerate(angles_rad):
                    polar[axis_index, radius_index, angle_index] = _component_line_g(
                        axis_index,
                        coordinate_m,
                        radius_m * np.cos(angle_rad),
                        radius_m * np.sin(angle_rad),
                    )
            for amplitude_index, amplitude_m in enumerate(target_amplitudes_m):
                for pattern_index, pattern in enumerate(cartesian_patterns):
                    cartesian[axis_index, amplitude_index, pattern_index] = _component_line_g(
                        axis_index,
                        coordinate_m,
                        amplitude_m * pattern[0],
                        amplitude_m * pattern[1],
                    )
        return on_axis, polar, cartesian

    full_on_axis, full_polar, full_cartesian = calculate(
        full_coordinate_m,
        radii_m,
        amplitudes_m,
    )
    central_on_axis, central_polar, central_cartesian = calculate(
        central_coordinate_m,
        radii_m,
        amplitudes_m,
    )

    _plot_polar_grid(
        figure_output / "01_polar_offsets_full_cell.png",
        full_coordinate_m,
        radii_m,
        angles_rad,
        full_polar,
        full_on_axis,
        central=False,
    )
    _plot_polar_grid(
        figure_output / "02_polar_offsets_central_zoom.png",
        central_coordinate_m,
        radii_m,
        angles_rad,
        central_polar,
        central_on_axis,
        central=True,
    )
    _plot_cartesian_grid(
        figure_output / "03_cartesian_offsets_full_cell.png",
        full_coordinate_m,
        amplitudes_m,
        full_cartesian,
        full_on_axis,
        central=False,
    )
    _plot_cartesian_grid(
        figure_output / "04_cartesian_offsets_central_zoom.png",
        central_coordinate_m,
        amplitudes_m,
        central_cartesian,
        central_on_axis,
        central=True,
    )

    fine_radii_m = 1.0e-3 * np.asarray((0.0, 0.1, 0.25, 0.5, 0.75, 1.0))
    fine_amplitudes_m = fine_radii_m.copy()
    fine_full_on_axis, fine_full_polar, fine_full_cartesian = calculate(
        full_coordinate_m,
        fine_radii_m,
        fine_amplitudes_m,
    )
    fine_central_on_axis, fine_central_polar, fine_central_cartesian = calculate(
        central_coordinate_m,
        fine_radii_m,
        fine_amplitudes_m,
    )
    _plot_polar_grid(
        fine_figure_output / "01_polar_offsets_0_to_1mm_full_cell.png",
        full_coordinate_m,
        fine_radii_m,
        angles_rad,
        fine_full_polar,
        fine_full_on_axis,
        central=False,
    )
    _plot_polar_grid(
        fine_figure_output / "02_polar_offsets_0_to_1mm_central_zoom.png",
        central_coordinate_m,
        fine_radii_m,
        angles_rad,
        fine_central_polar,
        fine_central_on_axis,
        central=True,
    )
    _plot_cartesian_grid(
        fine_figure_output / "03_cartesian_offsets_0_to_1mm_full_cell.png",
        full_coordinate_m,
        fine_amplitudes_m,
        fine_full_cartesian,
        fine_full_on_axis,
        central=False,
    )
    _plot_cartesian_grid(
        fine_figure_output / "04_cartesian_offsets_0_to_1mm_central_zoom.png",
        central_coordinate_m,
        fine_amplitudes_m,
        fine_central_cartesian,
        fine_central_on_axis,
        central=True,
    )

    np.savez_compressed(
        fine_data_output / "fine_offset_axis_linecuts_0_to_1mm.npz",
        full_coordinate_m=full_coordinate_m,
        central_coordinate_m=central_coordinate_m,
        radii_m=fine_radii_m,
        angles_rad=angles_rad,
        amplitudes_m=fine_amplitudes_m,
        cartesian_patterns=cartesian_patterns,
        full_on_axis_component_g=fine_full_on_axis,
        full_polar_component_g=fine_full_polar,
        full_cartesian_component_g=fine_full_cartesian,
        central_on_axis_component_g=fine_central_on_axis,
        central_polar_component_g=fine_central_polar,
        central_cartesian_component_g=fine_central_cartesian,
    )

    fine_angular_spread_g = np.ptp(fine_central_polar, axis=2)
    fine_zero_crossings_mm = {
        axis_name: {
            f"{1e3*radius_m:g}": _nearest_zero_mm(
                central_coordinate_m,
                fine_central_polar[axis_index, radius_index, 0],
            )
            for radius_index, radius_m in enumerate(fine_radii_m)
        }
        for axis_index, axis_name in enumerate(AXIS_NAMES)
    }
    fine_summary = {
        "field_model": "centered equal-power stretched-transition-equivalent surrogate",
        "true_transverse_radii_mm": (1.0e3 * fine_radii_m).tolist(),
        "polar_angles_deg": np.rad2deg(angles_rad).tolist(),
        "cartesian_amplitudes_mm": (1.0e3 * fine_amplitudes_m).tolist(),
        "cartesian_patterns_in_transverse_coordinates": cartesian_patterns.tolist(),
        "maximum_polar_angle_spread_g": float(np.max(fine_angular_spread_g)),
        "nearest_central_zero_crossing_mm_by_true_radius_mm": fine_zero_crossings_mm,
        "full_cell_range_mm": [
            float(1.0e3 * full_coordinate_m[0]),
            float(1.0e3 * full_coordinate_m[-1]),
        ],
        "central_zoom_range_mm": [
            float(1.0e3 * central_coordinate_m[0]),
            float(1.0e3 * central_coordinate_m[-1]),
        ],
        "configuration": cfg.metadata(),
    }
    (fine_data_output / "summary.json").write_text(
        json.dumps(fine_summary, indent=2),
        encoding="utf-8",
    )

    fine_readme = f"""# Fine radially offset effective-field line cuts

Each grid orders its columns as 0, 0.1, 0.25, 0.5, 0.75, and 1.0 mm.
The zero-offset column is the principal-axis reference.  Polar panels overlay
eight azimuths in 45-degree increments; their maximum numerical spread is
`{float(np.max(fine_angular_spread_g)):.6e} G`.

The Cartesian panels use `(u,v)=(a,0)`, `(0,a)`, `(a,a)`, and `(a,-a)`, where
`(u,v)` are the coordinates transverse to the plotted axis.  Diagonal cases
have true radius `sqrt(2)*a`.

Exact arrays are saved in `fine_offset_axis_linecuts_0_to_1mm.npz`.
"""
    (fine_figure_output / "README.md").write_text(fine_readme, encoding="utf-8")

    np.savez_compressed(
        data_output / "offset_axis_linecuts.npz",
        full_coordinate_m=full_coordinate_m,
        central_coordinate_m=central_coordinate_m,
        radii_m=radii_m,
        angles_rad=angles_rad,
        amplitudes_m=amplitudes_m,
        cartesian_patterns=cartesian_patterns,
        full_on_axis_component_g=full_on_axis,
        full_polar_component_g=full_polar,
        full_cartesian_component_g=full_cartesian,
        central_on_axis_component_g=central_on_axis,
        central_polar_component_g=central_polar,
        central_cartesian_component_g=central_cartesian,
    )

    angular_spread_g = np.ptp(central_polar, axis=2)
    zero_crossings_mm = {}
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        zero_crossings_mm[axis_name] = {
            f"{1e3*radius_m:g}": _nearest_zero_mm(
                central_coordinate_m,
                central_polar[axis_index, radius_index, 0],
            )
            for radius_index, radius_m in enumerate(radii_m)
        }
    summary = {
        "field_model": "centered equal-power stretched-transition-equivalent surrogate",
        "component_and_scan_axes": ["Bx along x", "By along y", "Bz along z"],
        "transverse_coordinate_pairs": {
            "Bx along x": ["y", "z"],
            "By along y": ["x", "z"],
            "Bz along z": ["x", "y"],
        },
        "true_transverse_radii_mm": (1.0e3 * radii_m).tolist(),
        "polar_angles_deg": np.rad2deg(angles_rad).tolist(),
        "cartesian_amplitudes_mm": (1.0e3 * amplitudes_m).tolist(),
        "cartesian_patterns_in_transverse_coordinates": cartesian_patterns.tolist(),
        "maximum_polar_angle_spread_g": float(np.max(angular_spread_g)),
        "nearest_central_zero_crossing_mm_by_true_radius_mm": zero_crossings_mm,
        "full_cell_range_mm": [
            float(1.0e3 * full_coordinate_m[0]),
            float(1.0e3 * full_coordinate_m[-1]),
        ],
        "central_zoom_range_mm": [
            float(1.0e3 * central_coordinate_m[0]),
            float(1.0e3 * central_coordinate_m[-1]),
        ],
        "configuration": cfg.metadata(),
    }
    (data_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    zero_lines = []
    for radius_m in radii_m:
        radius_key = f"{1e3*radius_m:g}"
        values = [zero_crossings_mm[name][radius_key] for name in AXIS_NAMES]
        zero_lines.append(
            f"| {radius_key} | " + " | ".join("none" if value is None else f"{value:.6f}" for value in values) + " |"
        )
    readme = f"""# Radially offset effective-field line cuts

These figures evaluate the centered, equal-power surrogate pMOT field along
each Cartesian axis while holding the two transverse coordinates fixed.

## Coordinate mapping

- `Bx along x`: transverse coordinates `(u,v)=(y,z)`.
- `By along y`: transverse coordinates `(u,v)=(x,z)`.
- `Bz along z`: transverse coordinates `(u,v)=(x,y)`.

The polar grids hold the *true* transverse radius fixed at 1, 2, 3, 4, or
5 mm and overlay azimuths from 0 through 315 degrees in 45-degree steps.  The
maximum computed difference between azimuths was
`{float(np.max(angular_spread_g)):.6e} G`; the curves overlap because the
analytic Gaussian intensity depends on transverse position only through
`u^2+v^2`.

The Cartesian grids show the requested `(a,0)`, `(0,a)`, `(a,a)`, and
`(a,-a)` examples.  The diagonal cases have true radius `sqrt(2)*a`, so they
should be compared with each other, not with the two radius-`a` curves.

## Nearest central zero crossings

Values are in millimeters along the scan axis for fixed true transverse radius.

| r_perp [mm] | Bx along x | By along y | Bz along z |
|---:|---:|---:|---:|
{chr(10).join(zero_lines)}

The exact numerical arrays are stored in `offset_axis_linecuts.npz`, and the
machine-readable metadata and zero locations are in `summary.json`.
"""
    (figure_output / "README.md").write_text(readme, encoding="utf-8")
    print(figure_output)


if __name__ == "__main__":
    main()
