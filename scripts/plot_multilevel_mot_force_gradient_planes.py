"""Plot conventional multilevel-MOT forces and their same-axis derivatives.

This diagnostic evaluates the replacement 24-state Section-12 population-rate
solver for stationary atoms in exact anti-Helmholtz fields with selectable
central axial gradients. Gravity is excluded from the radiation-force curves.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel.accelerated import observable, prepare


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "outputs"
    / "diagnostics"
    / "mot_multilevel"
    / "force_gradient_cartesian_planes_15_20gpcm_20261019"
)
FIGURE_ROOT = OUTPUT_ROOT / "figures"

AXIAL_GRADIENTS_G_PER_CM = (15.0, 20.0)
OFFSETS_MM = (0.0, 1.0, 2.0)
SCAN_LIMIT_MM = 12.0
PANEL_SPECS = (
    ("xy", "x", "y"),
    ("xy", "y", "x"),
    ("xz", "x", "z"),
    ("xz", "z", "x"),
    ("yz", "y", "z"),
    ("yz", "z", "y"),
)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

FORCE_COLOR = "#1261a0"
GRADIENT_COLOR = "#d95f02"


def _scan_coordinates_m() -> np.ndarray:
    """Use the prior comparison grid, including dense central sampling."""

    broad = np.linspace(-SCAN_LIMIT_MM, SCAN_LIMIT_MM, 121)
    central = np.linspace(-1.0, 1.0, 81)
    negative_ten_mm = np.linspace(-10.15, -9.85, 41)
    positive_ten_mm = np.linspace(9.85, 10.15, 41)
    values_mm = np.unique(
        np.round(
            np.concatenate(
                (broad, central, negative_ten_mm, positive_ten_mm, [0.0])
            ),
            decimals=9,
        )
    )
    return 1.0e-3 * values_mm


def _position_key(position_m: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(position_m, decimals=12))


def _line_positions(
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
    scan_m: np.ndarray,
) -> np.ndarray:
    positions = np.zeros((len(scan_m), 3), dtype=float)
    positions[:, AXIS_INDEX[scan_axis]] = scan_m
    positions[:, AXIS_INDEX[transverse_axis]] = 1.0e-3 * offset_mm
    return positions


def _all_unique_positions(scan_m: np.ndarray) -> np.ndarray:
    keyed: dict[tuple[float, float, float], np.ndarray] = {}
    for _, scan_axis, transverse_axis in PANEL_SPECS:
        for offset_mm in OFFSETS_MM:
            for position in _line_positions(
                scan_axis, transverse_axis, offset_mm, scan_m
            ):
                keyed[_position_key(position)] = position
    return np.asarray([keyed[key] for key in sorted(keyed)])


def _evaluate_forces(
    axial_gradient_g_per_cm: float,
    positions_m: np.ndarray,
    multilevel_config,
) -> tuple[dict[tuple[float, float, float], np.ndarray], object]:
    coil = default_anti_helmholtz_config(
        target_gradient_g_per_cm=axial_gradient_g_per_cm
    )
    accelerated_data = prepare(config=multilevel_config, coil=coil)
    zero_velocity = np.zeros(3)
    fallback_axis = np.array([0.0, 0.0, 1.0])
    forces: dict[tuple[float, float, float], np.ndarray] = {}

    start = perf_counter()
    total = len(positions_m)
    for index, position in enumerate(positions_m, start=1):
        force_n, _, _, _ = observable(
            position,
            zero_velocity,
            fallback_axis,
            accelerated_data,
        )
        forces[_position_key(position)] = np.asarray(force_n, dtype=float)
        if index == 1 or index % 500 == 0 or index == total:
            print(
                f"[{axial_gradient_g_per_cm:g} G/cm] positions "
                f"{index}/{total}; elapsed={perf_counter() - start:.1f} s",
                flush=True,
            )
    return forces, coil


def _assemble_rows(
    axial_gradient_g_per_cm: float,
    scan_m: np.ndarray,
    forces: dict[tuple[float, float, float], np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plane, scan_axis, transverse_axis in PANEL_SPECS:
        component = AXIS_INDEX[scan_axis]
        for offset_mm in OFFSETS_MM:
            positions = _line_positions(
                scan_axis,
                transverse_axis,
                offset_mm,
                scan_m,
            )
            force_n = np.asarray(
                [forces[_position_key(position)][component] for position in positions]
            )
            force_gradient_n_per_m = np.gradient(
                force_n,
                scan_m,
                edge_order=2,
            )
            for index, coordinate_m in enumerate(scan_m):
                rows.append(
                    {
                        "axial_gradient_g_per_cm": axial_gradient_g_per_cm,
                        "plane": plane,
                        "scan_axis": scan_axis,
                        "transverse_axis": transverse_axis,
                        "transverse_offset_mm": offset_mm,
                        "scan_coordinate_mm": 1.0e3 * coordinate_m,
                        "same_axis_force_n": force_n[index],
                        "same_axis_force_zN": 1.0e21 * force_n[index],
                        "same_axis_force_gradient_n_per_m": (
                            force_gradient_n_per_m[index]
                        ),
                        "same_axis_force_gradient_1e19_n_per_m": (
                            1.0e19 * force_gradient_n_per_m[index]
                        ),
                    }
                )
    return rows


def _panel_rows(
    rows: list[dict[str, object]],
    *,
    axial_gradient_g_per_cm: float,
    plane: str,
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
) -> list[dict[str, object]]:
    subset = [
        row
        for row in rows
        if row["axial_gradient_g_per_cm"] == axial_gradient_g_per_cm
        and row["plane"] == plane
        and row["scan_axis"] == scan_axis
        and row["transverse_axis"] == transverse_axis
        and row["transverse_offset_mm"] == offset_mm
    ]
    subset.sort(key=lambda row: float(row["scan_coordinate_mm"]))
    if len(subset) != 271:
        raise RuntimeError(
            f"Expected 271 samples for {axial_gradient_g_per_cm:g} G/cm, "
            f"{plane}/{scan_axis}/{transverse_axis}/{offset_mm:g} mm; "
            f"found {len(subset)}"
        )
    return subset


def _cut_description(scan_axis: str, transverse_axis: str, offset_mm: float) -> str:
    fixed = {"x": 0.0, "y": 0.0, "z": 0.0}
    fixed[transverse_axis] = offset_mm
    other_axes = [axis for axis in "xyz" if axis != scan_axis]
    return ", ".join(f"{axis}={fixed[axis]:+g} mm" for axis in other_axes)


def _plot_offset(
    rows: list[dict[str, object]],
    axial_gradient_g_per_cm: float,
    offset_mm: float,
) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 13.5), sharex=True)

    for panel_axis, (plane, scan_axis, transverse_axis) in zip(
        axes.flat, PANEL_SPECS, strict=True
    ):
        subset = _panel_rows(
            rows,
            axial_gradient_g_per_cm=axial_gradient_g_per_cm,
            plane=plane,
            scan_axis=scan_axis,
            transverse_axis=transverse_axis,
            offset_mm=offset_mm,
        )
        coordinate_mm = np.asarray(
            [row["scan_coordinate_mm"] for row in subset], dtype=float
        )
        force_zN = np.asarray([row["same_axis_force_zN"] for row in subset])
        gradient_scaled = np.asarray(
            [row["same_axis_force_gradient_1e19_n_per_m"] for row in subset]
        )

        panel_axis.axhline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
        panel_axis.axvline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
        panel_axis.plot(
            coordinate_mm,
            force_zN,
            color=FORCE_COLOR,
            linewidth=2.2,
            zorder=3,
        )

        gradient_axis = panel_axis.twinx()
        gradient_axis.plot(
            coordinate_mm,
            gradient_scaled,
            color=GRADIENT_COLOR,
            linewidth=2.0,
            linestyle="--",
            zorder=2,
        )
        gradient_axis.axhline(0.0, color="0.25", linewidth=0.7, alpha=0.45)

        panel_axis.set_title(
            f"{plane.upper()} plane: along {scan_axis}-axis\n"
            f"{_cut_description(scan_axis, transverse_axis, offset_mm)}",
            fontsize=11.5,
        )
        panel_axis.set_ylabel(rf"$F_{scan_axis}$ [zN]", color=FORCE_COLOR)
        gradient_axis.set_ylabel(
            rf"$\partial F_{scan_axis}/\partial {scan_axis}$ "
            r"[$10^{-19}$ N/m]",
            color=GRADIENT_COLOR,
        )
        panel_axis.tick_params(axis="y", colors=FORCE_COLOR)
        gradient_axis.tick_params(axis="y", colors=GRADIENT_COLOR)
        panel_axis.grid(True, which="major", alpha=0.22)
        panel_axis.grid(True, which="minor", alpha=0.08)
        panel_axis.minorticks_on()
        panel_axis.set_xlim(-SCAN_LIMIT_MM, SCAN_LIMIT_MM)
        panel_axis.set_xlabel(f"{scan_axis} [mm]")

    if offset_mm == 0.0:
        offset_label = "On-axis"
        filename_suffix = "on_axis"
    else:
        offset_label = f"+{offset_mm:g} mm in-plane transverse offset"
        filename_suffix = f"offset_{offset_mm:g}mm"

    fig.suptitle(
        "Regular Multilevel MOT: Force and Same-Axis Force Gradient",
        fontsize=17,
        y=0.995,
    )
    fig.text(
        0.5,
        0.969,
        f"{axial_gradient_g_per_cm:g} G/cm axial magnetic-field gradient — "
        f"{offset_label}",
        ha="center",
        va="top",
        fontsize=14,
    )
    legend_handles = [
        Line2D([0], [0], color=FORCE_COLOR, linewidth=2.2, label="same-axis force"),
        Line2D(
            [0],
            [0],
            color=GRADIENT_COLOR,
            linewidth=2.0,
            linestyle="--",
            label="same-axis spatial force gradient",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.006),
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.98, 0.925), h_pad=2.2, w_pad=2.4)

    output_path = FIGURE_ROOT / (
        f"multilevel_mot_force_gradient_{axial_gradient_g_per_cm:g}gpcm_"
        f"{filename_suffix}.png"
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_csv(rows: list[dict[str, object]]) -> Path:
    path = OUTPUT_ROOT / "multilevel_mot_force_gradient_linecuts_15_20gpcm.csv"
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _central_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for gradient in AXIAL_GRADIENTS_G_PER_CM:
        for offset in OFFSETS_MM:
            for plane, scan_axis, transverse_axis in PANEL_SPECS:
                origin = next(
                    row
                    for row in rows
                    if row["axial_gradient_g_per_cm"] == gradient
                    and row["transverse_offset_mm"] == offset
                    and row["plane"] == plane
                    and row["scan_axis"] == scan_axis
                    and row["transverse_axis"] == transverse_axis
                    and abs(float(row["scan_coordinate_mm"])) < 1.0e-15
                )
                summary.append(
                    {
                        "axial_gradient_g_per_cm": gradient,
                        "transverse_offset_mm": offset,
                        "plane": plane,
                        "scan_axis": scan_axis,
                        "same_axis_force_at_scan_origin_zN": origin[
                            "same_axis_force_zN"
                        ],
                        "same_axis_force_gradient_at_scan_origin_n_per_m": origin[
                            "same_axis_force_gradient_n_per_m"
                        ],
                    }
                )
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    multilevel_config = replace(
        default_multilevel_mot_config(),
        include_gravity=False,
    )
    scan_m = _scan_coordinates_m()
    positions_m = _all_unique_positions(scan_m)
    rows: list[dict[str, object]] = []
    coils = {}

    for axial_gradient_g_per_cm in AXIAL_GRADIENTS_G_PER_CM:
        forces, coil = _evaluate_forces(
            axial_gradient_g_per_cm,
            positions_m,
            multilevel_config,
        )
        coils[str(axial_gradient_g_per_cm)] = asdict(coil)
        rows.extend(
            _assemble_rows(
                axial_gradient_g_per_cm,
                scan_m,
                forces,
            )
        )

    figure_paths = [
        _plot_offset(rows, gradient, offset)
        for gradient in AXIAL_GRADIENTS_G_PER_CM
        for offset in OFFSETS_MM
    ]
    csv_path = _write_csv(rows)

    manifest = {
        "schema": "pmot.multilevel-mot-force-gradient-cartesian-planes.v2",
        "scientific_status": (
            "local stationary-force diagnostic from the replacement physical "
            "24-state population-rate kernel"
        ),
        "solver": (
            "compiled Section-12 solver tested against the reference 24-state "
            "implementation"
        ),
        "configuration": asdict(multilevel_config),
        "anti_helmholtz_coils": coils,
        "sampling": {
            "stationary_atom": True,
            "gravity_included": False,
            "axial_gradients_g_per_cm": list(AXIAL_GRADIENTS_G_PER_CM),
            "scan_range_mm": [-SCAN_LIMIT_MM, SCAN_LIMIT_MM],
            "scan_sample_count_per_cut": len(scan_m),
            "transverse_offsets_mm": list(OFFSETS_MM),
            "offset_definition": (
                "positive in-plane coordinate perpendicular to the scan axis; "
                "the coordinate normal to the named plane is zero"
            ),
            "force_gradient_definition": (
                "numpy.gradient of the saved same-axis force with respect to the "
                "saved nonuniform scan coordinate, using edge_order=2"
            ),
        },
        "units": {
            "force_plot": "zN = 1e-21 N",
            "force_gradient_plot": "1e-19 N/m",
            "position_plot": "mm",
        },
        "central_summary": _central_summary(rows),
        "outputs": {
            "linecut_csv": str(csv_path.relative_to(REPOSITORY_ROOT)),
            "figures": [str(path.relative_to(REPOSITORY_ROOT)) for path in figure_paths],
        },
        "interpretation": (
            "A negative same-axis force gradient is locally restoring for a "
            "stationary atom. These line cuts are not a substitute for a full "
            "three-dimensional force-Jacobian stability analysis."
        ),
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    readme_path = OUTPUT_ROOT / "README.md"
    readme_path.write_text(
        "# Conventional multilevel-MOT force gradients at 15 and 20 G/cm\n\n"
        "Six 3-by-2 figures show the same-axis stationary radiation force and "
        "its numerical same-axis spatial derivative in the XY, XZ, and YZ "
        "planes. Separate figures cover transverse offsets of 0, +1, and +2 "
        "mm for 15 and 20 G/cm axial anti-Helmholtz gradients. Gravity is "
        "excluded.\n\n"
        "The orange curve is computed directly from the saved blue force samples "
        "with `numpy.gradient(force, coordinate, edge_order=2)`. Negative force "
        "gradient is locally restoring.\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(figure_paths)} figures to {FIGURE_ROOT}")
    for path in figure_paths:
        print(path)
    print(manifest_path)
    print(csv_path)


if __name__ == "__main__":
    main()
