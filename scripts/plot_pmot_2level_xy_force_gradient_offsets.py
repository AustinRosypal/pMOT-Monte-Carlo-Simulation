"""Plot pMOT_2Level XY force-gradient cuts at three transverse offsets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "three_model_force_gradient_review_20261019"
    / "pMOT_2Level_xy_linecuts_plus_minus_25mm.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "three_model_force_gradient_review_20261019"
    / "figures"
    / "pMOT_2Level_xy_force_gradient_offsets_0_1_2mm.png"
)

OFFSETS_MM = (0.0, 1.0, 2.0)
SCAN_LIMIT_MM = 25.0
TRAPPING_PATH_POWERS_W = (19.147243e-3, 19.147243e-3, 38.294486e-3)
GLASS_WALL_CENTERS_MM = (-10.0, 10.0)
GLASS_WALL_WIDTH_MM = 5.0
GLASS_WALL_COLOR = "#64748b"
X_COLOR = "#2563eb"
Y_COLOR = "#d97706"


def _scan_coordinates_m() -> np.ndarray:
    broad = np.linspace(-SCAN_LIMIT_MM, SCAN_LIMIT_MM, 251)
    central = np.linspace(-1.0, 1.0, 81)
    negative_focus = np.linspace(-10.15, -9.85, 41)
    positive_focus = np.linspace(9.85, 10.15, 41)
    values_mm = np.unique(
        np.round(
            np.concatenate((broad, central, negative_focus, positive_focus, [0.0])),
            decimals=9,
        )
    )
    return 1.0e-3 * values_mm


def _evaluate_xy_linecuts() -> pd.DataFrame:
    simple = replace(default_simple_mot_config(), include_gravity=False)
    cooling = build_pmot_2level_cooling_beams(simple_config=simple)
    trapping = build_pmot_2level_trapping_beams()
    polarizability = load_pmot_2level_polarizability_table()
    scan_m = _scan_coordinates_m()
    rows: list[dict[str, object]] = []

    for scan_axis, transverse_axis, component in (("x", "y", 0), ("y", "x", 1)):
        for offset_mm in OFFSETS_MM:
            positions = np.zeros((len(scan_m), 3), dtype=float)
            positions[:, component] = scan_m
            positions[:, 1 - component] = 1.0e-3 * offset_mm
            force_n = np.empty(len(scan_m), dtype=float)
            previous_axis = np.array([0.0, 0.0, 1.0])

            for index, position in enumerate(positions):
                observable = evaluate_pmot_2level_local_observable(
                    cooling,
                    trapping,
                    position,
                    np.zeros(3),
                    path_powers_w=TRAPPING_PATH_POWERS_W,
                    previous_axis=previous_axis,
                    simple_config=simple,
                    polarizability_table=polarizability,
                )
                force_n[index] = observable.force_n[component]
                if observable.stark.quantization_axis_from_vector_shift:
                    previous_axis = np.asarray(
                        observable.stark.quantization_axis,
                        dtype=float,
                    )

            gradient_n_per_m = np.gradient(force_n, scan_m, edge_order=2)
            for coordinate_m, force, gradient in zip(
                scan_m,
                force_n,
                gradient_n_per_m,
                strict=True,
            ):
                rows.append(
                    {
                        "model": "pMOT_2Level",
                        "plane": "xy",
                        "scan_axis": scan_axis,
                        "transverse_axis": transverse_axis,
                        "transverse_offset_mm": offset_mm,
                        "scan_coordinate_mm": 1.0e3 * coordinate_m,
                        "same_axis_force_n": force,
                        "same_axis_force_gradient_n_per_m": gradient,
                        "same_axis_force_gradient_1e19_n_per_m": 1.0e19 * gradient,
                    }
                )
            print(
                f"evaluated {scan_axis} cut at {offset_mm:g} mm transverse offset",
                flush=True,
            )

    result = pd.DataFrame(rows)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(DATA_PATH, index=False)
    return result


def _xy_cut(
    frame: pd.DataFrame,
    *,
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
) -> pd.DataFrame:
    cut = frame[
        (frame["model"] == "pMOT_2Level")
        & (frame["plane"] == "xy")
        & (frame["scan_axis"] == scan_axis)
        & (frame["transverse_axis"] == transverse_axis)
        & (frame["transverse_offset_mm"] == offset_mm)
    ].sort_values("scan_coordinate_mm")
    if cut.empty:
        raise RuntimeError(
            f"No pMOT_2Level XY data for {scan_axis} cut at {offset_mm:g} mm"
        )
    return cut


def main() -> None:
    frame = _evaluate_xy_linecuts()
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 13.0), sharex=True)

    for axis, offset_mm in zip(axes, OFFSETS_MM, strict=True):
        for center_mm in GLASS_WALL_CENTERS_MM:
            axis.axvspan(
                center_mm - 0.5 * GLASS_WALL_WIDTH_MM,
                center_mm + 0.5 * GLASS_WALL_WIDTH_MM,
                color=GLASS_WALL_COLOR,
                alpha=0.12,
                zorder=0,
            )
            axis.axvline(
                center_mm,
                color="0.35",
                linestyle=":",
                linewidth=0.9,
                zorder=1,
            )

        x_cut = _xy_cut(
            frame,
            scan_axis="x",
            transverse_axis="y",
            offset_mm=offset_mm,
        )
        y_cut = _xy_cut(
            frame,
            scan_axis="y",
            transverse_axis="x",
            offset_mm=offset_mm,
        )
        axis.plot(
            x_cut["scan_coordinate_mm"],
            x_cut["same_axis_force_gradient_1e19_n_per_m"],
            color=X_COLOR,
            linewidth=2.2,
            label=r"$\partial F_x/\partial x$ ($y$ fixed)",
        )
        axis.plot(
            y_cut["scan_coordinate_mm"],
            y_cut["same_axis_force_gradient_1e19_n_per_m"],
            color=Y_COLOR,
            linewidth=2.2,
            linestyle="--",
            label=r"$\partial F_y/\partial y$ ($x$ fixed)",
        )
        axis.axhline(0.0, color="0.2", linewidth=0.9, alpha=0.8)
        axis.axvline(0.0, color="0.2", linewidth=0.9, alpha=0.8)
        axis.set_xlim(-SCAN_LIMIT_MM, SCAN_LIMIT_MM)
        axis.set_ylabel(r"Force gradient [$10^{-19}$ N/m]")
        axis.grid(True, which="major", alpha=0.22)
        axis.grid(True, which="minor", alpha=0.08)
        axis.minorticks_on()

        if offset_mm == 0.0:
            title = "On-axis: x cut at y=0; y cut at x=0"
        else:
            title = (
                f"+{offset_mm:g} mm transverse offset: "
                f"x cut at y=+{offset_mm:g} mm; "
                f"y cut at x=+{offset_mm:g} mm"
            )
        axis.set_title(title, fontsize=12)

    axes[-1].set_xlabel("Scan coordinate [mm]")
    figure.suptitle(
        "2-Level pMOT Force Gradients in the x-y Plane",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.973,
        "Stationary atom; gravity excluded; negative gradient is locally restoring",
        ha="center",
        va="top",
        fontsize=12,
    )
    figure.legend(
        handles=(
            Line2D(
                [0],
                [0],
                color=X_COLOR,
                linewidth=2.2,
                label=r"$\partial F_x/\partial x$",
            ),
            Line2D(
                [0],
                [0],
                color=Y_COLOR,
                linewidth=2.2,
                linestyle="--",
                label=r"$\partial F_y/\partial y$",
            ),
            Patch(
                facecolor=GLASS_WALL_COLOR,
                alpha=0.20,
                label="glass-cell walls (center ±10 mm, width 5 mm)",
            ),
        ),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.006),
        frameon=False,
    )
    figure.tight_layout(rect=(0.04, 0.055, 0.98, 0.94), h_pad=2.0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
