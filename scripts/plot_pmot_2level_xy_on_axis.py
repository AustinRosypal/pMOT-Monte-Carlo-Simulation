"""Plot the on-axis stationary XY cuts for the provisional two-level pMOT.

The two cuts are (x, 0, 0) and (0, y, 0).  Gravity is excluded.  Force and
force-gradient panels use ordinary linear axes so that dF_i/dx_i can be read
directly as the numerical derivative of the force curve above it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from explain_pmot_2level_capture_barrier import _build_model
from explain_pmot_2level_capture_barrier import _coordinates_m
from explain_pmot_2level_capture_barrier import _line_cut


OUTPUT_PATH = Path(
    "outputs/diagnostics/pMOT_2Level/capture_barrier_explainer_20261017/"
    "xy_plane_z0_on_axis_force_gradient_intensity.png"
)


def _decorate(axis, *, horizontal_zero: bool = True) -> None:
    # Five-millimetre-wide wall markers centered on +/-10 mm.  These are
    # annotations only and do not alter the beam or force calculation.
    axis.axvspan(-12.5, -7.5, color="#64748b", alpha=0.10)
    axis.axvspan(7.5, 12.5, color="#64748b", alpha=0.10)
    axis.axvline(0.0, color="0.35", linestyle=":", linewidth=0.9)
    axis.axvline(-10.0, color="0.35", linestyle=":", linewidth=0.9)
    axis.axvline(10.0, color="0.35", linestyle=":", linewidth=0.9)
    if horizontal_zero:
        axis.axhline(0.0, color="0.25", linewidth=0.7)
    axis.grid(alpha=0.2)


def main(*, offset_mm: float = 0.0, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simple, cooling, trapping, table = _build_model()
    coordinates_m = _coordinates_m()
    cuts = {
        "x": _line_cut(
            simple,
            cooling,
            trapping,
            table,
            coordinates_m,
            scan_axis="x",
            offset_axis="y",
            offset_mm=offset_mm,
            fixed_axis="z",
            fixed_mm=0.0,
        ),
        "y": _line_cut(
            simple,
            cooling,
            trapping,
            table,
            coordinates_m,
            scan_axis="y",
            offset_axis="x",
            offset_mm=offset_mm,
            fixed_axis="z",
            fixed_mm=0.0,
        ),
    }

    figure, axes = plt.subplots(3, 2, figsize=(15.0, 10.5), sharex="col")
    colors = {"x": "#2563eb", "y": "#d97706"}
    for column, axis_name in enumerate(("x", "y")):
        cut = cuts[axis_name]
        coordinate_mm = 1.0e3 * cut["coordinate_m"]
        color = colors[axis_name]
        axes[0, column].plot(
            coordinate_mm,
            cut["intensity_w_per_m2"],
            color=color,
            linewidth=1.35,
        )
        axes[1, column].plot(
            coordinate_mm,
            1.0e21 * cut["force_n"],
            color=color,
            linewidth=1.35,
        )
        axes[2, column].plot(
            coordinate_mm,
            1.0e19 * cut["force_gradient_n_per_m"],
            color=color,
            linewidth=1.35,
        )
        axes[0, column].set_yscale("log")
        if offset_mm == 0.0:
            column_title = "Along x-axis" if axis_name == "x" else "Along y-axis"
        elif axis_name == "x":
            column_title = f"Along x-axis, +{offset_mm:g} mm off y-axis"
        else:
            column_title = f"Along y-axis, +{offset_mm:g} mm off x-axis"
        axes[0, column].set_title(column_title)
        axes[2, column].set_xlabel(f"{axis_name} [mm]")
        for row in range(3):
            _decorate(axes[row, column], horizontal_zero=(row != 0))

    for column in range(2):
        axes[0, column].set_ylabel(r"total trap intensity [W/m$^2$]")
        axes[1, column].set_ylabel(rf"$F_{{{'x' if column == 0 else 'y'}}}$ [zN]")
        axes[2, column].set_ylabel(
            rf"$\partial F_{{{'x' if column == 0 else 'y'}}}/"
            rf"\partial {'x' if column == 0 else 'y'}$ [$10^{{-19}}$ N/m]"
        )

    figure.suptitle("2-Level pMOT Intensity and Force Curves in x-y Plane")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    # Independent finite-difference audit at representative central points.
    for axis_name, cut in cuts.items():
        coordinate = cut["coordinate_m"]
        force = cut["force_n"]
        gradient = cut["force_gradient_n_per_m"]
        audit = np.gradient(force, coordinate, edge_order=2)
        if not np.array_equal(gradient, audit):
            raise RuntimeError(f"stored {axis_name} gradient is not np.gradient(F, coordinate)")
    print(output_path)


def plot_offset_comparison(*, output_path: Path) -> None:
    """Overlay the 0, +1, and +2 mm transverse-offset cuts."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    simple, cooling, trapping, table = _build_model()
    coordinates_m = _coordinates_m()
    offset_colors = {0.0: "#2563eb", 1.0: "#d97706", 2.0: "#059669"}
    cuts = {}
    for offset_mm in offset_colors:
        cuts[(offset_mm, "x")] = _line_cut(
            simple,
            cooling,
            trapping,
            table,
            coordinates_m,
            scan_axis="x",
            offset_axis="y",
            offset_mm=offset_mm,
            fixed_axis="z",
            fixed_mm=0.0,
        )
        cuts[(offset_mm, "y")] = _line_cut(
            simple,
            cooling,
            trapping,
            table,
            coordinates_m,
            scan_axis="y",
            offset_axis="x",
            offset_mm=offset_mm,
            fixed_axis="z",
            fixed_mm=0.0,
        )

    figure, axes = plt.subplots(3, 2, figsize=(15.0, 10.5), sharex="col")
    for column, axis_name in enumerate(("x", "y")):
        for offset_mm, color in offset_colors.items():
            cut = cuts[(offset_mm, axis_name)]
            coordinate_mm = 1.0e3 * cut["coordinate_m"]
            axes[0, column].plot(
                coordinate_mm,
                cut["intensity_w_per_m2"],
                color=color,
                linewidth=1.35,
            )
            axes[1, column].plot(
                coordinate_mm,
                1.0e21 * cut["force_n"],
                color=color,
                linewidth=1.35,
            )
            axes[2, column].plot(
                coordinate_mm,
                1.0e19 * cut["force_gradient_n_per_m"],
                color=color,
                linewidth=1.35,
            )
        axes[0, column].set_yscale("log")
        axes[0, column].set_title(
            "Along x-axis" if axis_name == "x" else "Along y-axis"
        )
        axes[2, column].set_xlabel(f"{axis_name} [mm]")
        for row in range(3):
            _decorate(axes[row, column], horizontal_zero=(row != 0))

    for column in range(2):
        component = "x" if column == 0 else "y"
        axes[0, column].set_ylabel(r"total trap intensity [W/m$^2$]")
        axes[1, column].set_ylabel(rf"$F_{{{component}}}$ [zN]")
        axes[2, column].set_ylabel(
            rf"$\partial F_{{{component}}}/\partial {component}$ "
            rf"[$10^{{-19}}$ N/m]"
        )

    legend_handles = [
        Line2D(
            [], [], color=offset_colors[0.0], linewidth=1.6,
            label="On axis (0 mm offset)",
        ),
        Line2D(
            [], [], color=offset_colors[1.0], linewidth=1.6,
            label="+1 mm transverse offset",
        ),
        Line2D(
            [], [], color=offset_colors[2.0], linewidth=1.6,
            label="+2 mm transverse offset",
        ),
        Patch(
            facecolor="#64748b",
            alpha=0.18,
            edgecolor="none",
            label="Glass-cell walls (5 mm wide)",
        ),
        Line2D(
            [], [], color="0.35", linestyle=":", linewidth=0.9,
            label="Origin / wall center",
        ),
    ]
    figure.suptitle(
        "2-Level pMOT Intensity and Force Curves in x-y Plane: Offset Comparison\n"
        r"Corrected scattering normalization: $\Gamma/2=\pi\gamma_\nu$",
        y=0.995,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset-mm", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--compare-offsets", action="store_true")
    arguments = parser.parse_args()
    if arguments.compare_offsets:
        plot_offset_comparison(output_path=arguments.output)
    else:
        main(offset_mm=arguments.offset_mm, output_path=arguments.output)
