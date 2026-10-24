"""Plot the dense-refined provisional pMOT trapping-power optimum."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optimize_pmot_2level_trapping_powers import DENSE_COORDINATES_M
from optimize_pmot_2level_trapping_powers import PROJECT_ROOT
from optimize_pmot_2level_trapping_powers import _metrics


OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "pmot_2level_trapping_power_optimization_20261023"
    / "refined_optimum_force_gradient_comparison.png"
)
AXES = ("x", "y", "z")
CASES = {
    "equal 19.147243": np.asarray((19.147243, 19.147243, 19.147243)),
    "19.147243, 19.147243, 38.294486": np.asarray(
        (19.147243, 19.147243, 38.294486)
    ),
    "refined optimum": np.full(3, 31.524310589374092),
    "equal 38.294486": np.asarray((38.294486, 38.294486, 38.294486)),
}
COLORS = ("#2563eb", "#d97706", "#059669", "#dc2626")
STYLES = ("--", "-.", "-", ":")


def main() -> None:
    coordinate_mm = 1.0e3 * DENSE_COORDINATES_M
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), sharex=True)
    for (label, powers), color, style in zip(
        CASES.items(), COLORS, STYLES, strict=True
    ):
        curves, _ = _metrics(powers, DENSE_COORDINATES_M)
        powers_text = ", ".join(f"{power:.3f}" for power in powers)
        for column, axis in enumerate(AXES):
            force = curves[axis]
            gradient = np.gradient(force, DENSE_COORDINATES_M, edge_order=2)
            legend_label = f"{label}: ({powers_text}) mW"
            axes[0, column].plot(
                coordinate_mm,
                1.0e21 * force,
                color=color,
                linestyle=style,
                linewidth=2.2,
                label=legend_label,
            )
            axes[1, column].plot(
                coordinate_mm,
                1.0e19 * gradient,
                color=color,
                linestyle=style,
                linewidth=2.2,
                label=legend_label,
            )
    for column, axis in enumerate(AXES):
        fixed = "; ".join(f"{other}=0 mm" for other in AXES if other != axis)
        axes[0, column].set_title(f"Along {axis}-axis; {fixed}")
        axes[0, column].set_ylabel(rf"$F_{axis}$ [zN]")
        axes[1, column].set_ylabel(
            rf"$\partial F_{axis}/\partial {axis}$ [$10^{{-19}}$ N/m]"
        )
        axes[1, column].set_xlabel(f"{axis} position [mm]")
        for row in range(2):
            axes[row, column].axhline(0.0, color="0.3", linewidth=0.8)
            axes[row, column].axvline(0.0, color="0.3", linewidth=0.8)
            axes[row, column].grid(True, alpha=0.2)
            axes[row, column].set_xlim(-1.0, 1.0)
    figure.suptitle(
        "2-Level pMOT Central Force and Force-Gradient Power Comparison",
        fontsize=17,
    )
    figure.text(
        0.5,
        0.945,
        "Cooling power = 1.0571 mW per traveling beam; detuning = -15 MHz; "
        "stationary atom; gravity excluded",
        ha="center",
        fontsize=11.5,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.095, 0.98, 0.91), h_pad=2.0, w_pad=1.5)
    figure.savefig(OUTPUT_PATH, dpi=210, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
