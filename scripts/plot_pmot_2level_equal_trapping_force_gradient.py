"""Plot force and force gradients for equal-power pMOT trapping paths.

This is a provisional effective two-level diagnostic.  It includes the 780-nm
deterministic mean radiation-pressure force and the differential AC-Stark
detuning induced by the 1529-nm trapping light.  Gravity, conservative
trapping-light forces, recoil diffusion, and trapping-light scattering are
excluded.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "pMOT_2Level"
    / "equal_trapping_38p294486mW_cooling_1p0571mW_20261023"
)

COOLING_POWER_MW_PER_BEAM = 1.0571
COOLING_DETUNING_MHZ = -15.0
TRAPPING_PATH_POWERS_MW = np.asarray((38.294486, 38.294486, 38.294486))
TRAPPING_PATH_POWERS_W = 1.0e-3 * TRAPPING_PATH_POWERS_MW
AXIS_NAMES = ("x", "y", "z")
COLORS = ("#2563eb", "#f97316", "#059669")
GLASS_CENTERS_MM = (-10.0, 10.0)
GLASS_WIDTH_MM = 5.0


def _coordinate_grid_mm() -> np.ndarray:
    broad = np.linspace(-15.0, 15.0, 301)
    central = np.linspace(-2.0, 2.0, 321)
    negative_focus = np.linspace(-10.15, -9.85, 81)
    positive_focus = np.linspace(9.85, 10.15, 81)
    return np.unique(
        np.round(
            np.concatenate((broad, central, negative_focus, positive_focus)),
            decimals=9,
        )
    )


def _build_context():
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        include_gravity=False,
    )
    baseline = default_simple_mot_apparatus()
    apparatus = replace(
        baseline,
        cooling=replace(
            baseline.cooling,
            power_w_per_beam=1.0e-3 * COOLING_POWER_MW_PER_BEAM,
            detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    trapping = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    return simple, cooling, trapping, table


def _observable(simple, cooling, trapping, table, position_m):
    return evaluate_pmot_2level_local_observable(
        cooling,
        trapping,
        np.asarray(position_m, dtype=float),
        np.zeros(3),
        path_powers_w=TRAPPING_PATH_POWERS_W,
        previous_axis=(0.0, 0.0, 1.0),
        simple_config=simple,
        polarizability_table=table,
    )


def _force(simple, cooling, trapping, table, position_m) -> np.ndarray:
    return np.asarray(
        _observable(simple, cooling, trapping, table, position_m).force_n,
        dtype=float,
    )


def _shade_cell_walls(axis) -> None:
    for center_mm in GLASS_CENTERS_MM:
        axis.axvspan(
            center_mm - 0.5 * GLASS_WIDTH_MM,
            center_mm + 0.5 * GLASS_WIDTH_MM,
            color="#94a3b8",
            alpha=0.20,
            linewidth=0.0,
            zorder=0,
        )
        axis.axvline(
            center_mm,
            color="0.4",
            linestyle=":",
            linewidth=0.8,
            zorder=1,
        )


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    simple, cooling, trapping, table = _build_context()
    coordinate_mm = _coordinate_grid_mm()
    coordinate_m = 1.0e-3 * coordinate_mm
    forces_n = np.empty((coordinate_mm.size, 3))

    for component in range(3):
        for index, value_m in enumerate(coordinate_m):
            position = np.zeros(3)
            position[component] = value_m
            forces_n[index, component] = _force(
                simple,
                cooling,
                trapping,
                table,
                position,
            )[component]

    gradients_n_per_m = np.column_stack(
        [
            np.gradient(forces_n[:, component], coordinate_m, edge_order=2)
            for component in range(3)
        ]
    )

    origin = _observable(simple, cooling, trapping, table, np.zeros(3))
    raw_s0 = float(origin.rate_samples[0].saturation_parameter)
    common_shift_hz = float(origin.stark.transition_shifts_hz_by_q[1])
    effective_detuning_hz = float(
        origin.rate_samples[0].effective_detunings_hz_q_minus_pi_q_plus[1]
    )
    effective_saturation_per_beam = raw_s0 / (
        1.0 + (2.0 * effective_detuning_hz / simple.linewidth_hz) ** 2
    )

    central_step_m = 5.0e-6
    central_gradients_n_per_m = []
    for component in range(3):
        plus = np.zeros(3)
        minus = np.zeros(3)
        plus[component] = central_step_m
        minus[component] = -central_step_m
        central_gradients_n_per_m.append(
            (
                _force(simple, cooling, trapping, table, plus)[component]
                - _force(simple, cooling, trapping, table, minus)[component]
            )
            / (2.0 * central_step_m)
        )

    figure, axes = plt.subplots(2, 3, figsize=(17.2, 9.2), sharex="col")
    for component, (axis_name, color) in enumerate(zip(AXIS_NAMES, COLORS)):
        axes[0, component].plot(
            coordinate_mm,
            1.0e21 * forces_n[:, component],
            color=color,
            linewidth=1.9,
        )
        axes[1, component].plot(
            coordinate_mm,
            1.0e19 * gradients_n_per_m[:, component],
            color=color,
            linewidth=1.9,
        )
        axes[0, component].set_title(rf"Along {axis_name}-axis: $F_{axis_name}$")
        axes[1, component].set_xlabel(f"{axis_name} position [mm]")
        for row in range(2):
            _shade_cell_walls(axes[row, component])
            axes[row, component].axhline(0.0, color="black", linewidth=0.75)
            axes[row, component].axvline(0.0, color="black", linewidth=0.85)
            axes[row, component].grid(alpha=0.20)
            axes[row, component].set_xlim(-15.0, 15.0)

    axes[0, 0].set_ylabel("radiation-pressure force [zN]")
    axes[1, 0].set_ylabel(r"numerical force gradient [$10^{-19}$ N/m]")
    figure.legend(
        handles=[
            Patch(
                facecolor="#94a3b8",
                alpha=0.20,
                label="5-mm glass-wall bands centered at ±10 mm",
            )
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.963),
        frameon=True,
    )
    figure.suptitle(
        "Provisional 2-Level pMOT: Force and Same-Axis Force Gradient\n"
        r"$P_{trap,x}=P_{trap,y}=P_{trap,z}=38.294486$ mW; "
        r"$P_{cool}=1.0571$ mW/beam ($s_0="
        f"{raw_s0:.6f}$); stationary atom; gravity excluded\n"
        rf"$\Delta_{{AC}}(0)={1.0e-6 * common_shift_hz:+.6f}$ MHz, "
        rf"$\Delta_{{eff}}(0)={1.0e-6 * effective_detuning_hz:+.6f}$ MHz, "
        rf"$s_{{eff,beam}}(0)={effective_saturation_per_beam:.4f}$",
        fontsize=14.5,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure_path = OUTPUT_DIRECTORY / "force_gradient_equal_trapping_powers.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    rows: list[dict[str, float]] = []
    for index, coordinate in enumerate(coordinate_mm):
        row: dict[str, float] = {"coordinate_mm": float(coordinate)}
        for component, axis_name in enumerate(AXIS_NAMES):
            row[f"force_{axis_name}_n"] = float(forces_n[index, component])
            row[f"gradient_dF{axis_name}_d{axis_name}_n_per_m"] = float(
                gradients_n_per_m[index, component]
            )
        rows.append(row)
    csv_path = OUTPUT_DIRECTORY / "force_gradient_equal_trapping_powers.csv"
    _write_csv(csv_path, rows)

    manifest = {
        "status": "PROVISIONAL_TWO_LEVEL_PMOT_DIAGNOSTIC",
        "configuration": {
            "cooling_power_mw_per_traveling_beam": COOLING_POWER_MW_PER_BEAM,
            "cooling_raw_peak_saturation_per_beam": raw_s0,
            "cooling_detuning_mhz": COOLING_DETUNING_MHZ,
            "trapping_path_powers_mw_xyz": TRAPPING_PATH_POWERS_MW.tolist(),
            "trapping_helicities_xyz_incident_and_retro": [
                "sigma+",
                "sigma+",
                "sigma-",
            ],
            "stationary_atom": True,
            "gravity_included": False,
        },
        "origin": {
            "common_ac_stark_shift_mhz": 1.0e-6 * common_shift_hz,
            "effective_detuning_mhz": 1.0e-6 * effective_detuning_hz,
            "effective_saturation_per_beam": effective_saturation_per_beam,
            "effective_saturation_six_beam_total": (
                6.0 * effective_saturation_per_beam
            ),
            "force_n_xyz": list(origin.force_n),
            "central_force_gradients_n_per_m_xyz": [
                float(value) for value in central_gradients_n_per_m
            ],
            "central_gradient_step_um": 1.0e6 * central_step_m,
        },
        "excluded_physics": [
            "conservative 1529-nm gradient force",
            "1529-nm scattering and heating",
            "recoil diffusion",
            "coherent standing-wave structure",
            "nonadiabatic dynamics at the optical-spin zero",
        ],
        "outputs": {
            "figure": str(figure_path.relative_to(PROJECT_ROOT)),
            "linecuts_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
