"""Plot the provisional two-level pMOT force response versus cooling saturation.

The sweep varies only the six 780-nm cooling-component powers.  The trapping
light, cooling detuning, beam diameter, helicities, and all Stark-model inputs
are fixed at the current pMOT_2Level diagnostic values.

This is a provisional transition-level diagnostic, not a validated pMOT
prediction.  Gravity is excluded so the figures show radiation pressure only.
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
    / "saturation_sweep_s0_1_to_35_20261023"
)

SATURATION_VALUES = np.arange(1.0, 36.0, 2.0)
CURRENT_COOLING_POWER_W = 27.0e-3
COOLING_DETUNING_HZ = -15.0e6
FORCE_SAMPLE_OFFSET_M = 0.5e-3
GRADIENT_STEP_M = 5.0e-6
PATH_POWERS_W = np.asarray((19.147243e-3, 19.147243e-3, 38.294486e-3))
AXIS_NAMES = ("x", "y", "z")
COLORS = ("#2563eb", "#f97316", "#059669")
LINE_STYLES = ("-", "--", "-.")


def _power_w_from_s0(s0: float, *, saturation_intensity: float, radius_m: float) -> float:
    """Map peak raw saturation to Gaussian traveling-beam power."""

    return float(s0 * saturation_intensity * np.pi * radius_m**2 / 2.0)


def _s0_from_power_w(power_w: float, *, saturation_intensity: float, radius_m: float) -> float:
    return float(2.0 * power_w / (np.pi * radius_m**2 * saturation_intensity))


def _context(s0: float):
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=COOLING_DETUNING_HZ,
        include_gravity=False,
    )
    baseline = default_simple_mot_apparatus()
    power_w = _power_w_from_s0(
        s0,
        saturation_intensity=simple.saturation_intensity_w_per_m2,
        radius_m=baseline.cooling.beam_radius_m,
    )
    apparatus = replace(
        baseline,
        cooling=replace(
            baseline.cooling,
            power_w_per_beam=power_w,
            detuning_hz=COOLING_DETUNING_HZ,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    return simple, cooling, power_w


def _observable(simple, cooling, trapping, table, position_m):
    return evaluate_pmot_2level_local_observable(
        cooling,
        trapping,
        np.asarray(position_m, dtype=float),
        np.zeros(3),
        path_powers_w=PATH_POWERS_W,
        previous_axis=(0.0, 0.0, 1.0),
        simple_config=simple,
        polarizability_table=table,
    )


def _force_vector(simple, cooling, trapping, table, position_m) -> np.ndarray:
    return np.asarray(
        _observable(simple, cooling, trapping, table, position_m).force_n,
        dtype=float,
    )


def _axis_force_and_gradient(simple, cooling, trapping, table, axis: int):
    sample = np.zeros(3)
    sample[axis] = FORCE_SAMPLE_OFFSET_M
    force_at_offset = _force_vector(
        simple, cooling, trapping, table, sample
    )[axis]

    plus = np.zeros(3)
    minus = np.zeros(3)
    plus[axis] = GRADIENT_STEP_M
    minus[axis] = -GRADIENT_STEP_M
    gradient = (
        _force_vector(simple, cooling, trapping, table, plus)[axis]
        - _force_vector(simple, cooling, trapping, table, minus)[axis]
    ) / (2.0 * GRADIENT_STEP_M)
    return float(force_at_offset), float(gradient)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _shade_cell_walls(axis) -> None:
    for center_mm in (-10.0, 10.0):
        axis.axvspan(
            center_mm - 2.5,
            center_mm + 2.5,
            color="#94a3b8",
            alpha=0.20,
            linewidth=0.0,
        )


def _plot_saturation_sweep(
    rows: list[dict],
    *,
    selected_s0: float,
    current_s0: float,
    nominal_denominator: float,
    local_denominator: float,
    power_mw_per_s0: float,
) -> Path:
    s0 = np.asarray([row["s0_raw_per_beam"] for row in rows])
    force = np.asarray(
        [[row[f"force_{axis}_at_plus_0p5mm_n"] for axis in AXIS_NAMES] for row in rows]
    )
    gradient = np.asarray(
        [[row[f"gradient_dF{axis}_d{axis}_n_per_m"] for axis in AXIS_NAMES] for row in rows]
    )

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 9.2), sharex=True)
    for component, (axis_name, color, line_style) in enumerate(
        zip(AXIS_NAMES, COLORS, LINE_STYLES)
    ):
        axes[0].plot(
            s0,
            1.0e21 * force[:, component],
            color=color,
            linestyle=line_style,
            marker=("o", "s", "^")[component],
            markersize=4.0,
            label=rf"$F_{axis_name}$ at +0.5 mm on {axis_name}",
        )
        axes[1].plot(
            s0,
            1.0e19 * gradient[:, component],
            color=color,
            linestyle=line_style,
            marker=("o", "s", "^")[component],
            markersize=4.0,
            label=rf"$\partial F_{axis_name}/\partial {axis_name}|_0$",
        )

    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(
            current_s0,
            color="#64748b",
            linestyle=":",
            linewidth=1.8,
            label=rf"current 27 mW ($s_0={current_s0:.2f}$)",
        )
        axis.axvline(
            selected_s0,
            color="#7c3aed",
            linestyle=(0, (5, 2)),
            linewidth=1.6,
            label=rf"selected optimum ($s_0={selected_s0:g}$)",
        )
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=9, ncol=2)

    axes[0].set_ylabel("signed radiation-pressure force [zN]")
    axes[0].set_title(
        "Force at +0.5 mm versus raw single-beam saturation\n"
        "negative force is restoring at the positive-axis sample point"
    )
    axes[1].set_ylabel(r"origin force gradient [$10^{-19}$ N/m]")
    axes[1].set_xlabel(r"raw peak saturation of each traveling cooling beam, $s_0=I_0/I_{sat}$")
    axes[1].set_title("Central force gradient versus raw single-beam saturation")
    axes[1].set_xticks(SATURATION_VALUES)

    top_axis = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda value: value * power_mw_per_s0,
            lambda value: value / power_mw_per_s0,
        ),
    )
    top_axis.set_xlabel("cooling power per traveling beam [mW]")

    mapping_text = (
        rf"$P_{{beam}}={power_mw_per_s0:.6f}\,s_0$ mW"
        "\n"
        rf"nominal $s_{{eff,beam}}=s_0/{nominal_denominator:.3f}$ at $-15$ MHz"
        "\n"
        rf"local-origin $s_{{eff,beam}}=s_0/{local_denominator:.3f}$"
        "\n"
        rf"$S_{{eff,tot}}=6s_{{eff,beam}}$ at the symmetric origin"
    )
    axes[0].text(
        0.985,
        0.055,
        mapping_text,
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.91},
    )

    figure.suptitle(
        "Provisional 2-Level pMOT: Cooling Saturation Sweep\n"
        r"fixed trapping light; $\Delta_0=-15$ MHz; gravity excluded",
        fontsize=15,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    path = OUTPUT_DIRECTORY / "force_gradient_vs_saturation.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def _position_grid_mm() -> np.ndarray:
    broad = np.linspace(-15.0, 15.0, 301)
    central = np.linspace(-2.0, 2.0, 321)
    negative_focus = np.linspace(-10.15, -9.85, 81)
    positive_focus = np.linspace(9.85, 10.15, 81)
    # Round before uniquing so mathematically identical points contributed by
    # the nested grids do not survive as machine-epsilon-separated samples.
    # Such near-duplicates make a numerical derivative spuriously enormous.
    return np.unique(
        np.round(
            np.concatenate((broad, central, negative_focus, positive_focus)),
            decimals=9,
        )
    )


def _plot_position_response(
    simple,
    cooling,
    trapping,
    table,
    *,
    selected_s0: float,
    selected_power_mw: float,
    nominal_s_eff: float,
    local_s_eff: float,
) -> tuple[Path, list[dict]]:
    coordinate_mm = _position_grid_mm()
    coordinate_m = 1.0e-3 * coordinate_mm
    forces = np.empty((coordinate_mm.size, 3))
    rows: list[dict] = []
    for component in range(3):
        for index, value_m in enumerate(coordinate_m):
            position = np.zeros(3)
            position[component] = value_m
            forces[index, component] = _force_vector(
                simple, cooling, trapping, table, position
            )[component]
    gradients = np.column_stack(
        [
            np.gradient(forces[:, component], coordinate_m, edge_order=2)
            for component in range(3)
        ]
    )
    for index, coordinate in enumerate(coordinate_mm):
        row = {"coordinate_mm": float(coordinate)}
        for component, axis_name in enumerate(AXIS_NAMES):
            row[f"force_{axis_name}_n"] = float(forces[index, component])
            row[f"gradient_dF{axis_name}_d{axis_name}_n_per_m"] = float(
                gradients[index, component]
            )
        rows.append(row)

    figure, axes = plt.subplots(2, 3, figsize=(17.2, 9.0), sharex="col")
    for component, (axis_name, color) in enumerate(zip(AXIS_NAMES, COLORS)):
        axes[0, component].plot(
            coordinate_mm,
            1.0e21 * forces[:, component],
            color=color,
            linewidth=1.8,
        )
        axes[1, component].plot(
            coordinate_mm,
            1.0e19 * gradients[:, component],
            color=color,
            linewidth=1.8,
        )
        for row in range(2):
            _shade_cell_walls(axes[row, component])
            axes[row, component].axvline(0.0, color="black", linewidth=0.9)
            axes[row, component].axhline(0.0, color="black", linewidth=0.7)
            axes[row, component].grid(alpha=0.20)
            axes[row, component].set_xlim(-15.0, 15.0)
        axes[0, component].set_title(rf"along {axis_name}-axis: $F_{axis_name}$")
        axes[1, component].set_xlabel(f"{axis_name} position [mm]")

    axes[0, 0].set_ylabel("radiation-pressure force [zN]")
    axes[1, 0].set_ylabel(r"numerical force gradient [$10^{-19}$ N/m]")
    figure.legend(
        handles=[Patch(facecolor="#94a3b8", alpha=0.20, label="5-mm glass-wall bands")],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.965),
    )
    figure.suptitle(
        "Provisional 2-Level pMOT Force and Force Gradient at the Selected Saturation\n"
        rf"$s_0={selected_s0:g}$, $P={selected_power_mw:.6f}$ mW/beam, "
        rf"$s_{{eff,beam}}^{{nom}}={nominal_s_eff:.4f}$, "
        rf"$s_{{eff,beam}}^{{local}}(0)={local_s_eff:.4f}$; gravity excluded",
        fontsize=14.5,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = OUTPUT_DIRECTORY / "optimal_saturation_force_gradient_vs_position.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path, rows


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    trapping = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    baseline_simple = replace(default_simple_mot_config(), include_gravity=False)
    baseline_apparatus = default_simple_mot_apparatus()
    radius_m = baseline_apparatus.cooling.beam_radius_m
    saturation_intensity = baseline_simple.saturation_intensity_w_per_m2
    power_mw_per_s0 = 1.0e3 * _power_w_from_s0(
        1.0,
        saturation_intensity=saturation_intensity,
        radius_m=radius_m,
    )
    current_s0 = _s0_from_power_w(
        CURRENT_COOLING_POWER_W,
        saturation_intensity=saturation_intensity,
        radius_m=radius_m,
    )

    reference_simple, reference_cooling, _ = _context(current_s0)
    origin = _observable(
        reference_simple, reference_cooling, trapping, table, np.zeros(3)
    )
    central_shift_hz = float(origin.stark.transition_shifts_hz_by_q[1])
    local_detuning_hz = COOLING_DETUNING_HZ - central_shift_hz
    nominal_denominator = 1.0 + (
        2.0 * COOLING_DETUNING_HZ / baseline_simple.linewidth_hz
    ) ** 2
    local_denominator = 1.0 + (
        2.0 * local_detuning_hz / baseline_simple.linewidth_hz
    ) ** 2

    rows: list[dict] = []
    for s0 in SATURATION_VALUES:
        simple, cooling, power_w = _context(float(s0))
        axis_samples = [
            _axis_force_and_gradient(simple, cooling, trapping, table, component)
            for component in range(3)
        ]
        row = {
            "s0_raw_per_beam": float(s0),
            "peak_intensity_w_per_m2": float(s0 * saturation_intensity),
            "cooling_power_mw_per_beam": float(1.0e3 * power_w),
            "nominal_effective_saturation_per_beam": float(s0 / nominal_denominator),
            "nominal_effective_saturation_six_beam_total": float(
                6.0 * s0 / nominal_denominator
            ),
            "local_origin_effective_saturation_per_beam": float(
                s0 / local_denominator
            ),
            "local_origin_effective_saturation_six_beam_total": float(
                6.0 * s0 / local_denominator
            ),
        }
        for component, axis_name in enumerate(AXIS_NAMES):
            row[f"force_{axis_name}_at_plus_0p5mm_n"] = axis_samples[component][0]
            row[f"gradient_dF{axis_name}_d{axis_name}_n_per_m"] = axis_samples[component][1]
        rows.append(row)

    restoring = np.asarray(
        [
            [-row[f"gradient_dF{axis}_d{axis}_n_per_m"] for axis in AXIS_NAMES]
            for row in rows
        ]
    )
    weakest_axis_restoring = np.min(restoring, axis=1)
    selected_index = int(np.argmax(weakest_axis_restoring))
    selected_s0 = float(rows[selected_index]["s0_raw_per_beam"])
    selected_simple, selected_cooling, selected_power_w = _context(selected_s0)

    saturation_figure = _plot_saturation_sweep(
        rows,
        selected_s0=selected_s0,
        current_s0=current_s0,
        nominal_denominator=nominal_denominator,
        local_denominator=local_denominator,
        power_mw_per_s0=power_mw_per_s0,
    )
    position_figure, position_rows = _plot_position_response(
        selected_simple,
        selected_cooling,
        trapping,
        table,
        selected_s0=selected_s0,
        selected_power_mw=1.0e3 * selected_power_w,
        nominal_s_eff=selected_s0 / nominal_denominator,
        local_s_eff=selected_s0 / local_denominator,
    )

    sweep_csv = OUTPUT_DIRECTORY / "saturation_sweep.csv"
    position_csv = OUTPUT_DIRECTORY / "optimal_saturation_position_scan.csv"
    _write_csv(sweep_csv, rows)
    _write_csv(position_csv, position_rows)

    local_crossover_s0 = local_denominator / 6.0
    nominal_crossover_s0 = nominal_denominator / 6.0
    manifest = {
        "status": "PROVISIONAL_TWO_LEVEL_PMOT_DIAGNOSTIC",
        "physics_scope": {
            "force": "780-nm deterministic mean radiation pressure only",
            "gravity_included": False,
            "external_magnetic_field": False,
            "conservative_1529nm_force_included": False,
            "recoil_diffusion_included": False,
        },
        "fixed_parameters": {
            "cooling_detuning_mhz": 1.0e-6 * COOLING_DETUNING_HZ,
            "cooling_beam_diameter_mm": 2.0e3 * radius_m,
            "saturation_intensity_w_per_m2": saturation_intensity,
            "trapping_path_powers_mw_xyz": (1.0e3 * PATH_POWERS_W).tolist(),
            "force_sample_offset_mm": 1.0e3 * FORCE_SAMPLE_OFFSET_M,
            "central_gradient_step_um": 1.0e6 * GRADIENT_STEP_M,
        },
        "mapping": {
            "power_mw_per_beam_per_unit_s0": power_mw_per_s0,
            "current_power_mw_per_beam": 1.0e3 * CURRENT_COOLING_POWER_W,
            "current_s0_raw_per_beam": current_s0,
            "common_ac_stark_transition_shift_mhz_at_origin": 1.0e-6 * central_shift_hz,
            "effective_detuning_mhz_at_origin": 1.0e-6 * local_detuning_hz,
            "nominal_effective_saturation_denominator": nominal_denominator,
            "local_origin_effective_saturation_denominator": local_denominator,
            "nominal_six_beam_crossover_s0": nominal_crossover_s0,
            "local_origin_six_beam_crossover_s0": local_crossover_s0,
        },
        "selection_rule": (
            "maximize the weakest of -dFi/dxi across x, y, z over the requested "
            "discrete s0 grid"
        ),
        "selected": {
            "s0_raw_per_beam": selected_s0,
            "cooling_power_mw_per_beam": 1.0e3 * selected_power_w,
            "peak_intensity_w_per_m2": selected_s0 * saturation_intensity,
            "nominal_effective_saturation_per_beam": selected_s0 / nominal_denominator,
            "local_origin_effective_saturation_per_beam": selected_s0 / local_denominator,
            "force_gradients_n_per_m_xyz": [
                rows[selected_index][f"gradient_dF{axis}_d{axis}_n_per_m"]
                for axis in AXIS_NAMES
            ],
        },
        "outputs": {
            "saturation_figure": str(saturation_figure),
            "position_figure": str(position_figure),
            "sweep_csv": str(sweep_csv),
            "position_csv": str(position_csv),
        },
    }
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
