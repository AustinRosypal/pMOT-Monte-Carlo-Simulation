"""Compare three MOT force laws with configurable cooling powers.

The multilevel MOT and ``mot_strange`` retain the 0.1-mW repump components.
The provisional two-level pMOT uses equal configurable trapping-path powers.
Stationary atoms are sampled on the six on-axis Cartesian plane cuts and
gravity is excluded from every radiation-pressure evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import build_rate_equation_model
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import rate_equation_observable
from pmot.mot_multilevel.accelerated import observable as accelerated_observable
from pmot.mot_multilevel.accelerated import prepare as prepare_accelerated
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_strange.surrogate_effective_field import surrogate_effective_field_t
from pmot.mot_strange.surrogate_effective_field import surrogate_field_jacobians_t_per_m
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MULTILEVEL_COOLING_POWER_MW_PER_BEAM = 1.0571
DEFAULT_PMOT_COOLING_POWER_MW_PER_BEAM = 1.0571
COOLING_DETUNING_MHZ = -15.0
REPUMP_POWER_MW_PER_BEAM = 0.1
DEFAULT_PMOT_TRAPPING_PATH_POWER_MW = 38.294486
SCAN_LIMIT_MM = 12.0
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
PANEL_SPECS = (
    ("xy", "x", "y", "z"),
    ("xy", "y", "x", "z"),
    ("xz", "x", "z", "y"),
    ("xz", "z", "x", "y"),
    ("yz", "y", "z", "x"),
    ("yz", "z", "y", "x"),
)
MODEL_ORDER = ("mot_multilevel", "mot_strange", "pMOT_2Level")
MODEL_LABELS = {
    "mot_multilevel": "Multilevel MOT, (+10,+10,-20) G/cm",
    "mot_strange": "mot_strange, intensity-derived field",
}
MODEL_COLORS = {
    "mot_multilevel": "#2563eb",
    "mot_strange": "#d97706",
    "pMOT_2Level": "#be123c",
}
MODEL_STYLES = {
    "mot_multilevel": "--",
    "mot_strange": "-.",
    "pMOT_2Level": "-",
}
GLASS_CENTERS_MM = (-10.0, 10.0)
GLASS_WIDTH_MM = 5.0
GLASS_COLOR = "#64748b"


def _scan_coordinates_m() -> np.ndarray:
    broad = np.linspace(-SCAN_LIMIT_MM, SCAN_LIMIT_MM, 121)
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


def _multilevel_config(cooling_power_mw_per_beam: float):
    return replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=1.0e-3 * cooling_power_mw_per_beam,
        repump_power_w_per_beam=1.0e-3 * REPUMP_POWER_MW_PER_BEAM,
        cooling_detuning_rad_per_s=2.0 * np.pi * 1.0e6 * COOLING_DETUNING_MHZ,
        include_gravity=False,
    )


def _evaluate_multilevel_models(
    scan_m: np.ndarray,
    cooling_power_mw_per_beam: float,
) -> dict[str, dict[str, np.ndarray]]:
    config = _multilevel_config(cooling_power_mw_per_beam)
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=20.0)
    accelerated = prepare_accelerated(config=config, coil=coil)
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    fallback_axis = np.asarray((0.0, 0.0, 1.0))
    velocity = np.zeros(3)
    results = {
        model_name: {axis: np.empty(len(scan_m)) for axis in AXIS_INDEX}
        for model_name in ("mot_multilevel", "mot_strange")
    }
    start = perf_counter()

    for axis_name, component in AXIS_INDEX.items():
        for index, coordinate in enumerate(scan_m):
            position = np.zeros(3)
            position[component] = coordinate
            conventional_force, _, _, _ = accelerated_observable(
                position,
                velocity,
                fallback_axis,
                accelerated,
            )
            results["mot_multilevel"][axis_name][index] = conventional_force[component]

            strange = rate_equation_observable(
                model,
                beams,
                tuple(position),
                (0.0, 0.0, 0.0),
                None,
                config,
                tuple(fallback_axis),
                magnetic_field_function=surrogate_effective_field_t,
            )
            results["mot_strange"][axis_name][index] = strange.force_n[component]

        print(
            f"[model powers] completed multilevel and mot_strange {axis_name}-axis; "
            f"elapsed={perf_counter() - start:.1f} s",
            flush=True,
        )
    return results


def _evaluate_pmot(
    scan_m: np.ndarray,
    trapping_path_powers_w: np.ndarray,
    cooling_power_mw_per_beam: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
            power_w_per_beam=1.0e-3 * cooling_power_mw_per_beam,
            detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    trapping = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    result = {axis: np.empty(len(scan_m)) for axis in AXIS_INDEX}

    def observable(position):
        return evaluate_pmot_2level_local_observable(
            cooling,
            trapping,
            position,
            np.zeros(3),
            path_powers_w=trapping_path_powers_w,
            previous_axis=(0.0, 0.0, 1.0),
            simple_config=simple,
            polarizability_table=table,
        )

    for axis_name, component in AXIS_INDEX.items():
        for index, coordinate in enumerate(scan_m):
            position = np.zeros(3)
            position[component] = coordinate
            result[axis_name][index] = observable(position).force_n[component]

    origin = observable(np.zeros(3))
    effective_detuning_hz = float(
        origin.rate_samples[0].effective_detunings_hz_q_minus_pi_q_plus[1]
    )
    raw_s0 = float(origin.rate_samples[0].saturation_parameter)
    origin_metadata = {
        "raw_s0_per_beam": raw_s0,
        "common_ac_stark_shift_mhz": (
            1.0e-6 * float(origin.stark.transition_shifts_hz_by_q[1])
        ),
        "effective_detuning_mhz": 1.0e-6 * effective_detuning_hz,
        "effective_saturation_per_beam": raw_s0
        / (1.0 + (2.0 * effective_detuning_hz / simple.linewidth_hz) ** 2),
    }
    return result, origin_metadata


def _gradients(forces: dict[str, dict[str, np.ndarray]], scan_m: np.ndarray):
    return {
        model: {
            axis: np.gradient(values, scan_m, edge_order=2)
            for axis, values in axes.items()
        }
        for model, axes in forces.items()
    }


def _decorate_axis(axis) -> None:
    for center_mm in GLASS_CENTERS_MM:
        axis.axvspan(
            center_mm - 0.5 * GLASS_WIDTH_MM,
            center_mm + 0.5 * GLASS_WIDTH_MM,
            color=GLASS_COLOR,
            alpha=0.10,
            zorder=0,
        )
        axis.axvline(
            center_mm,
            color="0.35",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )
    axis.axhline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
    axis.axvline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
    axis.set_xlim(-12.5, 12.5)
    axis.grid(True, which="major", alpha=0.22)
    axis.grid(True, which="minor", alpha=0.08)
    axis.minorticks_on()


def _plot_overlay(
    values: dict[str, dict[str, np.ndarray]],
    scan_m: np.ndarray,
    *,
    quantity: str,
    output_path: Path,
    trapping_path_powers_mw: np.ndarray,
    multilevel_cooling_power_mw_per_beam: float,
    pmot_cooling_power_mw_per_beam: float,
) -> None:
    coordinate_mm = 1.0e3 * scan_m
    if quantity == "force":
        scale = 1.0e21
        title = "Same-Axis Force Comparison"
        ylabel = lambda axis: rf"$F_{axis}$ [zN]"
    elif quantity == "gradient":
        scale = 1.0e19
        title = "Same-Axis Force-Gradient Comparison"
        ylabel = lambda axis: rf"$\partial F_{axis}/\partial {axis}$ [$10^{{-19}}$ N/m]"
    else:
        raise ValueError("quantity must be 'force' or 'gradient'")

    figure, axes = plt.subplots(3, 2, figsize=(15.5, 12.8), sharex=True)
    for panel, (plane, scan_axis, transverse_axis, out_of_plane_axis) in zip(
        axes.flat, PANEL_SPECS, strict=True
    ):
        for model in MODEL_ORDER:
            label = (
                MODEL_LABELS[model]
                if model != "pMOT_2Level"
                else "2-level pMOT, equal "
                f"{trapping_path_powers_mw[0]:.6f}-mW trapping paths"
            )
            panel.plot(
                coordinate_mm,
                scale * values[model][scan_axis],
                color=MODEL_COLORS[model],
                linestyle=MODEL_STYLES[model],
                linewidth=2.0,
                label=label,
            )
        _decorate_axis(panel)
        panel.set_title(
            f"{plane.upper()} plane: along {scan_axis}-axis\n"
            f"{transverse_axis}=0 mm; {out_of_plane_axis}=0 mm",
            fontsize=11.5,
        )
        panel.set_ylabel(ylabel(scan_axis))
        panel.set_xlabel(f"{scan_axis} position [mm]")

    figure.suptitle(title, fontsize=17, y=0.995)
    figure.text(
        0.5,
        0.955,
        "Multilevel MOT and mot_strange: "
        f"{multilevel_cooling_power_mw_per_beam:.4f} mW per cooling traveling beam; "
        f"{REPUMP_POWER_MW_PER_BEAM:.1f} mW per repump traveling beam",
        ha="center",
        va="top",
        fontsize=12.3,
    )
    figure.text(
        0.5,
        0.929,
        f"2-level pMOT: {pmot_cooling_power_mw_per_beam:.4f} mW per cooling "
        "traveling beam; trapping-path powers = "
        f"({trapping_path_powers_mw[0]:.6f}, "
        f"{trapping_path_powers_mw[1]:.6f}, "
        f"{trapping_path_powers_mw[2]:.6f}) mW; all models: -15 MHz; "
        "stationary atom; gravity excluded",
        ha="center",
        va="top",
        fontsize=11.5,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            linestyle=MODEL_STYLES[model],
            linewidth=2.2,
            label=(
                MODEL_LABELS[model]
                if model != "pMOT_2Level"
                else "2-level pMOT, equal "
                f"{trapping_path_powers_mw[0]:.6f}-mW trapping paths"
            ),
        )
        for model in MODEL_ORDER
    ]
    handles.append(
        Patch(
            facecolor=GLASS_COLOR,
            alpha=0.18,
            label="glass-cell wall regions (center ±10 mm, width 5 mm)",
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.885), h_pad=2.0, w_pad=1.8)
    figure.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def _write_linecuts(
    path: Path,
    forces: dict[str, dict[str, np.ndarray]],
    gradients: dict[str, dict[str, np.ndarray]],
    scan_m: np.ndarray,
) -> None:
    fieldnames = (
        "model",
        "scan_axis",
        "scan_coordinate_mm",
        "same_axis_force_n",
        "same_axis_force_gradient_n_per_m",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODEL_ORDER:
            for axis in AXIS_INDEX:
                for index, coordinate in enumerate(scan_m):
                    writer.writerow(
                        {
                            "model": model,
                            "scan_axis": axis,
                            "scan_coordinate_mm": 1.0e3 * coordinate,
                            "same_axis_force_n": forces[model][axis][index],
                            "same_axis_force_gradient_n_per_m": gradients[model][axis][index],
                        }
                    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pmot-trapping-power-mw",
        type=float,
        default=DEFAULT_PMOT_TRAPPING_PATH_POWER_MW,
        help="Power in each of the three pMOT trapping paths, in mW.",
    )
    parser.add_argument(
        "--multilevel-cooling-power-mw",
        type=float,
        default=DEFAULT_MULTILEVEL_COOLING_POWER_MW_PER_BEAM,
        help="Cooling power per traveling beam for MOT and mot_strange, in mW.",
    )
    parser.add_argument(
        "--pmot-cooling-power-mw",
        type=float,
        default=DEFAULT_PMOT_COOLING_POWER_MW_PER_BEAM,
        help="Cooling power per traveling beam for pMOT_2Level, in mW.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trapping_path_powers_mw = np.full(3, args.pmot_trapping_power_mw, dtype=float)
    trapping_path_powers_w = 1.0e-3 * trapping_path_powers_mw
    power_token = f"{args.pmot_trapping_power_mw:.6f}".replace(".", "p")
    multilevel_cooling_token = f"{args.multilevel_cooling_power_mw:g}".replace(
        ".", "p"
    )
    pmot_cooling_token = f"{args.pmot_cooling_power_mw:g}".replace(".", "p")
    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "diagnostics"
        / (
            f"three_model_mot_{multilevel_cooling_token}mW_"
            f"pmot_cooling_{pmot_cooling_token}mW_trapping_"
            f"{power_token}mW_20261023"
        )
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    scan_m = _scan_coordinates_m()
    forces = _evaluate_multilevel_models(
        scan_m,
        args.multilevel_cooling_power_mw,
    )
    pmot_forces, pmot_origin = _evaluate_pmot(
        scan_m,
        trapping_path_powers_w,
        args.pmot_cooling_power_mw,
    )
    forces["pMOT_2Level"] = pmot_forces
    gradients = _gradients(forces, scan_m)

    force_figure = output_directory / "three_model_force_overlay_on_axis.png"
    gradient_figure = output_directory / "three_model_force_gradient_overlay_on_axis.png"
    _plot_overlay(
        forces,
        scan_m,
        quantity="force",
        output_path=force_figure,
        trapping_path_powers_mw=trapping_path_powers_mw,
        multilevel_cooling_power_mw_per_beam=args.multilevel_cooling_power_mw,
        pmot_cooling_power_mw_per_beam=args.pmot_cooling_power_mw,
    )
    _plot_overlay(
        gradients,
        scan_m,
        quantity="gradient",
        output_path=gradient_figure,
        trapping_path_powers_mw=trapping_path_powers_mw,
        multilevel_cooling_power_mw_per_beam=args.multilevel_cooling_power_mw,
        pmot_cooling_power_mw_per_beam=args.pmot_cooling_power_mw,
    )

    linecuts_path = output_directory / "three_model_model_specific_power_linecuts.csv"
    _write_linecuts(linecuts_path, forces, gradients, scan_m)
    origin_index = int(np.argmin(np.abs(scan_m)))
    strange_gradient = 100.0 * np.diag(
        surrogate_field_jacobians_t_per_m(np.zeros(3), step_m=1.0e-6)
    )
    manifest = {
        "status": "DIAGNOSTIC_COMPARISON_PMOT_PROVISIONAL",
        "common_configuration": {
            "cooling_detuning_mhz": COOLING_DETUNING_MHZ,
            "stationary_atom": True,
            "gravity_included": False,
        },
        "models": {
            "mot_multilevel": {
                "solver": "replacement 24-state Section-12 population-rate solver",
                "cooling_power_mw_per_traveling_beam": args.multilevel_cooling_power_mw,
                "repump_power_mw_per_traveling_beam": REPUMP_POWER_MW_PER_BEAM,
                "field_gradient_g_per_cm_xyz": [10.0, 10.0, -20.0],
            },
            "mot_strange": {
                "solver": "replacement 24-state Section-12 population-rate solver",
                "cooling_power_mw_per_traveling_beam": args.multilevel_cooling_power_mw,
                "repump_power_mw_per_traveling_beam": REPUMP_POWER_MW_PER_BEAM,
                "origin_effective_field_gradient_g_per_cm_xyz": strange_gradient.tolist(),
            },
            "pMOT_2Level": {
                "solver": "provisional effective F=2 to F'=3 saturated closure",
                "cooling_power_mw_per_traveling_beam": args.pmot_cooling_power_mw,
                "trapping_path_powers_mw_xyz": trapping_path_powers_mw.tolist(),
                "origin": pmot_origin,
                "external_magnetic_field": "zero",
                "repump_included": False,
            },
        },
        "origin_same_axis_force_gradients_1e19_n_per_m": {
            model: {
                axis: float(1.0e19 * gradients[model][axis][origin_index])
                for axis in AXIS_INDEX
            }
            for model in MODEL_ORDER
        },
        "outputs": {
            "force_figure": str(force_figure.relative_to(PROJECT_ROOT)),
            "gradient_figure": str(gradient_figure.relative_to(PROJECT_ROOT)),
            "linecuts_csv": str(linecuts_path.relative_to(PROJECT_ROOT)),
        },
        "caveat": (
            "The pMOT_2Level curve is provisional and excludes conservative "
            "1529-nm force, trapping-light scattering/heating, recoil diffusion, "
            "coherent standing waves, and nonadiabatic zero-crossing dynamics."
        ),
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
