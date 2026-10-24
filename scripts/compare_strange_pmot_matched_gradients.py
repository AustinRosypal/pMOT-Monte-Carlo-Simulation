"""Compare mot_strange and pMOT_2Level at reciprocally matched field gradients.

Two stationary, gravity-free radiation-force-gradient comparisons are made:

1. tune the pMOT trapping-path powers so its vector-Stark effective-field
   gradient at the origin matches the default ``mot_strange`` surrogate;
2. tune the ``mot_strange`` surrogate path powers so its origin field gradient
   matches the currently selected equal-power pMOT.

The multilevel ``mot_strange`` solver keeps 27 mW per cooling component and
0.1 mW per repump component.  The provisional effective two-level pMOT keeps
1.0571 mW per cooling component.  Both use -15 MHz cooling detuning.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import build_rate_equation_model
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import rate_equation_observable
from pmot.mot_multilevel.atomic_structure import BOHR_MAGNETON_J_PER_T
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_strange.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
)
from pmot.mot_strange.surrogate_effective_field import SurrogateEffectiveFieldConfig
from pmot.mot_strange.surrogate_effective_field import surrogate_effective_field_t
from pmot.mot_strange.surrogate_effective_field import surrogate_field_jacobians_t_per_m
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_differential_ac_stark_shift
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "strange_pmot_reciprocal_gradient_match_20261023"
)

COOLING_DETUNING_MHZ = -15.0
STRANGE_COOLING_POWER_MW = 27.0
STRANGE_REPUMP_POWER_MW = 0.1
PMOT_COOLING_POWER_MW = 1.0571
CURRENT_PMOT_PATH_POWERS_MW = np.full(3, 31.524310589374092)
FIELD_DIFFERENCE_STEP_M = 1.0e-6
SCAN_LIMIT_MM = 12.5
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
PANEL_SPECS = (
    ("xy", "x", "y", "z"),
    ("xy", "y", "x", "z"),
    ("xz", "x", "z", "y"),
    ("xz", "z", "x", "y"),
    ("yz", "y", "z", "x"),
    ("yz", "z", "y", "x"),
)
GLASS_CENTERS_MM = (-10.0, 10.0)
GLASS_WIDTH_MM = 5.0
GLASS_COLOR = "#64748b"
MODEL_COLORS = {"mot_strange": "#d97706", "pMOT_2Level": "#be123c"}
MODEL_STYLES = {"mot_strange": "--", "pMOT_2Level": "-"}


def _scan_coordinates_m() -> np.ndarray:
    broad = np.linspace(-SCAN_LIMIT_MM, SCAN_LIMIT_MM, 251)
    central = np.linspace(-1.0, 1.0, 81)
    negative_focus = np.linspace(-10.2, -9.8, 81)
    positive_focus = np.linspace(9.8, 10.2, 81)
    values_mm = np.unique(
        np.round(
            np.concatenate((broad, central, negative_focus, positive_focus, [0.0])),
            decimals=9,
        )
    )
    return 1.0e-3 * values_mm


def _build_contexts() -> dict[str, object]:
    multilevel_config = replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=1.0e-3 * STRANGE_COOLING_POWER_MW,
        repump_power_w_per_beam=1.0e-3 * STRANGE_REPUMP_POWER_MW,
        cooling_detuning_rad_per_s=2.0 * np.pi * 1.0e6 * COOLING_DETUNING_MHZ,
        include_gravity=False,
    )
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        include_gravity=False,
    )
    baseline_apparatus = default_simple_mot_apparatus()
    pmot_apparatus = replace(
        baseline_apparatus,
        cooling=replace(
            baseline_apparatus.cooling,
            power_w_per_beam=1.0e-3 * PMOT_COOLING_POWER_MW,
            detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        ),
    )
    return {
        "multilevel_config": multilevel_config,
        "multilevel_model": build_rate_equation_model(),
        "multilevel_beams": build_multilevel_mot_beams(config=multilevel_config),
        "simple_config": simple,
        "pmot_cooling": build_pmot_2level_cooling_beams(
            apparatus_config=pmot_apparatus,
            simple_config=simple,
        ),
        "pmot_trapping": build_pmot_2level_trapping_beams(),
        "polarizability": load_pmot_2level_polarizability_table(),
    }


def _pmot_effective_field_t(
    position_m: np.ndarray,
    powers_w: np.ndarray,
    contexts: dict[str, object],
) -> np.ndarray:
    sample = evaluate_differential_ac_stark_shift(
        contexts["pmot_trapping"],
        position_m,
        np.zeros(3),
        path_powers_w=powers_w,
        previous_axis=(0.0, 0.0, 1.0),
        polarizability_table=contexts["polarizability"],
    )
    return np.asarray(sample.net_vector_energy_vector_j) / BOHR_MAGNETON_J_PER_T


def _pmot_origin_gradient_g_per_cm(
    powers_mw: np.ndarray,
    contexts: dict[str, object],
) -> np.ndarray:
    powers_w = 1.0e-3 * np.asarray(powers_mw, dtype=float)
    diagonal_t_per_m = np.empty(3)
    for coordinate in range(3):
        displacement = np.zeros(3)
        displacement[coordinate] = FIELD_DIFFERENCE_STEP_M
        field_plus = _pmot_effective_field_t(displacement, powers_w, contexts)
        field_minus = _pmot_effective_field_t(-displacement, powers_w, contexts)
        diagonal_t_per_m[coordinate] = (
            field_plus[coordinate] - field_minus[coordinate]
        ) / (2.0 * FIELD_DIFFERENCE_STEP_M)
    return 100.0 * diagonal_t_per_m


def _pmot_gradient_response_g_per_cm_per_mw(
    contexts: dict[str, object],
) -> np.ndarray:
    response = np.empty((3, 3))
    for path in range(3):
        powers_mw = np.zeros(3)
        powers_mw[path] = 1.0
        response[:, path] = _pmot_origin_gradient_g_per_cm(powers_mw, contexts)
    return response


def _strange_origin_gradient_g_per_cm(
    config: SurrogateEffectiveFieldConfig,
) -> np.ndarray:
    jacobian = surrogate_field_jacobians_t_per_m(
        np.zeros(3),
        step_m=FIELD_DIFFERENCE_STEP_M,
        config=config,
    )
    return 100.0 * np.diag(jacobian)


def _strange_config_from_powers_mw(
    powers_mw: np.ndarray,
) -> SurrogateEffectiveFieldConfig:
    powers_w = 1.0e-3 * np.asarray(powers_mw, dtype=float)
    return replace(
        DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
        x_path_power_w=float(powers_w[0]),
        y_path_power_w=float(powers_w[1]),
        z_path_power_w=float(powers_w[2]),
    )


def _strange_gradient_response_g_per_cm_per_mw() -> np.ndarray:
    response = np.empty((3, 3))
    for path in range(3):
        powers_mw = np.zeros(3)
        powers_mw[path] = 1.0
        response[:, path] = _strange_origin_gradient_g_per_cm(
            _strange_config_from_powers_mw(powers_mw)
        )
    return response


def _solve_nonnegative_powers(
    response: np.ndarray,
    target_gradient: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    powers_mw, residuals, rank, _ = np.linalg.lstsq(
        response,
        target_gradient,
        rcond=None,
    )
    if rank < 3:
        raise RuntimeError(f"{label} gradient response matrix is rank deficient")
    if np.any(powers_mw < -1.0e-9):
        raise RuntimeError(f"{label} matching requires a negative path power")
    mismatch = response @ powers_mw - target_gradient
    if np.max(np.abs(mismatch)) > 1.0e-7:
        raise RuntimeError(
            f"{label} origin-gradient match failed: mismatch={mismatch.tolist()}"
        )
    return np.maximum(powers_mw, 0.0)


def _evaluate_scenario(
    scan_m: np.ndarray,
    *,
    strange_config: SurrogateEffectiveFieldConfig,
    pmot_powers_mw: np.ndarray,
    contexts: dict[str, object],
    scenario_name: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    forces = {
        model: {axis: np.empty(len(scan_m)) for axis in AXIS_INDEX}
        for model in ("mot_strange", "pMOT_2Level")
    }
    pmot_powers_w = 1.0e-3 * pmot_powers_mw
    fallback_axis = (0.0, 0.0, 1.0)

    def strange_field(position_m):
        return surrogate_effective_field_t(position_m, config=strange_config)

    started = perf_counter()
    for axis_name, component in AXIS_INDEX.items():
        for index, coordinate in enumerate(scan_m):
            position = np.zeros(3)
            position[component] = coordinate
            strange = rate_equation_observable(
                contexts["multilevel_model"],
                contexts["multilevel_beams"],
                tuple(position),
                (0.0, 0.0, 0.0),
                None,
                contexts["multilevel_config"],
                fallback_axis,
                magnetic_field_function=strange_field,
            )
            pmot = evaluate_pmot_2level_local_observable(
                contexts["pmot_cooling"],
                contexts["pmot_trapping"],
                position,
                np.zeros(3),
                path_powers_w=pmot_powers_w,
                previous_axis=fallback_axis,
                simple_config=contexts["simple_config"],
                polarizability_table=contexts["polarizability"],
            )
            forces["mot_strange"][axis_name][index] = strange.force_n[component]
            forces["pMOT_2Level"][axis_name][index] = pmot.force_n[component]
        print(
            f"[{scenario_name}] completed {axis_name}-axis; "
            f"elapsed={perf_counter() - started:.1f} s",
            flush=True,
        )

    gradients = {
        model: {
            axis: np.gradient(force, scan_m, edge_order=2)
            for axis, force in axes.items()
        }
        for model, axes in forces.items()
    }
    return forces, gradients


def _decorate_axis(axis, *, x_limit_mm: float) -> None:
    for center_mm in GLASS_CENTERS_MM:
        axis.axvspan(
            center_mm - 0.5 * GLASS_WIDTH_MM,
            center_mm + 0.5 * GLASS_WIDTH_MM,
            color=GLASS_COLOR,
            alpha=0.10,
            zorder=0,
        )
        axis.axvline(center_mm, color="0.35", linestyle=":", linewidth=0.9)
    axis.axhline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
    axis.axvline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
    axis.set_xlim(-x_limit_mm, x_limit_mm)
    axis.grid(True, which="major", alpha=0.22)
    axis.grid(True, which="minor", alpha=0.08)
    axis.minorticks_on()


def _format_tuple(values: np.ndarray, digits: int = 3) -> str:
    return "(" + ", ".join(f"{value:+.{digits}f}" for value in values) + ")"


def _format_power_tuple(values: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(value):.6f}" for value in values) + ")"


def _plot_scenario(
    gradients: dict[str, dict[str, np.ndarray]],
    scan_m: np.ndarray,
    *,
    title: str,
    strange_powers_mw: np.ndarray,
    pmot_powers_mw: np.ndarray,
    strange_gradient_g_per_cm: np.ndarray,
    pmot_gradient_g_per_cm: np.ndarray,
    output_path: Path,
    x_limit_mm: float = SCAN_LIMIT_MM,
) -> None:
    coordinate_mm = 1.0e3 * scan_m
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 12.7), sharex=True)
    for panel, (plane, scan_axis, transverse_axis, fixed_axis) in zip(
        axes.flat,
        PANEL_SPECS,
        strict=True,
    ):
        for model in ("mot_strange", "pMOT_2Level"):
            panel.plot(
                coordinate_mm,
                1.0e19 * gradients[model][scan_axis],
                color=MODEL_COLORS[model],
                linestyle=MODEL_STYLES[model],
                linewidth=2.1,
                label=model,
            )
        _decorate_axis(panel, x_limit_mm=x_limit_mm)
        panel.set_title(
            f"{plane.upper()} plane: along {scan_axis}-axis\n"
            f"{transverse_axis}=0 mm; {fixed_axis}=0 mm",
            fontsize=11.5,
        )
        panel.set_ylabel(
            rf"$\partial F_{scan_axis}/\partial {scan_axis}$ "
            r"[$10^{-19}$ N/m]"
        )
        panel.set_xlabel(f"{scan_axis} position [mm]")

    figure.suptitle(title, fontsize=17, y=0.995)
    figure.text(
        0.5,
        0.957,
        "Matched local effective-field gradients at the origin: "
        f"mot_strange {_format_tuple(strange_gradient_g_per_cm)} G/cm; "
        f"2-level pMOT {_format_tuple(pmot_gradient_g_per_cm)} G/cm",
        ha="center",
        va="top",
        fontsize=11.8,
    )
    figure.text(
        0.5,
        0.930,
        "mot_strange: 27 mW/cooling beam, 0.1 mW/repump beam, surrogate-path "
        f"powers={_format_power_tuple(strange_powers_mw)} mW; "
        "2-level pMOT: 1.0571 mW/cooling beam, trapping-path "
        f"powers={_format_power_tuple(pmot_powers_mw)} mW",
        ha="center",
        va="top",
        fontsize=10.8,
    )
    figure.text(
        0.5,
        0.905,
        "Both: -15 MHz cooling detuning; stationary atom; gravity excluded; "
        "force gradient is the numerical derivative of the displayed force law",
        ha="center",
        va="top",
        fontsize=10.8,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["mot_strange"],
            linestyle=MODEL_STYLES["mot_strange"],
            linewidth=2.3,
            label="mot_strange — replacement 24-state population-rate solver",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["pMOT_2Level"],
            linestyle=MODEL_STYLES["pMOT_2Level"],
            linewidth=2.3,
            label="pMOT_2Level — provisional effective two-level closure",
        ),
        Patch(
            facecolor=GLASS_COLOR,
            alpha=0.18,
            label="glass-cell wall regions (center ±10 mm, width 5 mm)",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.060, 0.98, 0.870), h_pad=2.0, w_pad=1.8)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _comparison_metrics(
    gradients: dict[str, dict[str, np.ndarray]],
    scan_m: np.ndarray,
) -> dict[str, object]:
    origin = int(np.argmin(np.abs(scan_m)))
    central = np.abs(scan_m) <= 2.0e-3
    result: dict[str, object] = {"axes": {}}
    for axis in AXIS_INDEX:
        strange = gradients["mot_strange"][axis]
        pmot = gradients["pMOT_2Level"][axis]
        difference = pmot - strange
        result["axes"][axis] = {
            "origin_force_gradient_1e19_n_per_m": {
                "mot_strange": float(1.0e19 * strange[origin]),
                "pMOT_2Level": float(1.0e19 * pmot[origin]),
            },
            "central_abs_rms_difference_1e19_n_per_m": float(
                1.0e19 * np.sqrt(np.mean(difference[central] ** 2))
            ),
            "central_relative_rms_difference": float(
                np.sqrt(np.mean(difference[central] ** 2))
                / max(np.sqrt(np.mean(strange[central] ** 2)), np.finfo(float).tiny)
            ),
        }
    return result


def _write_linecuts(
    path: Path,
    scenarios: dict[str, dict[str, object]],
    scan_m: np.ndarray,
) -> None:
    fieldnames = (
        "scenario",
        "model",
        "scan_axis",
        "scan_coordinate_mm",
        "same_axis_force_n",
        "same_axis_force_gradient_n_per_m",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_name, scenario in scenarios.items():
            forces = scenario["forces"]
            gradients = scenario["gradients"]
            for model in ("mot_strange", "pMOT_2Level"):
                for axis in AXIS_INDEX:
                    for index, coordinate in enumerate(scan_m):
                        writer.writerow(
                            {
                                "scenario": scenario_name,
                                "model": model,
                                "scan_axis": axis,
                                "scan_coordinate_mm": 1.0e3 * coordinate,
                                "same_axis_force_n": forces[model][axis][index],
                                "same_axis_force_gradient_n_per_m": gradients[model][axis][index],
                            }
                        )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    contexts = _build_contexts()

    default_strange = DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    default_strange_powers_mw = 1.0e3 * np.asarray(
        (
            default_strange.x_path_power_w,
            default_strange.y_path_power_w,
            default_strange.z_path_power_w,
        )
    )
    default_strange_gradient = _strange_origin_gradient_g_per_cm(default_strange)
    pmot_response = _pmot_gradient_response_g_per_cm_per_mw(contexts)
    pmot_matched_powers = _solve_nonnegative_powers(
        pmot_response,
        default_strange_gradient,
        label="pMOT_2Level",
    )
    pmot_matched_gradient = _pmot_origin_gradient_g_per_cm(
        pmot_matched_powers,
        contexts,
    )

    current_pmot_gradient = _pmot_origin_gradient_g_per_cm(
        CURRENT_PMOT_PATH_POWERS_MW,
        contexts,
    )
    strange_response = _strange_gradient_response_g_per_cm_per_mw()
    strange_matched_powers = _solve_nonnegative_powers(
        strange_response,
        current_pmot_gradient,
        label="mot_strange",
    )
    strange_matched = _strange_config_from_powers_mw(strange_matched_powers)
    strange_matched_gradient = _strange_origin_gradient_g_per_cm(strange_matched)

    scan_m = _scan_coordinates_m()
    scenario_specs = {
        "pmot_matches_default_strange": {
            "strange_config": default_strange,
            "strange_powers_mw": default_strange_powers_mw,
            "pmot_powers_mw": pmot_matched_powers,
            "strange_gradient_g_per_cm": default_strange_gradient,
            "pmot_gradient_g_per_cm": pmot_matched_gradient,
            "title": "Force-Gradient Overlay: pMOT Matched to mot_strange Field Gradients",
            "figure": OUTPUT_DIRECTORY / "pmot_matched_to_strange_force_gradients.png",
            "central_figure": OUTPUT_DIRECTORY / "pmot_matched_to_strange_force_gradients_central_2mm.png",
        },
        "strange_matches_current_pmot": {
            "strange_config": strange_matched,
            "strange_powers_mw": strange_matched_powers,
            "pmot_powers_mw": CURRENT_PMOT_PATH_POWERS_MW,
            "strange_gradient_g_per_cm": strange_matched_gradient,
            "pmot_gradient_g_per_cm": current_pmot_gradient,
            "title": "Force-Gradient Overlay: mot_strange Matched to Current pMOT Field Gradients",
            "figure": OUTPUT_DIRECTORY / "strange_matched_to_pmot_force_gradients.png",
            "central_figure": OUTPUT_DIRECTORY / "strange_matched_to_pmot_force_gradients_central_2mm.png",
        },
    }
    scenarios: dict[str, dict[str, object]] = {}
    for scenario_name, spec in scenario_specs.items():
        forces, gradients = _evaluate_scenario(
            scan_m,
            strange_config=spec["strange_config"],
            pmot_powers_mw=spec["pmot_powers_mw"],
            contexts=contexts,
            scenario_name=scenario_name,
        )
        _plot_scenario(
            gradients,
            scan_m,
            title=spec["title"],
            strange_powers_mw=spec["strange_powers_mw"],
            pmot_powers_mw=spec["pmot_powers_mw"],
            strange_gradient_g_per_cm=spec["strange_gradient_g_per_cm"],
            pmot_gradient_g_per_cm=spec["pmot_gradient_g_per_cm"],
            output_path=spec["figure"],
        )
        _plot_scenario(
            gradients,
            scan_m,
            title=spec["title"] + " — Central ±2 mm",
            strange_powers_mw=spec["strange_powers_mw"],
            pmot_powers_mw=spec["pmot_powers_mw"],
            strange_gradient_g_per_cm=spec["strange_gradient_g_per_cm"],
            pmot_gradient_g_per_cm=spec["pmot_gradient_g_per_cm"],
            output_path=spec["central_figure"],
            x_limit_mm=2.0,
        )
        scenarios[scenario_name] = {
            "forces": forces,
            "gradients": gradients,
            "metrics": _comparison_metrics(gradients, scan_m),
        }

    linecuts_path = OUTPUT_DIRECTORY / "matched_gradient_force_linecuts.csv"
    _write_linecuts(linecuts_path, scenarios, scan_m)
    manifest = {
        "status": "DIAGNOSTIC_COMPARISON_PMOT_PROVISIONAL",
        "common": {
            "cooling_detuning_mhz": COOLING_DETUNING_MHZ,
            "stationary_atom": True,
            "gravity_included": False,
            "mot_strange_cooling_power_mw_per_traveling_beam": STRANGE_COOLING_POWER_MW,
            "mot_strange_repump_power_mw_per_traveling_beam": STRANGE_REPUMP_POWER_MW,
            "pmot_2level_cooling_power_mw_per_traveling_beam": PMOT_COOLING_POWER_MW,
            "scan_range_mm": [-SCAN_LIMIT_MM, SCAN_LIMIT_MM],
        },
        "gradient_calibration": {
            "pmot_response_g_per_cm_per_mw": pmot_response.tolist(),
            "strange_response_g_per_cm_per_mw": strange_response.tolist(),
        },
        "scenarios": {},
        "outputs": {
            "linecuts_csv": str(linecuts_path.relative_to(PROJECT_ROOT)),
        },
        "caveat": (
            "Matching the three diagonal origin gradients does not make the models "
            "physically equivalent. mot_strange uses the replacement 24-state "
            "population-rate solver; pMOT_2Level uses a provisional effective "
            "two-level saturated closure with scalar/vector/tensor differential "
            "Stark shifts and omits conservative trap-light force, trap-light "
            "scattering/heating, recoil diffusion, coherent standing waves, and "
            "nonadiabatic zero-crossing dynamics."
        ),
    }
    for scenario_name, spec in scenario_specs.items():
        manifest["scenarios"][scenario_name] = {
            "mot_strange_path_powers_mw_xyz": spec["strange_powers_mw"].tolist(),
            "pmot_2level_trapping_path_powers_mw_xyz": spec["pmot_powers_mw"].tolist(),
            "mot_strange_origin_gradient_g_per_cm_xyz": spec[
                "strange_gradient_g_per_cm"
            ].tolist(),
            "pmot_2level_origin_gradient_g_per_cm_xyz": spec[
                "pmot_gradient_g_per_cm"
            ].tolist(),
            "force_gradient_overlap_metrics": scenarios[scenario_name]["metrics"],
            "figure": str(spec["figure"].relative_to(PROJECT_ROOT)),
            "central_figure": str(spec["central_figure"].relative_to(PROJECT_ROOT)),
        }
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
