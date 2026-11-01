"""Optimize the central x-axis restoring area for four current MOT models.

The bounded objective is the area below zero in the x-axis force-gradient
curve over [-5, +5] mm,

    A_- = integral max(-dF_x/dx, 0) dx,

subject to a meaningfully negative origin gradient.  The pMOT trapping paths
are constrained to the anti-Helmholtz/Gauss-law ratio P_x=P_y=P_z/2.  Gravity
is excluded and every curve is evaluated for a stationary atom.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import build_rate_equation_model
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import rate_equation_observable
from pmot.mot_multilevel.atomic_structure import BOHR_MAGNETON_J_PER_T
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_differential_ac_stark_shift
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.pmot_vec_2Level import evaluate_pmot_vec_2level_local_observable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "four_model_restoring_area_optimization_20261027"
)
MODELS = (
    "mot_multilevel",
    "pmot_derived_mot_strange",
    "pMOT_2Level",
    "pmot_vec_2Level",
)
LABELS = {
    "mot_multilevel": "24-state conventional MOT",
    "pmot_derived_mot_strange": "24-state pMOT-derived mot_strange",
    "pMOT_2Level": "Full 2-level pMOT",
    "pmot_vec_2Level": "Vector-only 2-level pMOT",
}
COLORS = {
    "mot_multilevel": "#111827",
    "pmot_derived_mot_strange": "#d97706",
    "pMOT_2Level": "#be123c",
    "pmot_vec_2Level": "#2563eb",
}
STYLES = {
    "mot_multilevel": "-.",
    "pmot_derived_mot_strange": "--",
    "pMOT_2Level": "-",
    "pmot_vec_2Level": ":",
}

HALF_WIDTH_M = 5.0e-3
ORIGIN_GRADIENT_LIMIT_SCALED = -0.1  # units of 1e-19 N/m
REPUMP_POWER_MW = 0.1
FALLBACK_AXIS = (0.0, 0.0, 1.0)
VELOCITY = np.zeros(3)
TRANSITION_ZEEMAN_FACTOR = 1.0
PMOT_POWER_MODE = "gauss"

# Bounds follow the project's existing power/detuning campaigns and the prior
# provisional trapping-power search.  The cooling lower bound is the s0=0.25
# campaign point; the upper bound is s0=125.
COOLING_POWER_BOUNDS_MW = (0.2642796195667887, 132.13980978339435)
COOLING_DETUNING_BOUNDS_MHZ = (-36.39, -3.0325)
MOT_AXIAL_GRADIENT_BOUNDS_G_PER_CM = (5.0, 40.0)
PMOT_XY_PATH_POWER_BOUNDS_MW = (0.1, 60.0)

_POSITIVE_MM = np.unique(
    np.concatenate(
        (
            np.asarray((0.0, 0.01, 0.025, 0.05, 0.075)),
            np.linspace(0.1, 5.0, 50),
        )
    )
)
OPTIMIZATION_POSITIVE_M = 1.0e-3 * _POSITIVE_MM
FINAL_COORDINATES_M = np.unique(
    np.round(
        np.concatenate(
            (
                np.linspace(-12.5e-3, 12.5e-3, 501),
                np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, 801),
                np.linspace(-11.0e-3, -9.0e-3, 161),
                np.linspace(9.0e-3, 11.0e-3, 161),
            )
        ),
        decimals=12,
    )
)

RATE_MODEL = build_rate_equation_model()
TRAPPING_BEAMS = build_pmot_2level_trapping_beams()
POLARIZABILITY_TABLE = load_pmot_2level_polarizability_table()
EVALUATION_COUNTS = {model: 0 for model in MODELS}


def _apparatus(cooling_power_mw: float, cooling_detuning_mhz: float):
    baseline = default_simple_mot_apparatus()
    return replace(
        baseline,
        cooling=replace(
            baseline.cooling,
            power_w_per_beam=1.0e-3 * cooling_power_mw,
            detuning_hz=1.0e6 * cooling_detuning_mhz,
        ),
    )


def _simple_configuration(cooling_detuning_mhz: float):
    return replace(
        default_simple_mot_config(),
        cooling_detuning_hz=1.0e6 * cooling_detuning_mhz,
        include_gravity=False,
    )


def _multilevel_configuration(cooling_power_mw: float, cooling_detuning_mhz: float):
    return replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=1.0e-3 * cooling_power_mw,
        cooling_detuning_rad_per_s=2.0 * np.pi * 1.0e6 * cooling_detuning_mhz,
        repump_power_w_per_beam=1.0e-3 * REPUMP_POWER_MW,
        include_gravity=False,
    )


def _trapping_powers_w(xy_power_mw: float) -> np.ndarray:
    if PMOT_POWER_MODE == "gauss":
        powers_mw = (xy_power_mw, xy_power_mw, 2.0 * xy_power_mw)
    elif PMOT_POWER_MODE == "spherical":
        powers_mw = (xy_power_mw, xy_power_mw, xy_power_mw)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"unknown pMOT power mode: {PMOT_POWER_MODE}")
    return 1.0e-3 * np.asarray(powers_mw)


def _effective_field_function(xy_power_mw: float):
    powers_w = _trapping_powers_w(xy_power_mw)

    def field(position_m):
        sample = evaluate_differential_ac_stark_shift(
            TRAPPING_BEAMS,
            position_m,
            VELOCITY,
            path_powers_w=powers_w,
            previous_axis=FALLBACK_AXIS,
            polarizability_table=POLARIZABILITY_TABLE,
        )
        return np.asarray(sample.net_vector_energy_vector_j, dtype=float) / (
            BOHR_MAGNETON_J_PER_T * TRANSITION_ZEEMAN_FACTOR
        )

    return field


def _force_evaluator(model: str, parameters: np.ndarray):
    cooling_power_mw, detuning_mhz, field_parameter = (
        float(value) for value in parameters
    )
    apparatus = _apparatus(cooling_power_mw, detuning_mhz)

    if model in {"mot_multilevel", "pmot_derived_mot_strange"}:
        config = _multilevel_configuration(cooling_power_mw, detuning_mhz)
        beams = build_multilevel_mot_beams(
            apparatus_config=apparatus,
            config=config,
        )
        if model == "mot_multilevel":
            coil = default_anti_helmholtz_config(
                target_gradient_g_per_cm=field_parameter
            )

            def evaluate(position_m):
                return rate_equation_observable(
                    RATE_MODEL,
                    beams,
                    tuple(position_m),
                    tuple(VELOCITY),
                    coil,
                    config,
                    FALLBACK_AXIS,
                ).force_n[0]

            return evaluate

        field_function = _effective_field_function(field_parameter)

        def evaluate(position_m):
            return rate_equation_observable(
                RATE_MODEL,
                beams,
                tuple(position_m),
                tuple(VELOCITY),
                None,
                config,
                FALLBACK_AXIS,
                magnetic_field_function=field_function,
            ).force_n[0]

        return evaluate

    simple = _simple_configuration(detuning_mhz)
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    powers_w = _trapping_powers_w(field_parameter)
    if model == "pMOT_2Level":

        def evaluate(position_m):
            return evaluate_pmot_2level_local_observable(
                cooling,
                TRAPPING_BEAMS,
                position_m,
                VELOCITY,
                path_powers_w=powers_w,
                previous_axis=FALLBACK_AXIS,
                simple_config=simple,
                polarizability_table=POLARIZABILITY_TABLE,
            ).force_n[0]

        return evaluate

    if model == "pmot_vec_2Level":

        def evaluate(position_m):
            return evaluate_pmot_vec_2level_local_observable(
                cooling,
                TRAPPING_BEAMS,
                position_m,
                VELOCITY,
                path_powers_w=powers_w,
                previous_axis=FALLBACK_AXIS,
                simple_config=simple,
            ).force_n[0]

        return evaluate

    raise ValueError(f"unknown model: {model}")


def _force_curve(
    model: str,
    parameters: np.ndarray,
    coordinates_m: np.ndarray,
) -> np.ndarray:
    evaluator = _force_evaluator(model, parameters)
    force = np.empty(len(coordinates_m))
    for index, coordinate_m in enumerate(coordinates_m):
        force[index] = evaluator(np.asarray((coordinate_m, 0.0, 0.0)))
    return force


def _optimization_curve(model: str, parameters: np.ndarray):
    positive_force = _force_curve(model, parameters, OPTIMIZATION_POSITIVE_M)
    coordinates = np.concatenate(
        (-OPTIMIZATION_POSITIVE_M[:0:-1], OPTIMIZATION_POSITIVE_M)
    )
    # Every retained configuration is centered and inversion symmetric.  The
    # mirrored optimization curve halves the expensive multilevel evaluations.
    force = np.concatenate((-positive_force[:0:-1], positive_force))
    return coordinates, force


def _curve_metrics(coordinates_m: np.ndarray, force_n: np.ndarray) -> dict[str, float]:
    gradient = np.gradient(force_n, coordinates_m, edge_order=2)
    origin_index = int(np.argmin(np.abs(coordinates_m)))
    restoring = gradient < 0.0
    restoring_area_n = float(
        np.trapezoid(np.maximum(-gradient, 0.0), coordinates_m)
    )
    anti_restoring_area_n = float(
        np.trapezoid(np.maximum(gradient, 0.0), coordinates_m)
    )

    left = origin_index
    right = origin_index
    if restoring[origin_index]:
        while left > 0 and restoring[left - 1]:
            left -= 1
        while right < len(restoring) - 1 and restoring[right + 1]:
            right += 1
        central_width_mm = 1.0e3 * (
            coordinates_m[right] - coordinates_m[left]
        )
    else:
        central_width_mm = 0.0

    return {
        "restoring_area_zN": 1.0e21 * restoring_area_n,
        "anti_restoring_area_zN": 1.0e21 * anti_restoring_area_n,
        "origin_gradient_1e19_n_per_m": 1.0e19 * float(gradient[origin_index]),
        "minimum_gradient_1e19_n_per_m": 1.0e19 * float(np.min(gradient)),
        "maximum_gradient_1e19_n_per_m": 1.0e19 * float(np.max(gradient)),
        "negative_gradient_fraction": float(np.mean(restoring)),
        "central_restoring_width_mm": float(central_width_mm),
        "force_peak_zN": 1.0e21 * float(np.max(np.abs(force_n))),
    }


def _objective(model: str, parameters: np.ndarray) -> float:
    EVALUATION_COUNTS[model] += 1
    try:
        coordinates, force = _optimization_curve(model, parameters)
        metrics = _curve_metrics(coordinates, force)
    except (ValueError, RuntimeError, FloatingPointError):
        return 1.0e6

    origin_gradient = metrics["origin_gradient_1e19_n_per_m"]
    violation = max(origin_gradient - ORIGIN_GRADIENT_LIMIT_SCALED, 0.0)
    objective = -metrics["restoring_area_zN"] + 1.0e4 * violation**2
    count = EVALUATION_COUNTS[model]
    if count % 25 == 0:
        print(
            f"{model}: evaluations={count}, area={metrics['restoring_area_zN']:.6g} zN, "
            f"origin_gradient={origin_gradient:.6g}e-19 N/m",
            flush=True,
        )
    return float(objective)


def _bounds(model: str):
    third = (
        MOT_AXIAL_GRADIENT_BOUNDS_G_PER_CM
        if model == "mot_multilevel"
        else PMOT_XY_PATH_POWER_BOUNDS_MW
    )
    return (
        COOLING_POWER_BOUNDS_MW,
        COOLING_DETUNING_BOUNDS_MHZ,
        third,
    )


def _optimize(model: str, *, maxiter: int, popsize: int) -> dict:
    started = perf_counter()
    seed = 20261027 + MODELS.index(model)
    result = differential_evolution(
        lambda values: _objective(model, np.asarray(values, dtype=float)),
        bounds=_bounds(model),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=2.0e-4,
        polish=True,
        updating="immediate",
        workers=1,
    )
    parameters = np.asarray(result.x, dtype=float)
    coordinates, force = _optimization_curve(model, parameters)
    metrics = _curve_metrics(coordinates, force)
    if metrics["origin_gradient_1e19_n_per_m"] > ORIGIN_GRADIENT_LIMIT_SCALED:
        raise RuntimeError(
            f"{model} optimum violates origin-gradient constraint: {metrics}"
        )
    print(
        f"{model} optimum={parameters.tolist()}, area={metrics['restoring_area_zN']:.9g} zN, "
        f"origin={metrics['origin_gradient_1e19_n_per_m']:.9g}e-19 N/m, "
        f"elapsed={perf_counter() - started:.1f}s",
        flush=True,
    )
    return {
        "parameters": parameters,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_fun": float(result.fun),
        "optimizer_nfev": int(result.nfev),
        "optimization_metrics": metrics,
    }


def _parameter_record(model: str, parameters: np.ndarray) -> dict[str, float]:
    cooling_power_mw, detuning_mhz, third = (
        float(value) for value in parameters
    )
    record = {
        "cooling_power_mw_per_component": cooling_power_mw,
        "cooling_detuning_mhz": detuning_mhz,
    }
    if model == "mot_multilevel":
        record.update(
            {
                "coil_gradient_x_g_per_cm": 0.5 * third,
                "coil_gradient_y_g_per_cm": 0.5 * third,
                "coil_gradient_z_g_per_cm": -third,
                "coil_axial_gradient_magnitude_g_per_cm": third,
                "repump_power_mw_per_component": REPUMP_POWER_MW,
            }
        )
    else:
        trapping_powers_mw = 1.0e3 * _trapping_powers_w(third)
        record.update(
            {
                "trapping_power_x_mw_per_path": float(trapping_powers_mw[0]),
                "trapping_power_y_mw_per_path": float(trapping_powers_mw[1]),
                "trapping_power_z_mw_per_path": float(trapping_powers_mw[2]),
            }
        )
        if model == "pmot_derived_mot_strange":
            record["repump_power_mw_per_component"] = REPUMP_POWER_MW
    return record


def _shade_walls(axis) -> None:
    for center_mm in (-10.0, 10.0):
        axis.axvspan(
            center_mm - 2.5,
            center_mm + 2.5,
            color="#94a3b8",
            alpha=0.13,
            linewidth=0.0,
        )


def _parameter_summary(model: str, record: dict[str, float]) -> str:
    if model == "mot_multilevel":
        return (
            f"Pcool={record['cooling_power_mw_per_component']:.5g} mW/beam; "
            f"detuning={record['cooling_detuning_mhz']:.5g} MHz; "
            f"grad=({record['coil_gradient_x_g_per_cm']:.5g}, "
            f"{record['coil_gradient_y_g_per_cm']:.5g}, "
            f"{record['coil_gradient_z_g_per_cm']:.5g}) G/cm"
        )
    return (
        f"Pcool={record['cooling_power_mw_per_component']:.5g} mW/beam; "
        f"detuning={record['cooling_detuning_mhz']:.5g} MHz; "
        f"Ptrap=({record['trapping_power_x_mw_per_path']:.5g}, "
        f"{record['trapping_power_y_mw_per_path']:.5g}, "
        f"{record['trapping_power_z_mw_per_path']:.5g}) mW/path"
    )


def _plot_individual(model: str, case: dict, output_path: Path) -> None:
    coordinate_mm = 1.0e3 * case["coordinates_m"]
    force_zN = 1.0e21 * case["force_n"]
    gradient_scaled = 1.0e19 * case["gradient_n_per_m"]
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    axes[0].plot(
        coordinate_mm,
        force_zN,
        color=COLORS[model],
        linestyle=STYLES[model],
        linewidth=2.2,
    )
    axes[1].plot(
        coordinate_mm,
        gradient_scaled,
        color=COLORS[model],
        linestyle=STYLES[model],
        linewidth=2.2,
    )
    axes[1].fill_between(
        coordinate_mm,
        gradient_scaled,
        0.0,
        where=gradient_scaled < 0.0,
        color=COLORS[model],
        alpha=0.12,
        interpolate=True,
    )
    for panel in axes:
        panel.axhline(0.0, color="0.25", linewidth=0.8)
        panel.axvline(0.0, color="0.25", linewidth=0.8)
        panel.axvspan(-5.0, 5.0, color="#22c55e", alpha=0.035)
        _shade_walls(panel)
        panel.grid(alpha=0.20)
        panel.set_xlim(-12.5, 12.5)
    axes[0].set_ylabel(r"$F_x$ [zN]")
    axes[1].set_ylabel(r"$\partial F_x/\partial x$ [$10^{-19}$ N/m]")
    axes[1].set_xlabel("x position [mm]; y = z = 0")
    figure.suptitle(f"Optimized {LABELS[model]}: x-Axis Force and Gradient", fontsize=16)
    figure.text(
        0.5,
        0.94,
        _parameter_summary(model, case["parameter_record"]),
        ha="center",
        fontsize=10.5,
    )
    metrics = case["metrics"]
    figure.text(
        0.5,
        0.02,
        (
            f"negative-gradient area over ±5 mm = {metrics['restoring_area_zN']:.6g} zN; "
            f"origin gradient = {metrics['origin_gradient_1e19_n_per_m']:.6g} ×10⁻¹⁹ N/m; "
            f"central restoring width = {metrics['central_restoring_width_mm']:.5g} mm"
        ),
        ha="center",
        fontsize=10.0,
    )
    figure.tight_layout(rect=(0.04, 0.06, 0.98, 0.90), h_pad=1.3)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_overlay(cases: dict[str, dict], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 9.2), sharex=True)
    for model in MODELS:
        case = cases[model]
        coordinate_mm = 1.0e3 * case["coordinates_m"]
        axes[0].plot(
            coordinate_mm,
            1.0e21 * case["force_n"],
            color=COLORS[model],
            linestyle=STYLES[model],
            linewidth=2.0,
            label=LABELS[model],
        )
        axes[1].plot(
            coordinate_mm,
            1.0e19 * case["gradient_n_per_m"],
            color=COLORS[model],
            linestyle=STYLES[model],
            linewidth=2.0,
        )
    for panel in axes:
        panel.axhline(0.0, color="0.25", linewidth=0.8)
        panel.axvline(0.0, color="0.25", linewidth=0.8)
        panel.axvspan(-5.0, 5.0, color="#22c55e", alpha=0.035)
        _shade_walls(panel)
        panel.grid(alpha=0.20)
        panel.set_xlim(-12.5, 12.5)
    axes[0].set_ylabel(r"$F_x$ [zN]")
    axes[1].set_ylabel(r"$\partial F_x/\partial x$ [$10^{-19}$ N/m]")
    axes[1].set_xlabel("x position [mm]; y = z = 0")
    figure.suptitle(
        "Independently Optimized MOT Configurations: x-Axis Force and Gradient",
        fontsize=16,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.04, 0.09, 0.98, 0.94), h_pad=1.3)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _final_case(model: str, result: dict) -> dict:
    parameters = result["parameters"]
    # Evaluate both signs explicitly for the saved production curve rather than
    # relying on the symmetry shortcut used by the optimizer.
    force = _force_curve(model, parameters, FINAL_COORDINATES_M)
    gradient = np.gradient(force, FINAL_COORDINATES_M, edge_order=2)
    objective_mask = np.abs(FINAL_COORDINATES_M) <= HALF_WIDTH_M + 1.0e-15
    metrics = _curve_metrics(
        FINAL_COORDINATES_M[objective_mask],
        force[objective_mask],
    )
    return {
        "parameters": parameters,
        "parameter_record": _parameter_record(model, parameters),
        "coordinates_m": FINAL_COORDINATES_M.copy(),
        "force_n": force,
        "gradient_n_per_m": gradient,
        "metrics": metrics,
        "optimizer": {
            key: value
            for key, value in result.items()
            if key not in {"parameters", "optimization_metrics"}
        },
    }


def _write_curves(cases: dict[str, dict], path: Path) -> None:
    fieldnames = (
        "model",
        "x_mm",
        "force_n",
        "force_zN",
        "force_gradient_n_per_m",
        "force_gradient_1e19_n_per_m",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODELS:
            case = cases[model]
            for x_m, force_n, gradient in zip(
                case["coordinates_m"],
                case["force_n"],
                case["gradient_n_per_m"],
                strict=True,
            ):
                writer.writerow(
                    {
                        "model": model,
                        "x_mm": 1.0e3 * x_m,
                        "force_n": force_n,
                        "force_zN": 1.0e21 * force_n,
                        "force_gradient_n_per_m": gradient,
                        "force_gradient_1e19_n_per_m": 1.0e19 * gradient,
                    }
                )


def _jsonable_case(case: dict) -> dict:
    return {
        "parameters": case["parameter_record"],
        "metrics": {key: float(value) for key, value in case["metrics"].items()},
        "optimizer": case["optimizer"],
    }


def main() -> None:
    global PMOT_POWER_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--popsize", type=int, default=7)
    parser.add_argument(
        "--pmot-power-mode",
        choices=("gauss", "spherical"),
        default="gauss",
        help=(
            "Constrain pMOT-like trapping paths to Px=Py=Pz/2 (gauss) or "
            "Px=Py=Pz (spherical)."
        ),
    )
    args = parser.parse_args()
    if args.maxiter <= 0 or args.popsize <= 0:
        raise ValueError("maxiter and popsize must be positive")
    PMOT_POWER_MODE = args.pmot_power_mode
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    optimization_results = {
        model: _optimize(model, maxiter=args.maxiter, popsize=args.popsize)
        for model in MODELS
    }
    cases = {
        model: _final_case(model, optimization_results[model])
        for model in MODELS
    }

    individual_plots = {}
    for model in MODELS:
        path = output_directory / f"{model}_optimized_force_gradient.png"
        _plot_individual(model, cases[model], path)
        individual_plots[model] = str(path.relative_to(PROJECT_ROOT))

    overlay_path = output_directory / "four_optimized_models_overlay.png"
    _plot_overlay(cases, overlay_path)
    curves_path = output_directory / "optimized_force_gradient_curves.csv"
    _write_curves(cases, curves_path)

    manifest = {
        "status": "BOUNDED_DIAGNOSTIC_OPTIMIZATION",
        "scientific_scope": (
            "stationary-atom deterministic mean 780-nm radiation pressure; "
            "gravity, recoil diffusion, conservative trap-light force, and "
            "1529-nm scattering/heating excluded"
        ),
        "objective": {
            "equation": "integral over [-5,+5] mm of max(-dFx/dx,0) dx",
            "interval_mm": [-5.0, 5.0],
            "origin_constraint_1e19_n_per_m": ORIGIN_GRADIENT_LIMIT_SCALED,
            "pmot_power_constraint": (
                "Px=Py=Pz/2"
                if PMOT_POWER_MODE == "gauss"
                else "Px=Py=Pz"
            ),
        },
        "optimized_variables_and_bounds": {
            "shared": {
                "cooling_power_mw_per_component": list(COOLING_POWER_BOUNDS_MW),
                "cooling_detuning_mhz": list(COOLING_DETUNING_BOUNDS_MHZ),
            },
            "mot_multilevel": {
                "coil_axial_gradient_magnitude_g_per_cm": list(
                    MOT_AXIAL_GRADIENT_BOUNDS_G_PER_CM
                )
            },
            "other_three": {
                "trapping_power_scale_mw_per_path": list(
                    PMOT_XY_PATH_POWER_BOUNDS_MW
                ),
                "trapping_power_mapping": (
                    "Px=Py=Pz/2"
                    if PMOT_POWER_MODE == "gauss"
                    else "Px=Py=Pz"
                ),
            },
        },
        "fixed_parameters": {
            "repump_power_mw_per_component_multilevel": REPUMP_POWER_MW,
            "cooling_beam_diameter_mm": 12.7,
            "cooling_wavelength_nm": 780.0,
            "repump_wavelength_nm": 780.232684,
            "trapping_wavelength_nm": 1529.268881,
            "cooling_and_trapping_helicities_xyz_both_directions": [
                "sigma+",
                "sigma+",
                "sigma-",
            ],
        },
        "optimizer": {
            "method": "scipy differential_evolution with bounded polish",
            "seed_base": 20261027,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "optimization_positive_samples": len(OPTIMIZATION_POSITIVE_M),
            "final_curve_samples": len(FINAL_COORDINATES_M),
            "evaluation_counts": EVALUATION_COUNTS,
        },
        "results": {model: _jsonable_case(cases[model]) for model in MODELS},
        "outputs": {
            "individual_plots": individual_plots,
            "overlay_plot": str(overlay_path.relative_to(PROJECT_ROOT)),
            "curves_csv": str(curves_path.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path = output_directory / "optimization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["results"], indent=2), flush=True)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
