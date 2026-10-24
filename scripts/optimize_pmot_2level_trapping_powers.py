"""Optimize provisional pMOT_2Level trapping powers over the central 2 mm.

The cooling configuration is fixed at 1.0571 mW per traveling beam and
-15 MHz.  The unrestricted search respects the exact x/y symmetry by using
``(P_xy, P_xy, P_z)``.  A second search imposes equal power on all three paths.

For each axis, the net restoring area is

    A_i = - integral_{-L}^{+L} dF_i/dx_i dx_i
        = F_i(-L) - F_i(+L),    L = 1 mm.

The robust optimization maximizes the least restoring local gradient anywhere
on any of the three axes inside the interval.  The weakest-axis integrated
area is a small secondary preference.  This prevents a large restoring patch
from hiding a narrow anti-restoring core at the origin.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar

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
    / "pmot_2level_trapping_power_optimization_20261023"
)
COOLING_POWER_MW_PER_BEAM = 1.0571
COOLING_DETUNING_MHZ = -15.0
HALF_WIDTH_M = 1.0e-3
_OPTIMIZATION_POSITIVE_MM = np.asarray(
    (0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
)
OPTIMIZATION_COORDINATES_M = 1.0e-3 * np.concatenate(
    (-_OPTIMIZATION_POSITIVE_MM[:0:-1], _OPTIMIZATION_POSITIVE_MM)
)
DENSE_COORDINATES_M = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, 161)
POWER_BOUNDS_MW = (0.0, 60.0)
AXES = ("x", "y", "z")
AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}
KNOWN_CASES_MW = {
    "equal 19.147243": np.asarray((19.147243, 19.147243, 19.147243)),
    "equal 38.294486": np.asarray((38.294486, 38.294486, 38.294486)),
    "19.147243, 19.147243, 38.294486": np.asarray(
        (19.147243, 19.147243, 38.294486)
    ),
}


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
    return (
        simple,
        build_pmot_2level_cooling_beams(
            apparatus_config=apparatus,
            simple_config=simple,
        ),
        build_pmot_2level_trapping_beams(),
        load_pmot_2level_polarizability_table(),
    )


SIMPLE, COOLING, TRAPPING, POLARIZABILITY_TABLE = _build_context()


def _observable(position_m: np.ndarray, powers_mw: np.ndarray):
    return evaluate_pmot_2level_local_observable(
        COOLING,
        TRAPPING,
        position_m,
        np.zeros(3),
        path_powers_w=1.0e-3 * np.asarray(powers_mw, dtype=float),
        previous_axis=(0.0, 0.0, 1.0),
        simple_config=SIMPLE,
        polarizability_table=POLARIZABILITY_TABLE,
    )


def _force_curves(powers_mw: np.ndarray, coordinates_m: np.ndarray):
    curves: dict[str, np.ndarray] = {}
    for axis, component in AXIS_INDEX.items():
        force = np.empty(len(coordinates_m))
        for index, coordinate in enumerate(coordinates_m):
            position = np.zeros(3)
            position[component] = coordinate
            force[index] = _observable(position, powers_mw).force_n[component]
        curves[axis] = force
    return curves


def _metrics(powers_mw: np.ndarray, coordinates_m: np.ndarray):
    powers = np.asarray(powers_mw, dtype=float)
    curves = _force_curves(powers, coordinates_m)
    metrics: dict[str, dict[str, float]] = {}
    for axis in AXES:
        force = curves[axis]
        gradient = np.gradient(force, coordinates_m, edge_order=2)
        net_area_n = float(force[0] - force[-1])
        restoring_area_n = float(
            np.trapezoid(np.maximum(-gradient, 0.0), coordinates_m)
        )
        anti_restoring_area_n = float(
            np.trapezoid(np.maximum(gradient, 0.0), coordinates_m)
        )
        origin_index = int(np.argmin(np.abs(coordinates_m)))
        metrics[axis] = {
            "net_restoring_area_zN": 1.0e21 * net_area_n,
            "restoring_area_zN": 1.0e21 * restoring_area_n,
            "anti_restoring_area_zN": 1.0e21 * anti_restoring_area_n,
            "negative_gradient_fraction": float(np.mean(gradient < 0.0)),
            "origin_gradient_1e19_n_per_m": float(
                1.0e19 * gradient[origin_index]
            ),
            "maximum_gradient_1e19_n_per_m": float(1.0e19 * np.max(gradient)),
            "minimum_gradient_1e19_n_per_m": float(1.0e19 * np.min(gradient)),
        }

    origin = _observable(np.zeros(3), powers)
    effective_detuning_hz = float(
        origin.rate_samples[0].effective_detunings_hz_q_minus_pi_q_plus[1]
    )
    metrics["origin"] = {
        "common_ac_stark_shift_mhz": float(
            1.0e-6 * origin.stark.transition_shifts_hz_by_q[1]
        ),
        "effective_detuning_mhz": 1.0e-6 * effective_detuning_hz,
        "total_cooling_saturation": float(
            origin.rate_samples[0].total_saturation_parameter
        ),
    }
    return curves, metrics


def _score(powers_mw: np.ndarray) -> float:
    _, metrics = _metrics(powers_mw, OPTIMIZATION_COORDINATES_M)
    weakest_net_area = min(
        metrics[axis]["net_restoring_area_zN"] for axis in AXES
    )
    weakest_local_restoring_gradient = min(
        -metrics[axis]["maximum_gradient_1e19_n_per_m"] for axis in AXES
    )
    # One zN accumulated across 2 mm corresponds to an average gradient of
    # 5e-19 N/m.  The small area term breaks near-degeneracies without allowing
    # a broad outer lobe to compensate for an anti-restoring central core.
    weakest_average_gradient = 5.0 * weakest_net_area
    return weakest_local_restoring_gradient + 0.02 * weakest_average_gradient


def _optimize_unrestricted() -> np.ndarray:
    def objective(values):
        pxy_mw, pz_mw = values
        return -_score(np.asarray((pxy_mw, pxy_mw, pz_mw)))

    result = differential_evolution(
        objective,
        bounds=(POWER_BOUNDS_MW, POWER_BOUNDS_MW),
        seed=20261023,
        popsize=6,
        maxiter=12,
        tol=1.0e-4,
        polish=True,
        updating="immediate",
        workers=1,
    )
    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise RuntimeError(f"unrestricted optimization failed: {result.message}")
    if not result.success:
        print(
            "unrestricted optimizer reached its iteration cap; using the "
            f"finite polished solution ({result.message})",
            flush=True,
        )
    return np.asarray((result.x[0], result.x[0], result.x[1]))


def _optimize_equal() -> np.ndarray:
    result = minimize_scalar(
        lambda power_mw: -_score(np.full(3, power_mw)),
        bounds=POWER_BOUNDS_MW,
        method="bounded",
        options={"xatol": 1.0e-6, "maxiter": 160},
    )
    if not result.success:
        raise RuntimeError(f"equal-power optimization failed: {result.message}")
    return np.full(3, result.x)


def _plot_curves(cases, path: Path) -> None:
    colors = {
        "equal 19.147243": "#2563eb",
        "equal 38.294486": "#dc2626",
        "19.147243, 19.147243, 38.294486": "#d97706",
        "optimized unrestricted": "#059669",
        "optimized equal power": "#7c3aed",
    }
    styles = {
        "equal 19.147243": "--",
        "equal 38.294486": ":",
        "19.147243, 19.147243, 38.294486": "-.",
        "optimized unrestricted": "-",
        "optimized equal power": (0, (5, 1)),
    }
    coordinate_mm = 1.0e3 * DENSE_COORDINATES_M
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), sharex=True)
    for column, axis in enumerate(AXES):
        for label, case in cases.items():
            force = case["curves"][axis]
            gradient = np.gradient(force, DENSE_COORDINATES_M, edge_order=2)
            power_text = ",".join(f"{value:.3f}" for value in case["powers_mw"])
            legend_label = f"{label}: ({power_text}) mW"
            axes[0, column].plot(
                coordinate_mm,
                1.0e21 * force,
                color=colors[label],
                linestyle=styles[label],
                linewidth=2.0,
                label=legend_label,
            )
            axes[1, column].plot(
                coordinate_mm,
                1.0e19 * gradient,
                color=colors[label],
                linestyle=styles[label],
                linewidth=2.0,
                label=legend_label,
            )
        axes[0, column].axhline(0.0, color="0.3", linewidth=0.8)
        axes[1, column].axhline(0.0, color="0.3", linewidth=0.8)
        axes[0, column].axvline(0.0, color="0.3", linewidth=0.8)
        axes[1, column].axvline(0.0, color="0.3", linewidth=0.8)
        axes[0, column].set_title(
            f"Along {axis}-axis; "
            + "; ".join(f"{other}=0 mm" for other in AXES if other != axis)
        )
        axes[0, column].set_ylabel(rf"$F_{axis}$ [zN]")
        axes[1, column].set_ylabel(
            rf"$\partial F_{axis}/\partial {axis}$ [$10^{{-19}}$ N/m]"
        )
        axes[1, column].set_xlabel(f"{axis} position [mm]")
        for row in range(2):
            axes[row, column].grid(True, alpha=0.2)
            axes[row, column].set_xlim(-1.0, 1.0)
    figure.suptitle(
        "2-Level pMOT Central Force and Force-Gradient Power Optimization",
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
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.095, 0.98, 0.91), h_pad=2.0, w_pad=1.5)
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def _plot_equal_scan(path: Path) -> list[dict[str, float]]:
    rows = []
    powers = np.linspace(0.0, 45.0, 91)
    for power in powers:
        candidate = np.full(3, power)
        _, metrics = _metrics(candidate, OPTIMIZATION_COORDINATES_M)
        rows.append(
            {
                "power_mw_per_path": float(power),
                "weakest_net_restoring_area_zN": float(
                    min(metrics[axis]["net_restoring_area_zN"] for axis in AXES)
                ),
                "largest_origin_gradient_1e19_n_per_m": float(
                    max(metrics[axis]["origin_gradient_1e19_n_per_m"] for axis in AXES)
                ),
                "weakest_local_restoring_gradient_1e19_n_per_m": float(
                    min(
                        -metrics[axis]["maximum_gradient_1e19_n_per_m"]
                        for axis in AXES
                    )
                ),
                "effective_detuning_mhz": float(
                    metrics["origin"]["effective_detuning_mhz"]
                ),
            }
        )
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
    area = np.asarray([row["weakest_net_restoring_area_zN"] for row in rows])
    detuning = np.asarray([row["effective_detuning_mhz"] for row in rows])
    axes[0].plot(powers, area, color="#7c3aed", linewidth=2.2)
    axes[0].axhline(0.0, color="0.3", linewidth=0.9)
    axes[0].set_ylabel("weakest-axis net restoring area [zN]")
    axes[0].set_title(r"$\min_i\,[F_i(-1\,\mathrm{mm})-F_i(+1\,\mathrm{mm})]$")
    axes[1].plot(powers, detuning, color="#0f766e", linewidth=2.2)
    axes[1].axhline(0.0, color="#dc2626", linestyle="--", linewidth=1.2)
    axes[1].axhline(-15.0, color="0.4", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel(r"origin $\Delta_{\rm eff}$ [MHz]")
    axes[1].set_xlabel("equal trapping power per path [mW]")
    for axis in axes:
        axis.grid(True, alpha=0.2)
    figure.suptitle("Equal-Power pMOT Restoring-Force Optimization", fontsize=16)
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.95))
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)
    return rows


def _jsonable_metrics(metrics):
    return {
        key: {name: float(value) for name, value in values.items()}
        for key, values in metrics.items()
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    unrestricted_powers = _optimize_unrestricted()
    equal_powers = _optimize_equal()
    requested_cases = dict(KNOWN_CASES_MW)
    requested_cases["optimized unrestricted"] = unrestricted_powers
    requested_cases["optimized equal power"] = equal_powers

    cases = {}
    for label, powers in requested_cases.items():
        curves, metrics = _metrics(powers, DENSE_COORDINATES_M)
        cases[label] = {
            "powers_mw": np.asarray(powers, dtype=float),
            "curves": curves,
            "metrics": metrics,
            "score": _score(powers),
        }
        print(
            f"{label}: powers={powers}, score={cases[label]['score']:.9g}, "
            f"detuning={metrics['origin']['effective_detuning_mhz']:.6f} MHz",
            flush=True,
        )

    curves_path = OUTPUT_DIRECTORY / "optimized_power_force_gradient_comparison.png"
    _plot_curves(cases, curves_path)
    scan_path = OUTPUT_DIRECTORY / "equal_power_optimization_scan.png"
    equal_scan = _plot_equal_scan(scan_path)

    csv_path = OUTPUT_DIRECTORY / "candidate_metrics.csv"
    fieldnames = (
        "case",
        "power_x_mw",
        "power_y_mw",
        "power_z_mw",
        "axis",
        "net_restoring_area_zN",
        "restoring_area_zN",
        "anti_restoring_area_zN",
        "negative_gradient_fraction",
        "origin_gradient_1e19_n_per_m",
        "maximum_gradient_1e19_n_per_m",
        "minimum_gradient_1e19_n_per_m",
        "origin_effective_detuning_mhz",
        "origin_common_ac_stark_shift_mhz",
        "optimization_score",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for label, case in cases.items():
            for axis in AXES:
                writer.writerow(
                    {
                        "case": label,
                        "power_x_mw": case["powers_mw"][0],
                        "power_y_mw": case["powers_mw"][1],
                        "power_z_mw": case["powers_mw"][2],
                        "axis": axis,
                        **case["metrics"][axis],
                        "origin_effective_detuning_mhz": case["metrics"]["origin"][
                            "effective_detuning_mhz"
                        ],
                        "origin_common_ac_stark_shift_mhz": case["metrics"][
                            "origin"
                        ]["common_ac_stark_shift_mhz"],
                        "optimization_score": case["score"],
                    }
                )

    scan_csv_path = OUTPUT_DIRECTORY / "equal_power_scan.csv"
    with scan_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=equal_scan[0].keys())
        writer.writeheader()
        writer.writerows(equal_scan)

    manifest = {
        "status": "PROVISIONAL_PMOT_2LEVEL_OPTIMIZATION",
        "objective": {
            "interval_mm": [-1.0, 1.0],
            "primary": (
                "maximize the least local restoring gradient -dF_i/dx_i over "
                "all axes and all sampled points in +/-1 mm"
            ),
            "secondary": (
                "add 0.02 times the weakest-axis average restoring gradient, "
                "obtained from F_i(-1 mm)-F_i(+1 mm)"
            ),
            "unrestricted_parameterization": "(P_x,P_y,P_z)=(P_xy,P_xy,P_z)",
            "search_bounds_mw_per_path": list(POWER_BOUNDS_MW),
        },
        "configuration": {
            "cooling_power_mw_per_traveling_beam": COOLING_POWER_MW_PER_BEAM,
            "cooling_detuning_mhz": COOLING_DETUNING_MHZ,
            "stationary_atom": True,
            "gravity_included": False,
        },
        "cases": {
            label: {
                "powers_mw_xyz": case["powers_mw"].tolist(),
                "score": float(case["score"]),
                "metrics": _jsonable_metrics(case["metrics"]),
            }
            for label, case in cases.items()
        },
        "outputs": {
            "comparison_figure": str(curves_path.relative_to(PROJECT_ROOT)),
            "equal_power_scan_figure": str(scan_path.relative_to(PROJECT_ROOT)),
            "candidate_metrics_csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "equal_power_scan_csv": str(scan_csv_path.relative_to(PROJECT_ROOT)),
        },
        "caveat": (
            "This optimizes only the provisional effective two-level radiation-"
            "pressure model. It is not a physical pMOT design optimization."
        ),
    }
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
