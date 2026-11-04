"""Plot one scalar/vector/tensor-selected two-level pMOT force and gradient.

The diagnostic evaluates a stationary atom at ``(x, 0, 0)`` with gravity
excluded.  It is a provisional transition-level force diagnostic, not a
capture, loading-rate, or trajectory calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import get_pmot_2level_pathway
from pmot.pMOT_2Level import PMOT_2LEVEL_PATHWAYS
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_component_polarizability_table
from pmot.pMOT_2Level import normalize_pmot_stark_component_code
from pmot.pMOT_2Level import pmot_stark_component_model_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_AXIS = (0.0, 0.0, 1.0)
VELOCITY_M_PER_S = np.zeros(3)


def _coordinate_grid_m() -> np.ndarray:
    return np.unique(
        np.round(
            np.concatenate(
                (
                    np.linspace(-12.5e-3, 12.5e-3, 501),
                    np.linspace(-5.0e-3, 5.0e-3, 801),
                    np.linspace(-11.0e-3, -9.0e-3, 161),
                    np.linspace(9.0e-3, 11.0e-3, 161),
                )
            ),
            decimals=12,
        )
    )


def _shade_glass_walls(axis) -> None:
    for center_mm in (-10.0, 10.0):
        axis.axvspan(
            center_mm - 2.5,
            center_mm + 2.5,
            color="#94a3b8",
            alpha=0.13,
            linewidth=0.0,
        )


def _metrics(
    coordinates_m: np.ndarray,
    force_n: np.ndarray,
    gradient_n_per_m: np.ndarray,
) -> dict[str, float]:
    objective_mask = np.abs(coordinates_m) <= 5.0e-3 + 1.0e-15
    x = coordinates_m[objective_mask]
    gradient = gradient_n_per_m[objective_mask]
    force = force_n[objective_mask]
    restoring = gradient < 0.0
    origin_index = int(np.argmin(np.abs(x)))
    left = origin_index
    right = origin_index
    if restoring[origin_index]:
        while left > 0 and restoring[left - 1]:
            left -= 1
        while right < len(restoring) - 1 and restoring[right + 1]:
            right += 1
        restoring_width_mm = 1.0e3 * (x[right] - x[left])
    else:
        restoring_width_mm = 0.0
    return {
        "negative_gradient_area_zN": 1.0e21
        * float(np.trapezoid(np.maximum(-gradient, 0.0), x)),
        "anti_restoring_area_zN": 1.0e21
        * float(np.trapezoid(np.maximum(gradient, 0.0), x)),
        "origin_gradient_1e19_n_per_m": 1.0e19 * float(gradient[origin_index]),
        "minimum_gradient_1e19_n_per_m": 1.0e19 * float(np.min(gradient)),
        "maximum_gradient_1e19_n_per_m": 1.0e19 * float(np.max(gradient)),
        "negative_gradient_fraction": float(np.mean(restoring)),
        "central_restoring_width_mm": float(restoring_width_mm),
        "force_peak_zN_within_5mm": 1.0e21 * float(np.max(np.abs(force))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cooling-power-mw", type=float, default=6.0)
    parser.add_argument("--cooling-detuning-mhz", type=float, default=-30.0)
    parser.add_argument("--trapping-power-mw", type=float, default=46.0)
    parser.add_argument("--pathway", choices=tuple(PMOT_2LEVEL_PATHWAYS), default="scalar_tensor_cancellation")
    parser.add_argument(
        "--components",
        default="svt",
        help=(
            "Enabled differential Stark ranks using s, v, and t; examples: "
            "svt, sv, vt, v, or none (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.cooling_power_mw <= 0.0 or args.trapping_power_mw <= 0.0:
        raise ValueError("cooling and trapping powers must be positive")
    if args.cooling_detuning_mhz >= 0.0:
        raise ValueError("cooling detuning must be negative for this diagnostic")
    components = normalize_pmot_stark_component_code(args.components)
    model_label = pmot_stark_component_model_label(components)

    output_directory = (args.output_directory or (
        PROJECT_ROOT / "outputs" / "diagnostics" / f"candidate_{args.pathway}_{model_label}"
    )).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    baseline_apparatus = default_simple_mot_apparatus()
    apparatus = replace(
        baseline_apparatus,
        cooling=replace(
            baseline_apparatus.cooling,
            power_w_per_beam=1.0e-3 * args.cooling_power_mw,
            detuning_hz=1.0e6 * args.cooling_detuning_mhz,
        ),
    )
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=1.0e6 * args.cooling_detuning_mhz,
        include_gravity=False,
    )
    cooling_beams = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    pathway = get_pmot_2level_pathway(args.pathway)
    trapping_beams = pathway.build_trapping_beams()
    polarizability_table = load_pmot_2level_component_polarizability_table(
        components
    )
    trapping_powers_w = 1.0e-3 * np.full(3, args.trapping_power_mw)

    coordinates_m = _coordinate_grid_m()
    force_n = np.empty_like(coordinates_m)
    for index, coordinate_m in enumerate(coordinates_m):
        observable = evaluate_pmot_2level_local_observable(
            cooling_beams,
            trapping_beams,
            np.asarray((coordinate_m, 0.0, 0.0)),
            VELOCITY_M_PER_S,
            path_powers_w=trapping_powers_w,
            previous_axis=FALLBACK_AXIS,
            simple_config=simple,
            polarizability_table=polarizability_table,
        )
        force_n[index] = observable.force_n[0]
    gradient_n_per_m = np.gradient(force_n, coordinates_m, edge_order=2)
    metrics = _metrics(coordinates_m, force_n, gradient_n_per_m)

    coordinate_mm = 1.0e3 * coordinates_m
    force_z_n = 1.0e21 * force_n
    gradient_scaled = 1.0e19 * gradient_n_per_m
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    axes[0].plot(coordinate_mm, force_z_n, color="#be123c", linewidth=2.3)
    axes[1].plot(coordinate_mm, gradient_scaled, color="#be123c", linewidth=2.3)
    axes[1].fill_between(
        coordinate_mm,
        gradient_scaled,
        0.0,
        where=gradient_scaled < 0.0,
        color="#be123c",
        alpha=0.13,
        interpolate=True,
    )
    for axis in axes:
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.axvline(0.0, color="0.25", linewidth=0.8)
        axis.axvspan(-5.0, 5.0, color="#22c55e", alpha=0.035)
        _shade_glass_walls(axis)
        axis.grid(alpha=0.20)
        axis.set_xlim(-12.5, 12.5)
    axes[0].set_ylabel(r"$F_x$ [zN]")
    axes[1].set_ylabel(r"$\partial F_x/\partial x$ [$10^{-19}$ N/m]")
    axes[1].set_xlabel("x position [mm]; y = z = 0; v = 0")
    figure.suptitle(
        f"{model_label}: x-Axis Force and Force Gradient",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.94,
        fill((
            f"Pcool = {args.cooling_power_mw:g} mW/component; "
            f"detuning = {args.cooling_detuning_mhz:g} MHz; "
            f"Ptrap = ({args.trapping_power_mw:g}, {args.trapping_power_mw:g}, "
            f"{args.trapping_power_mw:g}) mW/path; λ = {pathway.trapping_wavelength_nm:.9f} nm; gravity excluded"
        ), width=115),
        ha="center",
        va="top",
        fontsize=10.5,
    )
    figure.text(
        0.5,
        0.02,
        (
            "negative-gradient area over ±5 mm = "
            f"{metrics['negative_gradient_area_zN']:.6g} zN; "
            "origin gradient = "
            f"{metrics['origin_gradient_1e19_n_per_m']:.6g} ×10⁻¹⁹ N/m; "
            "central restoring width = "
            f"{metrics['central_restoring_width_mm']:.6g} mm"
        ),
        ha="center",
        fontsize=10.0,
    )
    figure.tight_layout(rect=(0.04, 0.06, 0.98, 0.86), h_pad=1.3)
    figure_path = output_directory / f"{model_label}_force_gradient.png"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    curves_path = output_directory / f"{model_label}_force_gradient.csv"
    with curves_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "x_mm",
                "force_n",
                "force_zN",
                "force_gradient_n_per_m",
                "force_gradient_1e19_n_per_m",
            )
        )
        writer.writerows(
            zip(
                coordinate_mm,
                force_n,
                force_z_n,
                gradient_n_per_m,
                gradient_scaled,
                strict=True,
            )
        )

    manifest = {
        "status": "PROVISIONAL_TRANSITION_LEVEL_DIAGNOSTIC",
        "parameters": {
            "model_label": model_label,
            "stark_components": components or "none",
            "pathway": pathway.key,
            "trapping_wavelength_nm": pathway.trapping_wavelength_nm,
            "cooling_power_mw_per_component": args.cooling_power_mw,
            "cooling_detuning_mhz": args.cooling_detuning_mhz,
            "trapping_power_mw_per_path_xyz": [args.trapping_power_mw] * 3,
            "position_cut": "(x, 0, 0)",
            "velocity_m_per_s": [0.0, 0.0, 0.0],
            "gravity_included": False,
        },
        "metrics_over_plus_minus_5mm": metrics,
        "outputs": {
            "figure": str(figure_path),
            "curves": str(curves_path),
        },
    }
    manifest_path = output_directory / f"{model_label}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
