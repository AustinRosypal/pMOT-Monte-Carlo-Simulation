"""Audit optical/effective-field geometry equivalence of mot_strange and pMOT_2Level.

This diagnostic deliberately stops before either radiation-force kernel.  It
compares the two Gaussian-envelope implementations after the pMOT path powers
have been calibrated to the default mot_strange origin gradients.  It also
records the independent vector-polarizability conversions and the pMOT common
scalar-plus-tensor shift at the vector-field zero.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pmot.configuration import PLANCK_CONSTANT_J_S
from pmot.configuration import SPEED_OF_LIGHT_M_PER_S
from pmot.configuration import VACUUM_PERMITTIVITY_F_PER_M
from pmot.mot_multilevel.atomic_structure import BOHR_MAGNETON_J_PER_T
from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import build_rate_equation_model
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import rate_equation_observable
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_strange.polarizability import DifferentialPolarizabilityTable
from pmot.mot_strange.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
)
from pmot.mot_strange.surrogate_effective_field import surrogate_effective_field_t
from pmot.mot_strange.surrogate_effective_field import surrogate_field_jacobians_t_per_m
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import evaluate_differential_ac_stark_shift
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "strange_pmot_geometry_equivalence_audit_20261023"
)
MATCHED_PMOT_POWERS_MW = np.asarray(
    (9.521254433005243, 9.521254433005243, 19.042508866010486)
)
CURRENT_PMOT_POWERS_MW = np.full(3, 31.524310589374092)
SCAN_LIMIT_MM = 12.5
AXES = "xyz"


def _scan_m() -> np.ndarray:
    broad = np.linspace(-SCAN_LIMIT_MM, SCAN_LIMIT_MM, 1001)
    negative_focus = np.linspace(-10.12, -9.88, 241)
    positive_focus = np.linspace(9.88, 10.12, 241)
    values_mm = np.unique(
        np.round(np.concatenate((broad, negative_focus, positive_focus)), 9)
    )
    return 1.0e-3 * values_mm


def _pmot_field_t(position_m, powers_w, beams, table) -> np.ndarray:
    sample = evaluate_differential_ac_stark_shift(
        beams,
        position_m,
        np.zeros(3),
        path_powers_w=powers_w,
        previous_axis=(0.0, 0.0, 1.0),
        polarizability_table=table,
    )
    return np.asarray(sample.net_vector_energy_vector_j) / BOHR_MAGNETON_J_PER_T


def _line_fields(scan_m, powers_mw, beams, table):
    strange = {}
    pmot = {}
    powers_w = 1.0e-3 * np.asarray(powers_mw)
    for component, axis in enumerate(AXES):
        positions = np.zeros((len(scan_m), 3))
        positions[:, component] = scan_m
        strange[axis] = np.asarray(surrogate_effective_field_t(positions))[:, component]
        pmot[axis] = np.asarray(
            [_pmot_field_t(point, powers_w, beams, table)[component] for point in positions]
        )
    return strange, pmot


def _relative_rms(reference, candidate, selector) -> float:
    difference = candidate[selector] - reference[selector]
    scale = np.sqrt(np.mean(reference[selector] ** 2))
    return float(np.sqrt(np.mean(difference**2)) / scale)


def _geometry_metrics(scan_m, strange, pmot):
    regions = {
        "central_2mm": np.abs(scan_m) <= 2.0e-3,
        "central_5mm": np.abs(scan_m) <= 5.0e-3,
        "full_12p5mm": np.abs(scan_m) <= 12.5e-3,
    }
    result = {}
    for axis in AXES:
        strange_gradient = np.gradient(strange[axis], scan_m, edge_order=2)
        pmot_gradient = np.gradient(pmot[axis], scan_m, edge_order=2)
        result[axis] = {}
        for region_name, selector in regions.items():
            result[axis][region_name] = {
                "field_relative_rms_difference": _relative_rms(
                    strange[axis], pmot[axis], selector
                ),
                "field_gradient_relative_rms_difference": _relative_rms(
                    strange_gradient, pmot_gradient, selector
                ),
            }
    return result


def _plot(
    scan_m,
    strange,
    pmot,
    output_path,
    *,
    x_limit_mm=SCAN_LIMIT_MM,
    title_suffix="",
):
    coordinate_mm = 1.0e3 * scan_m
    visible = np.abs(coordinate_mm) <= x_limit_mm
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), sharex="col")
    for column, axis in enumerate(AXES):
        strange_gradient = np.gradient(strange[axis], scan_m, edge_order=2)
        pmot_gradient = np.gradient(pmot[axis], scan_m, edge_order=2)
        axes[0, column].plot(
            coordinate_mm,
            1.0e4 * strange[axis],
            color="#d97706",
            linestyle="--",
            linewidth=2.0,
            label="mot_strange independent surrogate geometry",
        )
        axes[0, column].plot(
            coordinate_mm,
            1.0e4 * pmot[axis],
            color="#be123c",
            linewidth=1.8,
            label="pMOT_2Level shared trapping-beam geometry",
        )
        axes[1, column].plot(
            coordinate_mm,
            100.0 * strange_gradient,
            color="#d97706",
            linestyle="--",
            linewidth=2.0,
        )
        axes[1, column].plot(
            coordinate_mm,
            100.0 * pmot_gradient,
            color="#be123c",
            linewidth=1.8,
        )
        axes[0, column].set_title(f"Along {axis}-axis; other coordinates = 0")
        axes[0, column].set_ylabel(rf"$B_{{\mathrm{{eff}},{axis}}}$ [G]")
        axes[1, column].set_ylabel(
            rf"$\partial B_{{\mathrm{{eff}},{axis}}}/\partial {axis}$ [G/cm]"
        )
        axes[1, column].set_xlabel(f"{axis} position [mm]")
        for panel in axes[:, column]:
            panel.axhline(0.0, color="0.25", linewidth=0.8)
            panel.axvline(0.0, color="0.25", linewidth=0.8)
            panel.grid(alpha=0.20)
            panel.set_xlim(-x_limit_mm, x_limit_mm)
        field_visible = 1.0e4 * np.concatenate(
            (strange[axis][visible], pmot[axis][visible])
        )
        gradient_visible = 100.0 * np.concatenate(
            (strange_gradient[visible], pmot_gradient[visible])
        )
        for panel, values in (
            (axes[0, column], field_visible),
            (axes[1, column], gradient_visible),
        ):
            lower = float(np.min(values))
            upper = float(np.max(values))
            span = upper - lower
            padding = 0.08 * span if span > 0.0 else max(abs(upper), 1.0) * 0.08
            panel.set_ylim(lower - padding, upper + padding)
    figure.suptitle(
        "Geometry-Only Audit: Effective Fields Before Either Force Solver"
        + title_suffix,
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        "Both calibrated to (+4.973, +4.973, -9.945) G/cm at the origin; "
        "mot_strange paths=(10.8, 10.8, 21.6) mW; "
        "pMOT paths=(9.521254, 9.521254, 19.042509) mW",
        ha="center",
        va="top",
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
    figure.tight_layout(rect=(0.02, 0.065, 0.98, 0.90), h_pad=1.5, w_pad=1.5)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _stark_origin_summary(powers_mw, beams, table):
    sample = evaluate_differential_ac_stark_shift(
        beams,
        np.zeros(3),
        np.zeros(3),
        path_powers_w=1.0e-3 * np.asarray(powers_mw),
        previous_axis=(0.0, 0.0, 1.0),
        polarizability_table=table,
    )
    scalar = np.asarray(sample.component_scalar_energies_j)
    tensor = np.asarray(sample.component_tensor_energies_j)
    common_hz_by_path = [
        float(np.sum((scalar + tensor)[2 * index : 2 * index + 2]) / PLANCK_CONSTANT_J_S)
        for index in range(3)
    ]
    common_hz = float(sample.transition_shifts_hz_by_q[1])
    return {
        "powers_mw_xyz": np.asarray(powers_mw).tolist(),
        "vector_field_zero": bool(
            np.linalg.norm(sample.net_vector_energy_vector_j) < 1.0e-40
        ),
        "axis_is_fallback_at_origin": not sample.quantization_axis_from_vector_shift,
        "fallback_axis_xyz": list(sample.quantization_axis),
        "tensor_factors_by_component": list(sample.component_tensor_angular_factors),
        "common_scalar_plus_tensor_shift_mhz_by_path_xyz": [
            1.0e-6 * value for value in common_hz_by_path
        ],
        "total_common_scalar_plus_tensor_shift_mhz": 1.0e-6 * common_hz,
        "effective_pi_channel_detuning_mhz_for_minus15mhz_laser": (
            -15.0 - 1.0e-6 * common_hz
        ),
    }


def _vector_only_table(table):
    zeros = np.zeros_like(table.scalar_si)
    return DifferentialPolarizabilityTable(
        wavelengths_nm=table.wavelengths_nm,
        scalar_si=zeros,
        vector_si=table.vector_si,
        tensor_si=zeros,
        source_path=table.source_path,
    )


def _pmot_origin_force_gradient(
    cooling_power_mw,
    powers_mw,
    trapping_beams,
    table,
):
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=-15.0e6,
        include_gravity=False,
    )
    apparatus = default_simple_mot_apparatus()
    apparatus = replace(
        apparatus,
        cooling=replace(
            apparatus.cooling,
            power_w_per_beam=1.0e-3 * cooling_power_mw,
            detuning_hz=-15.0e6,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    step_m = 1.0e-6
    gradients = np.empty(3)
    for coordinate in range(3):
        displacement = np.zeros(3)
        displacement[coordinate] = step_m
        force_plus = evaluate_pmot_2level_local_observable(
            cooling,
            trapping_beams,
            displacement,
            np.zeros(3),
            path_powers_w=1.0e-3 * np.asarray(powers_mw),
            previous_axis=(0.0, 0.0, 1.0),
            simple_config=simple,
            polarizability_table=table,
        ).force_n[coordinate]
        force_minus = evaluate_pmot_2level_local_observable(
            cooling,
            trapping_beams,
            -displacement,
            np.zeros(3),
            path_powers_w=1.0e-3 * np.asarray(powers_mw),
            previous_axis=(0.0, 0.0, 1.0),
            simple_config=simple,
            polarizability_table=table,
        ).force_n[coordinate]
        gradients[coordinate] = (force_plus - force_minus) / (2.0 * step_m)
    return 1.0e19 * gradients


def _strange_origin_force_gradient():
    config = replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=27.0e-3,
        repump_power_w_per_beam=0.1e-3,
        cooling_detuning_rad_per_s=-2.0 * np.pi * 15.0e6,
        include_gravity=False,
    )
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    step_m = 1.0e-6
    gradients = np.empty(3)
    for coordinate in range(3):
        displacement = np.zeros(3)
        displacement[coordinate] = step_m
        samples = []
        for sign in (+1.0, -1.0):
            position = sign * displacement
            sample = rate_equation_observable(
                model,
                beams,
                tuple(position),
                (0.0, 0.0, 0.0),
                None,
                config,
                (0.0, 0.0, 1.0),
                magnetic_field_function=surrogate_effective_field_t,
            )
            samples.append(sample.force_n[coordinate])
        gradients[coordinate] = (samples[0] - samples[1]) / (2.0 * step_m)
    return 1.0e19 * gradients


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    beams = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    scan_m = _scan_m()
    strange, pmot = _line_fields(
        scan_m,
        MATCHED_PMOT_POWERS_MW,
        beams,
        table,
    )
    figure_path = OUTPUT_DIRECTORY / "geometry_only_effective_field_comparison.png"
    _plot(scan_m, strange, pmot, figure_path)
    central_figure_path = (
        OUTPUT_DIRECTORY / "geometry_only_effective_field_comparison_central_5mm.png"
    )
    _plot(
        scan_m,
        strange,
        pmot,
        central_figure_path,
        x_limit_mm=5.0,
        title_suffix=" — Central ±5 mm",
    )

    pbeam = beams[0]
    cfg = DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    seen_nm = pbeam.wavelength_m * 1.0e9
    alpha_vector = float(np.interp(seen_nm, table.wavelengths_nm, table.vector_si))
    csv_vector_shift_hz_per_w_per_m2 = (
        -2.0
        * alpha_vector
        / (
            SPEED_OF_LIGHT_M_PER_S
            * VACUUM_PERMITTIVITY_F_PER_M
            * PLANCK_CONSTANT_J_S
        )
    )
    csv_beff_t_per_w_per_m2 = (
        PLANCK_CONSTANT_J_S
        * csv_vector_shift_hz_per_w_per_m2
        / BOHR_MAGNETON_J_PER_T
    )
    strange_radii_um = [
        1.0e6 * np.sqrt(cfg.forward_rayleigh_range_m * cfg.wavelength_m / np.pi),
        1.0e6 * np.sqrt(cfg.return_rayleigh_range_m * cfg.wavelength_m / np.pi),
    ]
    origin_strange_gradient = 100.0 * np.diag(
        surrogate_field_jacobians_t_per_m(
            np.zeros(3), step_m=1.0e-6, config=cfg
        )
    )
    report = {
        "status": "GEOMETRY_AND_CONVERSION_AUDIT",
        "beam_parameters": {
            "mot_strange": {
                "wavelength_nm": 1.0e9 * cfg.wavelength_m,
                "waist_positions_mm_incident_retro": [
                    1.0e3 * cfg.forward_waist_position_m,
                    1.0e3 * cfg.return_waist_position_m,
                ],
                "rayleigh_ranges_um_incident_retro": [
                    1.0e6 * cfg.forward_rayleigh_range_m,
                    1.0e6 * cfg.return_rayleigh_range_m,
                ],
                "equivalent_waist_radii_um_incident_retro": strange_radii_um,
                "hard_cubic_cell_clipping_half_length_mm": 1.0e3 * cfg.cell_half_length_m,
            },
            "pMOT_2Level": {
                "wavelength_nm": seen_nm,
                "waist_positions_mm_incident_retro": [
                    1.0e3 * beams[0].waist_position_m[0],
                    1.0e3 * beams[1].waist_position_m[0],
                ],
                "rayleigh_ranges_um_incident_retro": [
                    1.0e6 * beams[0].rayleigh_range_m,
                    1.0e6 * beams[1].rayleigh_range_m,
                ],
                "waist_radii_um_incident_retro": [
                    1.0e6 * beams[0].waist_radius_m,
                    1.0e6 * beams[1].waist_radius_m,
                ],
                "hard_cell_clipping": False,
            },
        },
        "vector_conversion": {
            "csv_vector_shift_hz_per_w_per_m2": csv_vector_shift_hz_per_w_per_m2,
            "pMOT_2Level_beff_coefficient_t_per_w_per_m2": csv_beff_t_per_w_per_m2,
            "mot_strange_kappa_t_per_w_per_m2": cfg.kappa_t_per_w_per_m2,
            "absolute_kappa_ratio_strange_over_pmot": abs(
                cfg.kappa_t_per_w_per_m2 / csv_beff_t_per_w_per_m2
            ),
            "note": (
                "The 1363.34 Hz/(W/m^2) number already equals "
                "-2*alpha_vector/(c*epsilon0*h) for the CSV value. The corrected "
                "surrogate mapping converts it with -h/(mu_B*G), without another "
                "factor of two."
            ),
        },
        "matched_origin_gradients_g_per_cm": {
            "mot_strange": origin_strange_gradient.tolist(),
            "pMOT_2Level": [4.9725729169037585, 4.9725729169037585, -9.945145833807517],
        },
        "geometry_profile_differences": _geometry_metrics(scan_m, strange, pmot),
        "pMOT_origin_stark_state": {
            "matched_to_default_strange": _stark_origin_summary(
                MATCHED_PMOT_POWERS_MW, beams, table
            ),
            "current_equal_31p524mW": _stark_origin_summary(
                CURRENT_PMOT_POWERS_MW, beams, table
            ),
            "mot_strange_common_scalar_tensor_shift_mhz": 0.0,
        },
        "origin_force_gradient_attribution_1e19_n_per_m": {
            "mot_strange_24state_27mW_full_surrogate": (
                _strange_origin_force_gradient().tolist()
            ),
            "pMOT_2Level_full_stark_1p0571mW": _pmot_origin_force_gradient(
                1.0571,
                MATCHED_PMOT_POWERS_MW,
                beams,
                table,
            ).tolist(),
            "pMOT_2Level_vector_only_1p0571mW": _pmot_origin_force_gradient(
                1.0571,
                MATCHED_PMOT_POWERS_MW,
                beams,
                _vector_only_table(table),
            ).tolist(),
            "pMOT_2Level_full_stark_27mW": _pmot_origin_force_gradient(
                27.0,
                MATCHED_PMOT_POWERS_MW,
                beams,
                table,
            ).tolist(),
            "pMOT_2Level_vector_only_27mW": _pmot_origin_force_gradient(
                27.0,
                MATCHED_PMOT_POWERS_MW,
                beams,
                _vector_only_table(table),
            ).tolist(),
        },
        "figure": str(figure_path.relative_to(PROJECT_ROOT)),
        "central_figure": str(central_figure_path.relative_to(PROJECT_ROOT)),
        "conclusion": (
            "The branches do not use the same optical envelope implementation or "
            "the same focus parameters. The corrected vector conversion no longer "
            "contains the historical extra factor of two. Even after matching the origin Jacobian, "
            "their effective-field profiles differ away from the origin. Force "
            "differences additionally include unlike Stark closures, cooling powers, "
            "repumping, and rate-equation kernels."
        ),
    }
    report_path = OUTPUT_DIRECTORY / "audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
