"""Quantify the provisional two-level pMOT force law near the origin."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pmot.configuration import PLANCK_CONSTANT_J_S
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "pMOT_2Level"
    / "origin_gradient_diagnostic_20261020.json"
)
PATH_POWERS_W = (19.147243e-3, 19.147243e-3, 38.294486e-3)
AXES = "xyz"


def _context(*, cooling_power_mw: float, detuning_mhz: float):
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=detuning_mhz * 1.0e6,
        include_gravity=False,
    )
    baseline = default_simple_mot_apparatus()
    apparatus = replace(
        baseline,
        cooling=replace(
            baseline.cooling,
            power_w_per_beam=cooling_power_mw * 1.0e-3,
            detuning_hz=detuning_mhz * 1.0e6,
        ),
    )
    return simple, build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )


def _observable(simple, cooling, trapping, table, position):
    return evaluate_pmot_2level_local_observable(
        cooling,
        trapping,
        position,
        np.zeros(3),
        path_powers_w=PATH_POWERS_W,
        previous_axis=(0.0, 0.0, 1.0),
        simple_config=simple,
        polarizability_table=table,
    )


def _slopes(simple, cooling, trapping, table, step_um: float = 25.0):
    step_m = step_um * 1.0e-6
    slopes = {}
    for component, axis in enumerate(AXES):
        plus = np.zeros(3)
        minus = np.zeros(3)
        plus[component] = step_m
        minus[component] = -step_m
        plus_force = _observable(simple, cooling, trapping, table, plus).force_n[
            component
        ]
        minus_force = _observable(simple, cooling, trapping, table, minus).force_n[
            component
        ]
        slopes[axis] = (plus_force - minus_force) / (2.0 * step_m)
    return slopes


def main() -> None:
    trapping = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    simple, cooling = _context(cooling_power_mw=27.0, detuning_mhz=-15.0)
    origin = _observable(simple, cooling, trapping, table, np.zeros(3))

    convergence = {}
    for step_um in (1.0, 5.0, 10.0, 25.0, 50.0, 100.0):
        convergence[str(step_um)] = {
            axis: 1.0e19 * value
            for axis, value in _slopes(
                simple,
                cooling,
                trapping,
                table,
                step_um=step_um,
            ).items()
        }

    power_sweep = {}
    for power_mw in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 27.0):
        local_simple, local_cooling = _context(
            cooling_power_mw=power_mw,
            detuning_mhz=-15.0,
        )
        local_origin = _observable(
            local_simple,
            local_cooling,
            trapping,
            table,
            np.zeros(3),
        )
        power_sweep[str(power_mw)] = {
            "total_saturation_at_origin": (
                local_origin.rate_samples[0].total_saturation_parameter
            ),
            "gradient_1e19_n_per_m": {
                axis: 1.0e19 * value
                for axis, value in _slopes(
                    local_simple,
                    local_cooling,
                    trapping,
                    table,
                ).items()
            },
        }

    common_shift_mhz = 1.0e-6 * origin.stark.transition_shifts_hz_by_q[1]
    compensated_detuning_mhz = -15.0 + common_shift_mhz
    detuning_sweep = {}
    for detuning_mhz in (
        -15.0,
        -20.0,
        compensated_detuning_mhz,
        -25.0,
        -30.0,
    ):
        local_simple, local_cooling = _context(
            cooling_power_mw=27.0,
            detuning_mhz=detuning_mhz,
        )
        detuning_sweep[f"{detuning_mhz:.9g}"] = {
            "effective_detuning_at_origin_mhz": (
                detuning_mhz - common_shift_mhz
            ),
            "gradient_1e19_n_per_m": {
                axis: 1.0e19 * value
                for axis, value in _slopes(
                    local_simple,
                    local_cooling,
                    trapping,
                    table,
                ).items()
            },
        }

    combined_power_detuning_sweep = {}
    for power_mw in (0.25, 0.5, 1.0, 2.0, 5.0):
        local_simple, local_cooling = _context(
            cooling_power_mw=power_mw,
            detuning_mhz=compensated_detuning_mhz,
        )
        local_origin = _observable(
            local_simple,
            local_cooling,
            trapping,
            table,
            np.zeros(3),
        )
        combined_power_detuning_sweep[str(power_mw)] = {
            "nominal_detuning_mhz": compensated_detuning_mhz,
            "effective_detuning_at_origin_mhz": -15.0,
            "total_saturation_at_origin": (
                local_origin.rate_samples[0].total_saturation_parameter
            ),
            "gradient_1e19_n_per_m": {
                axis: 1.0e19 * value
                for axis, value in _slopes(
                    local_simple,
                    local_cooling,
                    trapping,
                    table,
                ).items()
            },
        }

    vector_frequency_gradient_mhz_per_mm = {}
    step_m = 1.0e-6
    for component, axis in enumerate(AXES):
        plus = np.zeros(3)
        minus = np.zeros(3)
        plus[component] = step_m
        minus[component] = -step_m
        plus_stark = _observable(
            simple,
            cooling,
            trapping,
            table,
            plus,
        ).stark
        minus_stark = _observable(
            simple,
            cooling,
            trapping,
            table,
            minus,
        ).stark
        plus_vector = np.asarray(plus_stark.net_vector_energy_vector_j)
        minus_vector = np.asarray(minus_stark.net_vector_energy_vector_j)
        gradient_hz_per_m = (
            plus_vector[component] - minus_vector[component]
        ) / (2.0 * step_m * PLANCK_CONSTANT_J_S)
        vector_frequency_gradient_mhz_per_mm[axis] = 1.0e-9 * gradient_hz_per_m

    result = {
        "configuration": {
            "cooling_power_mw_per_beam": 27.0,
            "nominal_detuning_mhz": -15.0,
            "trapping_path_powers_mw": [1.0e3 * value for value in PATH_POWERS_W],
        },
        "origin": {
            "total_saturation": origin.rate_samples[0].total_saturation_parameter,
            "transition_shifts_mhz_q_minus_pi_q_plus": [
                1.0e-6 * value for value in origin.stark.transition_shifts_hz_by_q
            ],
            "effective_detunings_mhz_q_minus_pi_q_plus": [
                1.0e-6 * value
                for value in origin.rate_samples[0].effective_detunings_hz_q_minus_pi_q_plus
            ],
            "axis": origin.stark.quantization_axis,
            "axis_from_vector_shift": origin.stark.quantization_axis_from_vector_shift,
            "force_n": origin.force_n,
        },
        "vector_frequency_gradient_mhz_per_mm": vector_frequency_gradient_mhz_per_mm,
        "central_difference_convergence_1e19_n_per_m": convergence,
        "power_sweep": power_sweep,
        "common_shift_compensated_nominal_detuning_mhz": compensated_detuning_mhz,
        "detuning_sweep": detuning_sweep,
        "combined_power_detuning_sweep": combined_power_detuning_sweep,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
