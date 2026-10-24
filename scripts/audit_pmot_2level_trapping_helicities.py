"""Audit two-level pMOT force Jacobians over trapping-beam helicities."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np

from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


POWER_W_PER_PATH = 38.294486e-3
POSITION_STEP_M = 10.0e-6
VELOCITY_STEP_M_PER_S = 0.01
OUTPUT_ROOT = Path("outputs/diagnostics/pMOT_2Level")


def _code(helicities: tuple[str, ...]) -> str:
    return "".join("+" if value == "sigma+" else "-" for value in helicities)


def _jacobians(cooling, trapping, simple, table) -> dict:
    def sample(position, velocity):
        return evaluate_pmot_2level_local_observable(
            cooling,
            trapping,
            position,
            velocity,
            path_powers_w=(POWER_W_PER_PATH,) * 3,
            previous_axis=(0.0, 0.0, 1.0),
            simple_config=simple,
            polarizability_table=table,
        )

    position_jacobian = np.empty((3, 3))
    velocity_jacobian = np.empty((3, 3))
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = POSITION_STEP_M
        plus = np.asarray(sample(offset, np.zeros(3)).force_n)
        minus = np.asarray(sample(-offset, np.zeros(3)).force_n)
        position_jacobian[:, axis] = (plus - minus) / (2.0 * POSITION_STEP_M)

        offset = np.zeros(3)
        offset[axis] = VELOCITY_STEP_M_PER_S
        plus = np.asarray(sample(np.zeros(3), offset).force_n)
        minus = np.asarray(sample(np.zeros(3), -offset).force_n)
        velocity_jacobian[:, axis] = (
            plus - minus
        ) / (2.0 * VELOCITY_STEP_M_PER_S)

    origin = sample(np.zeros(3), np.zeros(3))
    position_eigenvalues = np.linalg.eigvals(position_jacobian)
    velocity_eigenvalues = np.linalg.eigvals(velocity_jacobian)
    return {
        "origin_force_n": list(origin.force_n),
        "origin_force_magnitude_n": float(np.linalg.norm(origin.force_n)),
        "origin_axis_defined_by_vector_shift": (
            origin.stark.quantization_axis_from_vector_shift
        ),
        "origin_quantization_axis": list(origin.stark.quantization_axis),
        "origin_net_vector_energy_vector_j": list(
            origin.stark.net_vector_energy_vector_j
        ),
        "origin_ac_stark_shifts_hz_q_minus_pi_q_plus": list(
            origin.stark.transition_shifts_hz_by_q
        ),
        "position_jacobian_n_per_m": position_jacobian.tolist(),
        "position_eigenvalues_n_per_m": [
            [float(value.real), float(value.imag)] for value in position_eigenvalues
        ],
        "velocity_jacobian_n_s_per_m": velocity_jacobian.tolist(),
        "velocity_eigenvalues_n_s_per_m": [
            [float(value.real), float(value.imag)] for value in velocity_eigenvalues
        ],
        "all_position_restoring": bool(np.max(position_eigenvalues.real) < 0.0),
        "all_velocity_damping": bool(np.max(velocity_eigenvalues.real) < 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ac-stark-detuning-sign",
        choices=("minus", "plus"),
        default="minus",
        help=(
            "Sign before the current DeltaE_AC/h value. 'plus' is a diagnostic "
            "counterfactual, not the repository's physical U_e-U_g convention."
        ),
    )
    arguments = parser.parse_args()
    simple = default_simple_mot_config()
    cooling = build_pmot_2level_cooling_beams(simple_config=simple)
    table = load_pmot_2level_polarizability_table()
    if arguments.ac_stark_detuning_sign == "plus":
        # The production evaluator always applies ``-DeltaE_AC/h``. Negating
        # the differential table produces the exact summed-rate counterfactual
        # for ``+DeltaE_AC/h`` while leaving production sign semantics intact.
        table = replace(
            table,
            scalar_si=-table.scalar_si,
            vector_si=-table.vector_si,
            tensor_si=-table.tensor_si,
        )
    choices = tuple(product(("sigma+", "sigma-"), repeat=3))
    configurations = []
    for incident in choices:
        for retro in choices:
            trapping = build_pmot_2level_trapping_beams(
                incident_helicities_by_axis=incident,
                retro_helicities_by_axis=retro,
            )
            result = _jacobians(cooling, trapping, simple, table)
            configurations.append(
                {
                    "incident_code": _code(incident),
                    "retro_code": _code(retro),
                    "matched_paths": incident == retro,
                    **result,
                }
            )

    stable = [
        item
        for item in configurations
        if item["all_position_restoring"] and item["all_velocity_damping"]
    ]
    centered_stable = [
        item
        for item in stable
        if not item["origin_axis_defined_by_vector_shift"]
    ]
    manifest = {
        "ac_stark_detuning_sign": arguments.ac_stark_detuning_sign,
        "sign_convention_note": (
            "minus uses DeltaE_AC=U_e-U_g and is physical; plus is a "
            "counterfactual using the same tabulated numeric shift"
        ),
        "cooling_power_w_per_beam": cooling[0].intensity_beam.power_w,
        "cooling_detuning_hz": simple.cooling_detuning_hz,
        "cooling_component_helicities": [
            beam.circular_polarization for beam in cooling
        ],
        "trapping_power_w_per_path": POWER_W_PER_PATH,
        "trapping_power_status": (
            "historical 20 G/cm proxy scale; not a physical default"
        ),
        "position_step_m": POSITION_STEP_M,
        "velocity_step_m_per_s": VELOCITY_STEP_M_PER_S,
        "configuration_count": len(configurations),
        "stable_configuration_count": len(stable),
        "centered_stable_configuration_count": len(centered_stable),
        "stable_codes": [
            [item["incident_code"], item["retro_code"]] for item in stable
        ],
        "centered_stable_codes": [
            [item["incident_code"], item["retro_code"]]
            for item in centered_stable
        ],
        "configurations": configurations,
    }
    output_path = (
        OUTPUT_ROOT
        / f"trapping_helicity_audit_27mW_{arguments.ac_stark_detuning_sign}_20261015"
        / "summary.json"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "cooling_power_w_per_beam",
                    "cooling_detuning_hz",
                    "ac_stark_detuning_sign",
                    "trapping_power_w_per_path",
                    "configuration_count",
                    "stable_configuration_count",
                    "centered_stable_configuration_count",
                    "stable_codes",
                    "centered_stable_codes",
                )
            },
            indent=2,
        )
    )
    print("Matched configurations:")
    for item in configurations:
        if item["matched_paths"]:
            position = [value[0] for value in item["position_eigenvalues_n_per_m"]]
            velocity = [value[0] for value in item["velocity_eigenvalues_n_s_per_m"]]
            print(
                item["incident_code"],
                "position", position,
                "velocity", velocity,
                "restoring", item["all_position_restoring"],
                "damping", item["all_velocity_damping"],
            )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
