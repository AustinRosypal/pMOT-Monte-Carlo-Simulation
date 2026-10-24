"""Run two convergence-audited effective two-level pMOT trajectories."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.pMOT_2Level import simulate_pmot_2level_trajectory
from pmot.state import AtomState


HISTORICAL_DIAGNOSTIC_POWER_W_PER_PATH = 38.294486e-3
CORE_RADIUS_M = 2.0e-3
CORE_RESIDENCE_S = 5.0e-3


def _core_summary(times_s: np.ndarray, positions_m: np.ndarray) -> dict:
    inside = np.linalg.norm(positions_m, axis=1) <= CORE_RADIUS_M
    entries = int(np.sum((~inside[:-1]) & inside[1:]))
    if inside[0]:
        entries += 1
    longest = 0.0
    start_index = None
    for index, value in enumerate(inside):
        if value and start_index is None:
            start_index = index
        if start_index is not None and (not value or index == len(inside) - 1):
            end_index = index if value else index - 1
            longest = max(longest, float(times_s[end_index] - times_s[start_index]))
            start_index = None
    return {
        "core_entry_count": entries,
        "longest_continuous_core_residence_s": longest,
        "meets_early_trapped_criterion": bool(
            longest >= CORE_RESIDENCE_S or entries >= 2
        ),
    }


def _record_arrays(record) -> dict[str, np.ndarray]:
    return {
        "times_s": np.asarray(record.times_s),
        "positions_m": np.asarray(record.positions_m),
        "velocities_m_per_s": np.asarray(record.velocities_m_per_s),
        "forces_n": np.asarray(record.forces_n),
        "quantization_axes": np.asarray(record.quantization_axes),
        "axis_defined_by_vector_shift": np.asarray(
            record.axis_defined_by_vector_shift,
            dtype=bool,
        ),
        "sigma_plus_ac_stark_shift_hz": np.asarray(
            record.total_ac_stark_shifts_hz_for_sigma_plus
        ),
        "total_scattering_rate_per_s": np.asarray(
            record.total_scattering_rates_per_s
        ),
    }


def _trajectory_summary(arrays: dict[str, np.ndarray]) -> dict:
    position = arrays["positions_m"]
    velocity = arrays["velocities_m_per_s"]
    radius = np.linalg.norm(position, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    peak_shift_index = int(
        np.argmax(np.abs(arrays["sigma_plus_ac_stark_shift_hz"]))
    )
    peak_force_index = int(np.argmax(np.linalg.norm(arrays["forces_n"], axis=1)))
    summary = {
        "sample_count": int(len(arrays["times_s"])),
        "final_time_s": float(arrays["times_s"][-1]),
        "final_position_m": position[-1].tolist(),
        "final_velocity_m_per_s": velocity[-1].tolist(),
        "final_radius_m": float(radius[-1]),
        "final_speed_m_per_s": float(speed[-1]),
        "minimum_radius_m": float(np.min(radius)),
        "maximum_radius_m": float(np.max(radius)),
        "maximum_speed_m_per_s": float(np.max(speed)),
        "maximum_force_n": float(np.max(np.linalg.norm(arrays["forces_n"], axis=1))),
        "ac_stark_shift_hz_sigma_plus_range": [
            float(np.min(arrays["sigma_plus_ac_stark_shift_hz"])),
            float(np.max(arrays["sigma_plus_ac_stark_shift_hz"])),
        ],
        "peak_absolute_ac_stark_shift": {
            "time_s": float(arrays["times_s"][peak_shift_index]),
            "position_m": position[peak_shift_index].tolist(),
            "shift_hz": float(
                arrays["sigma_plus_ac_stark_shift_hz"][peak_shift_index]
            ),
        },
        "scattering_rate_per_s_range": [
            float(np.min(arrays["total_scattering_rate_per_s"])),
            float(np.max(arrays["total_scattering_rate_per_s"])),
        ],
        "axis_undefined_sample_count": int(
            np.sum(~arrays["axis_defined_by_vector_shift"])
        ),
        "peak_force_time_s": float(arrays["times_s"][peak_force_index]),
    }
    summary.update(_core_summary(arrays["times_s"], position))
    return summary


def _convergence_summary(
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
) -> dict:
    fine_indices = np.rint(
        coarse["times_s"] / (fine["times_s"][1] - fine["times_s"][0])
    ).astype(int)
    position_difference = coarse["positions_m"] - fine["positions_m"][fine_indices]
    velocity_difference = (
        coarse["velocities_m_per_s"] - fine["velocities_m_per_s"][fine_indices]
    )
    return {
        "final_position_difference_m": float(
            np.linalg.norm(position_difference[-1])
        ),
        "final_velocity_difference_m_per_s": float(
            np.linalg.norm(velocity_difference[-1])
        ),
        "maximum_sampled_position_difference_m": float(
            np.max(np.linalg.norm(position_difference, axis=1))
        ),
        "maximum_sampled_velocity_difference_m_per_s": float(
            np.max(np.linalg.norm(velocity_difference, axis=1))
        ),
        "classification_agrees": bool(
            _core_summary(coarse["times_s"], coarse["positions_m"])[
                "meets_early_trapped_criterion"
            ]
            == _core_summary(fine["times_s"], fine["positions_m"])[
                "meets_early_trapped_criterion"
            ]
        ),
    }


def _local_jacobians(
    cooling_beams,
    trapping_beams,
    power_w_per_path: float,
    simple_config,
    table,
) -> dict:
    def force(position, velocity):
        sample = evaluate_pmot_2level_local_observable(
            cooling_beams,
            trapping_beams,
            position,
            velocity,
            path_powers_w=(power_w_per_path,) * 3,
            previous_axis=(0.0, 0.0, 1.0),
            simple_config=simple_config,
            polarizability_table=table,
        )
        return np.asarray(sample.force_n), sample

    position_step_m = 10.0e-6
    velocity_step_m_per_s = 0.01
    position_jacobian = np.empty((3, 3))
    velocity_jacobian = np.empty((3, 3))
    for axis in range(3):
        displacement = np.zeros(3)
        displacement[axis] = position_step_m
        plus, _ = force(displacement, np.zeros(3))
        minus, _ = force(-displacement, np.zeros(3))
        position_jacobian[:, axis] = (plus - minus) / (2.0 * position_step_m)

        velocity = np.zeros(3)
        velocity[axis] = velocity_step_m_per_s
        plus, _ = force(np.zeros(3), velocity)
        minus, _ = force(np.zeros(3), -velocity)
        velocity_jacobian[:, axis] = (
            plus - minus
        ) / (2.0 * velocity_step_m_per_s)

    origin_force, origin = force(np.zeros(3), np.zeros(3))
    return {
        "position_step_m": position_step_m,
        "velocity_step_m_per_s": velocity_step_m_per_s,
        "origin_force_n": origin_force.tolist(),
        "origin_ac_stark_shifts_hz_q_minus_pi_q_plus": list(
            origin.stark.transition_shifts_hz_by_q
        ),
        "origin_effective_detunings_hz_first_beam_q_minus_pi_q_plus": list(
            origin.rate_samples[0].effective_detunings_hz_q_minus_pi_q_plus
        ),
        "position_force_jacobian_n_per_m": position_jacobian.tolist(),
        "position_force_jacobian_eigenvalues_n_per_m": np.linalg.eigvals(
            position_jacobian
        ).real.tolist(),
        "velocity_force_jacobian_n_s_per_m": velocity_jacobian.tolist(),
        "velocity_force_jacobian_eigenvalues_n_s_per_m": np.linalg.eigvals(
            velocity_jacobian
        ).real.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--power-mw-per-path",
        type=float,
        default=1.0e3 * HISTORICAL_DIAGNOSTIC_POWER_W_PER_PATH,
    )
    parser.add_argument("--time-step-us", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/pMOT_2Level/two_trajectory_probe_20261015"),
    )
    arguments = parser.parse_args()
    power_w_per_path = 1.0e-3 * arguments.power_mw_per_path
    time_step_s = 1.0e-6 * arguments.time_step_us
    if power_w_per_path < 0.0 or time_step_s <= 0.0:
        raise ValueError("power must be non-negative and timestep must be positive")

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    simple = default_simple_mot_config()
    cooling = build_pmot_2level_cooling_beams(simple_config=simple)
    trapping = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    powers = (power_w_per_path,) * 3
    cases = (
        {
            "name": "displaced_rest",
            "initial_state": AtomState(
                position_m=(0.5e-3, 0.0, 0.0),
                velocity_m_per_s=(0.0, 0.0, 0.0),
            ),
            "duration_s": 10.0e-3,
        },
        {
            "name": "axial_incoming",
            "initial_state": AtomState(
                position_m=(15.0e-3, 0.0, 0.0),
                velocity_m_per_s=(-17.0, 0.0, 0.0),
            ),
            "duration_s": 25.0e-3,
        },
    )

    manifest = {
        "model": "effective 5S1/2 F=2 to 5P3/2 F'=3 pMOT",
        "external_magnetic_field_t": [0.0, 0.0, 0.0],
        "cooling_power_w_per_beam": cooling[0].intensity_beam.power_w,
        "cooling_detuning_hz": simple.cooling_detuning_hz,
        "cooling_helicities_by_component": [
            beam.circular_polarization for beam in cooling
        ],
        "trapping_power_w_per_path": power_w_per_path,
        "trapping_power_status": (
            "historical 20 G/cm proxy diagnostic scale; not an apparatus default "
            "or power recommendation"
        ),
        "gravity_enabled": simple.include_gravity,
        "coarse_time_step_s": time_step_s,
        "fine_time_step_s": 0.5 * time_step_s,
        "polarizability_csv": str(table.source_path),
        "local_linear_diagnostics": _local_jacobians(
            cooling,
            trapping,
            power_w_per_path,
            replace(simple, include_gravity=False),
            table,
        ),
        "trajectories": {},
    }

    for case in cases:
        records = {}
        arrays_by_resolution = {}
        for label, step in (("coarse", time_step_s), ("fine", 0.5 * time_step_s)):
            record = simulate_pmot_2level_trajectory(
                cooling,
                trapping,
                case["initial_state"],
                case["duration_s"],
                step,
                path_powers_w=powers,
                simple_config=simple,
                polarizability_table=table,
            )
            arrays = _record_arrays(record)
            arrays_by_resolution[label] = arrays
            records[label] = _trajectory_summary(arrays)
            np.savez_compressed(
                output_dir / f"{case['name']}_{label}.npz",
                **arrays,
            )
        manifest["trajectories"][case["name"]] = {
            "initial_position_m": list(case["initial_state"].position_m),
            "initial_velocity_m_per_s": list(case["initial_state"].velocity_m_per_s),
            "duration_s": case["duration_s"],
            **records,
            "coarse_fine_convergence": _convergence_summary(
                arrays_by_resolution["coarse"],
                arrays_by_resolution["fine"],
            ),
        }

    manifest_path = output_dir / "summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
