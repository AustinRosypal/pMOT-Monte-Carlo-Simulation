"""Run a temporary seeded 100-shot effective two-level pMOT diagnostic.

This is deliberately a provisional basin probe, not a production capture or
loading campaign.  It uses the repository's full-sphere incident-disc launch
geometry, the current early capture criterion, two RK4 timesteps, and a longer
follow-up after the criterion is first met.  Any timestep disagreement or
trajectory that does not reach a terminal outcome is retained explicitly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from pmot.launch_geometry import sample_disc_points
from pmot.launch_geometry import sample_incident_disc_full_sphere
from pmot.configuration import RB87_MASS_KG
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.pMOT_2Level import rk4_step_pmot_2level
from pmot.state import AtomState


SEED = 20261015
SHOT_COUNT = 100
LAUNCH_RADIUS_M = 15.0e-3
IMPACT_DISC_RADIUS_M = 2.0e-3
MINIMUM_SPEED_M_PER_S = 0.5
MAXIMUM_SPEED_M_PER_S = 20.0
CORE_RADIUS_M = 2.0e-3
CORE_RESIDENCE_S = 5.0e-3
FOLLOW_UP_AFTER_CAPTURE_S = 25.0e-3
MAXIMUM_DURATION_S = 0.200
ESCAPE_RADIUS_M = 15.0e-3
HARD_ESCAPE_RADIUS_M = 50.0e-3
HISTORICAL_DIAGNOSTIC_POWER_W_PER_PATH = 38.294486e-3
WAIST_POINTS_M = np.asarray(
    [
        (-10.0e-3, 0.0, 0.0),
        (+10.0e-3, 0.0, 0.0),
        (0.0, -10.0e-3, 0.0),
        (0.0, +10.0e-3, 0.0),
        (0.0, 0.0, -10.0e-3),
        (0.0, 0.0, +10.0e-3),
    ],
    dtype=float,
)


_WORKER_CONTEXT = None


def _worker_context(power_w_per_path: float):
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is None:
        simple = default_simple_mot_config()
        _WORKER_CONTEXT = (
            build_pmot_2level_cooling_beams(simple_config=simple),
            build_pmot_2level_trapping_beams(),
            simple,
            load_pmot_2level_polarizability_table(),
            (power_w_per_path,) * 3,
        )
    return _WORKER_CONTEXT


def _minimum_segment_distance_to_points_m(
    start_m: np.ndarray,
    end_m: np.ndarray,
    points_m: np.ndarray,
) -> float:
    segment = end_m - start_m
    norm_squared = float(np.dot(segment, segment))
    if norm_squared == 0.0:
        return float(np.min(np.linalg.norm(points_m - start_m, axis=1)))
    fractions = ((points_m - start_m) @ segment) / norm_squared
    fractions = np.clip(fractions, 0.0, 1.0)
    closest = start_m + fractions[:, np.newaxis] * segment
    return float(np.min(np.linalg.norm(points_m - closest, axis=1)))


def _record_sample(
    storage: dict[str, list],
    time_s: float,
    state: AtomState,
    observable,
) -> None:
    storage["times_s"].append(float(time_s))
    storage["positions_m"].append(tuple(state.position_m))
    storage["velocities_m_per_s"].append(tuple(state.velocity_m_per_s))
    storage["forces_n"].append(tuple(observable.force_n))
    storage["quantization_axes"].append(tuple(observable.stark.quantization_axis))
    storage["axis_defined_by_vector_shift"].append(
        bool(observable.stark.quantization_axis_from_vector_shift)
    )
    storage["sigma_plus_ac_stark_shift_hz"].append(
        float(observable.stark.total_transition_shift_hz_for_sigma_plus)
    )
    storage["total_scattering_rate_per_s"].append(
        float(observable.total_scattering_rate_per_s)
    )


def _integrate(case: dict, time_step_s: float, power_w_per_path: float) -> dict:
    cooling, trapping, simple, table, powers = _worker_context(power_w_per_path)
    state = AtomState(
        position_m=tuple(case["initial_position_m"]),
        velocity_m_per_s=tuple(case["initial_velocity_m_per_s"]),
    )
    observable = evaluate_pmot_2level_local_observable(
        cooling,
        trapping,
        state.position_m,
        state.velocity_m_per_s,
        path_powers_w=powers,
        previous_axis=(0.0, 0.0, 1.0),
        simple_config=simple,
        polarizability_table=table,
    )
    saved = {
        "times_s": [],
        "positions_m": [],
        "velocities_m_per_s": [],
        "forces_n": [],
        "quantization_axes": [],
        "axis_defined_by_vector_shift": [],
        "sigma_plus_ac_stark_shift_hz": [],
        "total_scattering_rate_per_s": [],
    }
    _record_sample(saved, 0.0, state, observable)

    time_s = 0.0
    previous_inside_core = False
    current_core_residence_s = 0.0
    longest_core_residence_s = 0.0
    core_entry_count = 0
    capture_criterion_time_s = None
    entered_cell = False
    minimum_radius_m = float(np.linalg.norm(state.position_m))
    maximum_radius_m = minimum_radius_m
    minimum_waist_distance_m = float(
        np.min(np.linalg.norm(WAIST_POINTS_M - np.asarray(state.position_m), axis=1))
    )
    central_step_count = 0
    optical_restoring_step_count = 0
    optical_antirestoring_step_count = 0
    optical_damping_step_count = 0
    optical_heating_step_count = 0
    net_restoring_step_count = 0
    net_antirestoring_step_count = 0
    classification = "indeterminate"
    terminal_reason = "maximum duration reached without capture or escape"
    save_stride = max(1, int(round(50.0e-6 / time_step_s)))
    maximum_steps = int(round(MAXIMUM_DURATION_S / time_step_s))

    for step_index in range(maximum_steps):
        start_position = np.asarray(state.position_m, dtype=float)
        state, observable = rk4_step_pmot_2level(
            cooling,
            trapping,
            state,
            time_step_s,
            path_powers_w=powers,
            previous_axis=observable.stark.quantization_axis,
            simple_config=simple,
            polarizability_table=table,
        )
        time_s = (step_index + 1) * time_step_s
        position = np.asarray(state.position_m, dtype=float)
        velocity = np.asarray(state.velocity_m_per_s, dtype=float)
        radius_m = float(np.linalg.norm(position))
        minimum_radius_m = min(minimum_radius_m, radius_m)
        maximum_radius_m = max(maximum_radius_m, radius_m)
        minimum_waist_distance_m = min(
            minimum_waist_distance_m,
            _minimum_segment_distance_to_points_m(
                start_position,
                position,
                WAIST_POINTS_M,
            ),
        )

        inside_core = radius_m <= CORE_RADIUS_M
        if inside_core and not previous_inside_core:
            core_entry_count += 1
        if inside_core:
            current_core_residence_s += time_step_s
            longest_core_residence_s = max(
                longest_core_residence_s,
                current_core_residence_s,
            )
            if radius_m > 0.0:
                radial_direction = position / radius_m
                optical_force = np.asarray(observable.force_n, dtype=float)
                net_force = RB87_MASS_KG * np.asarray(
                    observable.acceleration_m_per_s2,
                    dtype=float,
                )
                optical_radial = float(np.dot(optical_force, radial_direction))
                net_radial = float(np.dot(net_force, radial_direction))
                optical_power = float(np.dot(optical_force, velocity))
                central_step_count += 1
                optical_restoring_step_count += int(optical_radial < 0.0)
                optical_antirestoring_step_count += int(optical_radial > 0.0)
                net_restoring_step_count += int(net_radial < 0.0)
                net_antirestoring_step_count += int(net_radial > 0.0)
                optical_damping_step_count += int(optical_power < 0.0)
                optical_heating_step_count += int(optical_power > 0.0)
        else:
            current_core_residence_s = 0.0
        previous_inside_core = inside_core

        if radius_m <= ESCAPE_RADIUS_M:
            entered_cell = True
        criterion_met = (
            longest_core_residence_s >= CORE_RESIDENCE_S
            or core_entry_count >= 2
        )
        if criterion_met and capture_criterion_time_s is None:
            capture_criterion_time_s = time_s

        outward = radius_m > 0.0 and float(np.dot(position, velocity)) > 0.0
        if entered_cell and radius_m >= ESCAPE_RADIUS_M and outward:
            classification = "escaped"
            terminal_reason = "left the 15 mm cell after entering it"
        elif radius_m >= HARD_ESCAPE_RADIUS_M and outward:
            classification = "escaped"
            terminal_reason = "crossed the 50 mm hard escape sphere"
        elif criterion_met and time_s >= FOLLOW_UP_AFTER_CAPTURE_S:
            classification = "trapped"
            terminal_reason = (
                "early capture criterion met and trajectory retained through 25 ms"
            )

        terminal = classification != "indeterminate"
        if step_index % save_stride == 0 or terminal:
            _record_sample(saved, time_s, state, observable)
        if terminal:
            break

    if saved["times_s"][-1] != time_s:
        _record_sample(saved, time_s, state, observable)

    final_position = np.asarray(state.position_m, dtype=float)
    final_velocity = np.asarray(state.velocity_m_per_s, dtype=float)
    summary = {
        "classification": classification,
        "terminal_reason": terminal_reason,
        "final_time_s": float(time_s),
        "step_count": int(round(time_s / time_step_s)),
        "final_position_m": final_position.tolist(),
        "final_velocity_m_per_s": final_velocity.tolist(),
        "final_radius_m": float(np.linalg.norm(final_position)),
        "final_speed_m_per_s": float(np.linalg.norm(final_velocity)),
        "minimum_radius_m": minimum_radius_m,
        "maximum_radius_m": maximum_radius_m,
        "minimum_waist_distance_m": minimum_waist_distance_m,
        "core_entry_count": core_entry_count,
        "longest_continuous_core_residence_s": longest_core_residence_s,
        "capture_criterion_time_s": capture_criterion_time_s,
        "central_step_count": central_step_count,
        "fraction_core_steps_optically_restoring": (
            optical_restoring_step_count / central_step_count
            if central_step_count
            else None
        ),
        "fraction_core_steps_optically_antirestoring": (
            optical_antirestoring_step_count / central_step_count
            if central_step_count
            else None
        ),
        "fraction_core_steps_net_restoring": (
            net_restoring_step_count / central_step_count
            if central_step_count
            else None
        ),
        "fraction_core_steps_net_antirestoring": (
            net_antirestoring_step_count / central_step_count
            if central_step_count
            else None
        ),
        "fraction_core_steps_optically_damping": (
            optical_damping_step_count / central_step_count
            if central_step_count
            else None
        ),
        "fraction_core_steps_optically_heating": (
            optical_heating_step_count / central_step_count
            if central_step_count
            else None
        ),
    }
    return {
        "summary": summary,
        "arrays": {key: np.asarray(value) for key, value in saved.items()},
    }


def _run_case_pair(case: dict, coarse_step_s: float, power_w_per_path: float) -> dict:
    coarse = _integrate(case, coarse_step_s, power_w_per_path)
    fine = _integrate(case, 0.5 * coarse_step_s, power_w_per_path)
    return {"case": case, "coarse": coarse, "fine": fine}


def _generate_cases(count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        disc = sample_incident_disc_full_sphere(index, LAUNCH_RADIUS_M, rng)
        point = sample_disc_points(
            disc,
            1,
            IMPACT_DISC_RADIUS_M,
            False,
            rng,
        )[0]
        speed = float(rng.uniform(MINIMUM_SPEED_M_PER_S, MAXIMUM_SPEED_M_PER_S))
        velocity = speed * np.asarray(point.incident_unit_vector, dtype=float)
        cases.append(
            {
                "shot_index": index,
                "initial_position_m": np.asarray(
                    point.initial_position_m,
                    dtype=float,
                ).tolist(),
                "initial_velocity_m_per_s": velocity.tolist(),
                "incident_unit_vector": np.asarray(
                    point.incident_unit_vector,
                    dtype=float,
                ).tolist(),
                "speed_m_per_s": speed,
                "impact_radius_m": float(point.s_m),
                "disc_theta_rad": float(disc.theta_rad),
                "disc_phi_rad": float(disc.phi_rad),
            }
        )
    return cases


def _flatten_row(result: dict) -> dict:
    case = result["case"]
    coarse = result["coarse"]["summary"]
    fine = result["fine"]["summary"]
    return {
        "shot_index": case["shot_index"],
        "speed_m_per_s": case["speed_m_per_s"],
        "impact_radius_m": case["impact_radius_m"],
        "initial_x_m": case["initial_position_m"][0],
        "initial_y_m": case["initial_position_m"][1],
        "initial_z_m": case["initial_position_m"][2],
        "initial_vx_m_per_s": case["initial_velocity_m_per_s"][0],
        "initial_vy_m_per_s": case["initial_velocity_m_per_s"][1],
        "initial_vz_m_per_s": case["initial_velocity_m_per_s"][2],
        "coarse_classification": coarse["classification"],
        "fine_classification": fine["classification"],
        "classification_agrees": coarse["classification"] == fine["classification"],
        "fine_final_time_s": fine["final_time_s"],
        "fine_minimum_radius_m": fine["minimum_radius_m"],
        "fine_final_radius_m": fine["final_radius_m"],
        "fine_core_entry_count": fine["core_entry_count"],
        "fine_longest_core_residence_s": fine[
            "longest_continuous_core_residence_s"
        ],
        "fine_minimum_waist_distance_m": fine["minimum_waist_distance_m"],
        "fine_fraction_core_steps_optically_restoring": fine[
            "fraction_core_steps_optically_restoring"
        ],
        "fine_fraction_core_steps_optically_antirestoring": fine[
            "fraction_core_steps_optically_antirestoring"
        ],
        "fine_fraction_core_steps_optically_damping": fine[
            "fraction_core_steps_optically_damping"
        ],
        "fine_fraction_core_steps_optically_heating": fine[
            "fraction_core_steps_optically_heating"
        ],
    }


def _aggregate(rows: list[dict]) -> dict:
    classifications = ("trapped", "escaped", "indeterminate")
    fine_counts = {
        label: sum(row["fine_classification"] == label for row in rows)
        for label in classifications
    }
    entered_core = [row for row in rows if row["fine_core_entry_count"] > 0]
    restoring_samples = [
        row["fine_fraction_core_steps_optically_restoring"]
        for row in entered_core
        if row["fine_fraction_core_steps_optically_restoring"] is not None
    ]
    damping_samples = [
        row["fine_fraction_core_steps_optically_damping"]
        for row in entered_core
        if row["fine_fraction_core_steps_optically_damping"] is not None
    ]
    return {
        "fine_classification_counts": fine_counts,
        "dual_timestep_agreement_count": int(
            sum(bool(row["classification_agrees"]) for row in rows)
        ),
        "dual_timestep_disagreement_shot_indices": [
            int(row["shot_index"])
            for row in rows
            if not row["classification_agrees"]
        ],
        "fine_entered_2mm_core_count": len(entered_core),
        "fine_minimum_radius_m_range": [
            float(min(row["fine_minimum_radius_m"] for row in rows)),
            float(max(row["fine_minimum_radius_m"] for row in rows)),
        ],
        "mean_fraction_core_steps_optically_restoring_among_core_crossers": (
            float(np.mean(restoring_samples)) if restoring_samples else None
        ),
        "mean_fraction_core_steps_optically_antirestoring_among_core_crossers": (
            float(1.0 - np.mean(restoring_samples)) if restoring_samples else None
        ),
        "mean_fraction_core_steps_optically_damping_among_core_crossers": (
            float(np.mean(damping_samples)) if damping_samples else None
        ),
        "mean_fraction_core_steps_optically_heating_among_core_crossers": (
            float(1.0 - np.mean(damping_samples)) if damping_samples else None
        ),
        "shots_passing_within_50um_of_any_1529nm_waist": [
            int(row["shot_index"])
            for row in rows
            if row["fine_minimum_waist_distance_m"] < 50.0e-6
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=SHOT_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--time-step-us", type=float, default=5.0)
    parser.add_argument(
        "--power-mw-per-path",
        type=float,
        default=1.0e3 * HISTORICAL_DIAGNOSTIC_POWER_W_PER_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/diagnostics/pMOT_2Level/"
            "temporary_random_100_20261015"
        ),
    )
    arguments = parser.parse_args()
    if arguments.count <= 0 or arguments.workers <= 0:
        raise ValueError("count and workers must be positive")
    coarse_step_s = arguments.time_step_us * 1.0e-6
    power_w_per_path = arguments.power_mw_per_path * 1.0e-3
    if coarse_step_s <= 0.0 or power_w_per_path < 0.0:
        raise ValueError("timestep must be positive and power non-negative")

    output_dir = arguments.output_dir.resolve()
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    cases = _generate_cases(arguments.count, arguments.seed)
    start = time.perf_counter()
    completed_results = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                _run_case_pair,
                case,
                coarse_step_s,
                power_w_per_path,
            ): case["shot_index"]
            for case in cases
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed_results.append(result)
            shot_index = result["case"]["shot_index"]
            arrays = {}
            for resolution in ("coarse", "fine"):
                for name, value in result[resolution]["arrays"].items():
                    arrays[f"{resolution}_{name}"] = value
            np.savez_compressed(
                trajectories_dir / f"shot_{shot_index:03d}.npz",
                **arrays,
            )
            print(
                f"completed {completed_count:3d}/{arguments.count}: "
                f"shot {shot_index:03d} "
                f"{result['fine']['summary']['classification']}",
                flush=True,
            )

    completed_results.sort(key=lambda result: result["case"]["shot_index"])
    rows = [_flatten_row(result) for result in completed_results]
    with (output_dir / "trajectory_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "launches.json").open("w", encoding="utf-8") as stream:
        json.dump(cases, stream, indent=2)

    cooling, trapping, simple, table, _ = _worker_context(power_w_per_path)
    summary = {
        "status": "temporary provisional effective-two-level pMOT diagnostic",
        "scientific_limitations": [
            "effective F=2 to F'=3 differential Stark closure, not an m_F-resolved Hamiltonian",
            "no conservative 1529-nm Stark-gradient force",
            "no 1529-nm scattering or heating",
            "no recoil diffusion",
            "historical vector-gradient proxy trapping power, not an apparatus default",
            "not a capture-cross-section or loading-rate campaign",
        ],
        "seed": arguments.seed,
        "shot_count": arguments.count,
        "elapsed_wall_time_s": time.perf_counter() - start,
        "workers": arguments.workers,
        "launch_sampling": {
            "direction_distribution": "uniform full sphere",
            "launch_sphere_radius_m": LAUNCH_RADIUS_M,
            "impact_disc_radius_m": IMPACT_DISC_RADIUS_M,
            "impact_distribution": "uniform area",
            "velocity_direction": "parallel to inward disc normal",
            "speed_distribution": "uniform",
            "minimum_speed_m_per_s": MINIMUM_SPEED_M_PER_S,
            "maximum_speed_m_per_s": MAXIMUM_SPEED_M_PER_S,
        },
        "configuration": {
            "cooling_power_w_per_beam": cooling[0].intensity_beam.power_w,
            "cooling_detuning_hz": simple.cooling_detuning_hz,
            "cooling_helicities_by_component": [
                beam.circular_polarization for beam in cooling
            ],
            "trapping_power_w_per_path": power_w_per_path,
            "trapping_helicities_by_component": [beam.helicity for beam in trapping],
            "external_magnetic_field_t": [0.0, 0.0, 0.0],
            "gravity_enabled": simple.include_gravity,
            "polarizability_csv": str(table.source_path),
        },
        "classification": {
            "core_radius_m": CORE_RADIUS_M,
            "continuous_core_residence_s": CORE_RESIDENCE_S,
            "alternative_core_entry_count": 2,
            "minimum_capture_follow_up_s": FOLLOW_UP_AFTER_CAPTURE_S,
            "maximum_duration_s": MAXIMUM_DURATION_S,
            "coarse_time_step_s": coarse_step_s,
            "fine_time_step_s": 0.5 * coarse_step_s,
        },
        "aggregate": _aggregate(rows),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
