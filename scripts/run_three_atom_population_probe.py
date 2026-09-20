"""Generate a reproducible three-atom Section-12 diagnostic dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pmot.configuration import GRAVITY_ACCELERATION_M_PER_S2, HBAR_J_S, RB87_MASS_KG
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    RateEquationAtomState,
    build_multilevel_mot_beams,
    build_rate_equation_model,
    default_multilevel_mot_config,
    rate_equation_observable,
)
from pmot.mot_multilevel.coupling import wavevector_rad_per_m


SEED = 20260918
TIME_STEP_S = 5.0e-6
STEP_COUNT = 100


def _tolist(value):
    return np.asarray(value).tolist()


def _observable(model, beams, coil, config, position, velocity, previous_axis):
    return rate_equation_observable(
        model,
        beams,
        tuple(position),
        tuple(velocity),
        coil,
        config,
        tuple(previous_axis),
        store_rate_matrix=True,
        store_beam_transition_quantities=True,
    )


def _stage(model, beams, coil, config, position, velocity, previous_axis):
    observable = rate_equation_observable(
        model,
        beams,
        tuple(position),
        tuple(velocity),
        coil,
        config,
        tuple(previous_axis),
    )
    gravity = np.asarray(
        GRAVITY_ACCELERATION_M_PER_S2
        if config.include_gravity
        else (0.0, 0.0, 0.0),
        dtype=float,
    )
    acceleration = np.asarray(observable.force_n) / RB87_MASS_KG + gravity
    return {
        "position_m": position.copy(),
        "velocity_m_per_s": velocity.copy(),
        "force_n": np.asarray(observable.force_n),
        "acceleration_m_per_s2": acceleration,
        "k_position_m_per_s": velocity.copy(),
        "k_velocity_m_per_s2": acceleration.copy(),
    }


def _rk4_step(model, beams, coil, config, position, velocity, previous_axis):
    s1 = _stage(model, beams, coil, config, position, velocity, previous_axis)
    s2 = _stage(
        model,
        beams,
        coil,
        config,
        position + 0.5 * TIME_STEP_S * s1["k_position_m_per_s"],
        velocity + 0.5 * TIME_STEP_S * s1["k_velocity_m_per_s2"],
        previous_axis,
    )
    s3 = _stage(
        model,
        beams,
        coil,
        config,
        position + 0.5 * TIME_STEP_S * s2["k_position_m_per_s"],
        velocity + 0.5 * TIME_STEP_S * s2["k_velocity_m_per_s2"],
        previous_axis,
    )
    s4 = _stage(
        model,
        beams,
        coil,
        config,
        position + TIME_STEP_S * s3["k_position_m_per_s"],
        velocity + TIME_STEP_S * s3["k_velocity_m_per_s2"],
        previous_axis,
    )
    stages = (s1, s2, s3, s4)
    position_next = position + TIME_STEP_S * (
        s1["k_position_m_per_s"]
        + 2.0 * s2["k_position_m_per_s"]
        + 2.0 * s3["k_position_m_per_s"]
        + s4["k_position_m_per_s"]
    ) / 6.0
    velocity_next = velocity + TIME_STEP_S * (
        s1["k_velocity_m_per_s2"]
        + 2.0 * s2["k_velocity_m_per_s2"]
        + 2.0 * s3["k_velocity_m_per_s2"]
        + s4["k_velocity_m_per_s2"]
    ) / 6.0
    weighted_force = (
        s1["force_n"] + 2.0 * s2["force_n"] + 2.0 * s3["force_n"] + s4["force_n"]
    ) / 6.0
    return position_next, velocity_next, stages, weighted_force


def _serialize_stage(stage):
    return {name: _tolist(value) for name, value in stage.items()}


def _serialize_point(
    atom_index,
    step,
    position,
    velocity,
    observable,
    beams,
    stages,
    weighted_force,
    position_next,
    velocity_next,
    model,
):
    quantities = observable.beam_transition_quantities
    assert quantities is not None
    assert observable.rate_matrix_per_s is not None
    beam_rates = np.asarray(observable.beam_effective_scattering_rates_per_s)
    beam_forces = np.asarray(
        [
            HBAR_J_S * np.asarray(wavevector_rad_per_m(beam)) * rate
            for beam, rate in zip(beams, beam_rates)
        ]
    )
    beam_w = quantities.stimulated_coefficients_per_s
    return {
        "atom": atom_index,
        "step": step,
        "time_s": step * TIME_STEP_S,
        "position_m": _tolist(position),
        "velocity_m_per_s": _tolist(velocity),
        "magnetic_field_t": _tolist(observable.magnetic_field_t),
        "magnetic_field_magnitude_t": float(np.linalg.norm(observable.magnetic_field_t)),
        "quantization_axis": _tolist(observable.quantization_axis),
        "stimulated_coefficients_per_beam_s-1": _tolist(beam_w),
        "total_stimulated_coefficients_s-1": _tolist(np.sum(beam_w, axis=0)),
        "rate_matrix_s-1": _tolist(observable.rate_matrix_per_s),
        "populations": _tolist(observable.populations),
        "beam_effective_scattering_rates_s-1": _tolist(beam_rates),
        "beam_forces_n": _tolist(beam_forces),
        "total_force_n": _tolist(observable.force_n),
        "total_spontaneous_scattering_rate_s-1": (
            observable.total_spontaneous_scattering_rate_per_s
        ),
        "rk4": {
            "dt_s": TIME_STEP_S,
            "stages": [_serialize_stage(stage) for stage in stages],
            "weighted_optical_force_n": _tolist(weighted_force),
            "position_next_m": _tolist(position_next),
            "velocity_next_m_per_s": _tolist(velocity_next),
            "position_formula": "r[n+1] = r[n] + dt*(k1_r + 2*k2_r + 2*k3_r + k4_r)/6",
            "velocity_formula": "v[n+1] = v[n] + dt*(k1_v + 2*k2_v + 2*k3_v + k4_v)/6",
        },
        "matrix_index_order": [
            (
                f"g:F={model.structure.states[index].f},mF={model.structure.states[index].m_f}"
                if model.structure.states[index].is_ground
                else f"e:F'={model.structure.states[index].f},mF'={model.structure.states[index].m_f}"
            )
            for index in range(model.state_count)
        ],
    }


def generate_dataset():
    rng = np.random.default_rng(SEED)
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=config)
    coil = default_anti_helmholtz_config()
    ground_labels = [
        f"F={model.structure.states[index].f},m={model.structure.states[index].m_f}"
        for index in model.ground_indices
    ]
    excited_labels = [
        f"F'={model.structure.states[index].f},m'={model.structure.states[index].m_f}"
        for index in model.excited_indices
    ]
    atoms = []
    selected_steps = []
    for atom_index in range(1, 4):
        position = rng.uniform(-3.0e-3, 3.0e-3, size=3)
        velocity = rng.uniform(-2.0, 2.0, size=3)
        selected = sorted(int(value) for value in rng.choice(STEP_COUNT, size=3, replace=False))
        selected_steps.append(selected)
        initial_position = position.copy()
        initial_velocity = velocity.copy()
        previous_axis = np.asarray((0.0, 0.0, 1.0))
        observable = _observable(
            model, beams, coil, config, position, velocity, previous_axis
        )
        previous_axis = np.asarray(observable.quantization_axis)
        path = [{"step": 0, "position_m": _tolist(position)}]
        points = []
        for step in range(STEP_COUNT):
            position_next, velocity_next, stages, weighted_force = _rk4_step(
                model, beams, coil, config, position, velocity, previous_axis
            )
            if step in selected:
                points.append(
                    _serialize_point(
                        atom_index,
                        step,
                        position,
                        velocity,
                        observable,
                        beams,
                        stages,
                        weighted_force,
                        position_next,
                        velocity_next,
                        model,
                    )
                )
            position = position_next
            velocity = velocity_next
            observable = _observable(
                model, beams, coil, config, position, velocity, previous_axis
            )
            previous_axis = np.asarray(observable.quantization_axis)
            path.append({"step": step + 1, "position_m": _tolist(position)})
        atoms.append(
            {
                "atom": atom_index,
                "initial_position_m": _tolist(initial_position),
                "initial_velocity_m_per_s": _tolist(initial_velocity),
                "selected_steps": selected,
                "path": path,
                "points": points,
            }
        )

    return {
        "metadata": {
            "seed": SEED,
            "time_step_s": TIME_STEP_S,
            "step_count": STEP_COUNT,
            "duration_s": STEP_COUNT * TIME_STEP_S,
            "initial_position_distribution": "independent Cartesian uniform[-3,3] mm",
            "initial_velocity_distribution": "independent Cartesian uniform[-2,2] m/s",
            "arc_version": model.structure.arc_version,
            "state_count": model.state_count,
            "transition_count": model.transition_count,
            "beam_count": len(beams),
            "selected_steps": selected_steps,
        },
        "ground_labels": ground_labels,
        "excited_labels": excited_labels,
        "state_labels": atoms[0]["points"][0]["matrix_index_order"],
        "beam_labels": [beam.label for beam in beams],
        "atoms": atoms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generate_dataset(), separators=(",", ":")), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
