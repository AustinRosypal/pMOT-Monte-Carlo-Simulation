from __future__ import annotations

import numpy as np

from pmot.fields import build_mot_beams
from pmot.mot import GroundState
from pmot.mot import MOTAtomState
from pmot.mot import addressed_excited_states_for_beam_family
from pmot.mot import default_anti_helmholtz_config
from pmot.mot import polarization_weights_for_quantization_axis
from pmot.mot import simulate_mot_trajectory
from pmot.mot import transition_rate_samples


def test_polarization_projection_normalizes() -> None:
    beams = build_mot_beams()
    for beam in beams:
        weights = polarization_weights_for_quantization_axis(beam, beam.direction)
        assert np.isclose(sum(weights.values()), 1.0)


def test_repump_is_far_off_resonance_for_f2_cooling_manifold() -> None:
    beams = build_mot_beams()
    coil = default_anti_helmholtz_config()
    atom = MOTAtomState(
        position_m=(0.5e-3, 0.0, 1.0e-3),
        velocity_m_per_s=(0.0, 0.0, 0.0),
        ground_state=GroundState(f=2, m_f=2),
    )
    samples, _ = transition_rate_samples(beams, atom, coil)
    cooling_rate = sum(sample.scattering_rate_per_s for sample in samples if sample.beam_family == "cooling")
    repump_rate = sum(sample.scattering_rate_per_s for sample in samples if sample.beam_family == "repump")
    assert cooling_rate > 1.0e4
    assert repump_rate < 10.0


def test_repump_only_addresses_f1_to_fprime_2() -> None:
    atom_f1 = GroundState(f=1, m_f=0)
    cooling_states = addressed_excited_states_for_beam_family(atom_f1, "cooling")
    repump_states = addressed_excited_states_for_beam_family(atom_f1, "repump")
    assert cooling_states == []
    assert repump_states
    assert {state.f_prime for state in repump_states} == {2}


def test_f1_atom_scatters_from_repump_not_cooling() -> None:
    beams = build_mot_beams()
    coil = default_anti_helmholtz_config()
    atom = MOTAtomState(
        position_m=(0.0, 0.0, 1.0e-3),
        velocity_m_per_s=(0.0, 0.0, 0.0),
        ground_state=GroundState(f=1, m_f=0),
    )
    samples, _ = transition_rate_samples(beams, atom, coil)
    cooling_rate = sum(sample.scattering_rate_per_s for sample in samples if sample.beam_family == "cooling")
    repump_rate = sum(sample.scattering_rate_per_s for sample in samples if sample.beam_family == "repump")
    assert cooling_rate == 0.0
    assert repump_rate > 1.0e3


def test_single_atom_mot_trajectory_shapes() -> None:
    beams = build_mot_beams()
    coil = default_anti_helmholtz_config()
    atom = MOTAtomState(
        position_m=(1.0e-3, 0.0, 2.0e-3),
        velocity_m_per_s=(0.15, 0.0, 0.0),
        ground_state=GroundState(f=2, m_f=2),
    )
    record = simulate_mot_trajectory(beams, coil, atom, duration_s=5.0e-6, time_step_s=1.0e-6, seed=4)
    assert len(record.times_s) == len(record.positions_m) == len(record.velocities_m_per_s)
    assert len(record.event_counts) == len(record.times_s)
    assert len(record.axis_scattering_rates_per_s["horizontal_x"]) == len(record.times_s)


def test_vertical_single_atom_case_moves_toward_center() -> None:
    beams = build_mot_beams()
    coil = default_anti_helmholtz_config()
    atom = MOTAtomState(
        position_m=(0.0, 0.0, 8.0e-3),
        velocity_m_per_s=(0.0, 0.0, -0.2),
        ground_state=GroundState(f=2, m_f=2),
    )
    record = simulate_mot_trajectory(beams, coil, atom, duration_s=5.0e-3, time_step_s=1.0e-6, seed=5)
    assert record.positions_m[-1][2] < atom.position_m[2]
    assert record.velocities_m_per_s[-1][2] < atom.velocity_m_per_s[2]


def test_horizontal_x_single_atom_case_moves_toward_center() -> None:
    beams = build_mot_beams()
    coil = default_anti_helmholtz_config()
    atom = MOTAtomState(
        position_m=(8.0e-3, 0.0, 0.0),
        velocity_m_per_s=(-0.2, 0.0, 0.0),
        ground_state=GroundState(f=2, m_f=2),
    )
    record = simulate_mot_trajectory(beams, coil, atom, duration_s=2.0e-3, time_step_s=1.0e-6, seed=5)
    assert record.positions_m[-1][0] < atom.position_m[0]
    assert record.velocities_m_per_s[-1][0] < atom.velocity_m_per_s[0]


def test_horizontal_y_single_atom_case_moves_toward_center() -> None:
    beams = build_mot_beams()
    coil = default_anti_helmholtz_config()
    atom = MOTAtomState(
        position_m=(0.0, 8.0e-3, 0.0),
        velocity_m_per_s=(0.0, -0.2, 0.0),
        ground_state=GroundState(f=2, m_f=2),
    )
    record = simulate_mot_trajectory(beams, coil, atom, duration_s=2.0e-3, time_step_s=1.0e-6, seed=5)
    assert record.positions_m[-1][1] < atom.position_m[1]
    assert record.velocities_m_per_s[-1][1] < atom.velocity_m_per_s[1]
