from __future__ import annotations

import numpy as np
from dataclasses import replace

from pmot.state import AtomState
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_simple import build_simple_mot_beams
from pmot.mot_simple import default_simple_mot_config
from pmot.mot_simple import mean_force_n
from pmot.mot_simple import simulate_simple_mot_trajectory
from pmot.mot_simple import rk4_step
from pmot.configuration import HBAR_J_S
from pmot.configuration import MOTBeamConfig


def test_zero_field_molasses_damps_along_x() -> None:
    simple = default_simple_mot_config()
    beams = [beam for beam in build_simple_mot_beams(simple_config=simple) if beam.axis_name == "horizontal_x"]
    coil = default_anti_helmholtz_config()
    coil = type(coil)(
        radius_m=coil.radius_m,
        turns_per_coil=coil.turns_per_coil,
        current_a=0.0,
        center_separation_m=coil.center_separation_m,
    )
    representative_beam = beams[0]
    v_test = 0.2 * simple.linewidth_hz / (2.0 * np.pi / representative_beam.wavelength_m)
    force_plus, _, _ = mean_force_n(beams, AtomState((0.0, 0.0, 0.0), (v_test, 0.0, 0.0)), coil, simple)
    force_minus, _, _ = mean_force_n(beams, AtomState((0.0, 0.0, 0.0), (-v_test, 0.0, 0.0)), coil, simple)
    assert force_plus[0] < 0.0
    assert force_minus[0] > 0.0


def test_restoring_force_signs_match_axis_displacement() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(simple_config=simple)
    coil = default_anti_helmholtz_config()
    x_plus, _, _ = mean_force_n(beams, AtomState((1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0)), coil, simple)
    x_minus, _, _ = mean_force_n(beams, AtomState((-1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0)), coil, simple)
    y_plus, _, _ = mean_force_n(beams, AtomState((0.0, 1.0e-3, 0.0), (0.0, 0.0, 0.0)), coil, simple)
    y_minus, _, _ = mean_force_n(beams, AtomState((0.0, -1.0e-3, 0.0), (0.0, 0.0, 0.0)), coil, simple)
    z_plus, _, _ = mean_force_n(beams, AtomState((0.0, 0.0, 1.0e-3), (0.0, 0.0, 0.0)), coil, simple)
    z_minus, _, _ = mean_force_n(beams, AtomState((0.0, 0.0, -1.0e-3), (0.0, 0.0, 0.0)), coil, simple)
    assert x_plus[0] < 0.0 and x_minus[0] > 0.0
    assert y_plus[1] < 0.0 and y_minus[1] > 0.0
    assert z_plus[2] < 0.0 and z_minus[2] > 0.0


def test_simple_mot_trajectory_runs_and_moves_inward_along_z() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(simple_config=simple)
    coil = default_anti_helmholtz_config()
    record = simulate_simple_mot_trajectory(
        beams=beams,
        initial_state=AtomState(position_m=(0.0, 0.0, 10.0e-3), velocity_m_per_s=(0.0, 0.0, -0.1)),
        duration_s=1.0e-3,
        time_step_s=2.0e-6,
        coil_config=coil,
        simple_config=simple,
    )
    assert len(record.times_s) == len(record.positions_m) == len(record.velocities_m_per_s)
    assert record.positions_m[-1][2] < record.positions_m[0][2]


def test_authoritative_default_parameters_and_polarizations() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(simple_config=simple)
    assert simple.cooling_detuning_hz == -15.0e6
    assert simple.include_gravity
    assert all(beam.detuning_hz == -15.0e6 for beam in beams)
    assert all(beam.intensity_beam.power_w == 20.0e-3 for beam in beams)
    assert {beam.circular_polarization for beam in beams} == {"sigma+", "sigma-"}
    generic_beam_config = MOTBeamConfig(
        name="test",
        role="test",
        wavelength_m=780.0e-9,
        resonance_frequency_hz=384.0e12,
        detuning_hz=-15.0e6,
    )
    assert generic_beam_config.power_w_per_beam == 20.0e-3


def test_reversing_effective_polarization_reverses_restoring_force() -> None:
    simple = default_simple_mot_config()
    coil = default_anti_helmholtz_config()
    beams = build_simple_mot_beams(simple_config=simple)
    normal_force, _, _ = mean_force_n(beams, AtomState((1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0)), coil, simple)
    reversed_simple = replace(
        simple,
        axis_polarization_sign={axis: -sign for axis, sign in simple.axis_polarization_sign.items()},
    )
    reversed_beams = build_simple_mot_beams(simple_config=reversed_simple)
    reversed_force, _, _ = mean_force_n(
        reversed_beams,
        AtomState((1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0)),
        coil,
        reversed_simple,
    )
    assert normal_force[0] < 0.0
    assert reversed_force[0] > 0.0
    assert np.isclose(reversed_force[0], -normal_force[0], rtol=1.0e-10)


def test_zero_detuning_and_zero_field_have_no_preferred_direction() -> None:
    simple = replace(default_simple_mot_config(), cooling_detuning_hz=0.0, include_gravity=False)
    beams = build_simple_mot_beams(simple_config=simple)
    coil = replace(default_anti_helmholtz_config(), current_a=0.0)
    force, _, _ = mean_force_n(beams, AtomState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), coil, simple)
    assert np.linalg.norm(force) < 1.0e-30


def test_force_scale_and_rk4_timestep_convergence() -> None:
    simple = replace(default_simple_mot_config(), include_gravity=False)
    beams = build_simple_mot_beams(simple_config=simple)
    coil = default_anti_helmholtz_config()
    state = AtomState((1.0e-3, -0.7e-3, 0.8e-3), (0.4, -0.2, 0.1))
    force, samples, _ = mean_force_n(beams, state, coil, simple)
    per_beam_limit = HBAR_J_S * (2.0 * np.pi / beams[0].wavelength_m) * simple.linewidth_hz / 2.0
    assert np.linalg.norm(force) <= len(beams) * per_beam_limit

    one_step, _, _, _ = rk4_step(beams, state, 2.0e-6, coil, simple)
    half_step, _, _, _ = rk4_step(beams, state, 1.0e-6, coil, simple)
    two_half_steps, _, _, _ = rk4_step(beams, half_step, 1.0e-6, coil, simple)
    assert np.allclose(one_step.position_m, two_half_steps.position_m, rtol=1.0e-7, atol=1.0e-12)
    assert np.allclose(one_step.velocity_m_per_s, two_half_steps.velocity_m_per_s, rtol=1.0e-5, atol=1.0e-9)
    assert all(sample.scattering_rate_per_s >= 0.0 for sample in samples)
