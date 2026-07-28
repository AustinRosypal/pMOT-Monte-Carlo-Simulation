from __future__ import annotations

import numpy as np

from pmot.forces import AtomState
from pmot.mot import default_anti_helmholtz_config
from pmot.mot_simple import build_simple_mot_beams
from pmot.mot_simple import default_simple_mot_config
from pmot.mot_simple import mean_force_n
from pmot.mot_simple import simulate_simple_mot_trajectory


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
