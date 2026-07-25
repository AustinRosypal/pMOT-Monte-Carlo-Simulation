from __future__ import annotations

import numpy as np

from pmot.configuration import STANDARD_GRAVITY_M_PER_S2
from pmot.fields import build_mot_beams
from pmot.forces import AtomState
from pmot.forces import gravitational_velocity_increment as simple_gravity_increment
from pmot.forces import simulate_scattering_trajectory
from pmot.mot import GroundState
from pmot.mot import MOTAtomState
from pmot.mot import default_anti_helmholtz_config
from pmot.mot import gravitational_velocity_increment as mot_gravity_increment
from pmot.mot import simulate_mot_trajectory


def test_simple_gravity_increment_points_down() -> None:
    increment = simple_gravity_increment(2.0e-3)
    assert increment[0] == 0.0
    assert increment[1] == 0.0
    assert np.isclose(increment[2], -2.0e-3 * STANDARD_GRAVITY_M_PER_S2)


def test_mot_gravity_increment_points_down() -> None:
    increment = mot_gravity_increment(2.0e-3)
    assert increment[0] == 0.0
    assert increment[1] == 0.0
    assert np.isclose(increment[2], -2.0e-3 * STANDARD_GRAVITY_M_PER_S2)


def test_simple_scattering_trajectory_falls_without_scattering() -> None:
    record = simulate_scattering_trajectory(
        beams=[],
        initial_state=AtomState(position_m=(0.0, 0.0, 0.0), velocity_m_per_s=(0.0, 0.0, 0.0)),
        duration_s=1.0e-4,
        time_step_s=1.0e-4,
        seed=0,
        active_transition="repump",
    )
    assert record.velocities_m_per_s[-1][2] < 0.0
    assert record.positions_m[-1][2] < 0.0


def test_mot_trajectory_falls_without_scattering() -> None:
    coil = default_anti_helmholtz_config()
    record = simulate_mot_trajectory(
        beams=[],
        coil_config=coil,
        initial_state=MOTAtomState(
            position_m=(0.0, 0.0, 0.0),
            velocity_m_per_s=(0.0, 0.0, 0.0),
            ground_state=GroundState(f=1, m_f=0),
        ),
        duration_s=1.0e-6,
        time_step_s=1.0e-6,
        seed=0,
    )
    assert record.velocities_m_per_s[-1][2] < 0.0
    assert record.positions_m[-1][2] < 0.0
