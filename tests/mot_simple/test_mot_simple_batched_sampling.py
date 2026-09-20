from __future__ import annotations

from dataclasses import replace

import numpy as np

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_simple.batched_sampling import (
    _acceleration_batch,
    _beam_arrays,
    classify_initial_conditions_batch,
    classify_trajectory_batch,
)
from pmot.mot_simple.configuration import (
    default_simple_mot_apparatus,
    default_simple_mot_config,
)
from pmot.mot_simple.sampling import CaptureSearchConfig, classify_trajectory
from pmot.mot_simple.simulation import acceleration_m_per_s2, build_simple_mot_beams
from pmot.launch_geometry import PointSample
from pmot.state import AtomState


def _point() -> PointSample:
    return PointSample(
        disc_index=0,
        point_index=0,
        theta_rad=1.2,
        phi_rad=2.1,
        theta_prime_rad=0.3,
        s_m=4.0e-3,
        radial_distance_m=15.0e-3,
        initial_position_m=(12.0e-3, -7.0e-3, 5.0e-3),
        incident_unit_vector=(-0.8, 0.48, -0.36),
        launch_axis_unit_vector=(-0.8, 0.48, -0.36),
    )


def test_vectorized_acceleration_matches_scalar_force_law() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(default_simple_mot_apparatus(), simple)
    coil = default_anti_helmholtz_config()
    positions = np.asarray(
        [[12.0e-3, -7.0e-3, 5.0e-3], [1.0e-3, 2.0e-3, -3.0e-3]]
    )
    velocities = np.asarray([[-2.0, 1.2, -0.9], [0.4, -0.2, 0.8]])
    batch = _acceleration_batch(
        positions, velocities, _beam_arrays(beams), coil, simple
    )
    scalar = np.asarray(
        [
            acceleration_m_per_s2(
                beams,
                AtomState(tuple(position), tuple(velocity)),
                coil,
                simple,
            )[0]
            for position, velocity in zip(positions, velocities, strict=True)
        ]
    )
    assert np.allclose(batch, scalar, rtol=2.0e-13, atol=2.0e-11)


def test_batched_classifier_matches_scalar_short_trajectories() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(default_simple_mot_apparatus(), simple)
    coil = default_anti_helmholtz_config()
    search = replace(
        CaptureSearchConfig(),
        max_simulation_time_s=0.5e-3,
        time_step_s=5.0e-6,
    )
    speeds = (0.0, 2.0, 12.0, 30.0)
    point = _point()
    batch = classify_trajectory_batch(beams, point, speeds, coil, simple, search)
    scalar = tuple(
        classify_trajectory(beams, point, speed, coil, simple, search)
        for speed in speeds
    )
    assert [item.termination_reason for item in batch] == [
        item.termination_reason for item in scalar
    ]
    for actual, expected in zip(batch, scalar, strict=True):
        assert actual.trapped == expected.trapped
        assert actual.core_entry_count == expected.core_entry_count
        assert actual.entered_trap_core == expected.entered_trap_core
        assert actual.elapsed_time_s == expected.elapsed_time_s
        assert np.isclose(actual.minimum_radius_m, expected.minimum_radius_m, atol=1e-14)
        assert np.allclose(actual.final_position_m, expected.final_position_m, atol=1e-13)
        assert np.allclose(
            actual.final_velocity_m_per_s, expected.final_velocity_m_per_s, atol=1e-10
        )


def test_batched_classifier_preserves_trapped_event_priority() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(default_simple_mot_apparatus(), simple)
    coil = default_anti_helmholtz_config()
    point = replace(_point(), initial_position_m=(0.0, 0.0, 0.0))
    search = replace(
        CaptureSearchConfig(),
        max_simulation_time_s=0.0,
        bounded_core_residence_s=0.0,
    )
    result = classify_trajectory_batch(beams, point, (0.0, 1.0), coil, simple, search)
    assert all(item.trapped for item in result)
    assert all(item.termination_reason == "bounded_core_residence" for item in result)


def test_batched_classifier_matches_scalar_terminal_events_after_steps() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(default_simple_mot_apparatus(), simple)
    coil = default_anti_helmholtz_config()
    point = replace(
        _point(),
        initial_position_m=(1.0e-3, 0.0, 0.0),
        incident_unit_vector=(1.0, 0.0, 0.0),
        launch_axis_unit_vector=(1.0, 0.0, 0.0),
    )
    search = replace(
        CaptureSearchConfig(),
        max_simulation_time_s=0.1e-3,
        bounded_core_residence_s=20.0e-6,
        escape_radius_m=2.5e-3,
        time_step_s=5.0e-6,
    )
    speeds = (0.0, 1000.0)
    batch = classify_trajectory_batch(beams, point, speeds, coil, simple, search)
    scalar = tuple(
        classify_trajectory(beams, point, speed, coil, simple, search)
        for speed in speeds
    )
    assert [item.termination_reason for item in batch] == [
        "bounded_core_residence",
        "escaped",
    ]
    assert [item.termination_reason for item in batch] == [
        item.termination_reason for item in scalar
    ]
    assert np.allclose(
        [item.final_radius_m for item in batch],
        [item.final_radius_m for item in scalar],
        atol=1.0e-13,
    )


def test_generalized_batch_matches_distinct_scalar_launch_rays() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(default_simple_mot_apparatus(), simple)
    coil = default_anti_helmholtz_config()
    search = replace(
        CaptureSearchConfig(),
        max_simulation_time_s=0.75e-3,
        time_step_s=5.0e-6,
    )
    points = (
        _point(),
        replace(
            _point(),
            disc_index=1,
            point_index=3,
            initial_position_m=(-9.0e-3, 11.0e-3, -4.0e-3),
            incident_unit_vector=(0.6, -0.64, 0.48),
            launch_axis_unit_vector=(0.6, -0.64, 0.48),
        ),
        replace(
            _point(),
            disc_index=2,
            point_index=7,
            initial_position_m=(3.0e-3, 4.0e-3, 14.0e-3),
            incident_unit_vector=(-0.2, -0.4, -0.8944271909999159),
            launch_axis_unit_vector=(-0.2, -0.4, -0.8944271909999159),
        ),
    )
    speeds = np.asarray((0.0, 7.5, 24.0))
    positions = np.asarray([point.initial_position_m for point in points])
    velocities = speeds[:, None] * np.asarray(
        [point.incident_unit_vector for point in points]
    )
    batch = classify_initial_conditions_batch(
        beams, positions, velocities, coil, simple, search
    )
    scalar = tuple(
        classify_trajectory(beams, point, speed, coil, simple, search)
        for point, speed in zip(points, speeds, strict=True)
    )
    assert [item.termination_reason for item in batch] == [
        item.termination_reason for item in scalar
    ]
    for actual, expected in zip(batch, scalar, strict=True):
        assert actual.trapped == expected.trapped
        assert actual.core_entry_count == expected.core_entry_count
        assert actual.entered_trap_core == expected.entered_trap_core
        assert actual.elapsed_time_s == expected.elapsed_time_s
        assert np.isclose(actual.minimum_radius_m, expected.minimum_radius_m, atol=1e-14)
        assert np.allclose(actual.final_position_m, expected.final_position_m, atol=1e-13)
        assert np.allclose(
            actual.final_velocity_m_per_s,
            expected.final_velocity_m_per_s,
            atol=1e-10,
        )


def test_generalized_batch_does_not_mutate_initial_state_arrays() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(default_simple_mot_apparatus(), simple)
    coil = default_anti_helmholtz_config()
    search = replace(
        CaptureSearchConfig(),
        max_simulation_time_s=20.0e-6,
        time_step_s=5.0e-6,
    )
    positions = np.asarray(((1.0e-3, 0.0, 0.0), (3.0e-3, 1.0e-3, 0.0)))
    velocities = np.asarray(((0.1, 0.0, 0.0), (-0.2, 0.3, 0.0)))
    original_positions = positions.copy()
    original_velocities = velocities.copy()
    classify_initial_conditions_batch(
        beams, positions, velocities, coil, simple, search
    )
    assert np.array_equal(positions, original_positions)
    assert np.array_equal(velocities, original_velocities)


def test_generalized_batch_validates_shapes_and_finite_values() -> None:
    simple = default_simple_mot_config()
    beams = build_simple_mot_beams(default_simple_mot_apparatus(), simple)
    coil = default_anti_helmholtz_config()
    search = CaptureSearchConfig()
    with np.testing.assert_raises_regex(ValueError, "shape"):
        classify_initial_conditions_batch(
            beams, np.zeros(3), np.zeros((1, 3)), coil, simple, search
        )
    with np.testing.assert_raises_regex(ValueError, "match"):
        classify_initial_conditions_batch(
            beams, np.zeros((2, 3)), np.zeros((1, 3)), coil, simple, search
        )
    with np.testing.assert_raises_regex(ValueError, "finite"):
        classify_initial_conditions_batch(
            beams,
            np.asarray(((np.nan, 0.0, 0.0),)),
            np.zeros((1, 3)),
            coil,
            simple,
            search,
        )
