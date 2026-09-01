"""Checks for efficient full-MOT capture and loading sampling."""

import numpy as np

from pmot.launch_geometry import PointSample
from pmot.mot_multilevel.rate_capture import (
    RateCaptureSearchConfig,
    classify_rate_trajectory,
    generate_rate_capture_launches,
)


def test_rate_capture_launch_geometry_is_reproducible_and_area_bounded() -> None:
    search = RateCaptureSearchConfig(disc_count=3, points_per_disc=5, seed=17)
    first_discs, first_points = generate_rate_capture_launches(search)
    second_discs, second_points = generate_rate_capture_launches(search)
    assert first_discs == second_discs
    assert first_points == second_points
    assert len(first_discs) == 3
    assert len(first_points) == 15
    assert all(0.0 <= point.s_m <= search.disc_radius_m for point in first_points)
    assert all(np.isclose(np.linalg.norm(point.incident_unit_vector), 1.0) for point in first_points)
    assert search.phase_space == "full_sphere"


def test_full_sphere_launch_geometry_covers_both_coordinate_signs() -> None:
    search = RateCaptureSearchConfig(
        disc_count=100,
        points_per_disc=1,
        seed=19,
        phase_space="full_sphere",
    )
    discs, points = generate_rate_capture_launches(search)
    directions = np.asarray([disc.outward_unit_vector for disc in discs])
    assert len(points) == 100
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.all(np.any(directions > 0.0, axis=0))
    assert np.all(np.any(directions < 0.0, axis=0))


def test_continuous_core_residence_is_a_trapped_candidate() -> None:
    point = PointSample(
        disc_index=0,
        point_index=0,
        theta_rad=0.0,
        phi_rad=0.0,
        theta_prime_rad=0.0,
        s_m=0.0,
        radial_distance_m=1.0e-3,
        initial_position_m=(1.0e-3, 0.0, 0.0),
        incident_unit_vector=(0.0, 0.0, 0.0),
        launch_axis_unit_vector=(0.0, 0.0, 0.0),
    )
    search = RateCaptureSearchConfig(
        disc_count=1,
        points_per_disc=1,
        time_step_s=5.0e-6,
        max_simulation_time_s=25.0e-6,
        bounded_core_residence_s=10.0e-6,
    )
    classification = classify_rate_trajectory(point, 0.0, search)
    assert classification.trapped
    assert classification.termination_reason == "bounded_core_residence"
