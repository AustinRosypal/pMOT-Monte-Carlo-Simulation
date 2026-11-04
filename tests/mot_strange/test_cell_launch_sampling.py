import numpy as np
import pytest

from pmot.launch_geometry import build_incident_disc_from_angles
from pmot.mot_strange.cell_launch_sampling import (
    disc_area_inside_cell_m2, sample_disc_points_inside_cell,
)
from pmot.mot_strange.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG as FIELD, inside_surrogate_cell,
)
from pmot.mot_strange.surrogate_capture_campaign import CampaignConfig, build_launch_tasks


def test_axis_disc_area_matches_analytic_circle_minus_four_caps():
    disc = build_incident_disc_from_angles(0, 0.010, np.pi / 2, 0.0)
    radius = 0.015
    half = FIELD.cell_half_length_m
    cap = radius**2 * np.arccos(half / radius) - half * np.sqrt(radius**2 - half**2)
    expected = np.pi * radius**2 - 4 * cap
    assert disc_area_inside_cell_m2(disc, radius) == pytest.approx(expected, rel=1e-11)
    assert disc_area_inside_cell_m2(disc, 0.004) == pytest.approx(np.pi*0.004**2)


def test_oblique_disc_area_matches_independent_uniform_area_integration():
    from scipy.stats import qmc
    disc = build_incident_disc_from_angles(0, 0.010, 0.73, 0.41)
    radius = 0.015
    uniform = qmc.Sobol(2, scramble=True, seed=12).random_base2(18)
    r = radius * np.sqrt(uniform[:, 0])
    phi = 2*np.pi * uniform[:, 1]
    positions = (np.asarray(disc.center_position_m)
                 + (r*np.cos(phi))[:, None] * np.asarray(disc.basis_u)
                 + (r*np.sin(phi))[:, None] * np.asarray(disc.basis_v))
    fraction = np.mean(inside_surrogate_cell(positions))
    assert disc_area_inside_cell_m2(disc, radius) / (np.pi*radius**2) == pytest.approx(
        fraction, abs=1e-4
    )


def test_rejection_preserves_count_and_parallel_velocities_on_every_disc():
    cfg = CampaignConfig(seed=20261025, direction_discs=40, points_per_disc=20,
                         speeds_m_per_s=(1.0, 3.0), disc_radius_m=0.015,
                         in_cell_launches=True)
    tasks = build_launch_tasks(cfg)
    assert len(tasks) == 40 * 20 * 2
    assert np.all(inside_surrogate_cell([t.initial_position_m for t in tasks]))
    for d in range(40):
        group = [t for t in tasks if t.disc_index == d]
        assert {t.point_index for t in group} == set(range(20))
        assert len({t.incident_unit_vector for t in group}) == 1
        direction = np.asarray(group[0].incident_unit_vector)
        positions = np.asarray([t.initial_position_m for t in group])
        np.testing.assert_allclose(positions @ direction, -cfg.launch_radius_m, atol=1e-16)
        assert len({t.launch_area_m2 for t in group}) == 1
        assert 0 < group[0].launch_area_m2 <= cfg.sample_area_m2
    assert tasks == build_launch_tasks(cfg)


def test_sampler_reports_rejected_proposals_and_dense_point_indices():
    disc = build_incident_disc_from_angles(0, 0.010, 0.73, 0.41)
    points, area, proposed = sample_disc_points_inside_cell(
        disc, 200, 0.015, np.random.default_rng(73)
    )
    assert proposed > len(points) == 200
    assert [p.point_index for p in points] == list(range(200))
    assert np.all(inside_surrogate_cell([p.initial_position_m for p in points]))
    assert area < np.pi*0.015**2
