from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pytest

from pmot.launch_geometry import build_incident_disc_from_angles
from pmot.mot_strange.cell_launch_sampling import sample_disc_points_inside_cell, disc_area_inside_cell_m2
from pmot.mot_strange.surrogate_effective_field import calibrated_surrogate_field_config


@pytest.fixture
def sampling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('mot_strange_geometry_audit_sampling')


def baseline(disc, count, radius, config, seed):
    points, _, _ = sample_disc_points_inside_cell(disc, count, radius, np.random.default_rng(seed), config)
    return np.asarray([p.initial_position_m for p in points])


def test_nested_mixture_preserves_uniform_area_and_exact_reuse(sampling):
    cfg = replace(calibrated_surrogate_field_config(), cell_half_length_m=.1)
    disc = build_incident_disc_from_angles(0, .01, 1.1, .7)
    original = baseline(disc, 4000, .005, cfg, 421)
    before = original.copy()
    result = sampling.enlarged_disc_mixture(disc, original, .005, .01, cfg, np.random.default_rng(422))
    np.testing.assert_array_equal(original, before)
    reused = result['reuse_baseline']
    assert result['reuse_probability'] == pytest.approx(.25)
    assert abs(reused.mean()-.25) < .025
    np.testing.assert_array_equal(result['positions_m'][reused], original[reused])
    radii = np.linalg.norm(result['positions_m']-disc.center_position_m, axis=1)
    assert np.all(radii[~reused] > .005) and np.all(radii <= .01+1e-14)
    assert abs(np.mean(radii**2)-.01**2/2) < 2e-6
    np.testing.assert_allclose((result['positions_m']-disc.center_position_m) @ np.asarray(disc.outward_unit_vector), 0., atol=1e-14)


def test_clipped_mixture_has_correct_radial_cdf_and_inside_cell_points(sampling):
    cfg = calibrated_surrogate_field_config()
    disc = build_incident_disc_from_angles(0, .01, 1.05, .6)
    original = baseline(disc, 3000, .015, cfg, 423)
    result = sampling.enlarged_disc_mixture(disc, original, .015, .02, cfg, np.random.default_rng(424))
    assert np.all(np.abs(result['positions_m']) <= cfg.cell_half_length_m)
    radii = np.linalg.norm(result['positions_m']-disc.center_position_m, axis=1)
    for radius in (.008, .012, .015, .018):
        expected = disc_area_inside_cell_m2(disc, radius, cfg)/result['larger_area_m2']
        assert abs(np.mean(radii <= radius)-expected) < .035


def test_unchanged_radius_reuses_every_point(sampling):
    cfg = calibrated_surrogate_field_config()
    disc = build_incident_disc_from_angles(0, .01, 1.05, .6)
    original = baseline(disc, 20, .015, cfg, 425)
    result = sampling.enlarged_disc_mixture(disc, original, .015, .015, cfg, np.random.default_rng(426))
    assert result['reuse_baseline'].all() and result['annulus_proposal_count'] == 0
    np.testing.assert_array_equal(result['positions_m'], original)
    with pytest.raises(ValueError, match='nested'):
        sampling.enlarged_disc_mixture(disc, original, .015, .01, cfg, np.random.default_rng(426))


def test_upstream_disc_preserves_normal_but_requires_new_points(sampling):
    cfg = calibrated_surrogate_field_config()
    disc = build_incident_disc_from_angles(0, .01, 1.05, .6)
    moved = sampling.disc_at_distance(disc, .012)
    np.testing.assert_array_equal(moved.incident_unit_vector, disc.incident_unit_vector)
    np.testing.assert_allclose(moved.center_position_m, .012*np.asarray(disc.outward_unit_vector))
    original = baseline(disc, 20, .015, cfg, 427)
    with pytest.raises(ValueError, match='baseline positions'):
        sampling.enlarged_disc_mixture(moved, original, .015, .02, cfg, np.random.default_rng(428))
    with pytest.raises(ValueError, match='distance'):
        sampling.disc_at_distance(disc, float('nan'))
