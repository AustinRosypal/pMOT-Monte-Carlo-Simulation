from __future__ import annotations

import numpy as np
import pytest

from pmot.pmot.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
    LEGACY_ASYMMETRIC_SURROGATE_EFFECTIVE_FIELD_CONFIG,
    inside_surrogate_cell,
    numerical_field_jacobian_t_per_m,
    surrogate_component_intensities_w_per_m2,
    surrogate_effective_field_t,
    surrogate_field_jacobians_t_per_m,
)


def test_component_intensities_and_vector_field_have_expected_shapes() -> None:
    point = np.asarray((1.0e-3, -2.0e-3, 3.0e-3))
    assert surrogate_component_intensities_w_per_m2(point).shape == (6,)
    assert surrogate_effective_field_t(point).shape == (3,)
    points = np.asarray((point, -point))
    assert surrogate_component_intensities_w_per_m2(points).shape == (2, 6)
    assert surrogate_effective_field_t(points).shape == (2, 3)


def test_field_is_exactly_zero_outside_closed_cubic_cell() -> None:
    half_length = DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.cell_half_length_m
    boundary = np.asarray((half_length, 0.0, 0.0))
    outside = np.asarray((np.nextafter(half_length, np.inf), 0.0, 0.0))
    assert inside_surrogate_cell(boundary)
    assert not inside_surrogate_cell(outside)
    assert np.linalg.norm(surrogate_effective_field_t(boundary)) > 0.0
    np.testing.assert_array_equal(surrogate_effective_field_t(outside), np.zeros(3))
    np.testing.assert_array_equal(
        surrogate_component_intensities_w_per_m2(outside),
        np.zeros(6),
    )


def test_vector_sum_matches_beamwise_specification() -> None:
    cfg = DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    point = np.asarray((0.4e-3, -0.7e-3, 0.9e-3))
    intensity = surrogate_component_intensities_w_per_m2(point, cfg)
    expected = cfg.kappa_t_per_w_per_m2 * np.asarray(
        (
            intensity[0] - intensity[1],
            intensity[2] - intensity[3],
            -(intensity[4] - intensity[5]),
        )
    )
    np.testing.assert_allclose(surrogate_effective_field_t(point, cfg), expected)
    assert cfg.kappa_t_per_w_per_m2 == pytest.approx(-1.948e-7, rel=5.0e-4)


def test_origin_jacobian_has_intended_quadrupole_signs_and_ratio() -> None:
    jacobian = numerical_field_jacobian_t_per_m(step_m=1.0e-7)
    diagonal_g_per_cm = 100.0 * np.diag(jacobian)
    assert diagonal_g_per_cm[0] > 0.0
    assert diagonal_g_per_cm[1] > 0.0
    assert diagonal_g_per_cm[2] < 0.0
    assert diagonal_g_per_cm[0] == pytest.approx(diagonal_g_per_cm[1], rel=1.0e-12)
    assert abs(diagonal_g_per_cm[2] / diagonal_g_per_cm[0]) == pytest.approx(
        2.0,
        rel=1.0e-12,
    )
    np.testing.assert_allclose(
        jacobian - np.diag(np.diag(jacobian)),
        0.0,
        atol=1.0e-14,
    )


def test_center_balanced_ranges_give_exact_origin_zero_and_preserve_scale() -> None:
    cfg = DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    legacy = LEGACY_ASYMMETRIC_SURROGATE_EFFECTIVE_FIELD_CONFIG
    np.testing.assert_allclose(
        surrogate_effective_field_t(np.zeros(3), cfg),
        np.zeros(3),
        rtol=0.0,
        atol=1.0e-18,
    )
    assert cfg.forward_rayleigh_range_m * cfg.return_rayleigh_range_m == pytest.approx(
        legacy.forward_rayleigh_range_m * legacy.return_rayleigh_range_m,
        rel=1.0e-14,
    )
    assert np.linalg.norm(
        surrogate_effective_field_t(np.zeros(3), legacy)
    ) > 1.0e-5


def test_vectorized_field_jacobians_match_scalar_origin_helper() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (0.5e-3, -0.2e-3, 0.1e-3)))
    jacobians = surrogate_field_jacobians_t_per_m(points, step_m=1.0e-7)
    assert jacobians.shape == (2, 3, 3)
    np.testing.assert_allclose(
        jacobians[0],
        numerical_field_jacobian_t_per_m(step_m=1.0e-7),
        rtol=0.0,
        atol=1.0e-14,
    )

