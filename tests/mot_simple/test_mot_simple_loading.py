"""Checks for the vapor-distribution loading-rate calculation."""

import numpy as np
import pytest

from pmot.mot_simple.loading import (
    LOADING_RATE_PREFACTOR,
    SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3,
    calculate_loading_rate_from_spectrum,
    loading_integrand,
)


def test_corrected_vapor_distribution_constants() -> None:
    assert SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3 == pytest.approx(2.80e-7)
    assert LOADING_RATE_PREFACTOR == pytest.approx(9.1196e5)


def test_loading_rate_uses_corrected_prefactor() -> None:
    velocity = np.asarray([1.0, 2.0, 3.0])
    cross_section = np.asarray([3.0e-6, 2.0e-6, 0.5e-6])
    expected_integral = np.trapezoid(loading_integrand(velocity, cross_section), velocity)
    result = calculate_loading_rate_from_spectrum(velocity, cross_section)
    assert result.integral_value_m5_per_s4 == pytest.approx(expected_integral)
    assert result.loading_rate_atoms_per_s == pytest.approx(9.1196e5 * expected_integral)
