"""Checks for the vapor-distribution loading-rate calculation."""

import numpy as np
import pytest

from pmot.mot_simple.loading import (
    DEFAULT_RB87_PARTIAL_PRESSURE_TORR,
    DEFAULT_VAPOR_TEMPERATURE_K,
    LOADING_RATE_PREFACTOR,
    SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3,
    THERMAL_SCALE_M2_PER_S2,
    VaporLoadingParameters,
    calculate_loading_rate_from_spectrum,
    loading_integrand,
)


def test_corrected_vapor_distribution_constants() -> None:
    vapor = VaporLoadingParameters()
    assert DEFAULT_VAPOR_TEMPERATURE_K == pytest.approx(296.483)
    assert DEFAULT_RB87_PARTIAL_PRESSURE_TORR == pytest.approx(1.0e-10)
    assert vapor.rb87_partial_pressure_pa == pytest.approx(1.3332e-8)
    assert vapor.number_density_m3 == pytest.approx(3.256958507565e12)
    assert SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3 == pytest.approx(
        1.670273425505391e-7
    )
    assert THERMAL_SCALE_M2_PER_S2 == pytest.approx(5.67281110282826e4)
    assert LOADING_RATE_PREFACTOR == pytest.approx(5.44001124315979e5)


def test_loading_rate_uses_corrected_prefactor() -> None:
    velocity = np.asarray([1.0, 2.0, 3.0])
    cross_section = np.asarray([3.0e-6, 2.0e-6, 0.5e-6])
    expected_integral = np.trapezoid(loading_integrand(velocity, cross_section), velocity)
    result = calculate_loading_rate_from_spectrum(velocity, cross_section)
    assert result.integral_value_m5_per_s4 == pytest.approx(expected_integral)
    assert result.loading_rate_atoms_per_s == pytest.approx(
        LOADING_RATE_PREFACTOR * expected_integral
    )
    assert result.integral_value_m6_per_s4 == pytest.approx(expected_integral)


def test_loading_rate_pressure_is_configurable_and_linear() -> None:
    velocity = np.asarray([0.0, 1.0, 2.0])
    cross_section = np.asarray([1.0e-6, 1.0e-6, 1.0e-6])
    default = calculate_loading_rate_from_spectrum(velocity, cross_section)
    doubled = calculate_loading_rate_from_spectrum(
        velocity,
        cross_section,
        VaporLoadingParameters(rb87_partial_pressure_torr=2.0e-10),
    )
    assert doubled.loading_rate_atoms_per_s == pytest.approx(
        2.0 * default.loading_rate_atoms_per_s
    )
