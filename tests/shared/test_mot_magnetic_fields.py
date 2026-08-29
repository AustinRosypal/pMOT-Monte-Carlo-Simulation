from math import isclose
import numpy as np

from pmot.magnetic_fields import anti_helmholtz_axial_gradient_t_per_m
from pmot.magnetic_fields import anti_helmholtz_field_t
from pmot.magnetic_fields import current_for_target_axial_gradient_a
from pmot.magnetic_fields import default_anti_helmholtz_config


def test_default_config_hits_target_gradient():
    config = default_anti_helmholtz_config()
    gradient = anti_helmholtz_axial_gradient_t_per_m(
        radius_m=config.radius_m,
        turns_per_coil=config.turns_per_coil,
        current_a=config.current_a,
    )
    assert isclose(gradient, 0.1, rel_tol=1e-12, abs_tol=1e-12)


def test_center_field_vanishes():
    config = default_anti_helmholtz_config()
    bx, by, bz = anti_helmholtz_field_t(0.0, 0.0, 0.0, config)
    assert isclose(float(bx), 0.0, abs_tol=1e-15)
    assert isclose(float(by), 0.0, abs_tol=1e-15)
    assert isclose(float(bz), 0.0, abs_tol=1e-15)


def test_current_solver_scales_linearly():
    current_low = current_for_target_axial_gradient_a(0.05, radius_m=0.04, turns_per_coil=50)
    current_high = current_for_target_axial_gradient_a(0.10, radius_m=0.04, turns_per_coil=50)
    assert isclose(current_high, 2.0 * current_low, rel_tol=1e-12, abs_tol=1e-12)


def test_field_is_linear_and_divergence_free_near_center():
    config = default_anti_helmholtz_config()
    displacement = 0.1e-3
    bx_plus, _, _ = anti_helmholtz_field_t(displacement, 0.0, 0.0, config)
    bx_minus, _, _ = anti_helmholtz_field_t(-displacement, 0.0, 0.0, config)
    _, by_plus, _ = anti_helmholtz_field_t(0.0, displacement, 0.0, config)
    _, by_minus, _ = anti_helmholtz_field_t(0.0, -displacement, 0.0, config)
    _, _, bz_plus = anti_helmholtz_field_t(0.0, 0.0, displacement, config)
    _, _, bz_minus = anti_helmholtz_field_t(0.0, 0.0, -displacement, config)
    gx = (float(bx_plus) - float(bx_minus)) / (2.0 * displacement)
    gy = (float(by_plus) - float(by_minus)) / (2.0 * displacement)
    gz = (float(bz_plus) - float(bz_minus)) / (2.0 * displacement)
    assert np.isclose(gx, gy, rtol=1.0e-5)
    assert np.isclose(gz, -2.0 * gx, rtol=2.0e-5)
    assert np.isclose(gx + gy + gz, 0.0, atol=2.0e-6)
