from math import isclose

from pmot.mot import anti_helmholtz_axial_gradient_t_per_m
from pmot.mot import anti_helmholtz_field_t
from pmot.mot import current_for_target_axial_gradient_a
from pmot.mot import default_anti_helmholtz_config


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

