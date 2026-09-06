from math import isclose
import numpy as np

from pmot.magnetic_fields import anti_helmholtz_axial_gradient_t_per_m
from pmot.magnetic_fields import anti_helmholtz_field_t
from pmot.magnetic_fields import current_for_target_axial_gradient_a
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.magnetic_fields import _coil_singularity_mask
from pmot.magnetic_fields import _within_absolute_tolerance


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


def test_direct_absolute_tolerance_matches_numpy_isclose_edge_cases():
    target = 0.04
    atol = 1.0e-15
    offsets = np.asarray(
        [
            -np.inf,
            -np.nextafter(atol, np.inf),
            -atol,
            -np.nextafter(atol, 0.0),
            0.0,
            np.nextafter(atol, 0.0),
            atol,
            np.nextafter(atol, np.inf),
            np.inf,
            np.nan,
        ]
    )
    values = target + offsets
    expected = np.isclose(values, target, atol=atol, rtol=0.0)
    actual = _within_absolute_tolerance(values, target, atol)
    assert np.array_equal(actual, expected)


def test_direct_absolute_tolerance_matches_numpy_isclose_random_finite_values():
    rng = np.random.default_rng(20260905)
    targets = (0.0, 0.02, -0.02, 0.04)
    for target in targets:
        values = target + rng.normal(scale=3.0e-15, size=10_000)
        expected = np.isclose(values, target, atol=1.0e-15, rtol=0.0)
        actual = _within_absolute_tolerance(values, target, 1.0e-15)
        assert np.array_equal(actual, expected)


def test_direct_coil_singularity_mask_matches_legacy_isclose_definition():
    config = default_anti_helmholtz_config()
    atol = 1.0e-15
    rng = np.random.default_rng(8675309)
    rho = rng.uniform(0.0, 0.08, size=4096)
    z = rng.uniform(-0.08, 0.08, size=4096)
    rho[:5] = config.radius_m + np.asarray(
        (0.0, atol, -atol, np.nextafter(atol, np.inf), -np.nextafter(atol, np.inf))
    )
    z[:5] = -config.half_separation_m
    rho[5:10] = config.radius_m
    z[5:10] = config.half_separation_m + np.asarray(
        (0.0, atol, -atol, np.nextafter(atol, np.inf), -np.nextafter(atol, np.inf))
    )
    rho[-2:] = (np.nan, np.inf)
    z[-2:] = (np.nan, np.inf)
    expected = np.isclose(
        rho, config.radius_m, atol=atol, rtol=0.0
    ) & (
        np.isclose(
            z, -config.half_separation_m, atol=atol, rtol=0.0
        )
        | np.isclose(
            z, config.half_separation_m, atol=atol, rtol=0.0
        )
    )
    assert np.array_equal(
        _coil_singularity_mask(rho, z, config, atol=atol), expected
    )


def test_vectorized_field_matches_individual_scalar_evaluations_exactly():
    config = default_anti_helmholtz_config()
    rng = np.random.default_rng(42)
    positions = rng.uniform(-0.025, 0.025, size=(128, 3))
    positions[:5, :2] = np.asarray(
        (
            (0.0, 0.0),
            (1.0e-19, 0.0),
            (0.0, -1.0e-19),
            (2.0e-18, 0.0),
            (0.0, -2.0e-18),
        )
    )
    vector = np.column_stack(
        anti_helmholtz_field_t(
            positions[:, 0], positions[:, 1], positions[:, 2], config
        )
    )
    scalar = np.asarray(
        [
            tuple(
                float(np.asarray(component))
                for component in anti_helmholtz_field_t(x, y, z, config)
            )
            for x, y, z in positions
        ]
    )
    assert np.array_equal(vector, scalar, equal_nan=True)
