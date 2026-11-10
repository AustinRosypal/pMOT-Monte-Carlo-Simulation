import numpy as np
import pytest

from pmot.loading import VaporLoadingParameters
from pmot.mot_strange.loading_scan_inputs import (
    ScanPoint,
    cooling_configuration,
    current_symmetric_surrogate,
    field_axes,
    field_config,
    sample_launch_geometry,
    scan_points,
)
from pmot.mot_strange.loading_scan_statistics import loading_by_ray, loading_interval, plan_production
from pmot.mot_strange.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG as GEOMETRY,
    surrogate_effective_field_t,
)


def test_diameter_scan_keeps_repump_radius_and_explicit_cooling_wavelength():
    points, scans = scan_points()
    assert len(points) == len(set(points)) == 69
    assert [p.trap_power_mw for p in scans["trap_power_axial"]] == [
        *range(1, 10, 2), 10, *range(11, 40, 2), 40,
    ]
    cfg, beams = cooling_configuration(ScanPoint(cooling_diameter_mm=30.0))
    assert cfg.include_gravity and cfg.repump_detuning_rad_per_s == 0.0
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]
    assert len(cooling) == len(repump) == 6
    for beam in cooling:
        assert beam.wavelength_m == 780.241209686e-9
        assert beam.power_w == pytest.approx(27e-3)
        assert beam.beam_radius_m == pytest.approx(0.015)
    for beam in repump:
        assert beam.power_w == pytest.approx(0.1e-3)
        assert beam.beam_radius_m == pytest.approx(0.00635)


def test_axial_sign_is_implemented_by_helicity_and_not_negative_power():
    axial = field_config(ScanPoint(trap_power_mw=40), GEOMETRY, isotropic_axial_helicity=1)
    iso = field_config(ScanPoint("isotropic", 40), GEOMETRY, isotropic_axial_helicity=1)
    axial_metadata = field_axes(axial)
    iso_metadata = field_axes(iso)
    assert axial_metadata["trapping_power_mw_xyz"] == [20, 20, 40]
    assert iso_metadata["trapping_power_mw_xyz"] == [40, 40, 40]
    axial_gradient = np.array(axial_metadata["origin_surrogate_field_jacobian_g_per_cm"])
    iso_gradient = np.array(iso_metadata["origin_surrogate_field_jacobian_g_per_cm"])
    np.testing.assert_allclose(axial_gradient, np.diag([0.5, 0.5, -1]) @ iso_gradient)
    assert np.trace(axial_gradient) == pytest.approx(0.0, abs=1e-8)


def test_current_geometry_keeps_the_field_centered_at_the_exact_wavelength():
    geometry = current_symmetric_surrogate()
    assert geometry.wavelength_m == 1529.268881e-9
    assert geometry.forward_waist_position_m == -.010
    assert geometry.return_waist_position_m == .010
    assert geometry.forward_rayleigh_range_m == geometry.return_rayleigh_range_m
    assert geometry.kappa_t_per_w_per_m2 == pytest.approx(-9.740758714569487e-8, rel=1e-10)
    field = field_config(ScanPoint(), geometry, isotropic_axial_helicity=1)
    np.testing.assert_allclose(surrogate_effective_field_t((0, 0, 0), field), 0, atol=1e-15)
    assert field_axes(field)["origin_surrogate_field_jacobian_g_per_cm"][2][2] == pytest.approx(-5.222603)


def test_clipped_launch_areas_weight_disc_rates_and_unknown_flux_fails_closed():
    positions, normals, areas = sample_launch_geometry(3, 4, 20261105, GEOMETRY)
    more_positions, _, _ = sample_launch_geometry(3, 6, 20261105, GEOMETRY)
    np.testing.assert_array_equal(positions, more_positions[:, :4])
    np.testing.assert_allclose(np.einsum("dpk,dk->dp", positions, normals), -0.01)
    assert np.all(np.abs(positions) < GEOMETRY.cell_half_length_m)
    assert np.all((areas > 0) & (areas <= np.pi * 0.015**2))
    masks = np.ones((3, 4, 3), dtype=int)
    masks[:, :, 0] = -1
    vapor = VaporLoadingParameters()
    rates = loading_by_ray(masks, np.arange(3.0), areas, vapor)
    np.testing.assert_allclose(rates[:, 0] / rates[0, 0], areas / areas[0])
    assert loading_interval(np.zeros(3))["relative_half_width"] is None
    masks[0, 0, 1] = -1
    with pytest.raises(ValueError, match="positive-flux"):
        loading_by_ray(masks, np.arange(3.0), areas, vapor)


def test_zero_capture_in_pilot_does_not_certify_precision():
    plan = plan_production(np.zeros((32, 16)))
    assert plan["ray_count"] == plan["max_rays"] == 12800
    assert plan["predicted_relative_half_width"] is None
    assert plan["predicted_target_met"] is False
