from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ellipe, ellipk

from pmot.configuration import COOLING_WAVELENGTH_M
from pmot.magnetic_fields import anti_helmholtz_field_t, default_anti_helmholtz_config
from pmot.mot_simple.accelerated import acceleration, elliptic_ke, magnetic_field, pack_parameters, rk4, classify_numeric
from pmot.mot_simple.batched_sampling import classify_initial_conditions_batch, _acceleration_batch, _beam_arrays
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_simple.sampling import CaptureSearchConfig
from pmot.mot_simple.simulation import build_simple_mot_beams


def test_exact_cooling_wavelength_and_elliptic_integrals():
    assert COOLING_WAVELENGTH_M == 780.241209686e-9
    for m in np.r_[0., np.linspace(.0001, .99, 100), .999999]:
        k, e = elliptic_ke(m)
        assert np.isclose(k, ellipk(m), rtol=3e-15)
        assert np.isclose(e, ellipe(m), rtol=2e-13)


@pytest.mark.parametrize("gradient", [2., 20., 40.])
def test_compiled_field_and_acceleration_match_reference(gradient):
    simple = default_simple_mot_config()
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=gradient)
    beams = build_simple_mot_beams(simple_config=simple)
    data, c, a = pack_parameters(beams, coil, simple)
    rng = np.random.default_rng(3107)
    positions = np.r_[np.zeros((1, 3)), [[0,0,.001]], rng.uniform(-.025, .025, (80,3))]
    velocities = rng.uniform(-40., 40., positions.shape)
    expected_field = np.column_stack(anti_helmholtz_field_t(*positions.T, coil))
    actual_field = np.asarray([magnetic_field(r, c) for r in positions])
    np.testing.assert_allclose(actual_field, expected_field, rtol=2e-11, atol=3e-16)
    expected_acc = _acceleration_batch(positions, velocities, _beam_arrays(beams), coil, simple)
    actual_acc = np.asarray([acceleration(r, v, data, c, a) for r,v in zip(positions, velocities)])
    np.testing.assert_allclose(actual_acc, expected_acc, rtol=2e-10, atol=5e-8)


def test_compiled_classifications_and_endpoints_match_reference():
    simple = default_simple_mot_config()
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=20.)
    beams = build_simple_mot_beams(simple_config=simple)
    positions = np.asarray([[0,0,0], [.031,0,0], [.015,.002,0], [.015,0,0], [0,0,.015]])
    velocities = np.asarray([[0,0,0], [2,0,0], [-40,0,0], [-12,0,0], [0,0,-3.]])
    search = replace(CaptureSearchConfig(), max_simulation_time_s=.008)
    expected = classify_initial_conditions_batch(beams, positions, velocities, coil, simple, search)
    actual = classify_initial_conditions_batch(beams, positions, velocities, coil, simple,
                                              replace(search, trajectory_backend="numba"))
    for ref, compiled in zip(expected, actual):
        assert ref.termination_reason == compiled.termination_reason
        assert ref.trapped == compiled.trapped
        assert ref.core_entry_count == compiled.core_entry_count
        assert ref.elapsed_time_s == compiled.elapsed_time_s
        np.testing.assert_allclose(ref.final_position_m, compiled.final_position_m, atol=2e-12, rtol=1e-9)
        np.testing.assert_allclose(ref.final_velocity_m_per_s, compiled.final_velocity_m_per_s, atol=2e-9, rtol=1e-9)


def test_deadline_fallthrough_minimum_includes_saved_final_position():
    simple=replace(default_simple_mot_config(),include_gravity=False)
    coil=replace(default_anti_helmholtz_config(),current_a=0.)
    beams=[replace(b,intensity_beam=replace(b.intensity_beam,power_w=0.))
           for b in build_simple_mot_beams(simple_config=simple)]
    data,c,a=pack_parameters(beams,coil,simple)
    result=classify_numeric(np.asarray([.02,0.,0.]),np.asarray([-.001,0.,0.]),
                            data,c,a,np.asarray([1.25e-6,1.,.002,.03,2.,.005]))
    code,entered,entries,elapsed,minimum,position,velocity=result
    assert code==0 and not entered and entries==0
    assert elapsed==1.
    assert minimum <= np.linalg.norm(position)
    assert minimum==pytest.approx(.019,abs=1e-11)
    np.testing.assert_allclose(velocity,[-.001,0.,0.],atol=1e-15)
