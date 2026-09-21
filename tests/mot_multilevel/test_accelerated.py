from dataclasses import replace
import numpy as np
import pytest
from scipy.special import ellipk, ellipe

from pmot.magnetic_fields import default_anti_helmholtz_config, anti_helmholtz_field_t
from pmot.mot_multilevel import (build_rate_equation_model, build_multilevel_mot_beams,
    default_multilevel_mot_config, rate_equation_observable)
from pmot.mot_multilevel.accelerated import prepare, elliptic_ke, magnetic_field, observable, step, classify
from pmot.mot_multilevel.capture import _rk4_step, CaptureSearchConfig, classify_capture_trajectory, generate_capture_launches


def test_elliptic_and_coil_fields_against_shared_reference():
    for m in np.r_[np.linspace(0, .99, 101), .99999]:
        np.testing.assert_allclose(elliptic_ke(m), (ellipk(m), ellipe(m)), rtol=3e-14)
    coil = default_anti_helmholtz_config()
    controls = prepare()[-1]
    rng = np.random.default_rng(971)
    positions = np.vstack([rng.uniform(-.035, .035, (120, 3)), np.eye(3)*.001, np.zeros(3)])
    for r in positions:
        np.testing.assert_allclose(magnetic_field(r, controls),
            np.array(anti_helmholtz_field_t(*r, coil)), rtol=1e-9, atol=2e-16)


@pytest.mark.parametrize('power,detuning', [(.0003,-3e6),(.027,-15e6),(.13,-36.42e6)])
def test_compiled_populations_and_force_match_24_state_reference(power, detuning):
    cfg = replace(default_multilevel_mot_config(), cooling_power_w_per_beam=power,
                  cooling_detuning_rad_per_s=2*np.pi*detuning)
    data, model, coil = prepare(cfg), build_rate_equation_model(), default_anti_helmholtz_config()
    beams = build_multilevel_mot_beams(config=cfg)
    rng = np.random.default_rng(392)
    for i in range(35):
        r = np.zeros(3) if i < 5 else rng.uniform(-.025,.025,3)
        v = rng.uniform(-40,40,3)
        axis = np.array([0.,0.,1.])
        force, populations, rates, _ = observable(r,v,axis,data)
        reference = rate_equation_observable(model,beams,tuple(r),tuple(v),coil,cfg)
        np.testing.assert_allclose(populations,reference.populations,rtol=3e-6,atol=3e-10)
        np.testing.assert_allclose(force,reference.force_n,rtol=2e-6,atol=2e-29)
        np.testing.assert_allclose(rates,reference.beam_effective_scattering_rates_per_s,rtol=3e-6,atol=2e-8)


def test_compiled_rk4_stages_and_escape_match_reference():
    cfg, coil = default_multilevel_mot_config(), default_anti_helmholtz_config()
    data, model = prepare(cfg), build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=cfg)
    r, v, axis = np.array([.001,.002,-.001]), np.array([1.,-2.,.3]), np.array([0.,0.,1.])
    fast = step(r,v,axis,5e-6,data)
    slow = _rk4_step(r,v,5e-6,axis,model,beams,coil,cfg)
    for a,b in zip(fast,slow):
        np.testing.assert_allclose(a,b,rtol=2e-10,atol=1e-13)
    search=CaptureSearchConfig(disc_count=1,points_per_disc=1,maximum_simulation_time_s=.003)
    point=generate_capture_launches(search)[1][0]
    result=classify(np.array(point.initial_position_m),100*np.array(point.incident_unit_vector),5e-6,.003,data)
    reference=classify_capture_trajectory(point,100,search)
    assert result[0] == 0 and reference.termination_reason == 'escaped'
    np.testing.assert_allclose(result[2],reference.final_position_m,atol=1e-11)
