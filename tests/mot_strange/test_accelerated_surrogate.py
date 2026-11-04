from dataclasses import replace
import numpy as np
import pytest

from pmot.mot_multilevel import (build_rate_equation_model, build_multilevel_mot_beams,
    default_multilevel_mot_config, rate_equation_observable)
from pmot.mot_strange import accelerated_surrogate as fast
from pmot.mot_strange.surrogate_effective_field import (
    calibrated_surrogate_field_config, surrogate_effective_field_t,
)


@pytest.fixture(scope='module')
def environment():
    cfg = default_multilevel_mot_config()
    field = calibrated_surrogate_field_config()
    return cfg, field, fast.prepare(cfg, field), build_rate_equation_model(), build_multilevel_mot_beams(config=cfg)


def test_surrogate_field_and_physical_force_match_reference(environment):
    cfg, field, data, model, beams = environment
    rng = np.random.default_rng(2026102503)
    positions = np.vstack([np.zeros(3), np.eye(3)*.001,
                           rng.uniform(-.014, .014, (25, 3)), [[.015, 0, 0]]])
    for r in positions:
        v = rng.uniform(-35., 35., 3)
        axis = np.array([0., 0., 1.])
        b = surrogate_effective_field_t(r, field)
        np.testing.assert_allclose(fast.field(r, data), b, rtol=1e-11, atol=1e-16)
        actual = fast.observable(r, v, axis, data)
        reference = rate_equation_observable(model, beams, tuple(r), tuple(v), None,
            cfg, magnetic_field_function=lambda pos: surrogate_effective_field_t(pos, field))
        np.testing.assert_allclose(actual[0], reference.force_n, rtol=4e-6, atol=3e-29)
        np.testing.assert_allclose(actual[1], reference.populations, rtol=4e-6, atol=5e-10)
    np.testing.assert_allclose(data[10][:3], 0.)


def test_rk4_recomputes_local_environment_at_every_stage(environment):
    cfg, field, data, model, beams = environment
    r, v = np.array([.001, .002, -.001]), np.array([1., -2., .3])
    axis = np.array([0., 0., 1.])
    h = 5e-6
    from pmot.configuration import RB87_MASS_KG
    def accel(pos, vel):
        obs = rate_equation_observable(model, beams, tuple(pos), tuple(vel), None, cfg,
            tuple(axis), magnetic_field_function=lambda pos: surrogate_effective_field_t(pos, field))
        return np.asarray(obs.force_n)/RB87_MASS_KG - np.array([0., 0., 9.80665])
    a1 = accel(r, v)
    v2 = v+h/2*a1; a2 = accel(r+h/2*v, v2)
    v3 = v+h/2*a2; a3 = accel(r+h/2*v2, v3)
    v4 = v+h*a3; a4 = accel(r+h*v3, v4)
    rn = r+h/6*(v+2*v2+2*v3+v4)
    vn = v+h/6*(a1+2*a2+2*a3+a4)
    actual = fast.step(r, v, axis, h, data)
    np.testing.assert_allclose(actual[0], rn, atol=1e-14)
    np.testing.assert_allclose(actual[1], vn, atol=1e-12)


@pytest.mark.parametrize('r,v,duration,expected', [
    ([0., 0., 0.], [0., 0., 0.], .006, 1),
    ([.014, 0., 0.], [100., 0., 0.], .001, 0),
    ([.010, 0., 0.], [-1., 0., 0.], 1e-6, -1),
])
def test_adaptive_and_fixed_preserve_capture_wall_and_timeout(environment, r, v, duration, expected):
    data = environment[2]
    r, v = np.array(r), np.array(v)
    assert fast.classify(r, v, 2.5e-6, duration, data)[0] == expected
    for refinement in (1., 2.):
        assert fast.integrate(r, v, duration, data, refinement)[0] == expected
