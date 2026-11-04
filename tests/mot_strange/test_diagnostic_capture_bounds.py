import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('complete_mot_strange_diagnostic_masks')


def test_known_capture_integral_bounds_coincide(diagnostic):
    masks = np.array([[[1, 1, 0]], [[0, 1, 0]]], dtype=np.int8)
    area = np.array([1e-4, 2e-4]); velocity = np.array([1., 2., 3.])
    lower, upper, interval = diagnostic.calculate_bounds(masks, area, velocity)
    np.testing.assert_array_equal(lower, upper)
    weight = velocity**3*np.exp(-velocity**2/diagnostic.original.THERMAL_SCALE_M2_PER_S2)
    expected = [diagnostic.original.LOADING_RATE_PREFACTOR*area[i]*np.trapezoid(
        np.r_[0., masks[i, 0]*weight], np.r_[0., velocity]) for i in range(2)]
    np.testing.assert_allclose(lower, expected)
    assert interval[0] <= np.mean(expected) <= interval[1]


def test_indeterminate_mask_widens_upper_integral_without_inventing_capture(diagnostic):
    masks = np.array([[[1, -1, 0]], [[0, 1, -1]]], dtype=np.int8)
    area = np.array([1e-4, 2e-4]); velocity = np.array([1., 2., 3.])
    lower, upper, interval = diagnostic.calculate_bounds(masks, area, velocity)
    all_escape = np.where(masks < 0, 0, masks)
    all_capture = np.where(masks < 0, 1, masks)
    lo, _, _ = diagnostic.calculate_bounds(all_escape, area, velocity)
    _, hi, _ = diagnostic.calculate_bounds(all_capture, area, velocity)
    np.testing.assert_array_equal(lower, lo)
    np.testing.assert_array_equal(upper, hi)
    assert np.all(upper > lower) and interval[1] > interval[0]


def test_empty_capture_has_nonzero_finite_sample_upper_bound(diagnostic):
    lower, upper, interval = diagnostic.calculate_bounds(
        np.zeros((4, 2, 3), dtype=np.int8), np.full(4, 1e-4), np.array([1., 2., 3.]))
    assert not np.any(lower) and not np.any(upper)
    assert interval[0] == 0 and interval[1] > 0


def test_diagnostic_continues_after_retained_unknown_without_reclassifying_it(diagnostic, tmp_path, monkeypatch):
    source, output = tmp_path/'source', tmp_path/'derived'
    monkeypatch.setattr(diagnostic, 'SOURCE', source)
    monkeypatch.setattr(diagnostic, 'OUTPUT', output)
    monkeypatch.setattr(diagnostic, 'PRIOR_DIAGNOSTIC', tmp_path/'prior_diagnostic')
    monkeypatch.setattr(diagnostic, 'VELOCITIES', np.array([.25, .5, .75]))
    initial = source/'pilot/rays/d000_p000.npz'
    initial.parent.mkdir(parents=True)
    np.savez_compressed(initial, speeds=np.array([.25]), codes=np.array([-1]),
        evaluations=np.zeros((1, 12)), evaluation_methods=np.array([0]), status=np.array('indeterminate'))
    before = initial.read_bytes()
    evaluated = []
    def resolve(r, direction, speed, data, records, methods):
        evaluated.append(speed)
        return 0
    monkeypatch.setattr(diagnostic, 'resolve_adaptive', resolve)
    monkeypatch.setattr(diagnostic, 'resolve_fixed_diagnostic', lambda *args: 0)
    result = diagnostic.collect_ray((0, 0, np.zeros(3), np.array([1., 0., 0.]), None))
    assert evaluated == [.5, .75]
    assert result['unknown_nodes'] == 1
    assert initial.read_bytes() == before
    with np.load(output/'rays/d000_p000.npz') as saved:
        np.testing.assert_array_equal(saved['codes'], [-1, 0, 0])
        assert str(saved['status']) == 'diagnostic_complete'
