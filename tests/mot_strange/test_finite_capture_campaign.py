import importlib
from pathlib import Path
import json

import numpy as np
import pytest


@pytest.fixture
def finite(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('run_mot_strange_finite_capture_campaign')


def endpoint(code=-1, time=2., position=(.01, 0, 0), first=-1.):
    return code, time, np.asarray(position), np.zeros(3), first


def test_short_timeout_is_never_promoted_to_horizon_retention(finite):
    assert finite.finite_code(endpoint(time=.4), .4) == finite.UNKNOWN
    assert finite.finite_code(endpoint(), 2.) == finite.RETAINED
    assert finite.finite_code(endpoint(first=.1), 2.) == finite.UNKNOWN
    assert finite.finite_code(endpoint(time=1.9), 2.) == finite.UNKNOWN


def test_retained_outcome_stays_distinct_from_wall_and_core(finite):
    assert finite.finite_code(endpoint(code=0, time=.02), .05) == finite.WALL
    assert finite.finite_code(endpoint(code=1, time=.006, first=.006), .05) == finite.CORE
    assert finite.RETAINED not in (finite.WALL, finite.CORE)
    with pytest.raises(ValueError, match='outside observation window'):
        finite.finite_code(endpoint(code=1, time=2.1, first=2.1), 2.)
    assert finite.finite_code(endpoint(position=(float('nan'), 0, 0)), 2.) == finite.UNKNOWN


def test_two_settings_must_agree_on_horizon_outcome_and_location(finite):
    assert finite.pair_code(endpoint(), endpoint(position=(.011, 0, 0)), 2.) == finite.RETAINED
    assert finite.pair_code(endpoint(), endpoint(code=0), 2.) == finite.UNKNOWN
    assert finite.pair_code(endpoint(), endpoint(position=(.001, 0, 0)), 2.) == finite.UNKNOWN
    assert finite.pair_code(endpoint(), endpoint(time=1.9), 2.) == finite.UNKNOWN


def test_pilot_proof_requires_two_independent_settings(finite):
    rows = [[1., 5e-4, 2., -1., 2., .01, 0., 0., 0., 0., 0., -1.],
            [1., 2.5e-4, 2., -1., 2., .01, 0., 0., 0., 0., 0., -1.]]
    assert finite.existing_proof(rows, [1, 1], 1.) == finite.RETAINED
    assert finite.existing_proof(rows[:1], [1], 1.) == finite.UNKNOWN
    assert finite.existing_proof(rows, [0, 1], 1.) == finite.UNKNOWN
    assert finite.existing_proof(rows, [1, 1], 1., fixed_only=True) == finite.UNKNOWN
    rows[0][2] = .4
    assert finite.existing_proof(rows, [1, 1], 1.) == finite.UNKNOWN


def test_pilot_conversion_preserves_events_and_source_bytes(finite, tmp_path, monkeypatch):
    base, source, output = tmp_path/'base', tmp_path/'source', tmp_path/'finite'
    monkeypatch.setattr(finite, 'BASE_OUTPUT', base)
    monkeypatch.setattr(finite, 'PILOT_SOURCE', source)
    monkeypatch.setattr(finite, 'OUTPUT', output)
    monkeypatch.setattr(finite, 'SPEEDS', np.array([1., 2., 3.]))
    monkeypatch.setattr(finite.original, 'PILOT_DISCS', 2)
    monkeypatch.setattr(finite.original, 'POINTS', 1)
    (base/'pilot').mkdir(parents=True)
    np.savez_compressed(base/'pilot/geometry.npz', positions_m=np.zeros((2, 1, 3)))
    (source/'validation').mkdir(parents=True)
    for name in ('additional_fixed_nodes', 'remaining_fixed_nodes'):
        (source/f'validation/{name}.json').write_text(json.dumps({'cases': []}))
    (source/'validation/core_retention_audit.json').write_text(json.dumps({'passed': True}))
    (source/'rays').mkdir()
    before = {}
    for d in range(2):
        rows = []
        for speed, raw, time, duration, first in ((1., 1., .006, .05, .006),
                                                 (2., 0., .02, .05, -1.),
                                                 (3., -1., 2., 2., -1.)):
            for step in (.0005, .00025):
                rows.append([speed, step, duration, raw, time, .01, 0., 0., 0., 0., 0., first])
        p = source/'rays'/f'd{d:03d}_p000.npz'
        np.savez_compressed(p, speeds=np.array([1., 2., 3.]), evaluations=np.asarray(rows),
            evaluation_methods=np.ones(6, dtype=int), position_m=np.array([.01, 0., 0.]),
            direction=np.array([-1., 0., 0.]))
        before[p] = p.read_bytes()
    finite.convert_pilot()
    for p, content in before.items():
        assert p.read_bytes() == content
        with np.load(output/'pilot/rays'/p.name) as saved:
            np.testing.assert_array_equal(saved['codes'], [1, 0, 2])
            np.testing.assert_array_equal(saved['retained_locations'], [-1, -1, 0])
            assert float(saved['observation_horizon_s']) == 2.


def test_zero_capture_statistics_keep_nonzero_sampling_bound(finite, tmp_path, monkeypatch):
    monkeypatch.setattr(finite, 'OUTPUT', tmp_path)
    monkeypatch.setattr(finite.original, 'POINTS', 1)
    finite.STAT_CACHE.clear()
    (tmp_path/'production/rays').mkdir(parents=True)
    (tmp_path/'campaign_manifest.json').write_text(json.dumps({'thermal_conditions': {'temperature_k': 296.179}}))
    for d in range(2):
        np.savez_compressed(tmp_path/f'production/rays/d{d:03d}_p000.npz',
            status=np.array('resolved'), observation_horizon_s=np.array(2.),
            speeds=finite.SPEEDS, codes=np.zeros(len(finite.SPEEDS), dtype=np.int8),
            retained_locations=np.full(len(finite.SPEEDS), -1))
    row = finite.statistics('production', 2, None, None, np.array([1e-4, 1e-4]))
    assert row['loading_rate_atoms_per_s'] == 0
    assert row['loading_95_half_width_atoms_per_s'] > 0
    assert not row['precision_pass']
    finite.STAT_CACHE.clear()
