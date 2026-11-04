import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def refined(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    diagnostic = importlib.import_module('complete_mot_strange_diagnostic_masks')
    old_velocity, old_output = diagnostic.ce.VELOCITIES, diagnostic.original.OUTPUT
    module = importlib.import_module('run_mot_strange_finite_capture_refined_campaign')
    try:
        yield module
    finally:
        diagnostic.ce.VELOCITIES = old_velocity
        diagnostic.original.OUTPUT = old_output


def test_grid_is_nested_and_resolves_half_intervals(refined):
    fine = refined.SPEEDS[refined.SPEEDS <= 80]
    np.testing.assert_array_equal(fine[1::2], np.arange(.25, 80.001, .25))
    assert len(fine) == 640
    np.testing.assert_array_equal(refined.SPEEDS[-8:], np.arange(85., 121., 5.))


def test_refinement_evaluates_missing_nodes_and_keeps_proven_events(refined, tmp_path, monkeypatch):
    path = tmp_path/'ray.npz'
    monkeypatch.setattr(refined, 'SPEEDS', np.array([.125, .25, .375, .5]))
    monkeypatch.setattr(refined.ce, 'VELOCITIES', np.array([.125, .25, .375, .5]))
    monkeypatch.setattr(refined.ce, 'boundary_node_indices', lambda codes: set())
    np.savez_compressed(path, status=np.array('refinement_pending'), speeds=np.array([.25, .5]),
        codes=np.array([0, 0]), evaluations=np.zeros((0, 12)), evaluation_methods=np.zeros(0, dtype=int))
    calls = []
    def resolve(r, direction, speed, data, records, methods):
        calls.append(speed)
        return 0
    monkeypatch.setattr(refined, 'resolve_node', resolve)
    result = refined.worker((None, 0, 0, np.array([.01, 0., 0.]), np.array([-1., 0., 0.]), path, 'pilot', None))
    assert result['status'] == 'resolved'
    assert calls == [.125, .375]
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['speeds'], [.125, .25, .375, .5])
        np.testing.assert_array_equal(saved['codes'], [0, 0, 0, 0])


def test_resolved_old_grid_is_rejected_instead_of_skipped(refined, tmp_path):
    path = tmp_path/'wrong.npz'
    np.savez_compressed(path, status=np.array('resolved'), speeds=np.array([.25, .5]))
    with pytest.raises(AssertionError):
        refined.worker((None, 0, 0, np.zeros(3), np.ones(3), path, 'pilot', None))


def test_short_or_inconsistent_horizon_results_remain_unknown(refined):
    good = (-1, 2., np.array([.01, 0., 0.]), np.zeros(3), -1.)
    short = (-1, .4, np.array([.01, 0., 0.]), np.zeros(3), -1.)
    assert refined.finite_code(short, .4) == -1
    assert refined.pair_code(good, short, 2.) == -1
    assert refined.pair_code(good, good, 2.) == 2
