import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def revised(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('mot_strange_retained_capture_v4r1')


def result(radius=.01, first=-1., end=2.):
    return -1, end, np.array([radius, 0., 0.]), np.zeros(3), first, radius


def records(model, code_result=None):
    rows = []
    for dt in (1.5625e-7, 7.8125e-8):
        model.append(rows, 33.625, 0, dt, code_result or result())
    return rows


def test_extended_retention_requires_three_mutually_converged_levels(revised, monkeypatch):
    sequence = iter([result(.0101), result(.01), result(.010004), result(.010008)])
    monkeypatch.setattr(revised.budget, 'retained_adaptive', lambda *args: next(sequence))
    assert revised.refined_consensus(np.zeros(3), np.ones(3), 33.625, None, records(revised)) == revised.OFFCORE


def test_adjacent_agreement_cannot_hide_total_span(revised, monkeypatch):
    sequence = iter([result(.01), result(.01), result(.010006), result(.010012)])
    monkeypatch.setattr(revised.budget, 'retained_adaptive', lambda *args: next(sequence))
    assert revised.refined_consensus(np.zeros(3), np.ones(3), 33.625, None, records(revised)) == -1


def test_extended_solution_cannot_contradict_fixed_category(revised, monkeypatch):
    monkeypatch.setattr(revised.budget, 'retained_adaptive', lambda *args: result(.001, first=.01))
    assert revised.refined_consensus(np.zeros(3), np.ones(3), 33.625, None, records(revised)) == -1


def test_early_returns_remain_unknown(revised, monkeypatch):
    monkeypatch.setattr(revised.budget, 'retained_adaptive', lambda *args: result(end=.4))
    assert revised.refined_consensus(np.zeros(3), np.ones(3), 33.625, None, records(revised)) == -1


def test_missing_or_conflicting_fixed_evidence_fails_closed(revised, monkeypatch):
    def forbidden(*args):
        raise AssertionError('No extended integration without fixed evidence')
    monkeypatch.setattr(revised.budget, 'retained_adaptive', forbidden)
    assert revised.refined_consensus(np.zeros(3), np.ones(3), 33.625, None, []) == -1
    rows = records(revised)
    revised.append(rows, 33.625, 0, 1e-6, result(.001, first=.01))
    assert revised.refined_consensus(np.zeros(3), np.ones(3), 33.625, None, rows) == -1


def test_parent_reuse_creates_missing_destination_directory(revised, monkeypatch, tmp_path):
    parent = tmp_path/'parent'
    source = parent/'pilot/rays/d000_p000.npz'
    source.parent.mkdir(parents=True)
    r, direction = np.zeros(3), np.array([1., 0., 0.])
    np.savez_compressed(source, status=np.array('resolved'), speeds=revised.SPEEDS,
        codes=np.zeros(len(revised.SPEEDS), dtype=np.int8), position_m=r,
        direction=direction, observation_horizon_s=np.array(2.),
        retention_window_s=np.array(.025))
    monkeypatch.setattr(revised, 'PREVIOUS_OUTPUT', parent)
    monkeypatch.setattr(revised, 'OLD_OUTPUT', tmp_path/'older_missing')
    target = tmp_path/'new/pilot/rays'/source.name
    assert not target.parent.exists()
    result = revised.worker((None, 0, 0, r, direction, target, 'pilot', None))
    assert result['status'] == 'resolved'
    assert target.read_bytes() == source.read_bytes()
    import json
    lineage = json.loads(target.with_suffix('.lineage.json').read_text())
    assert lineage['sha256'] == revised.original.sha(source)
    assert lineage['source_status'] == 'resolved'

