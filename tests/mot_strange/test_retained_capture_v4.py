import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def revised(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('mot_strange_retained_capture_v4')


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
