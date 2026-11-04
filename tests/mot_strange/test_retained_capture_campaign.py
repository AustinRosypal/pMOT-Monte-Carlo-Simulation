import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def retained(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('mot_strange_retained_capture')


def result(raw=-1, end=2., radius=.001, first=.01, maximum=.0011):
    return raw, end, np.array([radius, 0., 0.]), np.zeros(3), first, maximum


def test_event_without_full_observation_is_never_retention(retained):
    assert retained.outcome(result(raw=1, end=.01)) == -1
    assert retained.outcome(result(end=.4)) == -1
    assert retained.outcome(result(maximum=float('nan'))) == -1
    assert retained.outcome(result(maximum=.0001)) == -1


def test_final_window_and_core_event_both_required(retained):
    assert retained.outcome(result()) == retained.CENTRAL
    assert retained.outcome(result(maximum=.0021)) == retained.OTHER
    assert retained.outcome(result(first=-1)) == retained.OTHER
    assert retained.outcome(result(radius=.004, maximum=.005)) == retained.OFFCORE


def test_transient_wall_is_not_central_loading(retained):
    a = result(raw=0, end=.044397, radius=.0143, first=.010522)
    assert retained.outcome(a) == retained.TRANSIENT_WALL
    assert retained.primary(retained.outcome(a)) == retained.WALL
    b = result(raw=0, end=.044397, radius=.0143, first=-1)
    assert retained.pair_outcome(a, b) == retained.WALL_HISTORY_UNCERTAIN


def test_retained_pair_requires_category_and_endpoint_convergence(retained):
    assert retained.pair_outcome(result(), result()) == retained.CENTRAL
    assert retained.pair_outcome(result(), result(radius=.00102, maximum=.0011)) == -1
    assert retained.pair_outcome(result(), result(maximum=.0021)) == -1


def test_only_noncore_parent_proofs_can_be_reused(retained, tmp_path):
    p = tmp_path/'ray.npz'
    speeds = np.array([.125, .25, .375])
    # Two independent settings for each: wall, early core, retained off-core.
    records=[]; methods=[]
    for speed, raw, end, first in ((.125, 0, .01, -1), (.25, 1, .01, .01), (.375, -1, 2., -1)):
        for setting in (.0005, .00025):
            records.append([speed, setting, 2., raw, end, .005, 0., 0., 0., 0., 0., first]); methods.append(1)
    np.savez_compressed(p, status=np.array('resolved'), speeds=speeds, codes=[0, 1, 2],
                        retained_locations=[-1, -1, 0], evaluations=records,
                        evaluation_methods=methods, position_m=np.zeros(3), direction=np.ones(3))
    old = retained.SPEEDS
    retained.SPEEDS = speeds
    try:
        reuse, digest = retained.reusable_parent_proof(p, np.zeros(3), np.ones(3))
    finally:
        retained.SPEEDS = old
    assert reuse == {.125: retained.WALL, .375: retained.OFFCORE}
    assert len(digest) == 64
