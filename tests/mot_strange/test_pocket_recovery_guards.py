"""The recovery must fail closed unless the original probe left the cell."""

from types import SimpleNamespace
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import mot_strange_pocket_recovery as recovery


@pytest.mark.parametrize("offset,error", [
    (np.array([.5, 0., 0.]), "no unique ground steady state"),
    (np.array([2., 0., 0.]), "no unique ground steady state"),
    (np.array([100., 0., 0.]), "unrelated failure"),
])
def test_in_cell_and_unrelated_failures_propagate(monkeypatch, offset, error):
    observation = (2, 2., np.array([.005, 0., 0.]))
    data = [None] * 12
    data[11] = [0.] * 15 + [.014]
    monkeypatch.setattr(recovery, "_check", lambda obs, values: recovery.restoring.root(
        lambda x: (_ for _ in ()).throw(ValueError(error)), np.zeros(3)))
    monkeypatch.setattr(recovery, "_root", lambda fun, initial: fun(offset))
    monkeypatch.setattr(recovery, "least_squares", lambda *a, **k: pytest.fail("Invalid fallback"))
    with pytest.raises(ValueError, match=error):
        recovery.check_pocket(observation, data)


def test_only_out_of_cell_probe_uses_bounded_search(monkeypatch):
    observation = (2, 2., np.array([.005, 0., 0.]))
    data = [None] * 12
    data[11] = [0.] * 15 + [.014]
    def residual(offset):
        if np.max(np.abs(observation[2] + .001 * offset)) > .014:
            raise ValueError("no unique ground steady state")
        return offset
    def check(obs, values):
        recovery.restoring.root(residual, np.zeros(3))
        return {"status": "original_acceptance_result"}
    def bounded(fun, initial, *, bounds, **kwargs):
        low, high = bounds
        assert np.all(low >= -1.) and np.all(high <= 1.)
        assert np.max(np.abs(observation[2] + .001 * low)) <= .013
        assert np.max(np.abs(observation[2] + .001 * high)) <= .013
        np.testing.assert_array_equal(fun(initial), np.zeros(3))
        return SimpleNamespace(status=1, nfev=1)
    monkeypatch.setattr(recovery, "_check", check)
    monkeypatch.setattr(recovery, "_root", lambda fun, initial: fun(np.array([100., 0., 0.])))
    monkeypatch.setattr(recovery, "least_squares", bounded)
    result = recovery.check_pocket(observation, data)
    assert result["status"] == "original_acceptance_result"
    assert result["equilibrium_search_fallback"]["rejected_probe_position_m"] == [.10500000000000001, 0., 0.]


def test_fresh_worker_captures_original_checker_before_installing_wrapper():
    code = '''
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import resume_mot_strange_dense_pocket as runner
runner._initialize = lambda *args: None
runner.initialize()
import mot_strange_pocket_recovery as recovery
from pmot.mot_strange import loading_scan_restoring as restoring
assert recovery._check is not runner.checked_pocket
assert restoring.check_off_core_pocket is runner.checked_pocket
restoring.solver.acceleration = lambda *args: np.ones(3)
data = [None] * 12
data[10] = [0., 0., 0., 1e-12]
data[11] = [0.] * 15 + [.014]
zero = np.zeros(3)
center = np.array([.005, 0., 0.])
observation = (2, 2., center, zero, -1., center, .005, .005, 0.)
result = runner.checked_pocket(observation, data)
assert result['status'] == 'no_nearby_equilibrium'
'''
    subprocess.run(['uv', 'run', 'python', '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[2])


def test_windows_spawn_initializer_preserves_original_checker(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import resume_mot_strange_dense_pocket as runner

    arrays = tuple(np.empty(0) for _ in range(12))
    arguments = (62, arrays, runner.pilot.digest(arrays), np.zeros((1, 1, 3)),
                 np.zeros((1, 3)), tmp_path)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
                             initializer=runner.initialize, initargs=arguments) as pool:
        assert pool.submit(eval, "__import__('mot_strange_pocket_recovery')._check is not "
                           "__import__('resume_mot_strange_dense_pocket').checked_pocket").result()
