import numpy as np

from pmot.mot_strange import loading_scan_restoring as restoring
from pmot.mot_strange.loading_scan_retention import OFF_CORE


def test_off_core_witness_requires_stability_in_all_three_directions(monkeypatch):
    center = np.array([.005, 0., 0.])
    zero = np.zeros(3)
    data = [None] * 12
    data[10] = [0., 0., 0., 1e-12]
    data[11] = [0.] * 15 + [.014]
    observed = (OFF_CORE, 2., center, zero, -1., center, .005, .005, 0.)
    monkeypatch.setattr(restoring.solver, "field", lambda r, d: np.array([1e-5, 0., 0.]))
    monkeypatch.setattr(restoring, "observe", lambda *args: observed)

    def acceleration(position, velocity, axis, data):
        return -stiffness * (position - center) - 100. * velocity

    monkeypatch.setattr(restoring.solver, "acceleration", acceleration)
    stiffness = np.array([1e4, 1e4, 1e4])
    evidence = restoring.check_off_core_pocket(observed, data)
    assert evidence["status"] == "stable_pocket_and_return_verified"
    assert len(evidence["return_probes"]) == 14

    stiffness[-1] = 0.
    evidence = restoring.check_off_core_pocket(observed, data)
    assert evidence["status"] == "no_converged_local_stability"


def test_off_core_return_requires_localization_at_later_time(monkeypatch):
    center = np.array([.005, 0., 0.])
    zero = np.zeros(3)
    data = [None] * 12
    data[10] = [0., 0., 0., 1e-12]
    data[11] = [0.] * 15 + [.014]
    observed = (OFF_CORE, 2., center, zero, -1., center, .005, .005, 0.)
    monkeypatch.setattr(restoring.solver, "field", lambda r, d: np.array([1e-5, 0., 0.]))
    monkeypatch.setattr(
        restoring.solver, "acceleration",
        lambda position, velocity, axis, data: -1e4 * (position - center) - 100. * velocity,
    )
    offset = center + np.array([.0006, 0., 0.])
    monkeypatch.setattr(
        restoring, "observe",
        lambda *args: (OFF_CORE, .2, offset, zero, -1., offset, .0056, .0056, 0.),
    )
    assert restoring.check_off_core_pocket(observed, data)["status"] == "no_converged_return"
