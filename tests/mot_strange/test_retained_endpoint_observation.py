"""Analytic trajectories exercise final observations, independently of force QA."""
import importlib
from pathlib import Path
import numpy as np
import pytest
from numba import njit


@njit
def ballistic(r, v, axis, h, data):
    return r+h*v, v, axis


@njit
def zero_field(r, data):
    return np.zeros(3)


@njit
def no_wall(r, v, data):
    return False


@njit
def wall(r, v, data):
    return r[0] >= 1.


@pytest.fixture
def numerical(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('mot_strange_retained_numerics')


def test_rounding_remainder_endpoint_is_in_window_and_trace(numerical):
    # Ten additions of0.1 undershoot1.0. Observing a fixed count of nodes
    # misses the final propagated remainder; the analytic maximum is endpoint.
    result = numerical.propagate(np.array([1., 0., 0.]), np.array([1e5, 0., 0.]),
        .1, 1., (), np.array([1.]), ballistic, no_wall, zero_field)
    assert result[1] == 1.
    np.testing.assert_allclose(result[2], [100001., 0., 0.], rtol=0, atol=2e-10)
    assert result[5] == np.linalg.norm(result[2])
    assert result[6][0, 0] == 1.
    np.testing.assert_array_equal(result[6][0, 1:4], result[2])


def test_wall_at_final_endpoint_is_observed(numerical):
    result = numerical.propagate(np.zeros(3), np.array([1., 0., 0.]),
        .1, 1., (), np.array([1.]), ballistic, wall, zero_field)
    assert result[0] == 0
    assert result[1] == 1.


def test_final_observation_records_core_rule_without_early_exit(numerical):
    result = numerical.propagate(np.zeros(3), np.zeros(3), .001, .005, (),
        np.array([.005]), ballistic, no_wall, zero_field)
    assert result[0] == -1
    assert result[1] == .005
    assert result[4] == .005
    assert result[5] == 0.
