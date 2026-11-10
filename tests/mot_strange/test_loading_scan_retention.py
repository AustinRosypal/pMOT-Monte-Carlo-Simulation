import numpy as np
import pytest

from pmot.mot_strange.loading_scan_inputs import (
    ScanPoint, current_symmetric_surrogate, field_config, prepare,
)
from pmot.mot_strange.loading_scan_retention import CORE, WALL, agreed_category, observe


@pytest.fixture(scope="module")
def data():
    point = ScanPoint()
    field = field_config(point, current_symmetric_surrogate(), isotropic_axial_helicity=-1)
    return prepare(point, field)


def test_observation_does_not_terminate_at_first_core_event(data):
    r, v = np.zeros(3), np.zeros(3)
    horizon, window = .012, .005
    adaptive = observe(r, v, data, horizon, window, 2.)
    fixed = observe(r, v, data, horizon, window, 1., 2.5e-6)
    assert adaptive[1] == fixed[1] == horizon
    assert 0 < adaptive[4] < horizon
    assert adaptive[7] < .002
    assert agreed_category(adaptive, fixed) == CORE


def test_cell_wall_remains_terminal_before_retention_window(data):
    r, v = np.array([.014, 0., 0.]), np.array([80., 0., 0.])
    a = observe(r, v, data, .4, .1, 1.)
    b = observe(r, v, data, .4, .1, 2.)
    assert a[1] < .001 and b[1] < .001
    assert agreed_category(a, b) == WALL
