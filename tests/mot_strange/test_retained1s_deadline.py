from pathlib import Path
import importlib
import numpy as np
import pytest
from numba import njit


@pytest.fixture
def revised(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    module=importlib.import_module('run_mot_strange_retained1s_deadline')
    monkeypatch.setattr(module,'OUTPUT',tmp_path)
    monkeypatch.setattr(module.model,'HORIZON',1.)
    return module


def test_one_second_endpoint_required(revised):
    model=revised.model
    def result(end):return (-1,end,np.array([.001,0.,0.]),np.zeros(3),.02,.001)
    assert model.outcome(result(1.))==model.CENTRAL
    assert model.outcome(result(2.))==model.UNKNOWN
    assert model.outcome(result(.5))==model.UNKNOWN


def test_one_second_final_window_observed(revised):
    @njit
    def step(r,v,axis,h,data):return r+h*v,v,axis
    @njit
    def escaped(r,v,data):return False
    @njit
    def field(r,data):return np.zeros(3)
    result=revised.model.numerical.propagate(np.zeros(3),np.array([.001,0.,0.]),.003,1.,None,np.empty(0),step,escaped,field)
    assert result[1]==1.
    assert result[5]==pytest.approx(.001,abs=1e-14)


def test_deadline_checkpoint_keeps_unknown_separate(revised,monkeypatch):
    monkeypatch.setattr(revised,'stopped',lambda:True)
    result=revised.worker((7,2,np.zeros(3),np.array([1.,0.,0.]),None))
    assert result['status']=='administratively_stopped'
    with np.load(revised.OUTPUT/'production/rays/d007_p002.npz') as a:
        assert str(a['status'])=='administratively_stopped'
        assert len(a['codes'])==0
        assert float(a['observation_horizon_s'])==1.
