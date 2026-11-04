import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import t


def fixture_root(tmp_path, legacy=False):
    v=np.array([.125,.25,.375,.5]); area=np.array([2e-5,4e-5])
    manifest=dict(observation_horizon_s=2.,final_retention_window_s=.025,points_per_disc=1,
        positive_velocities_m_per_s=v.tolist(),outcomes={'1':'central final-window retention after core rule'},
        disc_radius_m=.015,source_sha256='fixture',thermal_conditions=dict(
            historical_loading_prefactor=911960.,thermal_scale_m2_per_s2=56670.,rb87_density_m3=5.45155695704e12))
    if legacy: del manifest['final_retention_window_s']
    (tmp_path/'campaign_manifest.json').write_text(json.dumps(manifest))
    folder=tmp_path/'production'; (folder/'rays').mkdir(parents=True)
    np.savez_compressed(folder/'geometry.npz',areas_m2=area)
    for d,codes in enumerate(([1,3,2,4],[0,1,5,0])):
        np.savez_compressed(folder/'rays'/f'd{d:03d}_p000.npz',status=np.array('resolved'),
            observation_horizon_s=np.array(2.),retention_window_s=np.array(.025),speeds=v,codes=codes)
    return v,area


def test_transient_events_excluded_and_disc_cluster_uncertainty(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    reducer=importlib.import_module('summarize_mot_strange_retained_discs')
    v,area=fixture_root(tmp_path)
    row,arrays=reducer.reduce_discs(tmp_path,2)
    weight=v**3*np.exp(-v*v/56670.)
    expected=911960.*.125*area*weight[:2]
    np.testing.assert_allclose(arrays['disc_loading_atoms_per_s'],expected)
    assert row['loading_rate_atoms_per_s']==pytest.approx(expected.mean())
    assert row['loading_95_half_width_atoms_per_s']==pytest.approx(t.ppf(.975,1)*abs(expected[1]-expected[0])/2)
    assert row['off_core_retention_integral_atoms_per_s']==pytest.approx(911960.*.125*area[0]*weight[2]/2)


def test_legacy_early_event_schema_rejected(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    reducer=importlib.import_module('summarize_mot_strange_retained_discs')
    fixture_root(tmp_path,legacy=True)
    with pytest.raises(KeyError): reducer.reduce_discs(tmp_path,2)
