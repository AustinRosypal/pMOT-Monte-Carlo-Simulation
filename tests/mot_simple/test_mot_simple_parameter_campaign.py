from copy import deepcopy
import gzip
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest
from scipy.stats import t

from pmot.loading import VaporLoadingParameters, calculate_loading_rate_from_spectrum
from pmot.mot_simple import parameter_campaign as campaign


def payload():
    return dict(velocity_m_per_s=[0., 1., 2., 3.],
                captured_counts=[[None,8,4,0],[None,6,3,0],[None,9,5,0]],
                points_per_disc=10,disc_area_m2=np.pi*.015**2)


def test_four_independent_grids_and_exact_settings():
    assert campaign.GRIDS["power"] == tuple(range(2,41,2))
    assert len(campaign.CONFIGURATIONS) == 64
    assert len(campaign.SCAN_INDICES["diameter"]) == 12
    for family, indices in campaign.SCAN_INDICES.items():
        for i in indices:
            for j in range(4):
                if j != campaign.FIELDS[family]:
                    assert campaign.CONFIGURATIONS[i][j] == campaign.BASELINE[j]
            apparatus, simple, coil, beams=campaign.build_configuration(campaign.point_for(i))
            assert len(beams)==6
            assert all(b.wavelength_m==780.241209686e-9 for b in beams)
            assert simple.cooling_detuning_hz==campaign.CONFIGURATIONS[i][1]*1e6


@pytest.mark.parametrize("temperature", [100.,296.483,500.])
def test_reweighting_integrates_discs_then_calculates_student_t(temperature):
    data=payload()
    original=deepcopy(data)
    result=campaign.reweight_counts(data,temperature)
    vapor=VaporLoadingParameters(temperature_k=temperature)
    rates=[]
    for counts in data["captured_counts"]:
        sigma=np.asarray([0,*counts[1:]])/data["points_per_disc"]*data["disc_area_m2"]
        rates.append(calculate_loading_rate_from_spectrum(np.asarray(data["velocity_m_per_s"]),sigma,vapor).loading_rate_atoms_per_s)
    expected_mean=np.mean(rates)
    expected_half=t.ppf(.975,2)*np.std(rates,ddof=1)/np.sqrt(3)
    assert result["mean"]==pytest.approx(expected_mean)
    assert result["upper"]==pytest.approx(expected_mean+expected_half)
    assert result["lower"]==pytest.approx(expected_mean-expected_half)
    assert data==original
    assert campaign.reweight_counts(data,temperature,2e-10)["mean"]==pytest.approx(2*expected_mean)


def test_positive_speed_unknown_fails_closed():
    data=payload();data["captured_counts"][0][1]=None
    with pytest.raises(ValueError,match="positive-flux"):
        campaign.reweight_counts(data)


def test_raw_and_effective_power_axes_keep_separate_conversions():
    import matplotlib.pyplot as plt
    figures=campaign.plot_figures({0:payload()},save=False)
    try:
        for figure,field in zip(figures[:2],("s0","seff")):
            power_axis=figure.axes[0].child_axes[0]
            value=campaign.config_record(0)[field]
            assert power_axis._functions[0](value)==pytest.approx(10.)
            assert power_axis._functions[1](10.)==pytest.approx(value)
    finally:
        for figure in figures: plt.close(figure)


def test_plan_has_fixed_sample_size_and_excludes_pilot():
    results=[dict(elapsed_wall_time_s=10.,relative_half_width=.3)]*5
    plan=campaign.select_plan(results,20,24)
    assert plan["pilot_excluded_from_production"]
    assert min(plan["fixed_rays_per_configuration"].values())>=625
    assert plan["search"]["trajectory_backend"]=="numba"
    assert plan["projected_production_hours"]<24


def test_publication_preserves_full_audit_and_commits_metadata_last(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,"CACHE",tmp_path/"cache")
    monkeypatch.setattr(campaign,"STATISTICS",tmp_path/"published")
    monkeypatch.setattr(campaign,"FIGURES",tmp_path/"figures")
    source=campaign.paths_for(0,working=True)
    source.statistics.mkdir(parents=True);source.figures.mkdir(parents=True)
    audit=b'disc,point,all_nodes\n0,1,"[true, false, null]"\n'
    (source.statistics/"capture_endpoint_audit.csv").write_bytes(audit)
    (source.statistics/"capture_velocity_partial_samples.csv").write_text("duplicate partial checkpoint")
    campaign.write_json(source.metadata_json,dict(status="completed"))
    campaign.publish_point(0,"production",source)
    destination=campaign.paths_for(0)
    path=destination.statistics/"capture_endpoint_audit.csv.gz"
    with gzip.open(path,"rb") as stream:
        assert stream.read()==audit
    publication=campaign.read_json(destination.statistics/"publication.json")
    assert publication["files_sha256"][path.name]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert campaign.read_json(destination.metadata_json)["status"]=="completed"
    assert not (destination.statistics/"capture_velocity_partial_samples.csv").exists()


@pytest.mark.skipif(shutil.which("node") is None,reason="Node required for browser/Python parity check")
def test_browser_temperature_reweighting_matches_python(tmp_path):
    from pmot.mot_simple.parameter_campaign_html import TEMPLATE
    from pmot.configuration import RB87_MASS_KG
    function="function rates"+TEMPLATE.split("function rates",1)[1].split("function fmt",1)[0]
    p=payload()
    p["critical"]=float(t.ppf(.975,len(p["captured_counts"])-1))
    script=(f"const DATA={{mass_kg:{RB87_MASS_KG}}},kb=1.380649e-23,pressure=1e-10*133.32;\n"+function+
            "\nconsole.log(JSON.stringify([100,296.483,500].map(T=>rates("+json.dumps(p)+",T))));")
    path=tmp_path/"browser-reweight.js";path.write_text(script,encoding="utf-8")
    results=json.loads(subprocess.check_output([shutil.which("node"),str(path)],text=True))
    for temperature,result in zip((100.,296.483,500.),results):
        expected=campaign.reweight_counts(p,temperature)
        for field in ("mean","lower","upper"):
            assert result[field]==pytest.approx(expected[field],rel=2e-13)
