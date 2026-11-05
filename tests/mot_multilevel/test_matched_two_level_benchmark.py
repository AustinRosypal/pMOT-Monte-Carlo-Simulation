from dataclasses import replace
import base64
import json
import shutil
import subprocess

import numpy as np
import pytest

from pmot.mot_multilevel import matched_benchmark as b
from pmot.mot_multilevel.accelerated import classify as original_classifier
from pmot.mot_multilevel.matched_capture import RayEvaluator,solve_ray,classify_numeric
from pmot.mot_multilevel.matched_campaign import reweight_counts
from pmot.mot_multilevel.rate_equations import build_rate_equation_model
from scipy.stats import t
from pmot.configuration import RB87_MASS_KG


def test_confirmed_grids_and_exact_original_configurations():
    points,scans=b.definitions()
    original=b.read_json(b.BENCHMARK/"campaign_manifest.json")
    assert len(points)==124
    assert [points[i].repump_power_mw for i in scans["repump_power"]]==list(b.REPUMP_POWERS_MW)
    assert [points[i].repump_detuning_mhz for i in scans["repump_detuning"]]==list(range(-10,11))
    for cfg,point in zip(original["configurations"],points[:64]):
        assert point.physical_values[:4]==(cfg["power_mw_per_beam"],cfg["detuning_mhz"],cfg["diameter_mm"],cfg["gradient_g_per_cm"])
        assert point.physical_values[4:]==(.1,12.7,0.)
        assert point.benchmark_index==cfg["index"]
    for index in scans["cooling_diameter_matched_repump"]:
        assert points[index].repump_diameter_mm==points[index].cooling_diameter_mm
        assert points[index].cooling_power_mw==10.
        assert points[index].repump_power_mw==.1


def test_saved_geometry_is_loaded_and_sampling_counts_match():
    positions,directions,digest=b.load_master_geometry()
    points,scans=b.definitions()
    assert positions.shape==(901,50,3)
    assert digest==b.sha256(b.BENCHMARK/"launch_geometry_master.csv")
    for point in points[:64]:
        saved=b.read_json(b.BENCHMARK/"production_plan.json")["searches"][str(point.index)]
        assert b.sample_count(point)==(saved["disc_count"],saved["points_per_disc"])
    for i,j in zip(scans["cooling_diameter"],scans["cooling_diameter_matched_repump"]):
        assert b.sample_count(points[i])==b.sample_count(points[j])
    np.testing.assert_array_equal(directions,directions[:,0:1,:].repeat(50,axis=1))


def test_independent_cooling_and_repump_geometry_and_powers():
    point=b.MatchedPoint(0,cooling_diameter_mm=5.,repump_diameter_mm=30.,repump_power_mw=20.,repump_detuning_mhz=3.)
    cfg,app,coil,beams=b.configuration(point);data=b.prepared(point)
    assert cfg.repump_detuning_rad_per_s==2*np.pi*3e6
    assert len(beams)==12
    for j,beam in enumerate(beams):
        if beam.family=="cooling":
            assert beam.power_w==.01 and beam.beam_radius_m==.0025
            assert beam.wavelength_m==780.241209686e-9
        else:
            assert beam.power_w==.02 and beam.beam_radius_m==.015
        assert data[1][j,0]==beam.power_w
        assert data[1][j,1]==beam.beam_radius_m**2
        assert data[1][j,3]==2*np.pi/beam.wavelength_m
    model=build_rate_equation_model()
    f0=(model.transition_ground_f==1)&(model.transition_excited_f==0)
    assert np.any(data[2][6:,f0]>0),"repump F=1 -> F'=0 channels must be retained"


def test_compiled_capture_history_matches_existing_physical_classifier():
    data=b.prepared(b.MatchedPoint(0))
    for r,v in (([.015,0.,0.],[-10.,0.,0.]),([.031,0.,0.],[1.,0.,0.])):
        r,v=np.asarray(r),np.asarray(v)
        old=original_classifier(r,v,5e-6,.01,data)
        new=classify_numeric(r,v,5e-6,.01,data)
        assert (old[0]==1)==(new[0]>0)
        assert (old[0]==0)==(new[0]==0)
        assert old[1]==new[1]
        np.testing.assert_allclose(old[2],new[4],rtol=1e-12,atol=1e-14)
        assert new[2]<=new[3]


class SyntheticEvaluator(RayEvaluator):
    def __init__(self,rule):
        self.rule=rule;self.records=[];self.cache={}
    def evaluate(self,speed,dt,duration):
        key=(float(speed),dt,duration)
        if key not in self.cache:
            row=np.zeros(15);row[:4]=[speed,dt,duration,self.rule(speed,dt,duration)]
            self.cache[key]=row;self.records.append(row)
        return self.cache[key]


def test_zero_threshold_finite_speed_capture_island_is_preserved():
    evaluator=SyntheticEvaluator(lambda speed,dt,duration:1 if 2<=speed<=5 else 0)
    result=solve_ray(evaluator)
    assert result["mode"]=="direct_zero_threshold_grid"
    assert np.all(result["mask"][(b.VELOCITIES>=2)&(b.VELOCITIES<=5)]==1)
    assert result["mask"][0]==0 and result["mask"][-1]==0
    assert result["nonmonotone"]


def test_positive_premise_failure_uses_converged_direct_mask():
    def rule(speed,dt,duration):
        if duration<.2:return 1 if speed<=10 else 0
        if duration==.2:return 1 if speed<=(10 if dt>3e-6 else 11) else 0
        return 1 if speed<=12 else 0
    result=solve_ray(SyntheticEvaluator(rule))
    assert result["mode"]=="direct_failed_positive_boundary_grid"
    np.testing.assert_array_equal(result["mask"],b.VELOCITIES<=12)


def test_unknown_zero_is_retained_but_positive_timeout_fails_closed():
    zero=SyntheticEvaluator(lambda speed,dt,duration:-1 if speed==0 else 0)
    result=solve_ray(zero)
    assert result["mask"][0]==-1 and np.all(result["mask"][1:]==0)
    assert (0.,.625e-6,2.) in zero.cache
    bad=SyntheticEvaluator(lambda speed,dt,duration:-1 if speed in (0,.5) else 0)
    with pytest.raises(RuntimeError,match="positive-flux"):
        solve_ray(bad)


def test_representative_direct_grid_does_not_fill_a_positive_capture_gap():
    evaluator=SyntheticEvaluator(lambda speed,dt,duration:1 if speed<=10 or 20<=speed<=22 else 0)
    result=solve_ray(evaluator,representative_grid=True)
    assert result["mode"]=="direct_representative_grid_disagrees_with_scalar"
    assert result["mask"][60]==0 and result["mask"][84]==1


def test_indeterminate_zero_has_exact_zero_flux_not_imputed_cross_section():
    payload=dict(velocity_m_per_s=[0.,.25,.5,.75],captured_counts=[[None,2,1,0],[None,1,0,0]],
                 points_per_disc=2,disc_area_m2=np.pi*.015**2)
    result=reweight_counts(payload)
    assert result["mean"]>0 and np.isfinite(result["lower"])
    original=dict(payload);original["captured_counts"]=[[0,2,1,0],[0,1,0,0]]
    assert reweight_counts(original)["mean"]==result["mean"]
    bad=dict(payload);bad["captured_counts"]=[[None,None,1,0],[None,1,0,0]]
    with pytest.raises(ValueError,match="positive-flux"):reweight_counts(bad)


@pytest.mark.skipif(shutil.which("node") is None,reason="Node required for browser reweighting parity")
def test_lossless_packed_browser_counts_reweight_like_python(tmp_path):
    from pmot.mot_multilevel.matched_report_html import TEMPLATE
    payload=dict(velocity_m_per_s=[0.,1.,2.,3.],captured_counts=[[None,5,2,0],[None,3,1,0],[None,6,3,0]],
                 points_per_disc=10,disc_area_m2=np.pi*.015**2)
    packed=dict(payload);rows=packed.pop("captured_counts")
    packed.update(disc_count=3,critical=float(t.ppf(.975,2)),
                  counts_base64=base64.b64encode(bytes(255 if v is None else v for row in rows for v in row)).decode())
    function="function rates"+TEMPLATE.split("function rates",1)[1].split("function fmt",1)[0]
    source=f"const DATA={{mass_kg:{RB87_MASS_KG}}},kb=1.380649e-23,pressure=1e-10*133.32;\n"+function
    source+="\nconsole.log(JSON.stringify([100,296.483,500].map(T=>rates("+json.dumps(packed)+",T))));"
    path=tmp_path/"packed-browser-math.js";path.write_text(source,encoding="utf-8")
    result=json.loads(subprocess.check_output([shutil.which("node"),str(path)],text=True))
    for T,row in zip((100.,296.483,500.),result):
        expected=reweight_counts(payload,T)
        for field in ("mean","lower","upper"):assert row[field]==pytest.approx(expected[field],rel=2e-13)


def test_comparison_views_map_both_diameter_variants_to_same_two_level_grid():
    from pmot.mot_multilevel.matched_reporting import report_bundle
    bundle=report_bundle(True)
    assert len(bundle["views"])==6
    assert not any(v["family"].startswith("repump_") for v in bundle["views"])
    fixed=next(v for v in bundle["views"] if v["family"]=="cooling_diameter")
    matched=next(v for v in bundle["views"] if v["family"]=="cooling_diameter_matched_repump")
    assert fixed["series"][0]["ids"]==matched["series"][0]["ids"]
    for i,j in zip(fixed["series"][1]["ids"],matched["series"][1]["ids"]):
        assert bundle["points"][i]["cooling_diameter_mm"]==bundle["points"][j]["cooling_diameter_mm"]
