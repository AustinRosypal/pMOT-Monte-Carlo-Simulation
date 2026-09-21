import ast
from pathlib import Path

import numpy as np
import pytest

from pmot.mot_multilevel.relationship_campaign import (
    loading_points,RAW_SATURATION,EFFECTIVE_SATURATION,LOADING_DETUNING,
    FORCE_DETUNING,REFERENCE_ISAT,BEAM_RADIUS,REFERENCE_GAMMA,
    cluster_interval,parabolic_minimum)
from pmot.mot_multilevel.campaign_execution import geometry,PILOT_SEED,PRODUCTION_SEED


def test_exact_historical_coordinates_and_independent_scans():
    assert len(loading_points())==75
    assert len(RAW_SATURATION)==24
    assert EFFECTIVE_SATURATION==tuple(np.arange(.25,5.001,.25))
    assert LOADING_DETUNING==tuple(np.arange(-.5,-6.001,-.25))
    assert len(FORCE_DETUNING)==111 and FORCE_DETUNING[-1]==-6
    archive=Path(__file__).resolve().parents[2]/'src/pmot/mot_error/refined_relationship_campaign.py'
    tree=ast.parse(archive.read_text(encoding='utf-8'))
    raw=next(n.value for n in tree.body if isinstance(n,ast.AnnAssign) and n.target.id=='RAW_SATURATION_VALUES')
    assert RAW_SATURATION==ast.literal_eval(raw)
    for point in loading_points():
        s0=2*point.cooling_power_w/(np.pi*BEAM_RADIUS**2*REFERENCE_ISAT)
        if point.study=='02_raw_saturation':
            assert s0==pytest.approx(point.coordinate)
        if point.study=='03_effective_saturation':
            assert s0/(1+4*(point.detuning_rad_s/REFERENCE_GAMMA)**2)==pytest.approx(point.coordinate)
        if point.study in ('01_radius','04_detuning'):
            assert point.cooling_power_w==.027


def test_nested_full_sphere_geometry_and_paired_radius_scaling():
    small,directions=geometry(PRODUCTION_SEED,64,32,3)
    big,directions2=geometry(PRODUCTION_SEED,64,64,30)
    centers=-.015*directions[:,None,:]
    np.testing.assert_array_equal(directions,directions2)
    np.testing.assert_allclose((small-centers)*10,big[:,:32]-centers,atol=1e-16)
    np.testing.assert_allclose(np.sum((big-centers)*directions[:,None,:],axis=2),0,atol=1e-16)
    normalized_area=np.sum((big-centers)**2,axis=2)/.03**2
    assert abs(np.mean(normalized_area)-.5)<.025
    assert np.all(np.min(directions,axis=0)<-.8) and np.all(np.max(directions,axis=0)>.8)
    pilot,_=geometry(PILOT_SEED,64,32,3)
    assert not np.array_equal(pilot,small)


def test_cluster_uncertainty_uses_discs_and_extremum_is_not_zero():
    means=np.array([1.,2.,3.,4.])
    m,h=cluster_interval(means)
    assert m==2.5
    assert h==pytest.approx(3.182446305284263*np.std(means,ddof=1)/2)
    x=np.arange(0,10,.25)
    v,interior=parabolic_minimum(x,(x-3.123)**2-7)
    assert interior and v==pytest.approx(3.123)


def test_new_campaign_does_not_import_archived_solver():
    package=Path(__file__).resolve().parents[2]/'src/pmot/mot_multilevel'
    for file in package.glob('*.py'):
        tree=ast.parse(file.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert 'mot_error' not in (node.module or '')
            elif isinstance(node,ast.Import):
                assert all('mot_error' not in a.name for a in node.names)


def test_workers_receive_precomputed_atomic_data_without_arc_calls():
    from pmot.mot_multilevel import campaign_execution
    import inspect
    for fn in (campaign_execution._ray_worker,campaign_execution._duration_worker):
        tree=ast.parse(inspect.getsource(fn))
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                       and n.func.id=='prepare' for n in ast.walk(tree))


def test_variance_allocation_separates_directions_from_impact_points():
    from pmot.mot_multilevel.campaign_execution import variance_components,allocation_from_variance
    pure_direction=np.repeat(np.arange(4.)[:,None],16,axis=1)
    between,within=variance_components(pure_direction)
    assert between==pytest.approx(np.var(np.arange(4.),ddof=1))
    assert within==0
    pure_within=np.tile([-1.,1.],(8,1))
    between,within=variance_components(pure_within)
    assert between==0 and within==2
    zero=np.zeros(3)
    n64=allocation_from_variance(1.,0.,50.,1.,zero,zero,64)
    n128=allocation_from_variance(1.,0.,50.,1.,zero,zero,128)
    assert n128<n64
    assert allocation_from_variance(1.,.5,0.,1.,zero,zero,64)==allocation_from_variance(1.,.5,0.,1.,zero,zero,256)
    with pytest.raises(RuntimeError,match='zero-capture'):
        allocation_from_variance(0.,0.,0.,1.,zero,zero)


def test_every_setting_has_its_own_pilot_before_allocation():
    from pmot.mot_multilevel.campaign_execution import pilot_points
    pilot=pilot_points()
    assert len(pilot)==75
    assert len({p.key for p in pilot})==75
    assert {p.key for p in pilot}=={p.key for p in loading_points()}
    assert pilot[8].key=='04_detuning/022'


def test_loading_only_allocation_does_not_require_unrequested_spectrum_precision():
    from pmot.mot_multilevel.campaign_execution import allocation_from_variance
    args=(1.,.1,1.,1.,np.array([.2]),np.array([1.]))
    assert allocation_from_variance(*args,16,False)<allocation_from_variance(*args,16,True)


def test_full_and_empty_cross_sections_have_valid_nonzero_boundary_intervals():
    from pmot.mot_multilevel.campaign_execution import cross_section_interval
    masks=np.ones((992,16,3));masks[:,:,0]=0;masks[:496,:,2]=0
    area=np.pi*.003**2
    _,mean,half,lower,upper=cross_section_interval(masks,area)
    assert mean[0]==0 and mean[1]==area
    assert upper[0]>0 and lower[1]<area
    assert np.all(lower<=mean) and np.all(upper>=mean)
    assert np.all(mean-lower>=0) and np.all(upper-mean>=0)
    assert mean[2]==pytest.approx(area/2)
    assert half[2]>0
