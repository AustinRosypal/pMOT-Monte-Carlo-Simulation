import numpy as np
import pytest
from pmot.mot_multilevel.accelerated import prepare,retention_audit,classify
from pmot.mot_multilevel.adaptive import integrate


@pytest.mark.parametrize('position,speed',[
    ([.015,0,0],5.),([.015,0,0],10.),([0,.015,0],10.),
    ([0,0,.015],10.),([.015,.004,0],5.)])
def test_adaptive_retention_matches_fixed_rk4(position,speed):
    data=prepare();r=np.array(position);v=-r/np.linalg.norm(r)*speed
    reference=retention_audit(r,v,2.5e-6,.1,data)
    for refinement in (1.,2.):
        result=integrate(r,v,.1,data,refinement,True)
        assert result[0]!=0
        assert (result[5]<=.002)==bool(reference[0])
        np.testing.assert_allclose(result[2],reference[2],atol=1e-5,rtol=0)
        np.testing.assert_allclose(result[3],reference[3],atol=1e-3,rtol=0)


def test_adaptive_short_timeout_is_not_escape_or_capture():
    data=prepare();r=np.array([.015,0.,0.]);v=np.array([-1.,0.,0.])
    result=integrate(r,v,1e-6,data)
    assert result[0]==-1
    assert result[1]==pytest.approx(1e-6)
    reference=classify(r,v,.5e-6,1e-6,data)
    np.testing.assert_allclose(result[2],reference[2],atol=1e-12)


def test_capture_edge_checks_recover_pilot_near_boundary_case(tmp_path):
    from pmot.mot_multilevel.campaign_execution import geometry,_ray_worker,PILOT_SEED
    from pmot.mot_multilevel.relationship_campaign import loading_points
    point=next(p for p in loading_points() if p.key=='01_radius/007')
    positions,directions=geometry(PILOT_SEED,16,16,30.)
    r,direction=positions[15,5],directions[15]
    data=prepare(point.config())
    # Both adaptive refinements agree here, but their edge is shifted.
    assert integrate(r,7.25*direction,.05,data,1.)[0]==1
    assert integrate(r,7.25*direction,.05,data,2.)[0]==1
    assert classify(r,7.25*direction,2.5e-6,.05,data)[0]==0
    path=tmp_path/'ray.npz'
    result=_ray_worker((point,15,5,r,direction,path,'test',data))
    assert result['status']=='resolved'
    with np.load(path) as z:
        assert z['codes'][np.flatnonzero(z['speeds']==7.25)[0]]==0
        assert set(z['evaluation_methods'])=={0,1}


def test_boundary_neighbourhood_includes_both_edges_of_an_island():
    from pmot.mot_multilevel.campaign_execution import boundary_node_indices
    assert boundary_node_indices([0,0,0,1,1,0,0,0])==set(range(1,7))
    assert boundary_node_indices([0,0,0])==set()
