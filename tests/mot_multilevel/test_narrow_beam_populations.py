"""Independent high-precision checks for highly unequal optical pumping rates."""
from dataclasses import replace
import numpy as np
import pytest
from pmot.configuration import default_mot_apparatus_config
from pmot.magnetic_fields import default_anti_helmholtz_config, anti_helmholtz_field_t
from pmot.mot_multilevel.accelerated import prepare, observable
from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.simulation import build_multilevel_mot_beams
from pmot.mot_multilevel.rate_equations import build_rate_equation_model, build_beam_transition_quantities, assemble_rate_matrix


@pytest.mark.parametrize('coordinate', [.004, .01, .013, .014])
def test_narrow_cooling_wings_match_full_matrix(coordinate):
    mp = pytest.importorskip('mpmath')
    cfg=replace(default_multilevel_mot_config(),cooling_power_w_per_beam=.027*(2/12.7)**2)
    app=default_mot_apparatus_config()
    app=replace(app,cooling=replace(app.cooling,beam_diameter_m=.002))
    beams=[replace(b,beam_radius_m=.00635) if b.family=='repump' else b
           for b in build_multilevel_mot_beams(app,cfg)]
    data=list(prepare(cfg));data[1]=data[1].copy()
    for i,b in enumerate(beams):
        data[1][i,1]=b.beam_radius_m**2
        data[1][i,2]=(np.pi*b.beam_radius_m**2/b.wavelength_m)**2
    r=np.full(3,coordinate);v=np.ones(3)
    field=np.array(anti_helmholtz_field_t(*r,default_anti_helmholtz_config()))
    axis=field/np.linalg.norm(field)
    model=build_rate_equation_model()
    w=build_beam_transition_quantities(model,beams,tuple(r),tuple(v),np.linalg.norm(field),tuple(axis),cfg).stimulated_coefficients_per_s
    fast=observable(r,v,axis,tuple(data))
    if coordinate==.014:
        assert np.all(np.sum(w,axis=(0,1))[3:]==0)
        np.testing.assert_array_equal(fast[0],np.zeros(3))
        np.testing.assert_array_equal(fast[1][3:8],np.full(5,.2))
        np.testing.assert_array_equal(assemble_rate_matrix(w.sum(axis=0),model.spontaneous_decay_matrix_per_s)@fast[1],np.zeros(24))
        return
    with mp.workdps(400):
        matrix=mp.matrix(24,24)
        for e in range(16):
            for g in range(8):
                up=sum(mp.mpf(float(w[b,e,g])) for b in range(12))
                down=up+mp.mpf(float(model.spontaneous_decay_matrix_per_s[g,e]))
                matrix[8+e,g]=up;matrix[g,8+e]=down
                matrix[g,g]-=up;matrix[8+e,8+e]-=down
        for g in range(24):matrix[23,g]=1
        rhs=mp.matrix(24,1);rhs[23]=1
        expected=np.array([float(x) for x in mp.lu_solve(matrix,rhs)])
    np.testing.assert_allclose(fast[1],expected,rtol=4e-6,atol=4e-10)
