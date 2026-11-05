"""Independent physical-kernel checks at the matched benchmark inputs."""
from __future__ import annotations

from dataclasses import asdict,replace
from time import perf_counter

import numpy as np

from ..configuration import HBAR_J_S
from . import matched_benchmark as b
from .accelerated import observable,step,retention_audit
from .capture import _rk4_step
from .campaign_validation import event_occupations
from .matched_capture import classify_numeric,record_numeric,RayEvaluator,NODE_LEVELS,trapped,states_agree
from .rate_equations import build_rate_equation_model,rate_equation_observable


def validate_local(point,*,events=False):
    cfg,app,coil,beams=b.configuration(point);data=b.prepared(point)
    model=build_rate_equation_model();axis=np.array([0.,0.,1.])
    checks=[]
    cases=((np.zeros(3),np.zeros(3)),(np.array([.001,-.0007,.0004]),np.array([1.,-.4,.7])),
           (np.array([.005,.002,-.001]),np.array([-8.,1.,-.3])))
    for r,v in cases:
        fast=observable(r,v,axis,data)
        ref=rate_equation_observable(model,beams,tuple(r),tuple(v),coil,cfg,
                                    store_rate_matrix=events,store_beam_transition_quantities=events)
        np.testing.assert_allclose(fast[0],ref.force_n,rtol=3e-6,atol=3e-29)
        np.testing.assert_allclose(fast[1],ref.populations,rtol=4e-6,atol=4e-10)
        np.testing.assert_allclose(fast[2],ref.beam_effective_scattering_rates_per_s,rtol=4e-6,atol=3e-8)
        assert abs(sum(fast[1])-1.)<1e-12 and min(fast[1])>=-cfg.population_absolute_tolerance
        row=dict(position_m=r.tolist(),velocity_m_per_s=v.tolist(),reference_force_n=list(ref.force_n),
                 compiled_force_n=fast[0].tolist(),populations_normalized=True,passed=True)
        if events:
            estimates=event_occupations(ref.rate_matrix_per_s,ref.populations,.001,64,2026110400+point.index)
            sem=np.std(estimates,axis=0,ddof=1)/8
            error=np.abs(estimates.mean(axis=0)-ref.populations)
            assert np.all(error<=6*sem+1e-6),"event-engine population comparison failed"
            reward=np.zeros((24,3))
            for j,beam in enumerate(beams):
                w=ref.beam_transition_quantities.stimulated_coefficients_per_s[j]
                momentum=HBAR_J_S*2*np.pi/beam.wavelength_m*np.asarray(beam.direction)
                reward[:8]+=w.sum(axis=0)[:,None]*momentum
                reward[8:]-=w.sum(axis=1)[:,None]*momentum
            forces=estimates@reward;fsem=forces.std(axis=0,ddof=1)/8
            assert np.all(np.abs(forces.mean(axis=0)-ref.force_n)<=6*fsem+1e-29),"event-engine force comparison failed"
            row.update(event_populations=estimates.mean(axis=0).tolist(),event_population_sem=sem.tolist(),
                       event_force_n=forces.mean(axis=0).tolist(),event_force_sem_n=fsem.tolist())
        checks.append(row)
    # Direct comparison of one full RK4 step, including all stage populations.
    r,v=cases[1]
    fast=step(r,v,axis,5e-6,data);ref=_rk4_step(r,v,5e-6,axis,model,beams,coil,cfg)
    for x,y in zip(fast,ref):np.testing.assert_allclose(x,y,rtol=3e-10,atol=2e-13)
    zero=observable(np.zeros(3),np.zeros(3),axis,data)[0]
    assert np.linalg.norm(zero)<1e-28,"nonzero symmetric origin radiation force"
    return dict(configuration=asdict(point),status="passed",local_checks=checks,rk4_stage_comparison="passed",
                event_method="64 independent 1 ms frozen-environment Gillespie histories; 6-SEM population/force gate" if events else None,
                controls=data[-1].tolist())


def validate_baseline_trajectories():
    point=b.MatchedPoint(0,benchmark_index=0);data=b.prepared(point)
    checks=[]
    for axis_index,speed in ((0,5.),(0,15.),(1,10.),(2,10.)):
        r=np.zeros(3);r[axis_index]=.015
        v=np.zeros(3);v[axis_index]=-speed
        a=classify_numeric(r,v,5e-6,.05,data);f=classify_numeric(r,v,2.5e-6,.05,data)
        assert a[0]>=0 and f[0]>=0 and (a[0]>0)==(f[0]>0),"representative capture timestep disagreement"
        original=[]
        for dt in (5e-6,2.5e-6):
            original.append(retention_audit(r,v,dt,.1,data))
        if a[0]>0:
            assert all(x[0] for x in original),"early capture did not remain in core during final 25 ms of 100 ms diagnostic"
        difference=float(np.linalg.norm(original[0][2]-original[1][2]))
        assert difference<1e-5,"longer trajectory endpoints are not timestep converged"
        checks.append(dict(position_m=r.tolist(),velocity_m_per_s=v.tolist(),
                     coarse=record_numeric(speed,5e-6,.05,a).tolist(),fine=record_numeric(speed,2.5e-6,.05,f).tolist(),
                     final_25ms_within_core=[bool(x[0]) for x in original],long_endpoint_difference_m=difference,status="passed"))
    return checks


def validate_point_retention(point,folder):
    """Independently continue representative captured positive-speed rays.

    These are validation diagnostics, not a replacement capture definition.
    The original early criterion and every authoritative mask stay unchanged.
    """
    data=b.prepared(point);cases=[]
    for path in sorted((folder/"rays").glob("*.npz")):
        with np.load(path) as z:
            captured=np.flatnonzero((z["mask"]>0)&(b.VELOCITIES>0))
            if not len(captured):continue
            j=captured[len(captured)//2];speed=float(b.VELOCITIES[j])
            r=z["position_m"].copy();direction=z["direction"].copy();v=speed*direction
            evaluator=RayEvaluator(r,direction,data,records=z["evaluations"])
            disc,pindex=int(z["disc"]),int(z["point"])
        for limit,coarse_step,fine_step in NODE_LEVELS:
            a=evaluator.evaluate(speed,coarse_step,limit);f=evaluator.evaluate(speed,fine_step,limit)
            if states_agree(a[3],f[3]):break
        if not (trapped(a[3]) and trapped(f[3])):
            b.write_json(folder/"retention_validation.json",dict(status="failed",disc=disc,point=pindex,
                         speed_m_per_s=speed,reason="captured node did not reproduce under the dual-step ladder"))
            raise RuntimeError("representative captured mask node did not reproduce")
        capture_time=max(float(a[4]),float(f[4]));first_duration=max(.1,capture_time+.05)
        horizons=sorted({first_duration,max(.2,first_duration*2),max(.4,first_duration*3),max(1.,first_duration*4),max(2.,capture_time+.1)})
        attempts=[];passed=False
        for duration in horizons:
            coarse=retention_audit(r,v,coarse_step,duration,data)
            fine=retention_audit(r,v,fine_step,duration,data)
            difference=float(np.linalg.norm(coarse[2]-fine[2]))
            attempts.append(dict(duration_s=duration,coarse_retained=bool(coarse[0]),fine_retained=bool(fine[0]),
                         coarse_max_final_window_radius_m=float(coarse[4]),fine_max_final_window_radius_m=float(fine[4]),
                         endpoint_difference_m=difference))
            if coarse[0] and fine[0] and difference<1e-5:
                passed=True;break
            if coarse[1]<duration-1e-10 or fine[1]<duration-1e-10:break
        cases.append(dict(disc=disc,point=pindex,speed_m_per_s=speed,passed=passed,attempts=attempts,
                          coarse_step_s=coarse_step,fine_step_s=fine_step,node_evidence=np.asarray(evaluator.records).tolist()))
        if not passed:
            b.write_json(folder/"retention_validation.json",dict(status="failed",cases=cases))
            raise RuntimeError(f"representative early capture failed the longer-trajectory gate for {point.key}, ray {disc}/{pindex}")
        if len(cases)==3:break
    result=dict(status="passed",created_utc=b.now(),cases=cases,
                interpretation="Representative positive-speed early captures remain in the 2 mm core throughout the last 25 ms of a converged longer fixed-RK4 diagnostic; original capture masks unchanged.",
                no_positive_early_capture_cases=not bool(cases))
    b.write_json(folder/"retention_validation.json",result)
    return result


def run_preflight(points=None,*,thorough=True):
    started=perf_counter()
    points=points or b.definitions()[0]
    # Basic input/reference checks cover every requested physical setting.
    # Event comparisons cover baseline and high/low repumper controls.
    event_indices={0}
    for attribute in ("repump_power_mw","repump_diameter_mm","repump_detuning_mhz"):
        event_indices.update((min(points,key=lambda p:getattr(p,attribute)).index,
                              max(points,key=lambda p:getattr(p,attribute)).index))
    results=[]
    for point in points if thorough else points[:1]:
        result=validate_local(point,events=point.index in event_indices)
        results.append(result)
        b.write_json(b.STATISTICS/"validation/local_reference_checks.json",dict(status="running",checks=results))
        print(f"Validated physical inputs {point.index}/{len(points)-1}",flush=True)
    trajectories=validate_baseline_trajectories()
    result=dict(status="passed",created_utc=b.now(),elapsed_wall_s=perf_counter()-started,
                scope="all_requested_settings" if thorough else "baseline_only",
                checked_configurations=len(results),local_reference_checks=results,
                trajectories=trajectories,atomic_model=b.model_notes(),
                kernel_source_sha256={name:b.sha256(b.ROOT/"src/pmot/mot_multilevel"/name)
                    for name in ("accelerated.py","rate_equations.py","atomic_structure.py","simulation.py","configuration.py")},
                limits="Local steady-state, incoherent-light mean force; frozen-environment event comparisons test rate/population kernel, not a full recoil trajectory or adiabatic-relaxation model.")
    b.write_json(b.STATISTICS/"validation/preflight.json",result)
    return result
