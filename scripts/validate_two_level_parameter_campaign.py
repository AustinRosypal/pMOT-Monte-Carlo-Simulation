"""Independent force, speed-domain, and bounded-trajectory campaign checks."""
from dataclasses import replace
import json

import numpy as np

from pmot.state import AtomState
from pmot.mot_simple import parameter_campaign as campaign
from pmot.mot_simple.accelerated import classify_numeric, pack_parameters
from pmot.mot_simple.batched_sampling import classify_initial_conditions_batch
from pmot.mot_simple.simulation import mean_force_n, build_simple_mot_beams
from pmot.mot_simple.timeout_audit import DEFAULT_ADAPTIVE_AUDIT_LEVELS


def main():
    force_checks=[]
    for i in range(len(campaign.CONFIGURATIONS)):
        _,simple,coil,beams=campaign.build_configuration(campaign.point_for(i))
        optical=replace(simple,include_gravity=False)
        origin=mean_force_n(beams,AtomState((0.,0.,0.),(0.,0.,0.)),coil,optical)[0]
        if np.linalg.norm(origin)>1e-30: raise AssertionError("nonzero origin radiation force")
        for axis in range(3):
            r=np.zeros(3);r[axis]=.0001
            v=np.zeros(3);v[axis]=.05
            restoring=mean_force_n(beams,AtomState(tuple(r),(0.,0.,0.)),coil,optical)[0][axis]
            damping=mean_force_n(beams,AtomState((0.,0.,0.),tuple(v)),coil,optical)[0][axis]
            if not restoring<0 or not damping<0: raise AssertionError(f"non-restoring or anti-damping configuration {i}")
        force_checks.append(dict(index=i,status="passed"))
    cases=[0,campaign.SCAN_INDICES["power"][-1],campaign.SCAN_INDICES["detuning"][0],
           campaign.SCAN_INDICES["detuning"][-1],campaign.SCAN_INDICES["diameter"][0],
           campaign.SCAN_INDICES["diameter"][-1],campaign.SCAN_INDICES["gradient"][0],
           campaign.SCAN_INDICES["gradient"][-1]]
    speeds=np.arange(0.,80.01,.5)
    checks=[]
    for i in cases:
        _,simple,coil,beams=campaign.build_configuration(campaign.point_for(i))
        positions=np.repeat([[.015,0.,0.]],len(speeds),axis=0)
        velocities=np.column_stack((-speeds,np.zeros((len(speeds),2))))
        search=replace(campaign.search_for(1,1,0),max_simulation_time_s=.2)
        coarse=classify_initial_conditions_batch(beams,positions,velocities,coil,simple,search)
        fine=classify_initial_conditions_batch(beams,positions,velocities,coil,simple,replace(search,time_step_s=2.5e-6))
        coarse,fine=list(coarse),list(fine)
        for j in range(len(speeds)):
            for level in DEFAULT_ADAPTIVE_AUDIT_LEVELS:
                if (coarse[j].termination_reason not in ("timeout","non_finite")
                    and fine[j].termination_reason not in ("timeout","non_finite")
                    and coarse[j].trapped==fine[j].trapped): break
                for results,dt in ((coarse,level.coarse_time_step_s),(fine,level.fine_time_step_s)):
                    results[j]=classify_initial_conditions_batch(beams,positions[j:j+1],velocities[j:j+1],coil,simple,
                        replace(search,time_step_s=dt,max_simulation_time_s=level.duration_s))[0]
            if j>0 and (coarse[j].termination_reason in ("timeout","non_finite") or fine[j].termination_reason in ("timeout","non_finite") or coarse[j].trapped!=fine[j].trapped):
                raise AssertionError(f"unresolved representative positive speed {i}/{speeds[j]}")
        mask=np.asarray([x.trapped for x in fine])
        monotone=not np.any(np.diff(mask.astype(int))>0)
        if not monotone: raise AssertionError(f"positive scalar premise fails for representative axial ray at {i}")
        if np.any(mask[-10:]): raise AssertionError("speed domain does not extend beyond capture")
        # Suppress early trapping so the same physical trajectory is propagated
        # through a full 100 ms, then compare both timesteps and final confinement.
        data,c,a=pack_parameters(beams,coil,simple)
        captured_indices=np.flatnonzero(mask & (speeds>0))
        bound={}
        if len(captured_indices):
            speed=speeds[captured_indices[len(captured_indices)//2]]
            endpoints=[]
            for dt in (5e-6,2.5e-6):
                controls=np.asarray([dt,.100,.002,.03,1e9,1e9])
                result=classify_numeric(np.asarray([.015,0.,0.]),np.asarray([-speed,0.,0.]),data,c,a,controls)
                endpoints.append(result)
            r0,r1=endpoints[0][5],endpoints[1][5]
            v0,v1=endpoints[0][6],endpoints[1][6]
            if np.linalg.norm(r1)>.002 or np.linalg.norm(r0-r1)>1e-7 or np.linalg.norm(v0-v1)>1e-4:
                raise AssertionError(f"long-duration confinement/timestep check failed for {i}")
            bound=dict(speed_m_per_s=float(speed),duration_s=.1,final_radius_m=float(np.linalg.norm(r1)),
                       endpoint_position_difference_m=float(np.linalg.norm(r0-r1)),
                       endpoint_velocity_difference_m_per_s=float(np.linalg.norm(v0-v1)))
        checks.append(dict(index=i,speed_m_per_s=speeds.tolist(),
                           coarse_capture=[x.trapped if x.termination_reason not in ("timeout","non_finite") else None for x in coarse],
                           fine_capture=[x.trapped if x.termination_reason not in ("timeout","non_finite") else None for x in fine],
                           coarse_termination=[x.termination_reason for x in coarse],
                           fine_termination=[x.termination_reason for x in fine],
                           monotone=monotone,bounded_diagnostic=bound,status="passed"))
        print(f"Validated representative configuration {i}",flush=True)
    result=dict(status="passed",updated_utc=campaign.now(),force_checks=force_checks,
                representative_speed_and_confinement_checks=checks,
                note="Independent checks supplement, and do not replace, each production ray's endpoint/grid audit")
    campaign.write_json(campaign.STATISTICS/"preflight_validation.json",result)
    print(json.dumps(dict(status="passed",force_configurations=len(force_checks),representative_configurations=len(checks))))


if __name__=="__main__":
    main()
