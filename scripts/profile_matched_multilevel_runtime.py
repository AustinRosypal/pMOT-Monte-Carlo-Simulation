"""Read-only planning probe; never changes production evidence or its solver."""
from __future__ import annotations

from dataclasses import replace
from collections import Counter
from time import perf_counter
import json

import numpy as np
from scipy.stats import t

from pmot.mot_multilevel import matched_campaign as campaign
from pmot.mot_multilevel import matched_benchmark as b
from pmot.mot_multilevel.matched_capture import classify_numeric
from pmot.mot_multilevel.adaptive import integrate
from pmot.loading import VaporLoadingParameters


def main():
    manifest=b.read_json(b.STATISTICS/'campaign_manifest.json')
    assert manifest['source_sha256']==b.source_hashes()
    points,_=campaign.points_and_scans()
    data=b.prepared(points[0]);positions,directions,_=b.load_master_geometry()
    destination=b.STATISTICS/'validation/runtime_options_probe.json'
    folder=b.CACHE/'production/configuration_000/rays'
    cases=[]
    for d,p in ((0,0),(0,25),(40,0),(40,25),(80,0),(80,25)):
        with np.load(folder/f'd{d:04d}_p{p:02d}.npz') as z:
            mask=z['mask'];edges=np.flatnonzero(np.diff(mask.astype(int))!=0)
        speeds={0.,.25,10.,40.,80.}
        if len(edges):speeds.update(float(b.VELOCITIES[j]) for j in (edges[-1],edges[-1]+1))
        for speed in sorted(speeds):cases.append((points[0],d,p,speed))
    for point in (replace(points[0],cooling_diameter_mm=5.),
                  replace(points[0],cooling_diameter_mm=30.),
                  replace(points[0],repump_power_mw=20.),
                  replace(points[0],cooling_detuning_mhz=-40.)):
        for speed in (.25,10.,40.):cases.append((point,0,25,speed))
    # Compile before timing, including the adaptive solver's specialization.
    r=np.array([.015,0.,0.]);v=np.array([-10.,0.,0.])
    classify_numeric(r,v,5e-6,.001,data);integrate(r,v,.001,data,1.)
    print('Compilation finished; interleaved timings under current production load.',flush=True)
    rows=[]
    for i,(point,d,p,speed) in enumerate(cases):
        data=b.prepared(point);r=positions[d,p];v=speed*directions[d,p]
        row=dict(configuration=point.__dict__,disc=d,point=p,speed_m_per_s=speed,methods={})
        for name,value in (('rk4_5us',5e-6),('adaptive_1',1.),('rk4_2p5us',2.5e-6),
                           ('adaptive_2',2.),('rk4_10us',10e-6),('rk4_20us',20e-6)):
            started=perf_counter()
            result=(integrate(r,v,.2,data,value) if name.startswith('adaptive')
                    else classify_numeric(r,v,value,.2,data))
            elapsed=perf_counter()-started
            code=-1 if result[0]<0 else int(result[0]>0)
            row['methods'][name]=dict(wall_s=elapsed,code=code,trajectory_s=float(result[1]))
        rows.append(row)
        b.write_json(destination,dict(status='running',cases=rows))
        if (i+1)%10==0:print(f'Timed {i+1}/{len(cases)} trajectory cases',flush=True)
    timings={name:sum(r['methods'][name]['wall_s'] for r in rows) for name in rows[0]['methods']}
    differences={name:sum(r['methods'][name]['code']!=r['methods']['rk4_2p5us']['code'] for r in rows)
                 for name in timings}
    summaries=[];disc_ray_rates=[];mode_counts=Counter();mode_wall=Counter();mode_evaluations=Counter()
    vapor=VaporLoadingParameters();v=b.VELOCITIES
    weight=v**3*np.exp(-v*v/vapor.thermal_scale_m2_per_s2)
    area=np.pi*.015**2
    # First 100 complete direction discs. Planning diagnostics only; no sample
    # extension, stopping rule, or revised production interval is selected here.
    for d in range(100):
        ray_rates=[]
        for p in range(50):
            with np.load(folder/f'd{d:04d}_p{p:02d}.npz') as z:
                assert str(z['status'])=='completed'
                mask=z['mask'].astype(float);mask[0]=0.
                ray_rates.append(vapor.loading_rate_prefactor*area*np.trapezoid(mask*weight,v))
                mode=str(z['mode']);mode_counts[mode]+=1
                mode_wall[mode]+=float(z['elapsed_wall_s']);mode_evaluations[mode]+=len(z['evaluations'])
        disc_ray_rates.append(ray_rates)
    a=np.asarray(disc_ray_rates);mean=float(a.mean())
    within=float(np.var(a,axis=1,ddof=1).mean())
    between=max(0.,float(np.var(a.mean(axis=1),ddof=1))-within/50)
    for npoints in (10,16,20,25,50):
        nd=next(n for n in range(20,1500) if t.ppf(.975,n-1)*np.sqrt((between+within/npoints)/n)/mean<.1)
        summaries.append(dict(points_per_disc=npoints,estimated_discs_for_10_percent_baseline=nd,
                              estimated_total_rays=nd*npoints,
                              predicted_95_relative_half_width_at_200_discs=float(t.ppf(.975,199)*np.sqrt((between+within/npoints)/200)/mean)))
    output=dict(status='completed',created_utc=b.now(),scope='Planning-only baseline and selected extremes; production unchanged',
                cases=rows,timings_wall_s=timings,classifications_different_from_2p5us=differences,
                adaptive_pair_speedup=(timings['rk4_5us']+timings['rk4_2p5us'])/(timings['adaptive_1']+timings['adaptive_2']),
                cost_sample=dict(discs=100,points_per_disc=50,mode_counts=dict(mode_counts),
                                 mode_worker_wall_s=dict(mode_wall),mode_evaluations=dict(mode_evaluations)),
                sampling_planning=dict(baseline_mean=mean,within_disc_variance=within,between_disc_variance=between,
                                       estimated_allocations=summaries,
                                       limits='Only first 100 baseline discs; estimates are not a precision guarantee at other settings. A new reduced production design needs independent planning and predeclared sample counts.'),
                source_unchanged=manifest['source_sha256']==b.source_hashes())
    b.write_json(destination,output)
    print(json.dumps({k:v for k,v in output.items() if k!='cases'},indent=2),flush=True)


if __name__=='__main__':main()
