"""Resumable, velocity-resolved capture/loading campaign for Section 12.

The production mask contains directly evaluated nodes, never inferred scalar
thresholds. Each positive-speed node needs two agreeing, definitive numerical
classifications. Every attempted node is retained in a compressed audit file.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

from ..launch_geometry import sample_incident_disc_full_sphere, sample_disc_points
from ..loading import LOADING_RATE_PREFACTOR, THERMAL_SCALE_M2_PER_S2
from .accelerated import prepare, classify, retention_audit
from .adaptive import integrate as adaptive_integrate
from .relationship_campaign import (paths,write_json,write_csv,source_digest,
    loading_points,cluster_interval,run_force_sweep,RADII_MM,RAW_SATURATION,
    EFFECTIVE_SATURATION,LOADING_DETUNING,FORCE_DETUNING)

PILOT_SEED=2026091901
PRODUCTION_SEED=2026091902
PILOT_DISCS=16
PILOT_POINTS=16
# Uniform 0.25 m/s nodes include the historical 1--30 m/s plotting grid.
# Positive nodes below 1 m/s support the loading integral; zero flux anchors it.
VELOCITIES=np.arange(.25,80.001,.25)
# A diagnostic high-speed tail is checked for every ray. Any capture requests
# a fully populated 0.25 m/s extension, rather than integrating sparse probes.
TAIL_VELOCITIES=np.arange(85.,120.001,5.)
LADDER=((5e-6,2.5e-6,.05),(5e-6,2.5e-6,.1),
        (5e-6,2.5e-6,.2),(5e-6,2.5e-6,.4),
        (1.25e-6,.625e-6,1.),(1.25e-6,.625e-6,2.))
AUDIT_COLUMNS=('speed_m_per_s','dt_or_max_step_s','duration_s','code','elapsed_s',
               'x_m','y_m','z_m','vx_m_per_s','vy_m_per_s','vz_m_per_s','first_trapped_time_s')
_DATA_CACHE={}


def geometry(seed,discs,points,radius_mm):
    positions=np.empty((discs,points,3))
    directions=np.empty((discs,3))
    for d in range(discs):
        disc=sample_incident_disc_full_sphere(d,.015,np.random.default_rng(np.random.SeedSequence([seed,0,d])))
        samples=sample_disc_points(disc,points,radius_mm*.001,False,
            np.random.default_rng(np.random.SeedSequence([seed,1,d])))
        directions[d]=disc.incident_unit_vector
        positions[d]=[p.initial_position_m for p in samples]
    return positions,directions


def _atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
    for attempt in range(40):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt==39:
                raise
            sleep(.1)


def _resolve_fixed_node(position,direction,speed,data,records,methods):
    for coarse,fine,duration in LADDER:
        a=classify(position,speed*direction,coarse,duration,data)
        b=classify(position,speed*direction,fine,duration,data)
        for dt,result in ((coarse,a),(fine,b)):
            records.append([speed,dt,duration,float(result[0]),result[1],
                            *result[2],*result[3],result[4]])
            methods.append(0)
        if a[0]>=0 and a[0]==b[0]:
            return int(a[0])
    return -1


def _resolve_node(position,direction,speed,data,records,methods):
    for duration in (.05,.1,.2,.4,1.,2.):
        a=adaptive_integrate(position,speed*direction,duration,data,1.)
        b=adaptive_integrate(position,speed*direction,duration,data,2.)
        for refinement,result in ((1.,a),(2.,b)):
            records.append([speed,.0005/refinement,duration,float(result[0]),result[1],
                            *result[2],*result[3],result[4]])
            methods.append(1)
        if a[0]>=0 and a[0]==b[0]:return int(a[0])
        if a[0]>=0 and b[0]>=0:break
    return _resolve_fixed_node(position,direction,speed,data,records,methods)


def boundary_node_indices(codes):
    """Both sides and immediate neighbours of every resolved capture edge."""
    indices=set()
    for i in range(len(codes)-1):
        if codes[i]!=codes[i+1]:
            indices.update(range(max(0,i-1),min(len(codes),i+3)))
    return indices


def _ray_worker(payload):
    point,d,p,position,direction,path,mode,prepared_data=payload
    key=(point.cooling_power_w,point.detuning_rad_s)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key]=prepared_data
    data=_DATA_CACHE[key]
    path=Path(path)
    records=[];methods=[]
    started=perf_counter()
    # Resume a ray from already resolved nodes, preserving earlier attempts.
    existing={}
    if path.exists():
        with np.load(path) as old:
            records=old['evaluations'].tolist()
            methods=old['evaluation_methods'].tolist() if 'evaluation_methods' in old else [0]*len(records)
            existing=dict(zip(old['speeds'].tolist(),old['codes'].tolist()))
    speeds=list(VELOCITIES)+list(TAIL_VELOCITIES)
    codes=[]
    def save(status,error=''):
        _atomic_npz(path,speeds=np.array(speeds[:len(codes)]),codes=np.array(codes,dtype=np.int8),
            evaluations=np.asarray(records,dtype=float).reshape((-1,12)),
            evaluation_methods=np.array(methods,dtype=np.int8),
            position_m=position,direction=direction,status=np.array(status),
            error=np.array(error),
            elapsed_wall_s=np.array(perf_counter()-started))
    try:
        for index,speed in enumerate(speeds):
            code=existing.get(speed,-1)
            if code<0:
                code=_resolve_node(position,direction,speed,data,records,methods)
            codes.append(code)
            if index%64==63:
                save('running')
            if code<0:
                save('indeterminate')
                return {'disc':d,'point':p,'status':'indeterminate','speed':speed}
        # A pair of adaptive solvers can agree on a slightly shifted capture
        # edge. Check every edge with the validated fixed RK4 ladder, and
        # repeat if resolving an edge reveals additional adjacent nodes.
        fixed=set()
        for speed in existing:
            ev=[(r,m) for r,m in zip(records,methods) if r[0]==speed and m==0]
            if len(ev)>=2 and ev[-1][0][3]>=0 and ev[-1][0][3]==ev[-2][0][3]:fixed.add(speeds.index(speed))
        while True:
            needed=boundary_node_indices(codes)-fixed
            needed.update(i for i in range(len(VELOCITIES)-20,len(speeds)) if codes[i]==1 and i not in fixed)
            if not needed:break
            for i in sorted(needed):
                codes[i]=_resolve_fixed_node(position,direction,speeds[i],data,records,methods)
                fixed.add(i)
                if codes[i]<0:
                    save('indeterminate')
                    return {'disc':d,'point':p,'status':'indeterminate','speed':speeds[i]}
        tail_capture=any(c==1 for c in codes[len(VELOCITIES):])
        if tail_capture or any(c==1 for c in codes[len(VELOCITIES)-20:len(VELOCITIES)]):
            save('tail_censored')
            return {'disc':d,'point':p,'status':'tail_censored'}
        # Zero speed is not part of the cross-section spectrum. The exact
        # loading-integrand anchor g(0)=0 requires no classification at v=0.
        save('resolved')
        return {'disc':d,'point':p,'status':'resolved','wall_s':perf_counter()-started}
    except Exception as error:
        save('error',repr(error))
        return {'disc':d,'point':p,'status':'error','error':repr(error)}


def initialize_manifest():
    output=paths()['statistics']
    cfg={'schema':1,'model':'pmot.mot_multilevel Section-12 population rate equations',
         'source_sha256':source_digest(),'temperature':'deferred by user',
         'grids':{'radius_mm':RADII_MM,'raw_saturation':RAW_SATURATION,
                  'effective_saturation':EFFECTIVE_SATURATION,'loading_detuning':LOADING_DETUNING,
                  'force_detuning':FORCE_DETUNING},
         'loading_points':[asdict(p) for p in loading_points()],
         'velocities_m_per_s':VELOCITIES.tolist(),'tail_probes_m_per_s':TAIL_VELOCITIES.tolist(),
         'convergence_ladder':LADDER,'audit_columns':AUDIT_COLUMNS,
         'pilot_seed':PILOT_SEED,'production_seed':PRODUCTION_SEED,
         'pilot_discs':PILOT_DISCS,'pilot_points_per_disc':PILOT_POINTS,
         'allocation_policy':'revision 4: independent pilot every setting; nested variance allocation; 16 points/disc, at least 64 discs in increments of 16; common radius disc count; cross-section precision target only for the requested radius spectra',
         'trajectory_policy':{'method':'embedded Dormand-Prince 5(4); two refinements at every positive velocity node; fixed RK4 at every capture-edge neighbourhood and all disagreements',
             'relative_tolerances':[1e-5,2.5e-6],'position_absolute_tolerances_m':[1e-8,2.5e-9],
             'velocity_absolute_tolerances_m_per_s':[1e-5,2.5e-6],
             'max_steps_s':[.0005,.00025],'near_core_max_steps_s':[.0001,.00005],
             'evaluation_methods':{'0':'fixed RK4; second column is timestep','1':'adaptive DP5(4); second column is maximum step; refinement follows max step'},
             'retention_audits':'unchanged fixed RK4 dual-timestep long-duration audits'},
         'sampling_target':'10% relative 95% loading interval half-width; 5% of peak cross section in the main spectrum',
         'packages':{p:version(p) for p in ('numpy','scipy','numba','arc-alkali-rydberg-calculator')},
         'python':platform.python_version(),
         'zero_speed':'sigma(0) omitted; exact weighted-integrand anchor g(0)=0',
         'limits':'0.25 m/s velocity discretization to 80 m/s; 85--120 m/s tail probes every 5 m/s; no claim of mathematically excluding sub-grid islands'}
    cfg=json.loads(json.dumps(cfg))
    manifest=output/'campaign_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old!=cfg:
            raise RuntimeError('campaign manifest/code mismatch; preserve outputs and create an explicitly reviewed revision')
    else:
        write_json(manifest,cfg)
    return cfg


def cross_section_interval(masks,area):
    """Cluster interval with exact all-empty/full Bernoulli boundaries.

    Summing identical floating point areas can move the mean above the
    physical area by a few ulps. Identify boundaries from the binary evidence,
    rather than equality of that rounded mean with the area.
    """
    discs=masks.shape[0]
    disc_spectra=area*np.mean(masks,axis=1)
    mean,half=cluster_interval(disc_spectra)
    empty=np.all(masks==0,axis=(0,1));full=np.all(masks==1,axis=(0,1))
    mean=np.where(full,area,np.where(empty,0.,np.clip(mean,0.,area)))
    half=np.where(empty|full,0.,half)
    lower=np.maximum(0,mean-half);upper=np.minimum(area,mean+half)
    boundary_fraction=1-.025**(1/discs)
    upper=np.where(empty,area*boundary_fraction,upper)
    lower=np.where(full,area*(1-boundary_fraction),lower)
    return disc_spectra,mean,half,lower,upper


def _statistics(point,mode,discs,points):
    output=paths()['statistics']/mode/point.key
    # A captured tail on any ray requires dense extension for *all* rays.
    # Do not synthesize uncomputed high-speed masks for other rays.
    masks=np.empty((discs,points,len(VELOCITIES)),dtype=float)
    tail_any=False
    for d in range(discs):
        for p in range(points):
            with np.load(output/'rays'/f'd{d:03d}_p{p:03d}.npz') as data:
                if str(data['status'])!='resolved':
                    raise RuntimeError('unresolved ray cannot contribute to statistics')
                lookup=dict(zip(data['speeds'].tolist(),data['codes'].tolist()))
                masks[d,p]=[lookup[v] for v in VELOCITIES]
                tail_any=tail_any or any(lookup[v]==1 for v in TAIL_VELOCITIES)
    if tail_any:
        raise RuntimeError('capture above 80 m/s: dense high-speed extension required before loading integration')
    area=np.pi*(point.radius_mm*.001)**2
    disc_spectra,mean,half,lower,upper=cross_section_interval(masks,area)
    velocity=np.r_[0.,VELOCITIES]
    weighted=disc_spectra*VELOCITIES**3*np.exp(-VELOCITIES**2/THERMAL_SCALE_M2_PER_S2)
    loading=LOADING_RATE_PREFACTOR*np.trapezoid(np.column_stack((np.zeros(discs),weighted)),velocity,axis=1)
    rate,rate_half=cluster_interval(loading)
    # Nested 0.5/0.25 m/s integrals estimate velocity quadrature sensitivity.
    coarse=LOADING_RATE_PREFACTOR*np.trapezoid(
        np.column_stack((np.zeros(discs),weighted[:,1::2])),np.r_[0.,VELOCITIES[1::2]],axis=1)
    quadrature_error=abs(float(np.mean(coarse)-rate))
    relative_half=float(rate_half/rate) if rate>0 else None
    main=mean>=.2*np.max(mean) if np.max(mean)>0 else np.zeros(len(mean),dtype=bool)
    spectrum_precision=bool(np.max(mean)>0 and np.all(half[main]<=.05*np.max(mean)))
    summary={'study':point.study,'index':point.index,'coordinate':point.coordinate,
        'disc_count':discs,'points_per_disc':points,'sample_count':discs*points,
        'radius_mm':point.radius_mm,'loading_rate_atoms_per_s':float(rate),
        'loading_95_half_width_atoms_per_s':float(rate_half),
        'loading_relative_half_width':relative_half,
        'velocity_quadrature_difference_atoms_per_s':quadrature_error,
        'velocity_quadrature_pass':bool(rate>0 and quadrature_error<=.02*rate),
        'loading_precision_pass':bool(relative_half is not None and relative_half<=.1),
        'cross_section_precision_pass':spectrum_precision,
        'cross_section_relative_half_width_to_peak':float(np.max(half[main])/np.max(mean)) if np.any(main) else None,
        'maximum_capture_speed_on_grid_m_per_s':float(np.max(VELOCITIES[np.any(masks==1,axis=(0,1))])) if np.any(masks==1) else None,
        'mode':mode,'all_positive_nodes_dual_step_converged':True,
        'cross_section_precision_required':point.study=='01_radius',
        'nonmonotone_ray_count':int(np.count_nonzero(np.any(np.diff(masks,axis=2)>0,axis=2))),
        'publication_status':'pilot only' if mode=='pilot' else 'pending precision and duration gates'}
    write_json(output/'summary.json',summary)
    np.savez_compressed(output/'disc_statistics.npz',disc_cross_section_m2=disc_spectra,
        disc_loading_atoms_per_s=loading,velocity_m_per_s=VELOCITIES)
    write_csv(output/'cross_section.csv',[{'velocity_m_per_s':float(v),'cross_section_m2':float(s),
        'lower_95_m2':float(lo),'upper_95_m2':float(hi)} for v,s,lo,hi in zip(VELOCITIES,mean,lower,upper)])
    write_csv(output/'loading_by_disc.csv',[{'disc_index':i,'loading_rate_atoms_per_s':float(x)} for i,x in enumerate(loading)])
    return summary


def run_loading_point(point,mode,discs,points,executor):
    output=paths()['statistics']/mode/point.key
    seed=PILOT_SEED if mode=='pilot' else PRODUCTION_SEED
    positions,directions=geometry(seed,discs,points,point.radius_mm)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'geometry.npz',positions_m=positions,directions=directions)
    write_json(output/'configuration.json',{'point':asdict(point),'seed':seed,
        'discs':discs,'points_per_disc':points,'mode':mode})
    payloads=[]
    # Windows uses spawn, not fork. Pass immutable ARC-derived arrays from
    # the parent; workers must never open ARC's shared SQLite cache.
    prepared_data=prepare(point.config())
    for d in range(discs):
        for p in range(points):
            file=output/'rays'/f'd{d:03d}_p{p:03d}.npz'
            if file.exists():
                with np.load(file) as existing:
                    if str(existing['status'])=='resolved':
                        continue
            payloads.append((point,d,p,positions[d,p],directions[d],str(file),mode,prepared_data))
    print(f'{mode} {point.key}: {len(payloads)} of {discs*points} rays pending',flush=True)
    futures={executor.submit(_ray_worker,p):p for p in payloads}
    failures=[]
    for i,future in enumerate(as_completed(futures),1):
        result=future.result()
        if result['status']!='resolved':
            failures.append(result)
        if i%16==0 or result['status']!='resolved':
            write_json(paths()['statistics']/'progress.json',{'phase':mode,'point':asdict(point),
                'completed_rays':discs*points-len(payloads)+i,'total_rays':discs*points,
                'failures':failures,'pid':os.getpid()})
            print(f'{mode} {point.key}: {i}/{len(payloads)} new rays; failures={len(failures)}',flush=True)
    if failures:
        write_json(output/'failures.json',failures)
        raise RuntimeError(f'{point.key}: unresolved trajectories; aggregate results withheld')
    return _statistics(point,mode,discs,points)


def _duration_worker(payload):
    point,disc,pindex,speed,r,direction,coarse_dt,fine_dt,minimum_duration,data=payload
    attempts=[]
    for duration in (.1,.2,.4,1.,2.):
        if duration<minimum_duration:
            continue
        a=retention_audit(r,speed*direction,coarse_dt,duration,data)
        b=retention_audit(r,speed*direction,fine_dt,duration,data)
        error=float(np.linalg.norm(a[2]-b[2]))
        attempts.append({'duration_s':duration,'coarse_retained':bool(a[0]),'fine_retained':bool(b[0]),
            'coarse_final_position_m':a[2].tolist(),'fine_final_position_m':b[2].tolist(),
            'coarse_final_window_max_radius_m':float(a[4]),'fine_final_window_max_radius_m':float(b[4]),
            'position_difference_m':error})
        if a[0] and b[0] and error<1e-5:
            return {'disc':disc,'point':pindex,'speed_m_per_s':speed,'passed':True,'attempts':attempts}
        # A definitive outward escape after early capture disproves retention.
        if a[1]<duration-1e-12 or b[1]<duration-1e-12:
            break
    return {'disc':disc,'point':pindex,'speed_m_per_s':speed,'passed':False,'attempts':attempts}


def duration_audits(point,mode,discs,points,executor):
    output=paths()['statistics']/mode/point.key
    audit_path=output/'duration_audit.json'
    if audit_path.exists():
        report=json.loads(audit_path.read_text())
        if report['passed']:
            return report
        raise RuntimeError('saved longer-duration audit failed; scientific review required')
    jobs=[]
    prepared_data=prepare(point.config())
    for d in range(discs):
        candidates=[]
        for p in range(points):
            with np.load(output/'rays'/f'd{d:03d}_p{p:03d}.npz') as saved:
                captured=saved['speeds'][saved['codes']==1]
                if len(captured):
                    candidates.append((float(np.linalg.norm(saved['position_m'])),p,
                                       float(np.max(captured)),float(captured[len(captured)//2])))
        if not candidates:
            continue
        # The outermost captured ray in every direction disc is deliberately
        # challenging; check its highest and median captured grid speeds.
        _,p,high,middle=max(candidates)
        with np.load(output/'rays'/f'd{d:03d}_p{p:03d}.npz') as saved:
            for speed in sorted(set((high,middle))):
                ev=saved['evaluations'][saved['evaluations'][:,0]==speed]
                fine_dt=min(2.5e-6,float(np.min(ev[:,1])))
                duration=max(.1,float(np.max(ev[-2:,4]))+.025)
                jobs.append((point,d,p,speed,saved['position_m'],saved['direction'],
                             2*fine_dt,fine_dt,duration,prepared_data))
    results=list(executor.map(_duration_worker,jobs))
    report={'passed':bool(results) and all(r['passed'] for r in results),'cases':results,
            'selection':'outermost captured ray in every direction disc; highest and median captured grid speeds',
            'criterion':'both timesteps remain within 2 mm throughout final 25 ms; endpoints agree within 10 micrometers',
            'scope':'representative bounded-trajectory validation; not a lifetime guarantee'}
    write_json(audit_path,report)
    if not report['passed']:
        raise RuntimeError(f'{point.key}: longer-trajectory retention gate not established')
    return report


def pilot_points():
    all_points=loading_points()
    selected={('01_radius',0),('01_radius',4),('01_radius',7),
              ('02_raw_saturation',0),('02_raw_saturation',23),
              ('03_effective_saturation',19),
              ('04_detuning',0),('04_detuning',10),('04_detuning',22)}
    # Retain the original nine completed settings first, then fill the other
    # settings so allocation does not extrapolate variance between regimes.
    return ([p for p in all_points if (p.study,p.index) in selected]
            +[p for p in all_points if (p.study,p.index) not in selected])


def variance_components(samples):
    """Estimate Var(direction mean) and E[Var(impact point | direction)].

    For N discs and P points, Var(grand mean) = (between + within/P)/N.
    The observed variance of pilot disc means includes within/P_pilot;
    failing to remove that term overstates the required production disc count
    when production uses more impact points than the pilot.
    """
    samples=np.asarray(samples,dtype=float)
    if samples.ndim<2 or min(samples.shape[:2])<2:
        raise ValueError('variance components need at least two discs and two points')
    within=np.mean(np.var(samples,axis=1,ddof=1),axis=0)
    observed=np.var(np.mean(samples,axis=1),axis=0,ddof=1)
    between=np.maximum(0.,observed-within/samples.shape[1])
    return between,within


def allocation_from_variance(rate_mean,rate_between,rate_within,
                             spectrum_peak,spectrum_between,spectrum_within,
                             points_per_disc=64,require_spectrum=True):
    if rate_mean<=0 or spectrum_peak<=0:
        raise RuntimeError('zero-capture pilot cannot establish relative precision; enlarge pilot')
    rate_score=(rate_between+rate_within/points_per_disc)/(.1*rate_mean)**2
    spectrum_score=np.max(spectrum_between+spectrum_within/points_per_disc)/(.05*spectrum_peak)**2
    score=max(float(rate_score),float(spectrum_score)) if require_spectrum else float(rate_score)
    for discs in range(64,4097,16):
        # Fixed independent production size; 50% variance safety margin.
        if 1.5*student_t.ppf(.975,discs-1)**2*score<=discs:
            return discs
    raise RuntimeError('pilot implies more than 4096 discs; inspect allocation and rare-event evidence')


def point_variance_plan(row):
    output=paths()['statistics']/'pilot'/row['study']/f"{row['index']:03d}"
    n,p=row['disc_count'],row['points_per_disc']
    masks=np.empty((n,p,len(VELOCITIES)))
    for d in range(n):
        for j in range(p):
            with np.load(output/'rays'/f'd{d:03d}_p{j:03d}.npz') as saved:
                if str(saved['status'])!='resolved':
                    raise RuntimeError('allocation cannot use an unresolved pilot ray')
                masks[d,j]=saved['codes'][:len(VELOCITIES)]
    area=np.pi*(row['radius_mm']*.001)**2
    weight=VELOCITIES**3*np.exp(-VELOCITIES**2/THERMAL_SCALE_M2_PER_S2)
    rates=area*LOADING_RATE_PREFACTOR*np.trapezoid(
        np.concatenate((np.zeros((n,p,1)),masks*weight),axis=2),np.r_[0.,VELOCITIES],axis=2)
    rb,rw=variance_components(rates)
    sb,sw=variance_components(masks)
    spectrum=np.mean(masks,axis=(0,1))
    support=spectrum>=.2*np.max(spectrum)
    discs=allocation_from_variance(float(np.mean(rates)),rb,rw,float(np.max(spectrum)),sb[support],sw[support],
                                  16,row['study']=='01_radius')
    result={'point':f"{row['study']}/{row['index']:03d}",'disc_count':discs,'points_per_disc':16,
        'pilot_discs':n,'pilot_points_per_disc':p,'loading_mean_atoms_per_s':float(np.mean(rates)),
        'loading_between_disc_variance':float(rb),'loading_within_disc_variance':float(rw),
        'main_support_peak_capture_fraction':float(np.max(spectrum)),
        'pilot_loading_relative_half_width':row['loading_relative_half_width']}
    write_json(output/'variance_allocation.json',result)
    return result


def choose_sample_size(pilot_summaries):
    expected={p.key for p in loading_points()}
    actual={f"{r['study']}/{r['index']:03d}" for r in pilot_summaries}
    if actual!=expected:
        raise RuntimeError('all 75 settings require pilot evidence before production allocation')
    estimates=[point_variance_plan(row) for row in pilot_summaries]
    # Keep exactly the same normalized launch geometry and sample size across
    # all eight radii. Other independent parameter sweeps can use pointwise N.
    radius_discs=max(r['disc_count'] for r in estimates if r['point'].startswith('01_radius/'))
    allocations={r['point']:{'disc_count':radius_discs if r['point'].startswith('01_radius/') else r['disc_count'],
                            'points_per_disc':r['points_per_disc']} for r in estimates}
    plan={'schema':4,'point_allocations':allocations,'pilot_estimates':estimates,
          'paired_radius_disc_count':radius_discs,
          'precision_target':'10% relative 95% loading half-width; verified at every production point',
          'method':'independent per-setting pilot; nested variance components; 50% variance safety margin; minimum 64 direction clusters and 16 points/disc; 5%-of-peak cross-section target only for radius spectra',
          'production_seed':PRODUCTION_SEED,'pilot_seed':PILOT_SEED,
          'total_production_rays':sum(a['disc_count']*a['points_per_disc'] for a in allocations.values())}
    write_json(paths()['statistics']/'sampling_plan.json',plan)
    return plan


def plot_loading_results(rows):
    figures=paths()['figures']/'loading'
    figures.mkdir(parents=True,exist_ok=True)
    labels={'01_radius':('Sampling-disc radius [mm]','loading_vs_radius'),
            '02_raw_saturation':('Single-beam center saturation s₀','loading_vs_saturation'),
            '03_effective_saturation':('Single-beam center effective saturation','loading_vs_effective_saturation'),
            '04_detuning':('Cooling detuning Δ/Γ','loading_vs_detuning')}
    for study,(xlabel,name) in labels.items():
        group=[r for r in rows if r['study']==study]
        if not group:
            continue
        fig,ax=plt.subplots(figsize=(8.8,5.6),layout='constrained')
        ax.errorbar([r['coordinate'] for r in group],[r['loading_rate_atoms_per_s'] for r in group],
            yerr=[r['loading_95_half_width_atoms_per_s'] for r in group],fmt='o-',capsize=3)
        ax.set(xlabel=xlabel,ylabel='Loading rate [atoms/s]',title='Corrected population-rate MOT · 95% disc-cluster intervals')
        ax.grid(alpha=.25)
        if not all(r.get('publication_status')=='validated' for r in group):
            ax.text(.02,.98,'Convergence/precision qualification pending',transform=ax.transAxes,va='top',color='darkred')
        for ext in ('png','pdf'):fig.savefig(figures/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    radius=[r for r in rows if r['study']=='01_radius']
    if radius:
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
        for row in radius:
            table=np.genfromtxt(paths()['statistics']/'production'/row['study']/f"{row['index']:03d}"/'cross_section.csv',delimiter=',',names=True)
            mask=(table['velocity_m_per_s']>=1)&(table['velocity_m_per_s']<=30)
            t=table[mask];s=t['cross_section_m2']*1e6
            ax.errorbar(t['velocity_m_per_s'],s,
                yerr=np.array([s-t['lower_95_m2']*1e6,t['upper_95_m2']*1e6-s]),
                label=f"{row['radius_mm']:g} mm",lw=1,marker='.',ms=2,capsize=1)
        ax.set(xlabel='Launch speed [m/s]',ylabel='Capture cross section [mm²]',
               title='Corrected population-rate MOT · 95% disc-cluster intervals')
        ax.legend(ncol=2);ax.grid(alpha=.25)
        if not all(r.get('publication_status')=='validated' for r in radius):
            ax.text(.02,.98,'Convergence/precision qualification pending',transform=ax.transAxes,va='top',color='darkred')
        for ext in ('png','pdf'):fig.savefig(figures/f'cross_section_by_radius.{ext}',dpi=180)
        plt.close(fig)


def main(stage,workers):
    initialize_manifest()
    output=paths()['statistics']
    write_json(output/'process.json',{'pid':os.getpid(),'stage':stage,'workers':workers})
    if stage in ('validate','all'):
        import pytest
        validation=output/'validation'
        validation.mkdir(parents=True,exist_ok=True)
        root=Path(__file__).resolve().parents[3]
        exit_code=pytest.main([str(root/'tests/mot_multilevel'),str(root/'tests/shared/test_mot_magnetic_fields.py'),
                              '-q',f'--junitxml={validation / "pytest.xml"}'])
        write_json(validation/'test_status.json',{'exit_code':int(exit_code),'source_sha256':source_digest()})
        if exit_code:
            raise RuntimeError('campaign tests failed')
        from .campaign_validation import run_validation
        run_validation()
        if stage=='validate':return
    if stage=='all':run_force_sweep()
    if stage in ('pilot','production','all'):
        gate=output/'validation'/'event_and_trajectory_checks.json'
        if not gate.exists() or not json.loads(gate.read_text())['passed']:
            raise RuntimeError('validation must pass before capture statistics')
        test_gate=output/'validation'/'test_status.json'
        if not test_gate.exists() or json.loads(test_gate.read_text())!={'exit_code':0,'source_sha256':source_digest()}:
            raise RuntimeError('tests must pass for this exact source digest before capture statistics')
        adaptive_gate=output/'validation/hybrid_pilot_comparison.json'
        if not adaptive_gate.exists():raise RuntimeError('hybrid dense-pilot validation is required')
        audit=json.loads(adaptive_gate.read_text())
        if (audit['source_sha256']!=source_digest() or audit['mismatching_nodes'] or
                audit['rays']<500 or any(r['status']!='resolved' for r in audit['rows'])):
            raise RuntimeError('hybrid integrator validation failed or is stale')
        with ProcessPoolExecutor(max_workers=workers) as executor:
            if stage in ('pilot','all'):
                pilot=[]
                for p in pilot_points():
                    row=run_loading_point(p,'pilot',PILOT_DISCS,PILOT_POINTS,executor)
                    duration_audits(p,'pilot',PILOT_DISCS,PILOT_POINTS,executor)
                    pilot.append(row)
                plan=choose_sample_size(pilot)
            else:
                plan=json.loads((output/'sampling_plan.json').read_text())
            if stage=='pilot':return
            rows=[]
            for point in loading_points():
                allocation=plan['point_allocations'][point.key]
                row=run_loading_point(point,'production',allocation['disc_count'],allocation['points_per_disc'],executor)
                duration_audits(point,'production',allocation['disc_count'],allocation['points_per_disc'],executor)
                passed=(row['loading_precision_pass'] and row['velocity_quadrature_pass'] and
                        (point.study!='01_radius' or row['cross_section_precision_pass']))
                row['publication_status']='validated' if passed else 'precision target not met'
                write_json(output/'production'/point.key/'summary.json',row)
                rows.append(row)
                write_csv(output/'production_summary.csv',rows)
                plot_loading_results(rows)
            ready=all(r['publication_status']=='validated' for r in rows)
            write_json(output/'completion.json',{'all_75_loading_points_computed':True,
                'publication_ready':ready,'remaining':[] if ready else [f"{r['study']}/{r['index']:03d}" for r in rows if r['publication_status']!='validated'],
                'temperature':'deferred'})
    elif stage=='report':
        report={'manifest':str(output/'campaign_manifest.json')}
        for name in ('progress','sampling_plan','completion'):
            file=output/f'{name}.json'
            if file.exists():report[name]=json.loads(file.read_text())
        print(json.dumps(report,indent=2))
