"""Equal cooling/repump diameters at fixed peak intensities, 5--25 mm.

Uses the validated population-rate kernel and uniform launch sampler unchanged.
Old outputs remain immutable; common random numbers support paired comparisons.
"""
import run_population_diameter_campaign as base
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
from scipy.stats import t

CAMPAIGN='matched_repump_diameter_20261002_fixed_peak_intensities'
OLD=base.ROOT/'outputs/statistics/mot_multilevel_population_rate_v1/cooling_diameter_20260930_fixed_intensity_baseline'
INDICES=tuple(range(2,10))
PILOT_SEED,PRODUCTION_SEED=2026093001,2026093002


@dataclass(frozen=True)
class MatchedPoint(base.DiameterPoint):
    def config(self):
        return base.DiameterConfig(cooling_power_w_per_beam=self.cooling_power_w,
            repump_power_w_per_beam=.0001*(self.coordinate/12.7)**2,
            cooling_detuning_rad_per_s=self.detuning_rad_s,cooling_diameter_mm=self.coordinate)


def beams_for(config):
    app=base.default_mot_apparatus_config()
    app=replace(app,cooling=replace(app.cooling,beam_diameter_m=config.cooling_diameter_mm*.001))
    return base.build_multilevel_mot_beams(app,config)


def prepare_matched(config):
    data=list(base.prepare(config));data[1]=data[1].copy()
    for i,b in enumerate(beams_for(config)):
        data[1][i,1]=b.beam_radius_m**2
        data[1][i,2]=(np.pi*b.beam_radius_m**2/b.wavelength_m)**2
    return tuple(data)


def points():
    return [MatchedPoint(i,base.DIAMETERS[i],.027*(base.DIAMETERS[i]/12.7)**2,-2*np.pi*15e6) for i in INDICES]


def initialize(workers):
    base.rc.CAMPAIGN=CAMPAIGN
    base.ce.PILOT_SEED=PILOT_SEED;base.ce.PRODUCTION_SEED=PRODUCTION_SEED
    base.ce.prepare=prepare_matched;base.ce._ray_worker=base.diameter_ray_worker
    root=base.rc.paths()['statistics'];root.mkdir(parents=True,exist_ok=True)
    assert base.rc.source_digest()==base.CORE_SHA
    old_plan=json.loads((OLD/'production_5mm_plan.json').read_text())
    assert hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()==old_plan['adapter_sha256']
    assert json.loads((OLD/'production_5mm_completion.json').read_text())['final_audit_passed']
    for name in ('test_status','event_and_trajectory_checks'):
        certificate=json.loads((OLD/'validation'/f'{name}.json').read_text())
        assert certificate['source_sha256']==base.CORE_SHA
        assert certificate.get('passed',certificate.get('exit_code')==0)
    manifest={'source_sha256':base.CORE_SHA,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'baseline_adapter_sha256':old_plan['adapter_sha256'],'baseline_root':str(OLD),
        'diameters_mm':[p.coordinate for p in points()],'repump_diameter':'equal to cooling diameter',
        'cooling_power_w':'.027*(diameter_mm/12.7)**2','repump_power_w':'.0001*(diameter_mm/12.7)**2',
        'cooling_detuning_hz':-15e6,'repump_detuning_hz':0.,'sampling_radius_mm':15.,
        'pilot_seed':PILOT_SEED,'production_seed':PRODUCTION_SEED,'pilot_discs':32,'pilot_points_per_disc':64,
        'production_points_per_disc':16,'loading_95_relative_half_width_target':.05,
        'variance_safety_factor':1.5,'production_minimum':'baseline per-point direction count',
        'paired_comparison':'same seed/direction/point coordinates; compare first baseline N discs, with paired Student-t difference intervals',
        'physics':'unchanged corrected Section-12 mean-force model; gravity on; recoil diffusion off',
        'normalization':'unchanged historical prefactor; also report normalized-Maxwell R/n87'}
    file=root/'campaign_manifest.json'
    if file.exists():assert json.loads(file.read_text())==manifest,'Recorded revision required before source/plan changes'
    else:base.rc.write_json(file,manifest)
    parameters=json.loads((OLD/'production_parameters.json').read_text())
    parameters['comparison_baseline']=str(OLD)
    parameters['repump_diameter_policy']='equal to cooling diameter'
    parameters['repump_peak_intensity_mw_per_cm2']=2*.0001/(np.pi*.00635**2)/10
    parameters['settings']=[]
    for p in points():
        parameters['settings'].append({'diameter_mm':p.coordinate,'configuration':asdict(p.config()),
                                      'beams':[asdict(b) for b in beams_for(p.config())]})
        file=root/'parameters'/f'{p.index:03d}_compiled_inputs.npz'
        if not file.exists():
            file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(file,**dict(zip(('directions','optical','dipole_strengths','frequency_offsets','ground_indices',
                'excited_indices','polarization_q','zeeman_coefficients','excited_decay_rates','spontaneous_matrix','controls'),prepare_matched(p.config()))))
    base.rc.write_json(root/'parameters.json',parameters)
    return root


def validate_local(root):
    file=root/'validation/geometry_and_force.json'
    if file.exists():
        saved=json.loads(file.read_text());assert saved['passed'] and saved['source_sha256']==base.CORE_SHA
        return
    model,coil=base.build_rate_equation_model(),base.default_anti_helmholtz_config()
    rng=np.random.default_rng(2026100201)
    baseline_data=base.prepare(base.DiameterConfig())
    count=0;high_precision=0
    for p in points():
        cfg=p.config();data=prepare_matched(cfg);beams=beams_for(cfg)
        assert len(beams)==12 and all(b.beam_radius_m==p.coordinate*.0005 for b in beams)
        expected=np.array([2*(.027 if b.family=='cooling' else .0001)/(np.pi*.00635**2) for b in beams])
        actual=np.array([2*b.power_w/(np.pi*b.beam_radius_m**2) for b in beams])
        np.testing.assert_allclose(actual,expected,rtol=1e-14)
        if p.coordinate==12.7:
            for a,b in zip(data,baseline_data):np.testing.assert_array_equal(a,b)
        center=base.observable(np.zeros(3),np.array([3.,-2.,1.]),np.array([0.,0.,1.]),data)
        old_center=base.observable(np.zeros(3),np.array([3.,-2.,1.]),np.array([0.,0.,1.]),baseline_data)
        np.testing.assert_allclose(center[0],old_center[0],rtol=1e-12,atol=1e-30)
        np.testing.assert_allclose(center[1],old_center[1],rtol=1e-12,atol=1e-13)
        waist=p.coordinate*.0005
        positions=np.vstack([np.zeros(3),np.eye(3)*waist*.5,rng.uniform(-waist,waist,(12,3)),rng.uniform(-.02,.02,(8,3))])
        for r in positions:
            v=rng.uniform(-40,40,3)
            fast=base.observable(r,v,np.array([0.,0.,1.]),data)
            if p.coordinate<=7 and np.linalg.norm(r)>3*waist:
                force,populations=base.high_precision_reference(model,beams,r,v,coil,cfg);high_precision+=1
            else:
                ref=base.rate_equation_observable(model,beams,tuple(r),tuple(v),coil,cfg)
                force,populations=ref.force_n,ref.populations
            np.testing.assert_allclose(fast[0],force,rtol=3e-6,atol=3e-29)
            np.testing.assert_allclose(fast[1],populations,rtol=4e-6,atol=4e-10)
            count+=1
    base.rc.write_json(file,dict(passed=True,source_sha256=base.CORE_SHA,local_checks=count,
        high_precision_checks=high_precision,baseline_12p7_arrays_identical=True,both_peak_intensities_fixed=True))


def fresh_check(payload):
    point,disc,pindex,speed,r,direction,data=payload
    adaptive=base.ce._resolve_node(r,direction,speed,data,[],[])
    fixed=base.ce._resolve_fixed_node(r,direction,speed,data,[],[])
    return dict(diameter_mm=point.coordinate,disc=disc,point=pindex,speed_m_per_s=speed,
                adaptive=adaptive,fixed=fixed,passed=adaptive==fixed and fixed>=0)


def allocate(point,root,old_allocation):
    folder=root/'pilot'/point.key
    values=np.empty((32,64));weight=base.ce.VELOCITIES**3*np.exp(-base.ce.VELOCITIES**2/base.ce.THERMAL_SCALE_M2_PER_S2)
    for d in range(32):
        for p in range(64):
            with np.load(folder/'rays'/f'd{d:03d}_p{p:03d}.npz') as z:
                assert str(z['status'])=='resolved'
                values[d,p]=np.pi*.015**2*base.ce.LOADING_RATE_PREFACTOR*np.trapezoid(np.r_[0,z['codes'][:320]*weight],np.r_[0,base.ce.VELOCITIES])
    mean=float(values.mean());assert mean>0,'Zero capture requires additional evidence before relative-precision allocation'
    between,within=base.ce.variance_components(values)
    score=float(between+within/16)/(.05*mean)**2
    n=max(old_allocation['disc_count'],64,int(math.ceil(1.5*1.96**2*score/16))*16)
    while 1.5*t.ppf(.975,n-1)**2*score>n:n+=16
    timing=json.loads((folder/'timing.json').read_text())
    hours=(timing['capture_wall_s']/(32*64)*n*16+timing['retention_wall_s']/32*n)/3600
    return dict(key=point.key,diameter_mm=point.coordinate,disc_count=n,points_per_disc=16,rays=n*16,
        baseline_disc_count=old_allocation['disc_count'],estimated_hours=[1.25*hours,1.75*hours],
        between_disc_variance=float(between),within_disc_variance=float(within),pilot_mean_atoms_per_s=mean)


def render(root,rows,comparisons):
    if len(rows)!=8:return
    figure=base.rc.paths()['figures'];figure.mkdir(parents=True,exist_ok=True)
    ready=all(r['publication_status']=='validated' for r in rows)
    for coefficient in (False,True):
        y='loading_coefficient_m3_per_s' if coefficient else 'loading_rate_atoms_per_s'
        err='loading_coefficient_95_half_width_m3_per_s' if coefficient else 'loading_95_half_width_atoms_per_s'
        for overlay in (False,True):
            fig,ax=base.plt.subplots(figsize=(9,6),layout='constrained')
            if overlay:
                old=[json.loads((OLD/'production'/p.key/'summary.json').read_text()) for p in points()]
                ax.errorbar([r['cooling_diameter_mm'] for r in old],[r[y] for r in old],yerr=[r[err] for r in old],fmt='s--',capsize=3,label='Repump diameter fixed at 12.7 mm')
            ax.errorbar([r['cooling_diameter_mm'] for r in rows],[r[y] for r in rows],yerr=[r[err] for r in rows],fmt='o-',capsize=3,label='Repump diameter = cooling diameter')
            ax.set(xlabel='Cooling-beam Gaussian 1/e² diameter [mm]',ylabel='Loading coefficient R/n87 [m³/s]' if coefficient else 'Loading rate [atoms/s; historical vapor normalization]',
                title='Population-rate MOT · −15 MHz · fixed peak intensities\n15 mm sampling radius · 95% direction-disc cluster intervals')
            if overlay:ax.legend()
            if not ready:ax.text(.02,.98,'Qualification pending',transform=ax.transAxes,va='top',color='darkred')
            ax.grid(alpha=.25)
            name=('loading_coefficient' if coefficient else 'loading_rate')+('_comparison' if overlay else '_matched_repump')
            for ext in ('png','pdf'):fig.savefig(figure/f'{name}.{ext}',dpi=180)
            base.plt.close(fig)


def run(args):
    root=initialize(args.workers)
    base.rc.write_json(root/'process.json',dict(pid=os.getpid(),command=sys.argv,workers=args.workers,
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    validate_local(root)
    if args.stage=='validate':return
    old_plan=json.loads((OLD/'production_5mm_plan.json').read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gate=root/'validation/fresh_workers.json'
        if not gate.exists():
            positions,directions=base.ce.geometry(PILOT_SEED,32,64,15.)
            jobs=[(point,d,p,v,positions[d,p],directions[d],prepare_matched(point.config()))
                  for point in points() for d,p in ((0,0),(17,31)) for v in (.25,10.,40.,120.)]
            checks=list(pool.map(fresh_check,jobs));base.rc.write_json(gate,dict(passed=all(c['passed'] for c in checks),rows=checks))
        assert json.loads(gate.read_text())['passed'],'Fresh-worker validation failed'
        plan_path=root/'sampling_plan.json'
        if not plan_path.exists():
            allocations=[];issues=[]
            for point,old in zip(points(),old_plan['allocations']):
                folder=root/'pilot'/point.key;start=time.perf_counter()
                row=base.ce.run_loading_point(point,'pilot',32,64,pool);capture=time.perf_counter()-start
                start=time.perf_counter();retention=True
                try:base.ce.duration_audits(point,'pilot',32,64,pool)
                except RuntimeError as error:
                    if 'retention gate' not in str(error) and 'longer-duration audit failed' not in str(error):raise
                    retention=False;issues.append(dict(diameter_mm=point.coordinate,error=str(error)))
                if not (folder/'timing.json').exists():base.rc.write_json(folder/'timing.json',dict(capture_wall_s=capture,retention_wall_s=time.perf_counter()-start))
                if not row['velocity_quadrature_pass']:issues.append(dict(diameter_mm=point.coordinate,error='Pilot velocity quadrature gate failed'))
                allocation=allocate(point,root,old);allocations.append(allocation)
                base.rc.write_json(root/'pilot_progress.json',dict(completed_settings=len(allocations),allocations=allocations,issues=issues))
                print(json.dumps(allocation),flush=True)
            if issues:raise RuntimeError('Pilot qualification issues retained in pilot_progress.json; production withheld')
            gate=root/'validation/full_fixed_comparison.json'
            if not gate.exists():
                jobs=[(point,root/'pilot'/point.key/'rays'/f'd{d:03d}_p{p:03d}.npz',prepare_matched(point.config()))
                      for point in points() for d,p in ((0,0),(17,31))]
                checks=list(pool.map(base.fixed_comparison,jobs))
                base.rc.write_json(gate,dict(passed=all(not c['differences'] for c in checks),rows=checks))
            assert json.loads(gate.read_text())['passed'],'Full fixed-RK4 comparison failed'
            base.rc.write_json(plan_path,dict(allocations=allocations,production_seed=PRODUCTION_SEED,
                independent_of_pilot=True,relative_95_half_width_target=.05,variance_safety_factor=1.5,
                estimated_total_hours=[sum(a['estimated_hours'][i] for a in allocations) for i in (0,1)]))
        plan=json.loads(plan_path.read_text())
        assert json.loads((root/'validation/full_fixed_comparison.json').read_text())['passed']
        rows=[];comparisons=[];issues=[]
        for point,a in zip(points(),plan['allocations']):
            assert point.key==a['key']
            folder=root/'production'/point.key;start=time.perf_counter()
            try:row=base.ce.run_loading_point(point,'production',a['disc_count'],16,pool)
            except RuntimeError as error:
                issues.append(dict(diameter_mm=point.coordinate,stage='capture',error=str(error)))
                base.rc.write_json(root/'production_issues.json',issues);continue
            capture=time.perf_counter()-start;start=time.perf_counter();retention=True
            try:base.ce.duration_audits(point,'production',a['disc_count'],16,pool)
            except RuntimeError as error:
                retention=False;issues.append(dict(diameter_mm=point.coordinate,stage='retention',error=str(error)))
                base.rc.write_json(root/'production_issues.json',issues)
            if not (folder/'timing.json').exists():base.rc.write_json(folder/'timing.json',dict(capture_wall_s=capture,retention_wall_s=time.perf_counter()-start))
            row['loading_precision_pass']=row['loading_relative_half_width'] is not None and row['loading_relative_half_width']<=.05
            row['retention_pass']=retention
            row['publication_status']='validated' if retention and row['loading_precision_pass'] and row['velocity_quadrature_pass'] else 'qualification pending'
            row['cooling_diameter_mm']=row['repump_diameter_mm']=point.coordinate
            row['cooling_power_w_per_beam']=point.cooling_power_w
            row['repump_power_w_per_beam']=point.config().repump_power_w_per_beam
            density=base.ce.LOADING_RATE_PREFACTOR/(4/(np.sqrt(np.pi)*base.ce.THERMAL_SCALE_M2_PER_S2**1.5))
            row['loading_coefficient_m3_per_s']=row['loading_rate_atoms_per_s']/density
            row['loading_coefficient_95_half_width_m3_per_s']=row['loading_95_half_width_atoms_per_s']/density
            base.rc.write_json(folder/'summary.json',row);rows.append(row)
            with np.load(folder/'geometry.npz') as new,np.load(OLD/'production'/point.key/'geometry.npz') as old:
                for key in ('positions_m','directions'):np.testing.assert_array_equal(new[key][:a['baseline_disc_count']],old[key])
            with np.load(folder/'disc_statistics.npz') as new,np.load(OLD/'production'/point.key/'disc_statistics.npz') as old:
                difference=new['disc_loading_atoms_per_s'][:a['baseline_disc_count']]-old['disc_loading_atoms_per_s']
                delta,half=base.rc.cluster_interval(difference)
                if point.coordinate==12.7:np.testing.assert_array_equal(difference,np.zeros_like(difference))
            comparisons.append(dict(diameter_mm=point.coordinate,paired_discs=a['baseline_disc_count'],
                loading_difference_atoms_per_s=float(delta),difference_95_half_width_atoms_per_s=float(half),
                lower_95_atoms_per_s=float(delta-half),upper_95_atoms_per_s=float(delta+half),
                scope='Matched subset of new production using all baseline discs; pointwise paired Student-t interval',
                qualified=row['publication_status']=='validated'))
            base.rc.write_csv(root/'production_summary.csv',rows)
            base.rc.write_csv(root/'paired_comparison.csv',comparisons)
            print(json.dumps(row),flush=True)
        render(root,rows,comparisons)
        ready=len(rows)==8 and all(r['publication_status']=='validated' for r in rows)
        base.rc.write_json(root/'completion.json',dict(computed=True,publication_ready=ready,visual_review_pending=True,
            production_finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),issues=issues,
            unqualified_points=[r['cooling_diameter_mm'] for r in rows if r['publication_status']!='validated']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('validate','all'))
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        base.rc.write_json(base.rc.paths()['statistics']/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
