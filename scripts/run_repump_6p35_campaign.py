"""Two independent cooling-diameter cases with a 6.35 mm Gaussian repumper.

Keeps the established constant peak intensities and corrected steady-state
population-rate physics. Preserves every previous campaign and plot.
"""
import run_matched_repump_diameter_campaign as previous
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np

base = previous.base
CAMPAIGN = 'repump6p35_two_cooling_20261002_fixed_peak_intensities'
MATCHED = base.ROOT/'outputs/statistics/mot_multilevel_population_rate_v1'/previous.CAMPAIGN


@dataclass(frozen=True)
class Config(base.DiameterConfig):
    repump_diameter_mm: float = 6.35


@dataclass(frozen=True)
class Point(base.DiameterPoint):
    def config(self):
        return Config(cooling_power_w_per_beam=self.cooling_power_w,
            repump_power_w_per_beam=.000025,cooling_diameter_mm=self.coordinate,
            cooling_detuning_rad_per_s=self.detuning_rad_s)


def points():
    return [Point(i,d,.027*(d/12.7)**2,-2*np.pi*15e6) for i,d in enumerate((6.35,12.7))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def beams_for(cfg):
    return [replace(b,beam_radius_m=cfg.repump_diameter_mm*.0005) if b.family=='repump' else b
        for b in previous.beams_for(cfg)]


def prepare(cfg):
    data=list(base.prepare(cfg));data[1]=data[1].copy()
    for i,b in enumerate(beams_for(cfg)):
        data[1][i,1]=b.beam_radius_m**2
        data[1][i,2]=(np.pi*b.beam_radius_m**2/b.wavelength_m)**2
    return tuple(data)


def set_seeds(point):
    base.ce.PILOT_SEED=2026100201+10*point.index
    base.ce.PRODUCTION_SEED=2026100202+10*point.index


def initialize():
    base.rc.CAMPAIGN=CAMPAIGN
    base.ce.prepare=prepare;base.ce._ray_worker=base.diameter_ray_worker
    root=base.rc.paths()['statistics'];root.mkdir(parents=True,exist_ok=True)
    assert base.rc.source_digest()==base.CORE_SHA
    old_manifest=read(MATCHED/'campaign_manifest.json')
    assert sha(previous.__file__)==old_manifest['runner_sha256']
    assert sha(base.__file__)==old_manifest['baseline_adapter_sha256']
    assert read(MATCHED/'completion.json')['final_audit_passed']
    manifest=dict(source_sha256=base.CORE_SHA,runner_sha256=sha(__file__),
        helper_sha256=sha(previous.__file__),baseline_adapter_sha256=sha(base.__file__),
        cooling_diameters_mm=[6.35,12.7],repump_diameter_mm=6.35,
        repump_power_w_per_component=.000025,cooling_powers_w_per_component=[.00675,.027],
        cooling_detuning_hz=-15e6,repump_detuning_hz=0.,sampling_radius_mm=15.,
        pilot_discs=32,pilot_points_per_disc=64,production_points_per_disc=16,
        production_minimum_discs=64,variance_safety_factor=1.5,relative_95_half_width_target=.05,
        seeds=[dict(index=p.index,pilot=2026100201+10*p.index,production=2026100202+10*p.index) for p in points()],
        model='corrected Section-12 local-steady-state 24-state MOT; no recoil diffusion',
        qualification='Numerical convergence within local steady-state assumption; time-dependent population applicability remains separate',
        subsequent_task='After simulation and plot QA, review prior corrected 12.7 mm relationship findings for best supported powers/detunings and loading versus 27 mW cooling, 0.1 mW repump, -15 MHz cooling, zero repump detuning. Do not infer a joint optimum from independent scans.')
    file=root/'campaign_manifest.json'
    if file.exists():assert read(file)==manifest,'Recorded revision and revalidation required'
    else:base.rc.write_json(file,manifest)
    params=read(previous.OLD/'production_parameters.json')
    params['campaign']=CAMPAIGN
    params['repump_diameter_mm']=6.35
    params['repump_power_w_per_component']=.000025
    params['peak_intensity_policy']='Cooling and repump peaks unchanged from 27 mW / 0.1 mW at 12.7 mm'
    params['sampling']['pilot_seed']='per-case values in manifest'
    params['sampling']['production_seed']='per-case values in manifest'
    params['settings']=[dict(cooling_diameter_mm=p.coordinate,repump_diameter_mm=6.35,
        configuration=asdict(p.config()),beams=[asdict(b) for b in beams_for(p.config())]) for p in points()]
    base.rc.write_json(root/'parameters.json',params)
    for p in points():
        file=root/'parameters'/f'{p.index:03d}_compiled_inputs.npz'
        if not file.exists():
            file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(file,**{f'array_{i}':a for i,a in enumerate(prepare(p.config()))})
    return root


def validate(root):
    file=root/'validation/geometry_and_force.json'
    if file.exists():assert read(file)['passed'];return
    model,coil=base.build_rate_equation_model(),base.default_anti_helmholtz_config()
    rng=np.random.default_rng(2026100203);checks=high=0
    for p in points():
        cfg=p.config();beams=beams_for(cfg);data=prepare(cfg)
        for b in beams:
            diameter=p.coordinate if b.family=='cooling' else 6.35
            reference_power=.027 if b.family=='cooling' else .0001
            np.testing.assert_allclose(b.beam_radius_m,diameter*.0005,rtol=1e-14)
            np.testing.assert_allclose(2*b.power_w/(np.pi*b.beam_radius_m**2),
                2*reference_power/(np.pi*.00635**2),rtol=1e-14)
        r0=np.zeros(3);v0=np.array([3.,-2.,1.]);axis=np.array([0.,0.,1.])
        new=base.observable(r0,v0,axis,data);ref=base.observable(r0,v0,axis,base.prepare(base.DiameterConfig()))
        np.testing.assert_allclose(new[0],ref[0],rtol=1e-12,atol=1e-30)
        np.testing.assert_allclose(new[1],ref[1],rtol=1e-12,atol=1e-13)
        positions=np.vstack([r0,np.eye(3)*.003175,rng.uniform(-.006,.006,(16,3)),rng.uniform(-.02,.02,(12,3))])
        for r in positions:
            v=rng.uniform(-40,40,3);fast=base.observable(r,v,axis,data)
            if np.linalg.norm(r)>3*.003175:
                force,pop=base.high_precision_reference(model,beams,r,v,coil,cfg);high+=1
            else:
                reference=base.rate_equation_observable(model,beams,tuple(r),tuple(v),coil,cfg)
                force,pop=reference.force_n,reference.populations
            np.testing.assert_allclose(fast[0],force,rtol=3e-6,atol=3e-29)
            np.testing.assert_allclose(fast[1],pop,rtol=4e-6,atol=4e-10);checks+=1
    base.rc.write_json(file,dict(passed=True,source_sha256=base.CORE_SHA,local_checks=checks,
        high_precision_checks=high,peak_intensities_verified=True,central_force_and_populations_unchanged=True))


def audit_aggregate(folder,row):
    with np.load(folder/'disc_statistics.npz') as z:
        speed=z['velocity_m_per_s'];sigma=z['disc_cross_section_m2'];saved=z['disc_loading_atoms_per_s']
        np.testing.assert_array_equal(speed,base.ce.VELOCITIES)
        assert sigma.shape==(row['disc_count'],320) and np.all(np.isfinite(sigma))
        weighted=sigma*speed**3*np.exp(-speed**2/base.ce.THERMAL_SCALE_M2_PER_S2)
        values=base.ce.LOADING_RATE_PREFACTOR*np.trapezoid(np.column_stack((np.zeros(len(saved)),weighted)),np.r_[0.,speed],axis=1)
        np.testing.assert_allclose(values,saved,rtol=1e-12,atol=1e-8)
        mean,half=base.rc.cluster_interval(values)
        np.testing.assert_allclose([mean,half],[row['loading_rate_atoms_per_s'],row['loading_95_half_width_atoms_per_s']],rtol=1e-12)
    assert len(list((folder/'rays').glob('*.npz')))==row['sample_count']
    report=read(folder/'duration_audit.json')
    assert report['passed'] and report['cases'] and all(c['passed'] for c in report['cases'])
    for case in report['cases']:
        a=case['attempts'][-1]
        assert a['coarse_retained'] and a['fine_retained'] and a['position_difference_m']<1e-5
        assert max(a['coarse_final_window_max_radius_m'],a['fine_final_window_max_radius_m'])<=.002
    return dict(passed=True,retention_cases=len(report['cases']),
        files={name:sha(folder/name) for name in ('summary.json','disc_statistics.npz','duration_audit.json','geometry.npz','configuration.json')})


def render(rows):
    figure=base.rc.paths()['figures'];figure.mkdir(parents=True,exist_ok=True)
    curve=[read(MATCHED/'production'/p.key/'summary.json') for p in previous.points()]+[rows[0]]
    curve.sort(key=lambda r:r['cooling_diameter_mm'])
    baseline=[read(previous.OLD/'production'/p.key/'summary.json') for p in previous.points()]
    for coefficient in (False,True):
        y='loading_coefficient_m3_per_s' if coefficient else 'loading_rate_atoms_per_s'
        e='loading_coefficient_95_half_width_m3_per_s' if coefficient else 'loading_95_half_width_atoms_per_s'
        for overlay in (False,True):
            fig,ax=base.plt.subplots(figsize=(9,6),layout='constrained')
            if overlay:
                ax.errorbar([r['cooling_diameter_mm'] for r in baseline],[r[y] for r in baseline],
                    yerr=[r[e] for r in baseline],fmt='s--',capsize=3,label='Repump fixed at 12.7 mm')
            ax.errorbar([r['cooling_diameter_mm'] for r in curve],[r[y] for r in curve],
                yerr=[r[e] for r in curve],fmt='o-',capsize=3,label='Repump diameter = cooling diameter')
            if overlay:
                ax.errorbar([12.7],[rows[1][y]],yerr=[rows[1][e]],fmt='D',capsize=4,color='tab:green',label='Repump 6.35 mm; cooling 12.7 mm')
                ax.legend(fontsize=10)
            ax.set(xlabel='Cooling-beam Gaussian 1/e² diameter [mm]',
                ylabel='Loading coefficient R/n87 [m³/s]' if coefficient else 'Loading rate [atoms/s; historical vapor normalization]',
                title='Population-rate MOT · −15 MHz · fixed peak intensities\n15 mm sampling radius · 95% direction-disc cluster intervals')
            ax.grid(alpha=.25)
            name=('loading_coefficient' if coefficient else 'loading_rate')+('_comparison_with_6p35' if overlay else '_matched_with_6p35')
            for ext in ('png','pdf'):fig.savefig(figure/f'{name}.{ext}',dpi=180)
            base.plt.close(fig)
    base.rc.write_csv(base.rc.paths()['statistics']/'matched_curve_with_6p35.csv',
        [{k:r[k] for k in ('cooling_diameter_mm','repump_diameter_mm','loading_rate_atoms_per_s','loading_95_half_width_atoms_per_s','disc_count','points_per_disc')} for r in curve])


def run(args):
    root=initialize()
    base.rc.write_json(root/'process.json',dict(pid=os.getpid(),command=sys.argv,workers=args.workers,
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    validate(root)
    if args.stage=='validate':return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gate=root/'validation/fresh_workers.json'
        if not gate.exists():
            jobs=[]
            for p in points():
                set_seeds(p);pos,directions=base.ce.geometry(base.ce.PILOT_SEED,32,64,15.)
                jobs += [(p,d,j,v,pos[d,j],directions[d],prepare(p.config()))
                    for d,j in ((0,0),(7,10),(17,31),(31,63)) for v in (.25,10.,40.,120.)]
            checks=list(pool.map(previous.fresh_check,jobs))
            base.rc.write_json(gate,dict(passed=all(c['passed'] for c in checks),rows=checks))
        assert read(gate)['passed']
        plan_file=root/'sampling_plan.json'
        if not plan_file.exists():
            allocations=[];issues=[]
            for p in points():
                set_seeds(p);folder=root/'pilot'/p.key;start=time.perf_counter()
                row=base.ce.run_loading_point(p,'pilot',32,64,pool);capture=time.perf_counter()-start
                start=time.perf_counter()
                try:base.ce.duration_audits(p,'pilot',32,64,pool)
                except RuntimeError as error:issues.append(dict(index=p.index,stage='retention',error=str(error)))
                if not (folder/'timing.json').exists():base.rc.write_json(folder/'timing.json',dict(capture_wall_s=capture,retention_wall_s=time.perf_counter()-start))
                if not row['velocity_quadrature_pass']:issues.append(dict(index=p.index,stage='quadrature'))
                allocation=previous.allocate(p,root,dict(disc_count=64))
                allocation.pop('baseline_disc_count')
                allocation.update(pilot_seed=base.ce.PILOT_SEED,production_seed=base.ce.PRODUCTION_SEED,
                    repump_diameter_mm=6.35)
                allocations.append(allocation)
                base.rc.write_json(root/'pilot_progress.json',dict(allocations=allocations,issues=issues))
                print(json.dumps(allocation),flush=True)
            assert not issues,'Pilot gate failed; inspect retained evidence before production'
            gate=root/'validation/full_fixed_comparison.json'
            if not gate.exists():
                jobs=[(p,root/'pilot'/p.key/'rays'/f'd{d:03d}_p{j:03d}.npz',prepare(p.config()))
                    for p in points() for d,j in ((0,0),(17,31),(31,63))]
                checks=list(pool.map(base.fixed_comparison,jobs))
                base.rc.write_json(gate,dict(passed=all(not c['differences'] for c in checks),rows=checks))
            assert read(gate)['passed']
            base.rc.write_json(plan_file,dict(allocations=allocations,independent_of_pilot=True,
                independent_case_seeds=True,relative_95_half_width_target=.05,variance_safety_factor=1.5,
                frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                estimated_total_hours=[sum(a['estimated_hours'][i] for a in allocations) for i in (0,1)]))
        plan=read(plan_file);assert read(root/'validation/full_fixed_comparison.json')['passed']
        rows=[];issues=[];audits=[]
        for p,a in zip(points(),plan['allocations']):
            set_seeds(p);assert a['key']==p.key and a['production_seed']==base.ce.PRODUCTION_SEED
            folder=root/'production'/p.key;start=time.perf_counter()
            try:row=base.ce.run_loading_point(p,'production',a['disc_count'],16,pool)
            except RuntimeError as error:
                issues.append(dict(index=p.index,stage='capture',error=str(error)))
                base.rc.write_json(root/'production_issues.json',issues);continue
            capture=time.perf_counter()-start;start=time.perf_counter();retention=True
            try:base.ce.duration_audits(p,'production',a['disc_count'],16,pool)
            except RuntimeError as error:
                retention=False;issues.append(dict(index=p.index,stage='retention',error=str(error)))
                base.rc.write_json(root/'production_issues.json',issues)
            if not (folder/'timing.json').exists():base.rc.write_json(folder/'timing.json',dict(capture_wall_s=capture,retention_wall_s=time.perf_counter()-start))
            precision=row['loading_relative_half_width'] is not None and row['loading_relative_half_width']<=.05
            density=base.ce.LOADING_RATE_PREFACTOR/(4/(np.sqrt(np.pi)*base.ce.THERMAL_SCALE_M2_PER_S2**1.5))
            row.update(cooling_diameter_mm=p.coordinate,repump_diameter_mm=6.35,
                cooling_power_w_per_beam=p.cooling_power_w,repump_power_w_per_beam=.000025,
                loading_precision_pass=precision,retention_pass=retention,
                publication_status='validated' if precision and retention and row['velocity_quadrature_pass'] else 'qualification pending',
                loading_coefficient_m3_per_s=row['loading_rate_atoms_per_s']/density,
                loading_coefficient_95_half_width_m3_per_s=row['loading_95_half_width_atoms_per_s']/density)
            base.rc.write_json(folder/'summary.json',row)
            rows.append(row);base.rc.write_csv(root/'production_summary.csv',rows)
            if row['publication_status']=='validated':audits.append(audit_aggregate(folder,row))
            print(json.dumps(row),flush=True)
        ready=len(rows)==2 and all(r['publication_status']=='validated' for r in rows)
        if ready:
            render(rows)
            base.rc.write_json(root/'aggregate_audit.json',dict(passed=True,evidence=audits,source_sha256=base.CORE_SHA,runner_sha256=sha(__file__)))
        base.rc.write_json(root/'completion.json',dict(computed=True,publication_ready=ready,
            visual_review_pending=True,optimal_settings_review_pending=True,issues=issues,
            production_finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('validate','all'))
    parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    try:run(args)
    except Exception as error:
        base.rc.write_json(base.rc.paths()['statistics']/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
