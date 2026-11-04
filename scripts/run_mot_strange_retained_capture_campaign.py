"""Validated central retention at 2 s; immutable revision after the event pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scipy.stats import t
import mot_strange_retained_capture as model

original, parent, ce = model.original, model.parent, model.ce
OUTPUT, OLD_OUTPUT = model.OUTPUT, model.OLD_OUTPUT
CATEGORIES = (model.CENTRAL, model.OFFCORE, model.TRANSIENT_WALL, model.OTHER,
              model.WALL_HISTORY_UNCERTAIN)
STAT_CACHE = {}


def initialize():
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'validation').mkdir(exist_ok=True)
    old = original.read(OLD_OUTPUT/'campaign_manifest.json')
    assert original.SOURCE_SHA == old['source_sha256']
    assert original.sha(parent.__file__) == old['finite_runner_sha256']
    assert original.sha(parent.diagnostic.__file__) == old['diagnostic_helper_sha256']
    assert original.sha(original.ROOT/'uv.lock') == old['lock_sha256']
    manifest = dict(old)
    manifest.update(campaign=OUTPUT.name, parent_campaign=str(OLD_OUTPUT),
        finite_runner_sha256=original.sha(__file__), retained_helper_sha256=original.sha(model.__file__),
        parent_finite_runner_sha256=old['finite_runner_sha256'],
        compiled_inputs_sha256=original.sha(parent.BASE_OUTPUT/'compiled_inputs.npz'),
        human_decision='core rule by 2 s AND continuously inside central 2 mm core over final 25 ms',
        final_retention_window_s=model.WINDOW,
        central_acceptance='core rule met; actual endpoint 2 s; maximum radius on [1.975,2] s <=2 mm',
        numerical_retained_endpoint_tolerance_m=model.ENDPOINT_TOLERANCE_M,
        fixed_retention_pairs_s=model.FIXED_PAIRS,
        fixed_ladder_interpretation='all new retention checks use full 2 s with early exit disabled',
        reuse='only provenance-matched dual-proven wall/no-core horizon results; never early core exits',
        outcomes={'-1':'numerically indeterminate', '0':'wall loss, no prior core event at saved settings',
                  '1':'central final-window retention after core rule', '2':'off-core endpoint retained at 2 s',
                  '3':'wall loss after a core event in both saved settings',
                  '4':'inside-core endpoint but final-window/core-rule requirement not met',
                  '5':'proven wall loss with disagreeing prior core-event history'},
        secondary_wall_history='Wall-history disagreements do not change proven wall loss; transient-event statistics are diagnostic, not loading.',
        convergence='dual full-observation adaptive settings; retained endpoints/window radii within 10 micrometers; fixed all outcome edges; three full fixed speed grids; longer retention audits',
        physical_source_unchanged=True)
    # JSON roundtrip gives tuples and saved lists one stable representation.
    manifest = json.loads(json.dumps(manifest))
    path = OUTPUT/'campaign_manifest.json'
    if path.exists():
        assert original.read(path) == manifest, 'Retained revision manifest changed'
    else:
        original.write_json(path, manifest)
        for source in (Path(__file__), Path(model.__file__), Path(parent.__file__),
                       original.ROOT/'tests/mot_strange/test_retained_capture_campaign.py'):
            shutil.copy2(source, OUTPUT/source.name)
    gate = OUTPUT/'validation/retained_tests.json'
    if not gate.exists():
        result = subprocess.run([sys.executable, '-m', 'pytest',
            'tests/mot_strange/test_retained_capture_campaign.py', '-q'],
            cwd=original.ROOT, capture_output=True, text=True)
        original.write_json(gate, dict(passed=result.returncode == 0,
            runner_sha256=original.sha(__file__), helper_sha256=original.sha(model.__file__),
            stdout=result.stdout, stderr=result.stderr))
    tests = original.read(gate)
    assert tests['passed'] and tests['helper_sha256'] == original.sha(model.__file__)


def disc_statistics(mode, d, areas):
    key = mode, d
    if key in STAT_CACHE:
        return STAT_CACHE[key]
    cache = OUTPUT/mode/'discs'/f'd{d:05d}.npz'
    if cache.exists():
        with np.load(cache) as a:
            value = {k:a[k] for k in a.files}
        STAT_CACHE[key] = value
        return value
    codes=[]
    for p in range(original.POINTS):
        with np.load(OUTPUT/mode/'rays'/f'd{d:03d}_p{p:03d}.npz') as a:
            assert str(a['status']) == 'resolved' and float(a['retention_window_s']) == .025
            np.testing.assert_array_equal(a['speeds'], model.SPEEDS)
            assert np.all(a['codes']>=0)
            assert not np.any(a['codes'][len(ce.VELOCITIES):] == model.CENTRAL)
            codes.append(a['codes'][:len(ce.VELOCITIES)])
    codes=np.asarray(codes)
    spectra=np.asarray([areas[d]*(codes==c).mean(axis=0) for c in CATEGORIES])
    v=ce.VELOCITIES; weight=v**3*np.exp(-v*v/original.THERMAL_SCALE_M2_PER_S2)
    rates=original.LOADING_RATE_PREFACTOR*np.trapezoid(
        np.column_stack((np.zeros(len(CATEGORIES)),spectra*weight)),np.r_[0.,v],axis=1)
    coarse=original.LOADING_RATE_PREFACTOR*np.trapezoid(
        np.r_[0.,(spectra[0]*weight)[1::2]],np.r_[0.,v[1::2]])
    value=dict(spectra_m2=spectra,rates_atoms_per_s=rates,coarse_core_rate=np.array(coarse),
               outcome_node_counts=np.array([np.count_nonzero(codes==c) for c in range(6)]))
    ce._atomic_npz(cache,**value); STAT_CACHE[key]=value
    return value


def statistics(mode,discs,positions,directions,areas,*,provisional=True):
    rows=[disc_statistics(mode,d,areas) for d in range(discs)]
    rates=np.asarray([x['rates_atoms_per_s'] for x in rows]); spectra=np.asarray([x['spectra_m2'] for x in rows])
    mean=rates.mean(axis=0); half=t.ppf(.975,discs-1)*rates.std(axis=0,ddof=1)/np.sqrt(discs)
    v=ce.VELOCITIES; weight=v**3*np.exp(-v*v/original.THERMAL_SCALE_M2_PER_S2)
    maximum=original.LOADING_RATE_PREFACTOR*np.pi*.015**2*np.trapezoid(np.r_[0.,weight],np.r_[0.,v])
    for i in range(len(CATEGORIES)):
        if np.all(rates[:,i]==0): half[i]=maximum*(1-.025**(1/discs))
    quadrature=abs(float(np.mean([x['coarse_core_rate'] for x in rows]))-mean[0])
    row=dict(at_utc=original.utc(),mode=mode,status='provisional final-central-retention rate; final gates pending',
        observation_horizon_s=2.,final_retention_window_s=.025,
        core_rule='2 mm core, 5 ms continuous residence OR two entries, plus continuous final25ms central residence',
        complete_direction_discs=discs,points_per_disc=20,resolved_rays=discs*20,
        loading_rate_atoms_per_s=float(mean[0]),loading_95_half_width_atoms_per_s=float(half[0]),
        loading_ci95_atoms_per_s=[max(0.,float(mean[0]-half[0])),float(mean[0]+half[0])],
        loading_relative_half_width=float(half[0]/mean[0]) if mean[0]>0 else None,
        loading_coefficient_m3_per_s=float(mean[0]/original.N87),
        loading_coefficient_95_half_width_m3_per_s=float(half[0]/original.N87),
        off_core_retention_integral_atoms_per_s=float(mean[1]),off_core_retention_95_half_width_atoms_per_s=float(half[1]),
        observed_transient_core_then_wall_integral_atoms_per_s=float(mean[2]),
        observed_transient_core_then_wall_95_half_width_atoms_per_s=float(half[2]),
        other_inside_core_endpoint_integral_atoms_per_s=float(mean[3]),
        wall_loss_ambiguous_core_history_integral_atoms_per_s=float(mean[4]),
        outcome_node_counts=np.sum([x['outcome_node_counts'] for x in rows],axis=0).tolist(),
        thermal_conditions=original.read(OUTPUT/'campaign_manifest.json')['thermal_conditions'],
        precision_pass=bool(mean[0]>0 and half[0]/mean[0]<=.05),
        quadrature_pass=bool(mean[0]>0 and quadrature<=.02*mean[0]),
        velocity_quadrature_difference_atoms_per_s=float(quadrature),
        interval='95% Student-t direction-disc cluster interval; interim estimates never control stopping',
        interpretation='core event plus final25ms central retention at2s, not equilibrium or infinite-time capture',
        off_core_interpretation='off-core endpoint retention at2s; not validated3D trapping',
        transient_interpretation='observed secondary core-event history; core-history disagreement remains separate; never counted as retained loading',
        scope='surrogate conventional MOT, interior15mm aperture; aperture/placement checks pending')
    folder=OUTPUT/mode
    original.write_json(folder/'latest_estimate.json',row)
    original.write_json(folder/'estimates'/f'discs_{discs:05d}.json',row)
    ce._atomic_npz(folder/'disc_statistics.npz',velocity_m_per_s=v,disc_cross_section_m2=spectra[:,0],
        disc_off_core_retention_cross_section_m2=spectra[:,1],disc_all_outcome_spectra_m2=spectra,
        disc_loading_atoms_per_s=rates[:,0],disc_all_outcome_rates_atoms_per_s=rates,disc_areas_m2=areas[:discs])
    print('LOADING UPDATE '+json.dumps(row),flush=True)
    return row


def probe(payload):
    name,r,direction,speed,expected,data=payload
    records=[]
    code=model.resolve(r,direction,speed,data,records)
    fixed=model.resolve_fixed(r,direction,speed,data,records)
    passed=code>=0 and fixed>=0 and model.primary(code)==model.primary(fixed)
    if expected is not None: passed=passed and model.primary(fixed)==expected
    return dict(name=name,code=code,fixed_code=fixed,expected_primary=expected,
                passed=bool(passed),evaluations=records)


def full_fixed_worker(payload):
    d,p,data=payload
    target=OUTPUT/'validation/full_fixed_grid'/f'd{d:03d}_p{p:03d}.npz'
    if target.exists():
        with np.load(target) as a:
            if str(a['status'])=='complete':
                return dict(disc=d,point=p,passed=bool(a['passed']),reused=True)
    with np.load(OUTPUT/'pilot/rays'/target.name) as a:
        expected=a['codes'].copy(); r=a['position_m']; direction=a['direction']
    records=[]; codes=[]; differences=[]; history_changes=[]
    if target.exists():
        with np.load(target) as a:
            records=a['evaluations'].tolist(); codes=a['codes'].tolist()
            differences=a['disagreements'].tolist(); history_changes=a['history_changes'].tolist()
    for i in range(len(codes),len(model.SPEEDS)):
        speed=float(model.SPEEDS[i]); code=model.resolve_fixed(r,direction,speed,data,records)
        codes.append(code)
        if code<0 or model.primary(code)!=model.primary(int(expected[i])):
            differences.append([speed,int(expected[i]),code])
        elif code!=expected[i]: history_changes.append([speed,int(expected[i]),code])
        if i%16==15 or i==len(model.SPEEDS)-1:
            ce._atomic_npz(target,status=np.array('complete' if i==len(model.SPEEDS)-1 else 'running'),
                speeds=model.SPEEDS[:len(codes)],codes=np.asarray(codes),
                evaluations=np.asarray(records).reshape((-1,14)),passed=np.array(not differences),
                disagreements=np.asarray(differences).reshape((-1,3)),history_changes=np.asarray(history_changes).reshape((-1,3)))
    return dict(disc=d,point=p,passed=not differences,node_count=len(codes),
                disagreements=differences,secondary_history_changes=history_changes)


def longer_worker(payload):
    d,p,speed,r,direction,data=payload
    results=[model.fast.integrate(r,speed*direction,2.2,data,f,True) for f in (2.,4.)]
    # This is a deliberately longer check, separate from the exact2s acceptance.
    bounded=all(int(x[0])==-1 and abs(x[1]-2.2)<1e-9 and x[4]>=0 and x[5]<=.002 for x in results)
    separation=float(np.linalg.norm(results[0][2]-results[1][2]))
    return dict(disc=d,point=p,speed_m_per_s=speed,duration_s=2.2,refinements=[2.,4.],
        passed=bool(bounded and separation<=1e-5),endpoint_difference_m=separation,
        results=[dict(raw_code=int(x[0]),time_s=float(x[1]),position_m=x[2].tolist(),
                      velocity_m_per_s=x[3].tolist(),first_core_time_s=float(x[4]),
                      final_window_maximum_radius_m=float(x[5])) for x in results])


def longer_retention(mode,positions,directions,impacts,data,pool):
    path=OUTPUT/mode/'retention_audit.json'
    if path.exists():
        assert original.read(path)['passed']; return
    targets=[]
    for d in range(len(positions)):
        candidates=[]
        for p in range(20):
            with np.load(OUTPUT/mode/'rays'/f'd{d:03d}_p{p:03d}.npz') as a:
                speeds=a['speeds'][a['codes']==model.CENTRAL]
                if len(speeds): candidates.append((impacts[d,p],p,float(speeds[-1]),float(speeds[len(speeds)//2])))
        if candidates:
            _,p,high,middle=max(candidates)
            targets.extend((d,p,s,positions[d,p],directions[d],data) for s in sorted({high,middle}))
    cases=[]
    for future in as_completed([pool.submit(longer_worker,p) for p in targets]):
        cases.append(future.result())
        original.write_json(path.with_name('retention_progress.json'),dict(completed=len(cases),
            requested=len(targets),failures=sum(not c['passed'] for c in cases)))
    report=dict(passed=bool(cases) and all(c['passed'] for c in cases),cases=cases,
        criterion='2.2s full trajectories; final25ms inside2mm; dual refined adaptive endpoints within10um',
        selection='outermost central-capture ray on every disc, highest and median accepted speeds')
    original.write_json(path,report)
    assert report['passed'], 'Longer central retention failed; preserve evidence'


def freeze_plan(pilot):
    path=OUTPUT/'sampling_plan.json'
    if path.exists(): return original.read(path)
    # Same independently seeded pilot variance policy, recomputed for new masks.
    old_output=parent.OUTPUT
    parent.OUTPUT=OUTPUT
    try: plan=parent.freeze_plan(pilot)
    finally: parent.OUTPUT=old_output
    plan['acceptance_definition']='core rule plus final25ms central retention at2s'
    original.write_json(path,plan)
    return plan


def phase(mode,count,seed,data,pool):
    try:
        return original.run_phase(mode,count,seed,data,pool)
    except Exception as error:
        # Record failure BEFORE executor shutdown can wait on already queued rays.
        original.write_json(OUTPUT/'failure.json',dict(at_utc=original.utc(),mode=mode,
            error=repr(error),traceback=traceback.format_exc(),queued_work_may_be_draining=True))
        raise


def run(args):
    initialize()
    original.OUTPUT=OUTPUT; original.worker=model.worker; original.statistics=statistics
    original.write_json(OUTPUT/'process.json',dict(pid=os.getpid(),command=sys.argv,
                        started_at_utc=original.utc(),workers=args.workers))
    data=parent.load_data()
    positions,directions,areas,impacts=original.geometry('pilot',32,original.PILOT_SEED)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gate=OUTPUT/'validation/fresh_retained_workers.json'
        if not gate.exists():
            with np.load(OLD_OUTPUT/'production/rays/d773_p006.npz') as a:
                failed_r,failed_direction=a['position_m'],a['direction']
            payloads=[('transient_counterexample',failed_r,failed_direction,25.125,model.WALL,data),
                ('pilot_central',positions[0,4],directions[0],16.,model.CENTRAL,data),
                ('pilot_offcore',positions[0,4],directions[0],19.75,model.OFFCORE,data),
                ('fast_wall',positions[0,0],directions[0],120.,model.WALL,data)]
            cases=list(pool.map(probe,payloads))
            original.write_json(gate,dict(passed=all(c['passed'] for c in cases),cases=cases))
        assert original.read(gate)['passed'], 'Fresh retention workers failed'
        print('FINAL-RETENTION FRESH WORKERS PASSED',flush=True)
        if args.stage=='validate': return
        pilot,positions,directions,areas,impacts=phase('pilot',32,original.PILOT_SEED,data,pool)
        assert pilot['quadrature_pass'], 'New retention pilot quadrature failed'
        gate=OUTPUT/'validation/full_fixed_grid.json'
        if not gate.exists():
            cases=[]
            for future in as_completed([pool.submit(full_fixed_worker,(d,p,data)) for d,p in ((0,0),(17,13),(0,4))]):
                cases.append(future.result())
                original.write_json(OUTPUT/'validation/full_fixed_progress.json',dict(completed=len(cases),requested=3,cases=cases))
            original.write_json(gate,dict(passed=all(c['passed'] for c in cases),cases=cases,
                node_count=3*len(model.SPEEDS),secondary_wall_history_separate=True))
        assert original.read(gate)['passed'], 'New retention full fixed-grid gate failed'
        longer_retention('pilot',positions,directions,impacts,data,pool)
        plan=freeze_plan(pilot)
        if args.stage=='pilot': return
        row,positions,directions,areas,impacts=phase('production',plan['disc_count'],original.PRODUCTION_SEED,data,pool)
        longer_retention('production',positions,directions,impacts,data,pool)
        row.update(status='production computed; final aperture/placement and visual gates pending',retention_pass=True)
        original.write_json(OUTPUT/'production/summary.json',row)
        original.write_json(OUTPUT/'completion.json',dict(computed=True,at_utc=original.utc(),
            precision_pass=row['precision_pass'],quadrature_pass=row['quadrature_pass'],retention_pass=True,
            aperture_and_launch_placement_pending=True,visual_review_pending=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('validate','pilot','all'))
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    original.OUTPUT=parent.BASE_OUTPUT
    with original.campaign_lock():
        try: run(args)
        except Exception as error:
            if not (OUTPUT/'failure.json').exists():
                original.write_json(OUTPUT/'failure.json',dict(at_utc=original.utc(),
                    error=repr(error),traceback=traceback.format_exc()))
            raise
