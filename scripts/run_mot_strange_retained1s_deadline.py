"""User-authorized, separately reported 1 s continuation with a wall-clock cutoff.

The v4r1 force and numerical algorithms are imported unchanged. No 2 s masks
are reused as 1 s observations. The remaining preassigned direction discs form
a separate cohort; a deadline-truncated result is explicitly descriptive.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import shutil
import time
import traceback

import numpy as np
from scipy.stats import t as student_t
import mot_strange_retained_capture_v4r1 as model

ROOT = model.original.ROOT
OLD = model.OUTPUT
OUTPUT = OLD.with_name('calibrated_10_10_20Gpcm_full_sphere_r15mm_incell_20261025_retained1s_v5_deadline_20261028')
CONTROL = OUTPUT/'user_control.json'
KINDS = (1, 2, 3, 4, 5)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AdministrativeStop(Exception):
    pass


def stopped():
    deadline = datetime.fromisoformat(read(CONTROL)['hard_deadline_utc']).timestamp()
    # Leave five minutes for long in-flight numerical settings and finalization.
    return (OUTPUT/'STOP_REQUESTED.json').exists() or time.time() >= deadline-300


def check_stop():
    if stopped():
        raise AdministrativeStop('User wall-clock deadline or stop request')


def install_checks():
    model.HORIZON = 1.
    frozen_fast = model.fast
    frozen_budget = model.budget.retained_adaptive
    def integrate(*args, **kwargs):
        check_stop(); return frozen_fast.integrate(*args, **kwargs)
    def classify(*args, **kwargs):
        check_stop(); return frozen_fast.classify(*args, **kwargs)
    def budget(*args, **kwargs):
        check_stop(); return frozen_budget(*args, **kwargs)
    model.fast = SimpleNamespace(integrate=integrate, classify=classify, field=frozen_fast.field)
    model.budget.retained_adaptive = budget


def worker(payload):
    d, p, r, direction, data = payload
    target = OUTPUT/'production/rays'/f'd{d:03d}_p{p:03d}.npz'
    records, codes = [], []
    began = time.monotonic()
    def save(status, error=''):
        target.parent.mkdir(parents=True, exist_ok=True)
        model.ce._atomic_npz(target, status=np.array(status), error=np.array(error),
            speeds=model.SPEEDS[:len(codes)], codes=np.asarray(codes, dtype=np.int8),
            evaluations=np.asarray(records, dtype=float).reshape((-1, 14)),
            position_m=r, direction=direction, observation_horizon_s=np.array(1.),
            retention_window_s=np.array(.025), elapsed_wall_s=np.array(time.monotonic()-began))
    try:
        check_stop()
        if target.exists():
            with np.load(target) as a:
                assert float(a['observation_horizon_s']) == 1.
                np.testing.assert_array_equal(a['position_m'], r)
                np.testing.assert_array_equal(a['direction'], direction)
                if str(a['status']) == 'resolved':
                    return dict(disc=d, point=p, status='resolved')
                records = a['evaluations'].tolist(); codes = a['codes'].tolist()
        target.parent.mkdir(parents=True, exist_ok=True)
        for i in range(len(codes), len(model.SPEEDS)):
            check_stop()
            code = model.resolve(r, direction, float(model.SPEEDS[i]), data, records)
            codes.append(code)
            if i % 64 == 63 or code in (1, 2, 4): save('running')
            if code < 0:
                save('numerical_indeterminate'); return dict(disc=d, point=p, status='numerical_indeterminate')
        fixed = {i for i, speed in enumerate(model.SPEEDS) if model.existing_fixed(records, speed) >= 0}
        while True:
            needed = model.ce.boundary_node_indices([model.primary(c) for c in codes])-fixed
            if not needed: break
            for i in sorted(needed):
                check_stop()
                codes[i] = model.resolve_fixed(r, direction, float(model.SPEEDS[i]), data, records)
                fixed.add(i); save('running')
                if codes[i] < 0:
                    save('numerical_indeterminate'); return dict(disc=d, point=p, status='numerical_indeterminate')
        if any(c == 1 for c in codes[len(model.ce.VELOCITIES)-20:]):
            save('tail_censored'); return dict(disc=d, point=p, status='tail_censored')
        save('resolved'); return dict(disc=d, point=p, status='resolved')
    except AdministrativeStop as error:
        save('administratively_stopped', str(error))
        return dict(disc=d, point=p, status='administratively_stopped')
    except Exception as error:
        save('error', repr(error)); raise


def audit(payload):
    name, r, direction, speeds, data, duration = payload
    cases = []
    path = OUTPUT/'validation'/f'{name}.json'
    try:
        for speed in speeds:
            check_stop(); records = []
            if duration == 1.:
                left = model.resolve(r, direction, float(speed), data, records)
                right = model.resolve_fixed(r, direction, float(speed), data, records)
                passed = left >= 0 and right >= 0 and model.primary(left) == model.primary(right)
                case = dict(speed_m_per_s=float(speed), adaptive_code=left, fixed_code=right,
                            passed=bool(passed), evaluations=records)
            else:
                results = [model.fast.integrate(r, float(speed)*direction, duration, data, f, True) for f in (2., 4.)]
                passed = all(x[0] == -1 and abs(x[1]-duration)<1e-9 and x[4]>=0 and x[5]<=.002 for x in results)
                separation = float(np.linalg.norm(results[0][2]-results[1][2]))
                case = dict(speed_m_per_s=float(speed), duration_s=duration,
                            passed=bool(passed and separation<=1e-5), endpoint_difference_m=separation,
                            results=[list(x[:2])+[x[2].tolist(), x[3].tolist(), float(x[4]), float(x[5])] for x in results])
            cases.append(case)
            if len(cases)%16 == 0 or not case['passed']: write(path, dict(status='running', cases=cases))
            if not case['passed']:
                write(path, dict(status='failed', passed=False, cases=cases)); return dict(status='failed', name=name)
        write(path, dict(status='complete', passed=True, cases=cases))
        return dict(status='complete', name=name)
    except AdministrativeStop:
        write(path, dict(status='administratively_stopped', passed=False, cases=cases))
        return dict(status='administratively_stopped', name=name)


def cache_disc(d, area):
    path = OUTPUT/'production/discs'/f'd{d:05d}.npz'
    if path.exists(): return
    codes = []
    for p in range(20):
        with np.load(OUTPUT/'production/rays'/f'd{d:03d}_p{p:03d}.npz') as a:
            assert str(a['status']) == 'resolved' and float(a['observation_horizon_s']) == 1.
            np.testing.assert_array_equal(a['speeds'], model.SPEEDS)
            assert np.all(a['codes'] >= 0); codes.append(a['codes'][:640])
    codes = np.asarray(codes); v = model.ce.VELOCITIES
    spectra = np.asarray([area*(codes==c).mean(axis=0) for c in KINDS])
    w = v**3*np.exp(-v*v/model.original.THERMAL_SCALE_M2_PER_S2)
    rates = model.original.LOADING_RATE_PREFACTOR*np.trapezoid(np.c_[np.zeros(5), spectra*w], np.r_[0.,v], axis=1)
    coarse = model.original.LOADING_RATE_PREFACTOR*np.trapezoid(np.r_[0., (spectra[0]*w)[1::2]], np.r_[0.,v[1::2]])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.ce._atomic_npz(path, disc=np.array(d), area_m2=np.array(area), spectra_m2=spectra,
        rates_atoms_per_s=rates, coarse_rate_atoms_per_s=np.array(coarse), counts=np.bincount(codes.ravel(), minlength=6))


def summarize(final=False):
    inv = read(OUTPUT/'predecessor_inventory.json'); start = inv['new_disc_start']; end = inv['new_disc_end_exclusive']
    paths=[]
    for d in range(start,end):
        path=OUTPUT/'production/discs'/f'd{d:05d}.npz'
        if not path.exists(): break
        paths.append(path)
    n=len(paths); manifest=read(OUTPUT/'campaign_manifest.json')
    row=dict(at_utc=utc(), observation_horizon_s=1., final_retention_window_s=.025,
        status='deadline-limited descriptive result' if final else 'provisional 1s continuation',
        complete_direction_discs=n, resolved_rays=n*20, disc_start=start, disc_end_exclusive=start+n,
        planned_remaining_discs=end-start, thermal_conditions=manifest['thermal_conditions'],
        source_sha256=manifest['source_sha256'], combined_with_2s_results=False,
        interval='Nominal 95% Student-t independent-disc interval; administrative time truncation can select by computational cost, so unconditional coverage is not certified.',
        pending_gates=['final numerical convergence','production longer retention','precision','aperture/placement','final visual review'])
    arrays={}
    if n:
        rows=[]; spectra=[]; coarse=[]; counts=[]; areas=[]
        for path in paths:
            with np.load(path) as a:
                rows.append(a['rates_atoms_per_s']); spectra.append(a['spectra_m2']); coarse.append(float(a['coarse_rate_atoms_per_s'])); counts.append(a['counts']);areas.append(float(a['area_m2']))
        rates=np.asarray(rows); mean=rates.mean(axis=0)
        half=student_t.ppf(.975,n-1)*rates.std(axis=0,ddof=1)/np.sqrt(n) if n>1 else np.full(5,np.nan)
        v=model.ce.VELOCITIES; thermal=manifest['thermal_conditions']
        maximum=thermal['historical_loading_prefactor']*np.pi*.015**2*np.trapezoid(np.r_[0.,v**3*np.exp(-v*v/thermal['thermal_scale_m2_per_s2'])],np.r_[0.,v])
        half[np.all(rates==0,axis=0)]=maximum*(1-.025**(1/n))
        row.update(loading_rate_atoms_per_s=float(mean[0]), loading_95_half_width_atoms_per_s=float(half[0]),
            loading_ci95_atoms_per_s=[max(0.,float(mean[0]-half[0])),float(mean[0]+half[0])],
            loading_coefficient_m3_per_s=float(mean[0]/thermal['rb87_density_m3']),
            off_core_retention_integral_atoms_per_s=float(mean[1]),off_core_retention_95_half_width_atoms_per_s=float(half[1]),
            observed_transient_core_then_wall_integral_atoms_per_s=float(mean[2]),
            observed_transient_core_then_wall_95_half_width_atoms_per_s=float(half[2]),
            outcome_node_counts=np.sum(counts,axis=0).tolist(),
            velocity_quadrature_difference_atoms_per_s=float(abs(np.mean(coarse)-mean[0])),
            precision_pass=bool(n>1 and mean[0]>0 and half[0]<=.05*mean[0]))
        arrays=dict(velocity_m_per_s=v, disc_ids=np.arange(start,start+n),disc_all_outcome_spectra_m2=np.asarray(spectra),
                    disc_cross_section_m2=np.asarray(spectra)[:,0],disc_all_outcome_rates_atoms_per_s=rates,
                    disc_loading_atoms_per_s=rates[:,0],disc_areas_m2=np.asarray(areas))
    stem=OUTPUT/'reports'/('final_1s' if final else f'production_discs_{n:05d}')
    stem.parent.mkdir(exist_ok=True); write(stem.with_suffix('.json'),row)
    if arrays: model.ce._atomic_npz(stem.with_suffix('.npz'),**arrays)
    write(OUTPUT/'latest_estimate.json',dict(row,estimate_file=str(stem.with_suffix('.json'))))
    if final:
        report=['# Deadline-limited MOT continuation', '', f'Finalized: {utc()}', '',
            f'1-second cohort: {n} complete independent discs / {n*20} rays, IDs {start} through {start+n-1}.',
            'Acceptance: core event by 1 s plus continuous central 2 mm residence during [0.975, 1.000] s.',
            'Existing 2-second results remain separate; no mixed-horizon estimate.', '',
            '```json',json.dumps(row,indent=2),'```','',
            'Nominal intervals are descriptive. Incomplete discs and numerical unknowns are excluded, not relabeled escaped. Full study validation is not certified.',
            f'Prior 2-second run: {OLD}',f'Prior stopped inventory: {OUTPUT / "predecessor_inventory.json"}',
            f'1-second spectra and statistics: {stem.with_suffix(".npz")}',
            f'Raw 1-second rays: {OUTPUT / "production/rays"}',
            'Scope: surrogate conventional MOT, not physical pMOT; historical thermal inputs are assumed, not measured.']
        (OUTPUT/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return row


def initialize():
    model.HORIZON = 1.
    manifest=read(OLD/'campaign_manifest.json'); assert model.original.SOURCE_SHA==manifest['source_sha256']
    for name,key in [('mot_strange_retained_capture_v4r1.py','retained_helper_sha256'),('mot_strange_retained_numerics.py','observation_module_sha256'),('mot_strange_retained_budget.py','budget_module_sha256')]:
        assert sha(ROOT/'scripts'/name)==manifest[key]==sha(OLD/name)
    inv=read(OUTPUT/'predecessor_inventory.json')
    manifest.update(campaign=OUTPUT.name, observation_horizon_s=1., final_retention_window_s=.025,
        central_acceptance='Core event by 1s AND continuous central2mm residence throughout[0.975,1.000]s',
        human_decision=read(CONTROL)['user_instruction'],deadline=read(CONTROL),
        predecessor_complete_prefix_discs=inv['complete_prefix_discs'],
        numerical_revision='Unchanged v4r1 algorithms and10um tolerance; duration1s, finite budgets scale with duration.',
        allocation='Remaining preassigned production discs only, capped by user administrative deadline; no2s pilot precision claim for1s.',
        reuse='No2s classifications reused as1s evidence; exact preassigned geometry only.',
        runner_sha256=sha(__file__),new_gates_pending=True)
    manifest['outcomes']['2']='off-core endpoint retained at1s; not validated3D trapping'
    manifest['fixed_ladder_interpretation']='full1s with early core exit disabled'
    manifest['adaptive_budget_policy']='max(2million,ceil(1s/0.1ms*refinement)*1.5+10000), finite<=50million'
    write(OUTPUT/'campaign_manifest.json',manifest)
    for source in [Path(__file__),ROOT/'scripts/mot_strange_retained_capture_v4r1.py',ROOT/'scripts/mot_strange_retained_numerics.py',ROOT/'scripts/mot_strange_retained_budget.py']:
        shutil.copy2(source,OUTPUT/source.name)
    (OUTPUT/'production').mkdir(exist_ok=True)
    if not (OUTPUT/'production/geometry.npz').exists():shutil.copy2(OLD/'production/geometry.npz',OUTPUT/'production/geometry.npz')
    with np.load(OUTPUT/'production/geometry.npz') as a:return {k:a[k].copy() for k in a.files}


def run(workers):
    geometry=initialize(); inv=read(OUTPUT/'predecessor_inventory.json')
    write(OUTPUT/'process.json',dict(pid=os.getpid(),started_at_utc=utc(),workers=workers,command='run_mot_strange_retained1s_deadline.py run'))
    data=model.parent.load_data(); install_checks()
    with np.load(OLD/'pilot/geometry.npz') as a:pr=a['positions_m'];pd=a['directions']
    with ProcessPoolExecutor(max_workers=workers,initializer=install_checks) as pool:
        # Fresh1s adaptive/fixed checks; previous2s pass certificates are not copied.
        probes=[('fresh_pilot_central',pr[0,4],pd[0],[16.],data,1.),
                ('fresh_pilot_offcore',pr[0,4],pd[0],[19.75],data,1.),
                ('fresh_fast_wall',pr[0,0],pd[0],[120.],data,1.)]
        futures=[pool.submit(audit,p) for p in probes]
        for f in futures:
            result=f.result()
            if result['status']!='complete':raise RuntimeError('Fresh1s gate did not pass: '+str(result))
        write(OUTPUT/'validation/fresh_workers.json',dict(passed=True,at_utc=utc(),horizon_s=1.,probe_count=3))
        audits=[('full_fixed_d0p0',pr[0,0],pd[0],model.SPEEDS,data,1.),
                ('full_fixed_d17p13',pr[17,13],pd[17],model.SPEEDS,data,1.),
                ('full_fixed_d0p4',pr[0,4],pd[0],model.SPEEDS,data,1.),
                ('longer_pilot_1p1s',pr[0,4],pd[0],[16.],data,1.1)]
        pending={pool.submit(audit,p):('audit',p[0]) for p in audits}
        jobs=iter((d,p,geometry['positions_m'][d,p],geometry['directions'][d],data)
            for d in range(inv['new_disc_start'],inv['new_disc_end_exclusive']) for p in range(20))
        resolved=set(); completed=0; last_report=0.; last_count=0; requested=(inv['new_disc_end_exclusive']-inv['new_disc_start'])*20
        def submit():
            if stopped():return False
            payload=next(jobs,None)
            if payload is None:return False
            pending[pool.submit(worker,payload)]=('ray',(payload[0],payload[1]));return True
        for _ in range(workers-len(pending)):submit()
        while pending:
            done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
            for f in done:
                kind,identity=pending.pop(f); result=f.result()
                if kind=='ray' and result['status']=='resolved':
                    resolved.add(identity);completed+=1;d=identity[0]
                    if all((d,p) in resolved for p in range(20)):cache_disc(d,geometry['areas_m2'][d])
                elif result['status'] not in ('complete','administratively_stopped'):
                    write(OUTPUT/'failure.json',dict(at_utc=utc(),result=result));write(OUTPUT/'STOP_REQUESTED.json',dict(reason='Numerical or audit gate failed',at_utc=utc()))
                submit()
            prefix=inv['new_disc_start']
            while (OUTPUT/'production/discs'/f'd{prefix:05d}.npz').exists():prefix+=1
            count=prefix-inv['new_disc_start']
            if count>=2 and count//25>last_count//25:
                summarize();last_count=count
            if time.monotonic()-last_report>=30:
                write(OUTPUT/'progress.json',dict(at_utc=utc(),stage='draining' if stopped() else 'production',
                    complete_prefix_discs=count,first_disc_id=inv['new_disc_start'],completed_rays=completed,
                    requested_rays=requested,hard_deadline_utc=read(CONTROL)['hard_deadline_utc']))
                last_report=time.monotonic()
        write(OUTPUT/'stopped.json',dict(at_utc=utc(),reason='Administrative deadline' if stopped() else 'Allocated cohort computed',workers_drained=True))
    summarize(final=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['run','finalize']);parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    if args.mode=='finalize':print(json.dumps(summarize(final=True)))
    else:
        model.original.OUTPUT=model.parent.BASE_OUTPUT
        with model.original.campaign_lock():
            try:run(args.workers)
            except Exception as error:
                write(OUTPUT/'failure.json',dict(at_utc=utc(),error=repr(error),traceback=traceback.format_exc()));raise
