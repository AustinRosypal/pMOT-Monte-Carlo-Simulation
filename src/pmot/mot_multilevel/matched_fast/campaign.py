"""Resumable independent-pilot campaign with exact physical hybrid trajectories."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
# Higher-order Numba dispatcher cache indices must not be shared by workers:
# concurrent loading/saving can serialize a dead weak dispatcher reference.
# Private caches change no numerical code, force, tolerance or trajectory.
os.environ['NUMBA_CACHE_DIR']=str(Path(os.environ.get('LOCALAPPDATA',tempfile.gettempdir()))/'pmot-campaigns/matched_two_level_exact_D2_fast_20261104/compiled_cache'/f'process_{os.getpid()}')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import csv
import json
import hashlib
import shutil
import tarfile
import io
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from ..matched_campaign import atomic_npz,reweight_counts,write_csv,library_versions
from ..matched_capture import RayEvaluator,AUDIT_COLUMNS
from ..matched_validation import validate_point_retention
from ...loading import VaporLoadingParameters
from . import config as b
from .capture import HybridEvaluator,ADAPTIVE_COLUMNS,solve

_LOCK=None
_DATA={}
_POSITIONS=None
_DIRECTIONS=None
_BOUND_INDEX=None
_CONTROL_DIGEST=None


def points_and_scans():return b.definitions()


def acquire_lock():
    global _LOCK
    b.CACHE.mkdir(parents=True,exist_ok=True)
    handle=(b.CACHE/'runner.lock').open('a+b');handle.seek(0,2)
    if handle.tell()==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    _LOCK=handle
    b.write_json(b.STATISTICS/'run_owner.json',dict(pid=os.getpid(),started_utc=b.now(),command='matched_fast.campaign'))


def check_frozen():
    path=b.STATISTICS/'campaign_manifest.json'
    if path.exists():
        manifest=b.read_json(path)
        if manifest['source_sha256']!=b.source_hashes() or manifest['libraries']!=library_versions():
            raise RuntimeError('frozen source/library mismatch; preserve evidence and validate a recorded revision')


def initialize():
    points,scans=points_and_scans();previous=b.read_json(b.PREVIOUS/'campaign_manifest.json')
    assert previous['source_sha256']==b.original.source_hashes(),'original source changed during migration'
    migration=b.read_json(b.STATISTICS/'migration.json')
    manifest=dict(schema=1,campaign=b.NAME,points=[asdict(p) for p in points],scans=scans,
                  source_sha256=b.source_hashes(),libraries=library_versions(),model=b.model_notes(),
                  vapor=asdict(VaporLoadingParameters()),master_geometry_sha256=b.sha256(b.BENCHMARK/'launch_geometry_master.csv'),
                  pilot_disc_indices=list(b.PILOT_DISCS),pilot_point_indices=list(b.PILOT_POINTS),
                  validation_disc_indices=list(b.VALIDATION_DISCS),
                  statistical_policy=dict(target_relative_95_half_width=.08,variance_inflation=1.25,min_discs=128,max_rays_per_point=12800,
                    allocation='independent pilot only; predeclare all production counts before starting production',
                    pilot_and_original_data='excluded from all revised production intervals',
                    unresolved_precision='report achieved intervals honestly; no CI-triggered pooling'),
                  trajectory_policy=dict(adaptive='existing physical DP5(4), refinements 1 and 2 for direct nodes',
                    worker_inputs='fresh point-bound worker pool, parent-prepared immutable input copies, complete array fingerprints and actual-worker full24 reference checks; independent saved-node replays before publication',
                    fixed='original dual-step RK4 endpoints, every direct capture-edge neighbourhood, all adaptive disagreements, literal zero and long retention',
                    fixed_timeouts_s=[.2,.25,.4,1.,2.],velocity_m_per_s=b.VELOCITIES.tolist(),
                    capture='unchanged 2mm core: continuous5ms OR two entries separatedexit; outward30mm escape',
                    full_fixed_validation='one complete 321-node independently held-out ray at every physical setting; fail closed on positive-mask disagreement'),
                  authorization='User approved combined speedup, 12-36h target, progress and completion ETAs; future extensions requested as a capability.',
                  predecessor_manifest_sha256=b.sha256(b.PREVIOUS/'campaign_manifest.json'),migration=migration)
    path=b.STATISTICS/'campaign_manifest.json'
    if path.exists() and b.read_json(path)!=json.loads(json.dumps(manifest)):raise ValueError('frozen campaign manifest mismatch')
    b.write_json(path,manifest)
    for relative,digest in manifest['source_sha256'].items():
        target=b.STATISTICS/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(b.ROOT/relative,target)
        assert b.sha256(target)==digest
    return manifest


def data_digest(data):
    digest=hashlib.sha256()
    for array in data:
        digest.update(str((array.shape,str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def immutable_data(data):
    result=[]
    for array in data:
        array=np.array(array,copy=True,order='C');array.setflags(write=False);result.append(array)
    return tuple(result)


def verify_runtime_reference(index,data):
    """Check actual worker arrays and force, not a separate preparation path."""
    from ..rate_equations import build_rate_equation_model,rate_equation_observable
    from ..accelerated import observable
    point=points_and_scans()[0][index];cfg,_,coil,beams=b.configuration(point)
    expected=b.prepared(point)
    if data_digest(data)!=data_digest(expected):raise RuntimeError('actual worker arrays differ from requested controls')
    model=build_rate_equation_model();axis=np.array([0.,0.,1.])
    for r,v in ((np.array([.001,-.0007,.0004]),np.array([1.,-.4,.7])),
                (np.array([-.000745,.000439,-.001923]),np.array([.000482,-.000284,-.0000765]))):
        reference=rate_equation_observable(model,beams,tuple(r),tuple(v),coil,cfg)
        fast=observable(r,v,axis,data)
        np.testing.assert_allclose(fast[0],reference.force_n,rtol=3e-6,atol=3e-29)
        np.testing.assert_allclose(fast[1],reference.populations,rtol=4e-6,atol=4e-10)


def worker_init(index=None,prepared_data=None,expected_digest=None):
    global _POSITIONS,_DIRECTIONS,_BOUND_INDEX,_CONTROL_DIGEST,_DATA
    _POSITIONS,_DIRECTIONS,_=b.load_master_geometry()
    _DATA={};_BOUND_INDEX=index
    if index is not None:
        data=immutable_data(prepared_data)
        if data_digest(data)!=expected_digest:raise RuntimeError('worker initializer data fingerprint mismatch')
        verify_runtime_reference(index,data)
        _DATA[index]=data;_CONTROL_DIGEST=expected_digest


def worker_data(index):
    if _BOUND_INDEX is not None and index!=_BOUND_INDEX:raise RuntimeError('worker cannot cross physical configurations')
    if index not in _DATA:
        data=immutable_data(b.prepared(points_and_scans()[0][index]));verify_runtime_reference(index,data);_DATA[index]=data
    if _BOUND_INDEX is not None and data_digest(_DATA[index])!=_CONTROL_DIGEST:raise RuntimeError('runtime input arrays mutated')
    return _DATA[index]


def signature(point,stage):
    discs,points=b.sampling(point,stage)
    payload=dict(schema=1,configuration=asdict(point),stage=stage,disc_indices=discs,point_indices=points,
                 prepared_data_sha256=data_digest(b.prepared(point)),
                 master_geometry_sha256=b.sha256(b.BENCHMARK/'launch_geometry_master.csv'),
                 numerical_sources=b.source_hashes(),libraries=library_versions(),
                 campaign_manifest_sha256=b.sha256(b.STATISTICS/'campaign_manifest.json'))
    if stage=='production':payload['production_plan_sha256']=b.sha256(b.STATISTICS/'production_plan.json')
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return digest,payload


def paths(point,stage='production'):
    return b.CACHE/stage/point.key,b.STATISTICS/stage/point.key,b.FIGURES/stage/point.key


def ray_worker(task):
    stage,index,disc,point,path,digest,representative=task
    data=worker_data(index);r=_POSITIONS[disc,point];direction=_DIRECTIONS[disc,point]
    path=Path(path);fixed=[];adaptive=[];previous_wall=0.
    if path.exists():
        with np.load(path) as z:
            if str(z['signature'])!=digest:raise ValueError('ray signature mismatch')
            np.testing.assert_array_equal(z['position_m'],r);np.testing.assert_array_equal(z['direction'],direction)
            if str(z['status'])=='completed':return float(z['elapsed_wall_s'])
            fixed=z['evaluations'].copy();adaptive=z['adaptive_evaluations'].copy();previous_wall=float(z['elapsed_wall_s'])
    started=perf_counter()
    def save(e,status='running',result=None,error=''):
        payload=dict(signature=np.array(digest),status=np.array(status),error=np.array(error),
                     prepared_data_sha256=np.array(data_digest(data)),
                     stage=np.array(stage),configuration=np.array(index),disc=np.array(disc),point=np.array(point),
                     position_m=r,direction=direction,elapsed_wall_s=np.array(previous_wall+perf_counter()-started),
                     evaluations=np.asarray(e.records).reshape((-1,len(AUDIT_COLUMNS))),
                     adaptive_evaluations=np.asarray(e.adaptive_records).reshape((-1,len(ADAPTIVE_COLUMNS))))
        if result is not None:payload.update(mask=result['mask'],mode=np.array(result['mode']),
            boundary_json=np.array(json.dumps(result['boundary'])),complete_searches_json=np.array(json.dumps(result['complete_searches'])),
            nonmonotone=np.array(result['nonmonotone']))
        atomic_npz(path,**payload)
    evaluator=HybridEvaluator(r,direction,data,fixed,adaptive,checkpoint=save)
    try:
        result=solve(evaluator,representative);save(evaluator,'completed',result)
        return previous_wall+perf_counter()-started
    except BaseException as exc:
        save(evaluator,'failed',error=f'{type(exc).__name__}: {exc}');raise


def validation_worker(index):
    point=points_and_scans()[0][index];data=worker_data(index)
    disc=b.VALIDATION_DISCS[index%len(b.VALIDATION_DISCS)];p=(17*index)%50
    r=_POSITIONS[disc,p];direction=_DIRECTIONS[disc,p]
    folder=b.CACHE/'validation';folder.mkdir(parents=True,exist_ok=True)
    evaluator=HybridEvaluator(r,direction,data)
    hybrid=solve(evaluator,True)
    fixed=RayEvaluator(r,direction,data);mask=fixed.direct_mask()
    different=np.flatnonzero(mask[1:]!=hybrid['mask'][1:])+1
    atomic_npz(folder/f'configuration_{index:03d}.npz',configuration=np.array(index),disc=np.array(disc),point=np.array(p),
               prepared_data_sha256=np.array(data_digest(data)),
               position_m=r,direction=direction,mask_fixed=mask,mask_hybrid=hybrid['mask'],
               fixed_evaluations=np.asarray(fixed.records),hybrid_fixed_evaluations=np.asarray(evaluator.records),
               adaptive_evaluations=np.asarray(evaluator.adaptive_records))
    result=dict(index=index,disc=disc,point=p,status='passed' if not len(different) else 'failed',
                differing_positive_nodes=b.VELOCITIES[different].tolist(),
                fixed_node_count=len(fixed.records),adaptive_node_count=len(evaluator.adaptive_records),
                evidence=str(folder/f'configuration_{index:03d}.npz'))
    b.write_json(b.STATISTICS/'validation'/f'configuration_{index:03d}.json',result)
    return result


def validate(pool):
    original=b.read_json(b.PREVIOUS/'validation/preflight.json')
    assert original['status']=='passed' and original['checked_configurations']==124
    for name,digest in original['kernel_source_sha256'].items():assert b.sha256(b.ROOT/'src/pmot/mot_multilevel'/name)==digest
    todo=[];results=[]
    for point in points_and_scans()[0]:
        path=b.STATISTICS/'validation'/f'{point.key}.json'
        if path.exists() and b.read_json(path)['status']=='passed':results.append(b.read_json(path))
        else:todo.append(point.index)
    futures=[pool.submit(validation_worker,i) for i in todo]
    for future in as_completed(futures):
        result=future.result();results.append(result)
        b.write_json(b.STATISTICS/'validation/progress.json',dict(completed=len(results),expected=124,updated_utc=b.now(),last=result))
        print(f'[{b.now()}] full-mask numerical validation {len(results)}/124: {result["status"]}',flush=True)
        if result['status']!='passed':raise RuntimeError(f'held-out full-grid fixed/hybrid disagreement at configuration {result["index"]}')
    certificate=dict(status='passed',created_utc=b.now(),source_sha256=b.source_hashes(),
                     local_reference_certificate_sha256=b.sha256(b.PREVIOUS/'validation/preflight.json'),
                     checked_configurations=124,cases=sorted(results,key=lambda x:x['index']),
                     limits='Finite held-out numerical validation; local steady-state and incoherent-light assumptions remain. No temperature prediction.')
    b.write_json(b.STATISTICS/'validation/preflight.json',certificate)


def summarize(point,stage,digest):
    folder,_,_=paths(point,stage);discs,points=b.sampling(point,stage)
    counts=[];ray_rates=[];samples=[];walls=[]
    vapor=VaporLoadingParameters();v=b.VELOCITIES
    weight=v**3*np.exp(-v*v/vapor.thermal_scale_m2_per_s2);area=float(np.pi*.015**2)
    for d in discs:
        masks=[];values=[]
        for p in points:
            with np.load(folder/'rays'/f'd{d:04d}_p{p:02d}.npz') as z:
                assert str(z['signature'])==digest and str(z['status'])=='completed'
                mask=z['mask'].copy();assert np.all(mask[1:]>=0)
                masks.append(mask);zero=mask.astype(float);zero[0]=0.
                values.append(float(vapor.loading_rate_prefactor*area*np.trapezoid(zero*weight,v)))
                walls.append(float(z['elapsed_wall_s']))
                samples.append(dict(disc_index=d,point_index=p,mode=str(z['mode']),nonmonotone=bool(z['nonmonotone']),
                                    fixed_evaluations=len(z['evaluations']),adaptive_evaluations=len(z['adaptive_evaluations'])))
        a=np.asarray(masks);row=np.maximum(a,0).sum(axis=0).astype(object).tolist()
        if np.any(a[:,0]<0):row[0]=None
        counts.append(row);ray_rates.append(values)
    payload=dict(schema='pmot.capture-counts-by-disc.v1',configuration=asdict(point),stage=stage,
                 batch_id='independent-production-v1' if stage=='production' else 'excluded-pilot-v1',
                 sample_count_fixed_before_batch=stage=='production',
                 disc_indices=discs,point_indices=points,points_per_disc=len(points),disc_area_m2=area,
                 master_geometry_sha256=b.sha256(b.BENCHMARK/'launch_geometry_master.csv'),
                 velocity_m_per_s=v.tolist(),captured_counts=counts)
    b.write_json(folder/'capture_counts_by_disc.json',payload)
    rates=reweight_counts(payload)
    write_csv(folder/'capture_samples.csv',samples)
    write_csv(folder/'loading_rate_by_disc.csv',[dict(disc_index=d,loading_rate_atoms_per_s=x) for d,x in zip(discs,rates['disc_loading_rates_atoms_per_s'])])
    from scipy.stats import t
    spectrum=[]
    for j,speed in enumerate(v):
        if speed==0 and any(row[0] is None for row in counts):continue
        a=np.array([row[j] for row in counts],float)*area/len(points)
        mean=float(a.mean());half=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
        spectrum.append(dict(velocity_m_per_s=float(speed),capture_cross_section_m2=mean,lower_95_m2=max(0.,mean-half),upper_95_m2=min(area,mean+half)))
    write_csv(folder/'capture_cross_section.csv',spectrum)
    summary=dict(status='completed',configuration=asdict(point),stage=stage,signature=digest,disc_count=len(discs),points_per_disc=len(points),
                 sample_count=len(discs)*len(points),unresolved_positive_speed_count=0,
                 worker_wall_s=float(sum(walls)),mean_ray_worker_wall_s=float(np.mean(walls)),
                 loading={k:v for k,v in rates.items() if k!='disc_loading_rates_atoms_per_s'})
    b.write_json(folder/'summary.json',summary)
    if stage=='pilot':b.write_json(folder/'pilot_ray_loading_rates.json',dict(disc_indices=discs,point_indices=points,ray_loading_rates=ray_rates))
    return summary


def publish(point,stage):
    folder,destination,_=paths(point,stage);destination.mkdir(parents=True,exist_ok=True);hashes={}
    target=destination/'capture_evidence.tar.gz';temporary=target.with_suffix('.gz.tmp')
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:tar.add(folder/'rays',arcname='rays')
    b.replace_file(temporary,target);hashes[target.name]=b.sha256(target)
    for source in folder.iterdir():
        if not source.is_file() or source.name=='run_metadata.json':continue
        target=destination/source.name;temporary=target.with_suffix(target.suffix+'.tmp')
        shutil.copyfile(source,temporary);b.replace_file(temporary,target);hashes[target.name]=b.sha256(target)
    b.write_json(destination/'publication.json',dict(status='completed',published_utc=b.now(),files_sha256=hashes))
    b.write_json(destination/'run_metadata.json',b.read_json(folder/'run_metadata.json'))


def validate_saved_replays(point,stage,folder):
    """Independently replay actual saved nodes outside the producing workers."""
    from ..matched_capture import classify_numeric
    discs,points=b.sampling(point,stage)
    selected={(discs[j],points[j%len(points)]) for j in (0,len(discs)//2,len(discs)-1)}
    if stage=='pilot':selected.add((838,2))
    data=b.prepared(point);digest=data_digest(data);checks=[]
    for d,p in sorted(selected):
        with np.load(folder/'rays'/f'd{d:04d}_p{p:02d}.npz') as z:
            assert str(z['prepared_data_sha256'])==digest
            rows=z['evaluations'];positive=rows[(rows[:,0]>0)&(rows[:,3]>=0)]
            row=positive[0] if len(positive) else rows[0]
            result=classify_numeric(z['position_m'],float(row[0])*z['direction'],float(row[1]),float(row[2]),data)
        same_state=(result[0]<0)==(row[3]<0) and (result[0]>0)==(row[3]>0)
        difference=float(np.linalg.norm(result[4]-row[7:10]))
        passed=bool(same_state and difference<1e-9 and abs(result[1]-row[4])<1.1*row[1])
        checks.append(dict(disc=d,point=p,speed_m_per_s=float(row[0]),position_difference_m=difference,passed=passed))
        if not passed:
            b.write_json(folder/'runtime_replay_validation.json',dict(status='failed',checks=checks))
            raise RuntimeError('saved worker trajectory does not reproduce independently with verified requested controls')
    b.write_json(folder/'runtime_replay_validation.json',dict(status='passed',checks=checks,prepared_data_sha256=digest))


def update_eta(workers):
    points,_=points_and_scans();done=completed();now=datetime.now(timezone.utc)
    planpath=b.STATISTICS/'production_plan.json'
    if not planpath.exists():
        started=datetime.fromisoformat(b.read_json(b.STATISTICS/'migration.json')['authorized_utc'])
        value=dict(stage='independent_pilot_or_validation',updated_utc=b.now(),basis='Initial 12-36h target; recalculated after all pilots',
                   earliest_completion_utc=(started+timedelta(hours=12)).isoformat(),latest_completion_utc=(started+timedelta(hours=36)).isoformat())
    else:
        plan=b.read_json(planpath);seconds=0.
        for point in points:
            if point.index in done:continue
            allocation=plan['allocations'][str(point.index)];pilot=b.read_json(paths(point,'pilot')[1]/'summary.json')
            local=paths(point)[0]/'run_metadata.json';n=b.read_json(local).get('completed_samples',0) if local.exists() else 0
            seconds+=max(0,allocation['rays']-n)*pilot['mean_ray_worker_wall_s']/workers/.75
            seconds+=60. # Cold point-bound pool startup and runtime input checks.
        value=dict(stage='production',updated_utc=b.now(),remaining_estimated_wall_s=seconds,completed=len(done),expected=124,
                   basis='Per-setting measured independent-pilot worker cost, actual completed rays, 75% scheduling efficiency; uncertainty range includes numerical tails',
                   earliest_completion_utc=(now+timedelta(seconds=seconds*.7+600)).isoformat(),
                   latest_completion_utc=(now+timedelta(seconds=seconds*1.7+1800)).isoformat())
    b.write_json(b.STATISTICS/'eta.json',value)


def run_point(point,stage,pool,workers):
    check_frozen();folder,published,_=paths(point,stage);folder.mkdir(parents=True,exist_ok=True)
    data=b.prepared(point);expected_digest=data_digest(data)
    inputs_path=folder/'runtime_inputs.npz'
    if not inputs_path.exists():atomic_npz(inputs_path,**{f'array_{j:02d}':a for j,a in enumerate(data)},prepared_data_sha256=np.array(expected_digest))
    digest,payload=signature(point,stage);metadata=folder/'run_metadata.json'
    if metadata.exists():
        old=b.read_json(metadata)
        if old['signature']!=digest:raise ValueError('point signature mismatch')
        if old['status']=='completed' and (published/'publication.json').exists():return
    discs,points=b.sampling(point,stage);tasks=[];n=0
    for d in discs:
        for p in points:
            path=folder/'rays'/f'd{d:04d}_p{p:02d}.npz'
            if path.exists():
                with np.load(path) as z:
                    if str(z['signature'])!=digest:raise ValueError('checkpoint source/input mismatch')
                    if str(z['status'])=='completed':n+=1;continue
            tasks.append((stage,point.index,d,p,str(path),digest,bool(p==0 and d%40==0)))
    meta=dict(status='running',signature=digest,inputs=payload,started_utc=b.now(),expected_samples=len(discs)*len(points),completed_samples=n)
    b.write_json(metadata,meta);print(f'[{b.now()}] {stage} {point.key}: {n}/{meta["expected_samples"]}',flush=True)
    point_pool=ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(point.index,data,expected_digest))
    try:
        futures=[point_pool.submit(ray_worker,task) for task in tasks]
        for future in as_completed(futures):
            future.result();n+=1
            if n%25==0 or n==meta['expected_samples']:
                meta.update(completed_samples=n,updated_utc=b.now());b.write_json(metadata,meta)
                b.write_json(b.STATISTICS/'phase_progress.json',dict(stage=stage,configuration=point.index,completed_samples=n,expected_samples=meta['expected_samples'],updated_utc=b.now()))
                if n%200==0 or n==meta['expected_samples']:update_eta(workers)
                print(f'[{b.now()}] {stage} {point.key}: {n}/{meta["expected_samples"]}',flush=True)
    except BaseException:
        point_pool.shutdown(wait=True,cancel_futures=True);point_pool=None;raise
    finally:
        if point_pool is not None:point_pool.shutdown(wait=True)
    validate_saved_replays(point,stage,folder)
    if stage=='production':validate_point_retention(point,folder)
    check_frozen();summarize(point,stage,digest)
    meta.update(status='completed',completed_samples=n,updated_utc=b.now());b.write_json(metadata,meta)
    publish(point,stage)


def completed(stage='production'):
    result={}
    for point in points_and_scans()[0]:
        folder=paths(point,stage)[1]
        if (folder/'run_metadata.json').exists() and b.read_json(folder/'run_metadata.json')['status']=='completed':
            result[point.index]=b.read_json(folder/'capture_counts_by_disc.json')
    return result


def freeze_plan():
    allocations={}
    for point in points_and_scans()[0]:
        assert b.read_json(paths(point,'pilot')[1]/'runtime_replay_validation.json')['status']=='passed'
        pilot=b.read_json(paths(point,'pilot')[1]/'pilot_ray_loading_rates.json')
        allocations[str(point.index)]=b.allocation(point,pilot['ray_loading_rates'])
    plan=dict(schema=1,allocations=allocations,total_production_rays=sum(x['rays'] for x in allocations.values()),
              pilot_discs_excluded=list(b.PILOT_DISCS),validation_discs_excluded=list(b.VALIDATION_DISCS),
              all_counts_frozen_before_any_production=True,target=.08,coverage='pointwise95% Student-t across independent production direction discs')
    path=b.STATISTICS/'production_plan.json'
    if path.exists() and b.read_json(path)!=json.loads(json.dumps(plan)):raise ValueError('fixed production plan mismatch')
    b.write_json(path,plan)
    return plan


def audit_campaign():
    points,_=points_and_scans();data=completed();assert len(data)==124
    checks=[];positions,directions,_=b.load_master_geometry()
    for point in points:
        folder=paths(point)[1];publication=b.read_json(folder/'publication.json')
        for name,digest in publication['files_sha256'].items():assert b.sha256(folder/name)==digest
        payload=data[point.index];discs,points_indices=b.sampling(point)
        assert payload['disc_indices']==discs and payload['point_indices']==points_indices
        assert not set(discs)&set(b.PILOT_DISCS+b.VALIDATION_DISCS)
        assert b.read_json(folder/'retention_validation.json')['status']=='passed'
        summary=b.read_json(folder/'summary.json');rates=reweight_counts(payload)
        assert b.read_json(folder/'runtime_replay_validation.json')['status']=='passed'
        digest,_=signature(point,'production')
        expected_data_digest=data_digest(b.prepared(point))
        assert summary['signature']==b.read_json(folder/'run_metadata.json')['signature']==digest
        counts=np.zeros((len(discs),len(b.VELOCITIES)),int);unknown=np.zeros(len(discs),bool);seen=set()
        mapping={d:j for j,d in enumerate(discs)}
        with tarfile.open(folder/'capture_evidence.tar.gz','r:gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.npz'):continue
                with np.load(io.BytesIO(archive.extractfile(member).read())) as z:
                    assert str(z['signature'])==digest and str(z['status'])=='completed'
                    assert str(z['prepared_data_sha256'])==expected_data_digest
                    d,p=int(z['disc']),int(z['point']);assert (d,p) not in seen;seen.add((d,p))
                    assert d in mapping and p in points_indices
                    np.testing.assert_array_equal(z['position_m'],positions[d,p])
                    np.testing.assert_array_equal(z['direction'],directions[d,p])
                    mask=z['mask'];assert np.all(mask[1:]>=0)
                    counts[mapping[d]]+=np.maximum(mask,0);unknown[mapping[d]]|=mask[0]<0
        assert len(seen)==len(discs)*len(points_indices)
        expected=counts.astype(object).tolist()
        for j in np.flatnonzero(unknown):expected[j][0]=None
        assert expected==payload['captured_counts']
        for field in ('mean','lower','upper'):assert np.isclose(rates[field],summary['loading'][field],rtol=2e-12)
        checks.append(dict(index=point.index,relative_half_width=rates['relative_half_width']))
    check_frozen()
    result=dict(status='passed',configuration_count=124,checks=checks,audited_utc=b.now())
    b.write_json(b.STATISTICS/'final_audit.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('all','validate','pilot','production','report','audit'),default='all',nargs='?')
    parser.add_argument('--workers',type=int,default=20);args=parser.parse_args(argv)
    if not 1<=args.workers<=24:parser.error('workers must be1..24')
    initialize();check_frozen()
    from .reporting import refresh_report,comparison_manifest
    if args.stage=='report':refresh_report();return
    if args.stage=='audit':audit_campaign();refresh_report('completed');return
    acquire_lock();update_eta(args.workers)
    pool=ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init)
    try:
        certificate=b.STATISTICS/'validation/preflight.json'
        if not certificate.exists():validate(pool)
        pool.shutdown(wait=True);pool=None
        if args.stage=='validate':return
        if args.stage in ('all','pilot'):
            for point in points_and_scans()[0]:run_point(point,'pilot',pool,args.workers)
            freeze_plan();update_eta(args.workers)
        if args.stage=='pilot':return
        assert b.read_json(certificate)['status']=='passed'
        comparison_manifest();refresh_report()
        points,scans=points_and_scans();order=[0]
        for ids in scans.values():
            for i in ids:
                if i not in order:order.append(i)
        for index in order:
            run_point(points[index],'production',pool,args.workers);refresh_report();update_eta(args.workers)
        audit_campaign();refresh_report('completed')
    except BaseException as exc:
        b.write_json(b.STATISTICS/'failure.json',dict(time_utc=b.now(),error=f'{type(exc).__name__}: {exc}'))
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True);pool=None
        raise
    finally:
        if pool is not None:pool.shutdown(wait=True)


if __name__=='__main__':main()
