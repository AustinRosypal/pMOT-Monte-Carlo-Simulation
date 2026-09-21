"""Revision-4 validation against immutable dense, dual-RK4 pilot evidence."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.venv_pMOT_MC/Lib/site-packages'),str(ROOT/'src')]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pmot.mot_multilevel.adaptive import integrate
from pmot.mot_multilevel.accelerated import prepare
from pmot.mot_multilevel.relationship_campaign import loading_points,paths,write_json,source_digest


def worker(payload):
    point,path,data,hybrid,output=payload
    key=point.key
    with np.load(path) as z:
        r=z['position_m'];direction=z['direction'];speeds=z['speeds'];codes=z['codes'];old_s=float(z['elapsed_wall_s'])
    # Compile before the timer, including both refinement signatures.
    for refinement in (1.,2.):integrate(r,120*direction,.000001,data,refinement)
    if hybrid:
        from pmot.mot_multilevel.campaign_execution import _ray_worker
        from pmot.mot_multilevel.accelerated import classify
        classify(r,120*direction,5e-6,1e-6,data)
        target=output/'hybrid_benchmark'/key/path.name
        if target.exists():
            raise RuntimeError('benchmark file already exists; preserve it and select a new benchmark root')
        start=time.perf_counter()
        result=_ray_worker((point,0,0,r,direction,target,'benchmark',data))
        elapsed=time.perf_counter()-start
        with np.load(target) as z:
            actual=z['codes'];method=z['evaluation_methods']
            mismatch=[{'speed':float(s),'expected':int(e),'hybrid':int(a)}
                      for s,e,a in zip(speeds,codes,actual) if e!=a]
        return {'key':key,'file':str(path),'old_s':old_s,'new_pair_s':elapsed,
                'nodes':len(speeds),'mismatches':mismatch,'status':result['status'],
                'fixed_evaluations':int(np.count_nonzero(method==0)),
                'adaptive_evaluations':int(np.count_nonzero(method==1))}
    start=time.perf_counter();mismatches=[];counts=[0,0]
    for speed,expected in zip(speeds,codes):
        a=integrate(r,speed*direction,.05,data,1.)
        b=integrate(r,speed*direction,.05,data,2.)
        if a[0]<0 or b[0]<0:
            for duration in (.1,.2,.4,1.,2.):
                a=integrate(r,speed*direction,duration,data,1.)
                b=integrate(r,speed*direction,duration,data,2.)
                if a[0]>=0 and b[0]>=0:break
        counts[0]+=a[6]+b[6];counts[1]+=a[7]+b[7]
        if a[0]!=expected or b[0]!=expected:
            mismatches.append({'speed':float(speed),'expected':int(expected),'adaptive':[int(a[0]),int(b[0])],
                'times':[float(a[1]),float(b[1])]})
    return {'key':key,'file':str(path),'old_s':old_s,'new_pair_s':time.perf_counter()-start,
            'nodes':len(speeds),'mismatches':mismatches,'steps':counts}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--hybrid',action='store_true')
    parser.add_argument('--revision-dir',default='04_runtime')
    args=parser.parse_args();hybrid=args.hybrid
    base=paths()['statistics'];out=base/'revision_history'/args.revision_dir
    jobs=[]
    for point in loading_points():
        data=prepare(point.config())
        for f in sorted((base/'pilot'/point.key/'rays').glob('*.npz')):
            # Four dispersed pilot rays per setting, plus every nonmonotone ray.
            d=int(f.stem[1:4]);p=int(f.stem[6:9])
            with np.load(f) as z:nonmonotone=bool(np.any(np.diff(z['codes'][:320])>0))
            if (d,p) in ((0,0),(5,5),(10,10),(15,15)) or nonmonotone:
                jobs.append((point,f,data,hybrid,out))
    started=time.perf_counter();rows=[]
    print(f'{len(jobs)} dense rays selected',flush=True)
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            rows.append(f.result())
            if i%16==0:print(f'{i}/{len(jobs)} rays; mismatching nodes={sum(len(r["mismatches"]) for r in rows)}',flush=True)
    report={'source_sha256':source_digest(),'rays':len(rows),'nodes':sum(r['nodes'] for r in rows),
        'wall_s':time.perf_counter()-started,'mismatching_nodes':sum(len(r['mismatches']) for r in rows),'rows':rows}
    write_json(out/('hybrid_pilot_comparison.json' if hybrid else 'adaptive_pilot_comparison.json'),report)
    if hybrid and report['mismatching_nodes']==0 and all(r['status']=='resolved' for r in rows):
        write_json(base/'validation/hybrid_pilot_comparison.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
