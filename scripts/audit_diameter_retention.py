"""Recheck every early-positive node after a diameter retention gate fails.

This creates separate evidence only. It never rewrites the original capture
masks, treats timeouts as escapes, or releases an unqualified loading result.
"""
import run_population_diameter_campaign as runner
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pmot.mot_multilevel.accelerated import retention_audit

CAMPAIGN = 'cooling_diameter_20260930_fixed_intensity_baseline'
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def node(position, direction, speed, minimum_duration, data):
    attempts=[]
    for duration, coarse, fine in ((.1,5e-6,2.5e-6),(.2,5e-6,2.5e-6),
            (.4,5e-6,2.5e-6),(1.,1.25e-6,.625e-6),(2.,1.25e-6,.625e-6)):
        if duration < minimum_duration:
            continue
        results=[retention_audit(position,speed*direction,dt,duration,data) for dt in (coarse,fine)]
        delta=float(np.linalg.norm(results[0][2]-results[1][2]))
        observations=[]
        for dt,a in zip((coarse,fine),results):
            escaped=bool(a[1]<duration-1e-12 and np.linalg.norm(a[2])>=.03 and np.dot(a[2],a[3])>0)
            observations.append({'dt_s':dt,'retained':bool(a[0]),'escaped':escaped,
                'elapsed_s':float(a[1]),'position_m':a[2].tolist(),'velocity_m_per_s':a[3].tolist(),
                'final_window_max_radius_m':float(a[4])})
        attempts.append({'duration_s':duration,'endpoint_difference_m':delta,'observations':observations})
        if all(a['retained'] for a in observations) and delta<1e-5:
            return {'speed_m_per_s':speed,'code':1,'attempts':attempts}
        if all(a['escaped'] for a in observations):
            return {'speed_m_per_s':speed,'code':0,'attempts':attempts}
    return {'speed_m_per_s':speed,'code':-1,'attempts':attempts}


def ray(payload):
    filename, output, data = payload
    filename,output=Path(filename),Path(output)
    digest=hashlib.sha256(filename.read_bytes()).hexdigest()
    if output.exists():
        previous=json.loads(output.read_text())
        assert previous['input_sha256']==digest and previous['script_sha256']==SCRIPT_SHA
        return previous
    started=time.perf_counter()
    with np.load(filename) as saved:
        records=[]
        for speed in saved['speeds'][saved['codes']==1]:
            ev=saved['evaluations'][saved['evaluations'][:,0]==speed]
            minimum=max(.1,float(np.max(ev[:,4]))+.025)
            records.append(node(saved['position_m'],saved['direction'],float(speed),minimum,data))
    result={'ray':filename.name,'input_sha256':digest,'script_sha256':SCRIPT_SHA,
            'source_sha256':runner.CORE_SHA,'nodes':records,'elapsed_wall_s':time.perf_counter()-started}
    runner.rc.write_json(output,result)
    return result


def main(args):
    assert runner.rc.source_digest()==runner.CORE_SHA
    runner.rc.CAMPAIGN=CAMPAIGN
    root=runner.rc.paths()['statistics']
    output=root/'retention_recheck'/f'{args.index:03d}'
    diameter=runner.DIAMETERS[args.index]
    point=runner.DiameterPoint(args.index,diameter,.027*(diameter/12.7)**2,-2*np.pi*15e6)
    data=runner.prepare_diameter(point.config())
    original=root/'pilot'/point.key
    manifest={'source_sha256':runner.CORE_SHA,'script_sha256':SCRIPT_SHA,
              'runner_sha256':hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
              'diameter_mm':diameter,'method':'all early-positive pilot nodes; dual fixed RK4 retention/definitive escape ladder; separate immutable evidence'}
    if (output/'manifest.json').exists():
        assert json.loads((output/'manifest.json').read_text())==manifest
    else:
        runner.rc.write_json(output/'manifest.json',manifest)
    runner.rc.write_json(output/'process.json',{'pid':os.getpid(),'command':sys.argv,
        'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':args.workers})
    # Reproduce the original failure with both timesteps before the bulk audit.
    failed=json.loads((original/'duration_audit.json').read_text())
    validation=[]
    for case in failed['cases']:
        if case['passed']:
            continue
        filename=original/'rays'/f"d{case['disc']:03d}_p{case['point']:03d}.npz"
        with np.load(filename) as saved:
            check=node(saved['position_m'],saved['direction'],case['speed_m_per_s'],.1,data)
        validation.append({'original_case':case,'recheck':check})
        assert check['code']==0, 'original retention failure needs further diagnosis'
    runner.rc.write_json(output/'validation.json',{'passed':bool(validation),'cases':validation})
    payloads=[]
    for filename in sorted((original/'rays').glob('*.npz')):
        with np.load(filename) as saved:
            if np.any(saved['codes']==1):
                payloads.append((filename,output/'rays'/f'{filename.stem}.json',data))
    print(f'{diameter:g} mm: audit {len(payloads)} early-positive rays',flush=True)
    rows=[]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(ray,payload) for payload in payloads]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%16==0 or len(rows)==len(payloads):
                counts={str(code):sum(n['code']==code for r in rows for n in r['nodes']) for code in (-1,0,1)}
                progress={'completed_rays':len(rows),'total_rays':len(payloads),'node_counts':counts,
                          'elapsed_wall_s':time.perf_counter()-started}
                runner.rc.write_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    runner.rc.write_json(output/'completion.json',dict(progress,complete=True,
        all_nodes_definitive=counts['-1']==0,original_results_unchanged=True,
        remaining='Rebuild qualified masks with provenance, check quadrature and revise production allocation; no publication authorized by this audit alone.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--workers',type=int,default=16)
    main(parser.parse_args())
