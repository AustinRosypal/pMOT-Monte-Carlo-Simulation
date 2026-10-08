"""Collect independent diameter pilots without bypassing publication gates.

An unqualified setting is retained and recorded while the other independent
settings continue. This helper never starts production or publishes a curve.
"""
import run_population_diameter_campaign as runner
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def main(args):
    runner.rc.CAMPAIGN='cooling_diameter_20260930_fixed_intensity_baseline'
    root=runner.rc.paths()['statistics']
    assert runner.rc.source_digest()==runner.CORE_SHA
    original=json.loads((root/'campaign_manifest.json').read_text())
    assert original['source_sha256']==runner.CORE_SHA
    assert original['runner_sha256']==hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
    for name in ('event_and_trajectory_checks','geometry_and_force','fresh_worker_checks'):
        gate=json.loads((root/'validation'/f'{name}.json').read_text())
        assert gate['passed'] and gate['source_sha256']==runner.CORE_SHA
    assert json.loads((root/'validation/test_status.json').read_text())==dict(exit_code=0,source_sha256=runner.CORE_SHA)
    manifest={'core_source_sha256':runner.CORE_SHA,'runner_sha256':original['runner_sha256'],
        'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'start_index':args.start_index,'policy':'collect independent pilots; retain failed gates; production prohibited until reviewed',
        'pilot_seed':2026093001,'discs':runner.PILOT_N,'points_per_disc':runner.PILOT_P}
    target=root/'pilot_collection_manifest.json'
    if target.exists():
        assert json.loads(target.read_text())==manifest
    else:
        runner.rc.write_json(target,manifest)
    runner.rc.write_json(root/'pilot_collection_process.json',dict(pid=os.getpid(),command=sys.argv,
        workers=args.workers,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    runner.ce.PILOT_SEED=2026093001
    runner.ce.prepare=runner.prepare_diameter
    runner.ce._ray_worker=runner.diameter_ray_worker
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i in range(args.start_index,len(runner.DIAMETERS)):
            d=runner.DIAMETERS[i]
            point=runner.DiameterPoint(i,d,.027*(d/12.7)**2,-2*np.pi*15e6)
            folder=root/'pilot'/point.key
            start=time.perf_counter()
            row=runner.ce.run_loading_point(point,'pilot',runner.PILOT_N,runner.PILOT_P,pool)
            capture=time.perf_counter()-start
            start=time.perf_counter()
            problem=None
            try:
                runner.ce.duration_audits(point,'pilot',runner.PILOT_N,runner.PILOT_P,pool)
            except RuntimeError as error:
                if 'retention gate' not in str(error) and 'longer-duration audit failed' not in str(error):
                    raise
                problem=str(error)
            audit=time.perf_counter()-start
            if not (folder/'timing.json').exists():
                runner.rc.write_json(folder/'timing.json',dict(capture_wall_s=capture,retention_wall_s=audit))
            result={'diameter_mm':d,'retention_pass':problem is None,'retention_issue':problem,
                'velocity_quadrature_pass':row['velocity_quadrature_pass'],'production_authorized':False}
            if row['loading_rate_atoms_per_s']>0:
                result['provisional_allocation']=runner.allocation(row,root,args.workers)
            else:
                result['allocation_issue']='Zero-capture pilot requires additional evidence; relative-precision allocation withheld.'
            results.append(result)
            runner.rc.write_json(root/'pilot_collection_progress.json',dict(completed_settings=len(results),settings=results))
            print(json.dumps(result),flush=True)
    runner.rc.write_json(root/'pilot_collection_complete.json',dict(settings=results,
        production_authorized=False,remaining='Review 2 mm all-positive retention evidence and all failed gates; finish fixed-RK4 comparison; then freeze validated independent production allocations.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start-index',type=int,default=1)
    parser.add_argument('--workers',type=int,default=16)
    main(parser.parse_args())
