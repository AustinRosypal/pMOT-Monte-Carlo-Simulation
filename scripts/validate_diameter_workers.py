"""Fresh spawned-worker checks for compiled-cache and launch-state coverage."""
import run_population_diameter_campaign as runner
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pmot.mot_multilevel.campaign_execution import _resolve_node, _resolve_fixed_node


def check(payload):
    diameter, disc, point, r, direction, speed, data = payload
    # The adaptive call must be the first numerical call in the child process.
    adaptive = _resolve_node(r,direction,speed,data,[],[])
    fixed = _resolve_fixed_node(r,direction,speed,data,[],[])
    return {'diameter_mm':diameter,'disc':disc,'point':point,'speed_m_per_s':speed,
            'adaptive':adaptive,'fixed':fixed,'passed':adaptive==fixed and fixed>=0}


if __name__ == '__main__':
    root=runner.ROOT/'outputs/statistics/mot_multilevel_population_rate_v1/cooling_diameter_20260930_fixed_intensity_baseline'
    positions,directions=runner.ce.geometry(2026093001,32,64,15.)
    jobs=[]
    for i,diameter in enumerate(runner.DIAMETERS):
        point=runner.DiameterPoint(i,diameter,.027*(diameter/12.7)**2,-2*np.pi*15e6)
        data=runner.prepare_diameter(point.config())
        for disc,pindex in ((0,0),(17,31)):
            for speed in (.25,10.,40.,120.):
                jobs.append((diameter,disc,pindex,positions[disc,pindex],directions[disc],speed,data))
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(check,jobs))
    result={'source_sha256':runner.CORE_SHA,'cache_directory':runner.os.environ['NUMBA_CACHE_DIR'],
            'passed':all(r['passed'] for r in rows),'rows':rows}
    runner.rc.write_json(root/'validation/fresh_worker_checks.json',result)
    print('fresh worker checks passed:',result['passed'],flush=True)
    assert result['passed']
