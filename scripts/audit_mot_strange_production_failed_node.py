"""Read-only, single-process refinement of the first finite production failure.

Never edits production rays, changes their classifications, or resumes a pool.
The extra fixed settings are diagnostic evidence for a separately validated
revision; agreement here alone is not a campaign acceptance certificate.
"""
from pathlib import Path
import hashlib
import os
import shutil
import time

import numpy as np

import run_mot_strange_finite_capture_refined_campaign as campaign


def main():
    root = campaign.OUTPUT
    output = root/'validation/production_failed_node_20261026'
    output.mkdir(exist_ok=True)
    original = campaign.original
    manifest = original.read(root/'campaign_manifest.json')
    assert original.SOURCE_SHA == manifest['source_sha256']
    assert original.sha(campaign.__file__) == manifest['finite_runner_sha256']
    ray = root/'production/rays/d773_p006.npz'
    ray_hash = original.sha(ray)
    source_hash = original.sha(__file__)
    plan = dict(disc=773, point=6, speed_m_per_s=25.125, duration_s=2.,
                time_steps_s=[.3125e-6, .15625e-6, .078125e-6],
                physical_source_sha256=original.SOURCE_SHA,
                runner_sha256=source_hash, original_ray_sha256=ray_hash,
                purpose='diagnostic only; preserve original unknown and all campaign gates',
                single_process=True, worker_pool=False)
    path = output/'manifest.json'
    if path.exists():
        assert original.read(path) == plan
    else:
        original.write_json(path, plan)
        shutil.copy2(__file__, output/Path(__file__).name)
        shutil.copy2(ray, output/ray.name)
    original.write_json(output/'process.json', dict(pid=os.getpid(),
                        started_at_utc=original.utc(), command=os.sys.argv))
    with np.load(ray) as saved:
        r, direction = saved['position_m'], saved['direction']
        assert 25.125 in saved['speeds'][saved['codes'] < 0]
    data = campaign.load_data()
    cases = []
    began = time.perf_counter()
    for dt in plan['time_steps_s']:
        target = output/f'fixed_{dt:.12g}.json'
        if target.exists():
            row = original.read(target)
        else:
            start = time.perf_counter()
            result = campaign.fast.classify(r, 25.125*direction, dt, 2., data)
            row = dict(dt_s=dt, raw_code=int(result[0]),
                       finite_code=campaign.finite_code(result, 2.),
                       endpoint_time_s=float(result[1]), position_m=result[2].tolist(),
                       velocity_m_per_s=result[3].tolist(),
                       first_core_capture_time_s=float(result[4]),
                       wall_s=time.perf_counter()-start)
            original.write_json(target, row)
        cases.append(row)
        original.write_json(output/'progress.json', dict(completed=len(cases),
                            requested=len(plan['time_steps_s']), cases=cases))
        print(row, flush=True)
    assert original.sha(ray) == ray_hash, 'Original failed ray changed during audit'
    original.write_json(output/'completion.json', dict(completed_at_utc=original.utc(),
                        cases=cases, classifications_agree=len({c['finite_code'] for c in cases}) == 1,
                        original_ray_unchanged=True, campaign_acceptance=False,
                        wall_s=time.perf_counter()-began))


if __name__ == '__main__':
    main()
