"""Independent longer retention check of the refined production failure node."""
from pathlib import Path
import os
import shutil
import time
import numpy as np
import run_mot_strange_finite_capture_refined_campaign as campaign


def main():
    original = campaign.original
    root = campaign.OUTPUT
    output = root/'validation/production_failed_node_20261026/retention'
    output.mkdir(exist_ok=True)
    source = root/'production/rays/d773_p006.npz'
    manifest = dict(physical_source_sha256=original.SOURCE_SHA,
                    runner_sha256=original.sha(__file__),
                    original_ray_sha256=original.sha(source),
                    time_steps_s=[.15625e-6, .078125e-6], duration_s=.1,
                    early_exit=False, diagnostic_only=True)
    path = output/'manifest.json'
    if path.exists():
        assert original.read(path) == manifest
    else:
        original.write_json(path, manifest)
        shutil.copy2(__file__, output/Path(__file__).name)
    original.write_json(output/'process.json', dict(pid=os.getpid(),
                        started_at_utc=original.utc(), command=os.sys.argv))
    with np.load(source) as a:
        r, v = a['position_m'], 25.125*a['direction']
    data = campaign.load_data()
    rows = []
    for dt in manifest['time_steps_s']:
        path = output/f'fixed_{dt:.12g}.json'
        if path.exists():
            row = original.read(path)
        else:
            began = time.perf_counter()
            result = campaign.fast.classify(r, v, dt, .1, data, False)
            row = dict(dt_s=dt, raw_code=int(result[0]),
                       endpoint_time_s=float(result[1]), position_m=result[2].tolist(),
                       velocity_m_per_s=result[3].tolist(),
                       first_core_capture_time_s=float(result[4]),
                       final_window_max_radius_m=float(result[5]),
                       wall_s=time.perf_counter()-began,
                       bounded_final_core=bool(result[0] != 0 and result[5] <= .002))
            original.write_json(path, row)
        rows.append(row)
        original.write_json(output/'progress.json', dict(completed=len(rows), requested=2, cases=rows))
        print(row, flush=True)
    original.write_json(output/'completion.json', dict(cases=rows,
                        passed=all(row['bounded_final_core'] for row in rows),
                        completed_at_utc=original.utc(), original_ray_unchanged=True,
                        diagnostic_only=True))
    assert original.sha(source) == manifest['original_ray_sha256']


if __name__ == '__main__':
    main()
