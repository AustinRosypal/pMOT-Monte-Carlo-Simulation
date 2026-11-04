"""Bounded diagnostic trace using the prospective endpoint-observation loop.

Uses the frozen force, RK4 stages, inputs and retention definition. No campaign
mask, allocation, source snapshot or failed gate is edited by this diagnostic.
"""
from pathlib import Path
import os
import shutil
import time
import numpy as np
import mot_strange_retained_capture as model
import mot_strange_retained_numerics as numerical


def main():
    original = model.original
    root = model.OUTPUT
    output = root/'validation/production_failed_node_20261027/endpoint_trace_v1'
    output.mkdir(exist_ok=True)
    ray = root/'production/rays/d210_p000.npz'
    manifest = original.read(root/'campaign_manifest.json')
    assert original.SOURCE_SHA == manifest['source_sha256']
    assert original.sha(model.__file__) == manifest['retained_helper_sha256']
    plan = dict(disc=210, point=0, speed_m_per_s=33.625, duration_s=2.,
        steps_s=[.625e-6, .3125e-6],
        requested_trace_times_s=[0., .0001, .001, .005, .01, .02, .05, .1, .2,
                                 .5, 1., 1.5, 1.975, 2.],
        trace_convention='first actual step endpoint at or after each requested time; no interpolation',
        trace_columns=['time_s', 'x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s',
                       'axis_x', 'axis_y', 'axis_z', 'field_norm_t', 'first_core_s', 'window_max_m'],
        physical_source_sha256=original.SOURCE_SHA,
        helper_sha256=original.sha(model.__file__),
        diagnostic_sha256=original.sha(__file__),
        numerical_observation_sha256=original.sha(numerical.__file__),
        original_ray_sha256=original.sha(ray),
        purpose='localize endpoint sensitivity; observation-loop revision only; not campaign acceptance',
        force_and_step_unchanged=True, single_process=True, campaign_acceptance=False)
    path = output/'manifest.json'
    if path.exists():
        assert original.read(path) == plan
    else:
        original.write_json(path, plan)
        for p in (Path(__file__), Path(numerical.__file__), ray):
            shutil.copy2(p, output/p.name)
    assert not (output/'completion.json').exists(), 'Diagnostic already complete'
    original.write_json(output/'process.json', dict(pid=os.getpid(),
        started_at_utc=original.utc(), command=os.sys.argv))
    with np.load(ray) as a:
        r, direction = a['position_m'], a['direction']
    data = model.parent.load_data()
    cases, results = [], []
    for dt in plan['steps_s']:
        start = time.perf_counter()
        try:
            result = numerical.traced(r, 33.625*direction, dt, 2., data,
                                      plan['requested_trace_times_s'])
        except Exception as error:
            original.write_json(output/'failure.json', dict(at_utc=original.utc(),
                dt_s=dt, error=repr(error), campaign_acceptance=False))
            raise
        row = dict(dt_s=dt, raw_code=int(result[0]), time_s=float(result[1]),
            position_m=result[2].tolist(), velocity_m_per_s=result[3].tolist(),
            first_core_time_s=float(result[4]), final_window_maximum_radius_m=float(result[5]),
            retained_outcome=model.outcome(result), trace_rows=len(result[6]),
            window_includes_endpoint=result[5] >= np.linalg.norm(result[2]),
            wall_s=time.perf_counter()-start)
        row['window_includes_endpoint'] = bool(row['window_includes_endpoint'])
        stem = output/f'fixed_{dt:.12g}'
        np.savez_compressed(stem.with_suffix('.npz'), trace=result[6])
        original.write_json(stem.with_suffix('.json'), row)
        cases.append(row)
        results.append(result)
        original.write_json(output/'progress.json', dict(completed=len(cases), requested=2, cases=cases))
        print(row, flush=True)
    assert original.sha(ray) == plan['original_ray_sha256']
    original.write_json(output/'completion.json', dict(completed_at_utc=original.utc(),
        cases=cases, endpoint_difference_m=float(np.linalg.norm(results[0][2]-results[1][2])),
        window_difference_m=float(abs(results[0][5]-results[1][5])),
        pair_outcome=model.pair_outcome(*results), original_ray_unchanged=True,
        campaign_acceptance=False))


if __name__ == '__main__':
    main()
