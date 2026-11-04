"""Single-process diagnostic; never reclassifies or resumes the failed campaign."""
from pathlib import Path
import argparse
import os
import shutil
import time

import numpy as np
import mot_strange_retained_capture as model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinements', type=float, nargs=2, default=[16., 32.])
    parser.add_argument('--revision', default='')
    parser.add_argument('--fixed-time-steps', type=float, nargs=2)
    parser.add_argument('--budgeted-adaptive', action='store_true')
    args = parser.parse_args()
    assert all(x >= 16 and np.isfinite(x) for x in args.refinements)
    assert args.refinements[1] == 2*args.refinements[0]
    assert not args.revision or args.revision.replace('_', '').isalnum()
    if args.fixed_time_steps:
        assert not args.budgeted_adaptive
        assert args.revision
        assert all(0 < x < model.FIXED_PAIRS[-1][-1] for x in args.fixed_time_steps)
        assert args.fixed_time_steps[0] == 2*args.fixed_time_steps[1]
    original = model.original
    budget = None
    if args.budgeted_adaptive:
        import mot_strange_retained_budget as budget
        assert args.revision
    root = model.OUTPUT
    output = root/'validation/production_failed_node_20261027'
    if args.revision:
        output = output/args.revision
    output.mkdir(parents=True, exist_ok=True)
    ray = root/'production/rays/d210_p000.npz'
    manifest = original.read(root/'campaign_manifest.json')
    assert original.SOURCE_SHA == manifest['source_sha256']
    assert original.sha(model.__file__) == manifest['retained_helper_sha256']
    plan = dict(disc=210, point=0, speed_m_per_s=33.625, duration_s=2.,
                adaptive_refinements=[] if args.fixed_time_steps else args.refinements,
                fixed_time_steps_s=args.fixed_time_steps or [],
                adaptive_attempt_budget=6000000 if budget else 2000000,
                budgeted_numerics_sha256=original.sha(budget.__file__) if budget else None,
                original_ray_sha256=original.sha(ray),
                physical_source_sha256=original.SOURCE_SHA,
                helper_sha256=original.sha(model.__file__),
                diagnostic_sha256=original.sha(__file__), single_process=True,
                endpoint_tolerance_m=model.ENDPOINT_TOLERANCE_M,
                purpose='diagnostic only; raw unknown and all campaign gates preserved')
    target = output/'manifest.json'
    if target.exists():
        assert original.read(target) == plan
    else:
        original.write_json(target, plan)
        shutil.copy2(__file__, output/Path(__file__).name)
        shutil.copy2(ray, output/ray.name)
        if budget:
            shutil.copy2(budget.__file__, output/Path(budget.__file__).name)
    assert not (output/'completion.json').exists(), 'Diagnostic already complete'
    original.write_json(output/'process.json', dict(pid=os.getpid(),
                        started_at_utc=original.utc(), command=os.sys.argv))
    with np.load(ray) as saved:
        r, direction = saved['position_m'], saved['direction']
        assert 33.625 in saved['speeds'][saved['codes'] < 0]
        rows = saved['evaluations'][saved['evaluations'][:, 0] == 33.625]
        fixed = [model.unpack(row) for row in rows[-2:]]
    original.write_json(output/'saved_failure.json', dict(
        saved_setting_outcomes=[model.outcome(model.unpack(row)) for row in rows],
        last_pair_endpoint_difference_m=float(np.linalg.norm(fixed[0][2]-fixed[1][2])),
        last_pair_window_difference_m=float(abs(fixed[0][5]-fixed[1][5])),
        last_pair_outcome=model.pair_outcome(*fixed), campaign_acceptance=False))
    data = model.parent.load_data()
    began = time.perf_counter()
    cases, results = [], []
    settings = args.fixed_time_steps or plan['adaptive_refinements']
    method = 'fixed' if args.fixed_time_steps else 'adaptive'
    for parameter in settings:
        path = output/f'{method}_{parameter:.12g}.json'
        if path.exists():
            row = original.read(path)
            result = (row['raw_code'], row['time_s'], np.array(row['position_m']),
                      np.array(row['velocity_m_per_s']), row['first_core_time_s'],
                      row['final_window_maximum_radius_m'])
        else:
            start = time.perf_counter()
            try:
                if args.fixed_time_steps:
                    result = model.fast.classify(r, 33.625*direction, parameter, 2., data, False)
                else:
                    result = (budget.retained_adaptive(r, 33.625*direction, 2., data, parameter)
                              if budget else model.fast.integrate(r, 33.625*direction, 2., data, parameter, True))
            except Exception as error:
                original.write_json(output/'failure.json', dict(at_utc=original.utc(),
                    method=method, parameter=parameter, error=repr(error),
                    completed_cases=cases, campaign_acceptance=False,
                    original_ray_unchanged=original.sha(ray) == plan['original_ray_sha256']))
                raise
            row = dict(method=method, parameter=parameter, raw_code=int(result[0]), time_s=float(result[1]),
                       position_m=result[2].tolist(), velocity_m_per_s=result[3].tolist(),
                       first_core_time_s=float(result[4]),
                       final_window_maximum_radius_m=float(result[5]),
                       retained_outcome=model.outcome(result), wall_s=time.perf_counter()-start)
            row['dt_s' if args.fixed_time_steps else 'refinement'] = parameter
            original.write_json(path, row)
        cases.append(row)
        results.append(result)
        original.write_json(output/'progress.json', dict(completed=len(cases), requested=2, cases=cases))
        print(row, flush=True)
    assert original.sha(ray) == plan['original_ray_sha256']
    original.write_json(output/'completion.json', dict(completed_at_utc=original.utc(),
        cases=cases, pair_outcome=model.pair_outcome(*results),
        endpoint_difference_m=float(np.linalg.norm(results[0][2]-results[1][2])),
        window_difference_m=float(abs(results[0][5]-results[1][5])),
        original_ray_unchanged=True, campaign_acceptance=False,
        wall_s=time.perf_counter()-began))


if __name__ == '__main__':
    main()
