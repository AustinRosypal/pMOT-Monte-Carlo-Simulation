"""Full fixed-grid pilot cross-check, preserving all indeterminate outcomes."""
import complete_mot_strange_diagnostic_masks as diagnostic
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

import numpy as np

original = diagnostic.original
OUTPUT = diagnostic.OUTPUT
CASES = ((0, 0), (17, 13), (0, 4))


def worker(payload):
    d, p, data = payload
    diagnostic.ce.classify = diagnostic.fast.classify
    path = OUTPUT/'validation/full_fixed_grid'/f'd{d:03d}_p{p:03d}.npz'
    if path.exists():
        with np.load(path) as saved:
            if str(saved['status']) == 'complete':
                return dict(disc=d, point=p, passed=bool(saved['passed']),
                    disagreements=saved['disagreements'].tolist(), reused=True)
    with np.load(OUTPUT/'rays'/path.name) as saved:
        r, direction = saved['position_m'], saved['direction']
        existing_ev = saved['evaluations']; existing_methods = saved['evaluation_methods']
    with np.load(OUTPUT/'disc_capture_bounds_refined.npz') as saved:
        expected_main = saved['masks'][d, p]
    with np.load(OUTPUT/'rays'/path.name) as saved:
        expected = np.r_[expected_main, saved['codes'][len(expected_main):]]
    speeds = diagnostic.VELOCITIES
    records, methods, codes, reused_nodes = [], [], [], []
    disagreements = []
    began = time.perf_counter()
    prior_results = {}
    if path.exists():
        with np.load(path) as saved:
            records = saved['evaluations'].tolist(); methods = saved['evaluation_methods'].tolist()
            prior_results = dict(zip(saved['speeds'].tolist(), saved['codes'].tolist()))
    def save(status):
        diagnostic.ce._atomic_npz(path, status=np.array(status), speeds=speeds[:len(codes)],
            codes=np.asarray(codes), expected_codes=expected[:len(codes)],
            evaluations=np.asarray(records, dtype=float).reshape((-1, 12)),
            evaluation_methods=np.asarray(methods, dtype=np.int8),
            reused_nodes=np.asarray(reused_nodes),
            disagreements=np.asarray(disagreements, dtype=float).reshape((-1, 3)),
            passed=np.array(not disagreements), elapsed_wall_s=np.array(time.perf_counter()-began))
    for i, speed in enumerate(speeds):
        if speed in prior_results:
            code = int(prior_results[speed]); reused_nodes.append(i)
        else:
            prior = existing_ev[(existing_ev[:, 0] == speed) & (existing_methods == 0)]
            if len(prior) >= 2 and prior[-1, 3] >= 0 and prior[-1, 3] == prior[-2, 3]:
                code = int(prior[-1, 3]); reused_nodes.append(i)
                records.extend(prior[-2:].tolist()); methods.extend((0, 0))
            elif expected[i] < 0:
                audit = OUTPUT/'validation/remaining_fixed_nodes'/f'd{d:03d}_p{p:03d}_v{speed:06.2f}.json'
                result = original.read(audit)
                assert result['classifications_agree']
                code = int(result['code']); reused_nodes.append(i)
                for dt, row in zip(result['time_steps_s'], result['results']):
                    records.append([speed, dt, 2., row['code'], row['time_s'],
                        *row['position_m'], *row['velocity_m_per_s'], row['first_core_capture_time_s']])
                    methods.append(0)
            else:
                code = diagnostic.ce._resolve_fixed_node(r, direction, speed, data, records, methods)
        codes.append(code)
        if code != expected[i]:
            disagreements.append([float(speed), int(expected[i]), int(code)])
        if i % 16 == 15:
            save('running')
    save('complete')
    return dict(disc=d, point=p, passed=not disagreements, disagreements=disagreements,
                evaluated_nodes=len(codes), reused_nodes=len(reused_nodes),
                original_masks_unchanged=True, wall_s=time.perf_counter()-began)


def main():
    assert original.SOURCE_SHA == original.read(original.OUTPUT/'campaign_manifest.json')['source_sha256']
    path = OUTPUT/'validation/full_fixed_grid.json'
    if path.exists():
        print(original.read(path)); return
    original.write_json(OUTPUT/'validation/full_fixed_grid_process.json', dict(pid=os.getpid(),
        started_at_utc=original.utc(), cases=CASES, runner_sha256=original.sha(__file__)))
    with np.load(original.OUTPUT/'compiled_inputs.npz') as saved:
        data = tuple(saved[f'array_{i}'] for i in range(12))
    cases = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, (d, p, data)) for d, p in CASES]
        for future in as_completed(futures):
            cases.append(future.result())
            original.write_json(OUTPUT/'validation/full_fixed_grid_progress.json', dict(
                completed=len(cases), requested=len(CASES), failures=sum(not c['passed'] for c in cases)))
    result = dict(completed_at_utc=original.utc(), passed=all(c['passed'] for c in cases),
        cases=cases, source_sha256=original.SOURCE_SHA,
        scope='complete sampled speed grid, two originally predetermined pilot rays plus one challenging captured ray; preserved -1 is numerical agreement, not a terminal event')
    original.write_json(path, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
