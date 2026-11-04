"""Final dual-fixed-step audit of retained pilot nodes; preserve their labels."""
import complete_mot_strange_diagnostic_masks as diagnostic
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

import numpy as np

original = diagnostic.original
OUTPUT = diagnostic.OUTPUT


def retained_fixed_pair(path, speed):
    with np.load(path) as saved:
        ev = saved['evaluations']; methods = saved['evaluation_methods']
        pairs = []
        for dt in (1.25e-6, .625e-6):
            rows = ev[(ev[:, 0] == speed) & (ev[:, 1] == dt)
                      & (ev[:, 2] == 2.) & (methods == 0)]
            if not len(rows):
                return None
            row = rows[-1]
            pairs.append(dict(code=int(row[3]), time_s=float(row[4]),
                position_m=row[5:8].tolist(), velocity_m_per_s=row[8:11].tolist(),
                first_core_capture_time_s=float(row[11])))
        return pairs


def worker(payload):
    d, p, speed, r, direction, data = payload
    path = OUTPUT/'validation/remaining_fixed_nodes'/f'd{d:03d}_p{p:03d}_v{speed:06.2f}.json'
    if path.exists():
        return diagnostic.read(path)
    began = time.perf_counter()
    results = retained_fixed_pair(OUTPUT/'rays'/f'd{d:03d}_p{p:03d}.npz', speed)
    reused = results is not None
    if results is None:
        results = []
        for dt in (1.25e-6, .625e-6):
            result = diagnostic.fast.classify(r, speed*direction, dt, 2., data)
            results.append(dict(code=int(result[0]), time_s=float(result[1]),
                position_m=result[2].tolist(), velocity_m_per_s=result[3].tolist(),
                first_core_capture_time_s=float(result[4])))
            # Save each completed integration before beginning the next one.
            original.write_json(path.with_name(path.stem+'_partial.json'), dict(
                disc=d, point=p, speed_m_per_s=speed, completed_settings=len(results),
                time_steps_s=[1.25e-6, .625e-6], duration_s=2., results=results))
    agreement = results[0]['code'] == results[1]['code']
    separation = float(np.linalg.norm(np.asarray(results[0]['position_m'])-results[1]['position_m']))
    result = dict(disc=d, point=p, speed_m_per_s=speed, time_steps_s=[1.25e-6, .625e-6],
        duration_s=2., reused_existing_pair=reused, results=results,
        classifications_agree=agreement, endpoint_difference_m=separation,
        code=results[0]['code'] if agreement else -1,
        interpretation='nonterminal stays indeterminate under eventual-capture interpretation',
        wall_s=time.perf_counter()-began, source_sha256=original.SOURCE_SHA)
    original.write_json(path, result)
    return result


def main():
    with np.load(OUTPUT/'disc_capture_bounds_refined.npz') as saved:
        masks = saved['masks']; velocity = saved['velocity_m_per_s']
    with np.load(original.OUTPUT/'pilot/geometry.npz') as saved:
        positions, directions = saved['positions_m'], saved['directions']
    with np.load(original.OUTPUT/'compiled_inputs.npz') as saved:
        data = tuple(saved[f'array_{i}'] for i in range(12))
    path = OUTPUT/'validation/remaining_fixed_nodes.json'
    if path.exists():
        print(diagnostic.read(path)); return
    targets = [(int(d), int(p), float(velocity[i]), positions[d, p], directions[d], data)
               for d, p, i in np.argwhere(masks < 0)]
    original.write_json(OUTPUT/'validation/remaining_fixed_process.json', dict(pid=os.getpid(),
        started_at_utc=original.utc(), requested_cases=len(targets), runner_sha256=original.sha(__file__)))
    cases = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, target) for target in targets]
        for future in as_completed(futures):
            cases.append(future.result())
            original.write_json(OUTPUT/'validation/remaining_fixed_progress.json', dict(
                completed=len(cases), requested=len(targets),
                nonterminal_pairs=sum(c['code'] < 0 for c in cases),
                terminal_pairs=sum(c['code'] >= 0 for c in cases)))
    original.write_json(path, dict(completed_at_utc=original.utc(), cases=sorted(cases,
        key=lambda c: (c['disc'], c['point'], c['speed_m_per_s'])),
        classifications_agree=all(c['classifications_agree'] for c in cases),
        terminal_pairs=sum(c['code'] >= 0 for c in cases),
        nonterminal_pairs=sum(c['code'] < 0 for c in cases),
        original_masks_unchanged=True))
    print(dict(completed=len(cases), terminal=sum(c['code'] >= 0 for c in cases),
               nonterminal=sum(c['code'] < 0 for c in cases)), flush=True)


if __name__ == '__main__':
    main()
