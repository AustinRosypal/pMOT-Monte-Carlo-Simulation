"""Resolve delayed-escape pilot nodes in a provenance-preserving overlay."""
import complete_mot_strange_diagnostic_masks as diagnostic
from concurrent.futures import ProcessPoolExecutor
import os
import time

import numpy as np

original = diagnostic.original
OUTPUT = diagnostic.OUTPUT


def worker(payload):
    node, r, direction, data = payload
    speed = node['speed_m_per_s']
    began = time.perf_counter()
    results = [diagnostic.fast.classify(r, speed*direction, dt, 2., data)
               for dt in (1.25e-6, .625e-6)]
    code = int(results[0][0]) if results[0][0] == results[1][0] else -1
    return dict(disc=node['disc'], point=node['point'], speed_m_per_s=speed,
        code=code, time_steps_s=[1.25e-6, .625e-6], duration_s=2.,
        results=[dict(code=int(a[0]), time_s=float(a[1]), position_m=a[2].tolist(),
                      velocity_m_per_s=a[3].tolist(), first_core_capture_time_s=float(a[4])) for a in results],
        wall_s=time.perf_counter()-began)


def main():
    output = OUTPUT/'validation/additional_fixed_nodes.json'
    if output.exists():
        print(diagnostic.read(output)); return
    taxonomy = diagnostic.read(OUTPUT/'validation/indeterminate_node_taxonomy.json')['nodes']
    targets = [n for n in taxonomy if not n['both_adaptive_settings_nonterminal_at_two_seconds']]
    with np.load(original.OUTPUT/'pilot/geometry.npz') as saved:
        positions, directions = saved['positions_m'], saved['directions']
    with np.load(original.OUTPUT/'compiled_inputs.npz') as saved:
        data = tuple(saved[f'array_{i}'] for i in range(12))
    jobs = [(n, positions[n['disc'], n['point']], directions[n['disc']], data) for n in targets]
    original.write_json(OUTPUT/'validation/fixed_overlay_process.json', dict(pid=os.getpid(),
        started_at_utc=original.utc(), requested_cases=len(jobs)))
    cases = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for case in pool.map(worker, jobs):
            cases.append(case)
            original.write_json(OUTPUT/'validation/fixed_overlay_progress.json', dict(
                completed=len(cases), requested=len(jobs), resolved=sum(c['code'] >= 0 for c in cases)))
    original.write_json(output, dict(cases=cases, source_sha256=original.SOURCE_SHA,
        runner_sha256=original.sha(__file__), original_masks_unchanged=True))
    with np.load(OUTPUT/'disc_capture_bounds.npz') as saved:
        masks = saved['masks'].copy(); areas = saved['disc_areas_m2']; velocity = saved['velocity_m_per_s']
    for case in cases:
        index = int(np.flatnonzero(velocity == case['speed_m_per_s'])[0])
        assert masks[case['disc'], case['point'], index] == -1
        masks[case['disc'], case['point'], index] = case['code']
    lower, upper, envelope = diagnostic.calculate_bounds(masks, areas, velocity)
    report = dict(status='final rate withheld; refined diagnostic bounds only',
        at_utc=original.utc(), complete_direction_discs=len(masks),
        unknown_positive_speed_nodes=int(np.count_nonzero(masks < 0)),
        lower_capture_integral_atoms_per_s=float(lower.mean()),
        upper_capture_integral_atoms_per_s=float(upper.mean()),
        approximate_sampling_95_envelope_atoms_per_s=list(envelope),
        additional_fixed_resolved_nodes=sum(c['code'] >= 0 for c in cases),
        thermal_conditions=diagnostic.read(OUTPUT/'latest_capture_bounds.json')['thermal_conditions'],
        provenance='original masks retained; additional fixed-node evidence applied as a separate overlay')
    original.write_json(OUTPUT/'refined_capture_bounds.json', report)
    diagnostic.ce._atomic_npz(OUTPUT/'disc_capture_bounds_refined.npz', masks=masks,
        velocity_m_per_s=velocity, disc_areas_m2=areas,
        lower_loading_atoms_per_s=lower, upper_loading_atoms_per_s=upper)
    print(report, flush=True)


if __name__ == '__main__':
    main()
