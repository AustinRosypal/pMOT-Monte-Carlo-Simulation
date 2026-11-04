"""Audit the completed censored pilot; do not modify its classifications."""
import complete_mot_strange_diagnostic_masks as diagnostic
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import t

original = diagnostic.original
OUTPUT = diagnostic.OUTPUT


def main():
    report = diagnostic.read(OUTPUT/'diagnostic_completion.json')
    assert report['masks_complete']
    with np.load(OUTPUT/'disc_capture_bounds.npz') as saved:
        masks, velocity, areas = saved['masks'], saved['velocity_m_per_s'], saved['disc_areas_m2']
    weight = velocity**3*np.exp(-velocity**2/original.THERMAL_SCALE_M2_PER_S2)
    candidates = []
    for bound, capture in (('lower', masks == 1), ('upper', masks != 0)):
        samples = original.LOADING_RATE_PREFACTOR*areas[:, None]*np.trapezoid(
            np.concatenate((np.zeros((*capture.shape[:2], 1)), capture*weight), axis=2),
            np.r_[0., velocity], axis=2)
        mean = float(samples.mean())
        observed = float(samples.mean(axis=1).var(ddof=1))
        within = float(samples.var(axis=1, ddof=1).mean())
        between = max(0., observed-within/original.POINTS)
        n = 200
        conservative_variance = max(observed, between+within/original.POINTS)
        while 1.5*t.ppf(.975, n-1)**2*conservative_variance/(.05*mean)**2 > n:
            n += 25
        candidates.append(dict(bound=bound, pilot_mean_atoms_per_s=mean,
            pilot_variance_of_disc_means=observed, within_disc_variance=within,
            nonnegative_between_disc_variance=between, target_statistical_relative_half_width=.05,
            suggested_discs_at_20_points=n, suggested_rays=n*original.POINTS))
    original.write_json(OUTPUT/'prospective_allocation.json', dict(status='not frozen; observation-time definition pending',
        method='independent pilot, 1.5 variance safety factor; conservative nested variance cross-check',
        candidates=candidates))
    with np.load(original.OUTPUT/'pilot/geometry.npz') as saved:
        positions, directions, impacts = saved['positions_m'], saved['directions'], saved['impacts_m']
    with np.load(original.OUTPUT/'compiled_inputs.npz') as saved:
        data = tuple(saved[f'array_{i}'] for i in range(12))
    timeout_taxonomy = []
    for d, p, index in np.argwhere(masks < 0):
        speed = float(velocity[index])
        with np.load(OUTPUT/'rays'/f'd{d:03d}_p{p:03d}.npz') as saved:
            ev = saved['evaluations']; methods = saved['evaluation_methods']
            selected = ev[(ev[:, 0] == speed) & (methods == 1) & (ev[:, 2] == 2.)]
            retained = len(selected) >= 2 and np.all(selected[-2:, 3] == -1)
            timeout_taxonomy.append(dict(disc=int(d), point=int(p), speed_m_per_s=speed,
                both_adaptive_settings_nonterminal_at_two_seconds=bool(retained),
                endpoints_m=selected[-2:, 5:8].tolist(),
                velocities_m_per_s=selected[-2:, 8:11].tolist()))
    original.write_json(OUTPUT/'validation/indeterminate_node_taxonomy.json', dict(nodes=timeout_taxonomy,
        final_rate_status='withheld under the current eventual-capture interpretation'))
    path = OUTPUT/'validation/core_retention_audit.json'
    if path.exists():
        print(json.dumps(diagnostic.read(path))); return
    jobs = []
    for d in range(len(masks)):
        options = []
        for p in range(original.POINTS):
            captured = velocity[masks[d, p] == 1]
            if len(captured):
                options.append((impacts[d, p], p, float(captured.max()), float(captured[len(captured)//2])))
        if options:
            _, p, high, middle = max(options)
            for speed in sorted(set((high, middle))):
                jobs.append((d, p, speed, positions[d, p], directions[d], data))
    began = time.perf_counter()
    original.write_json(OUTPUT/'validation/audit_process.json', dict(pid=os.getpid(), started_at_utc=original.utc(),
        requested_cases=len(jobs), status='central-capture retention audit only'))
    cases = []
    with ProcessPoolExecutor(max_workers=16) as pool:
        for result in pool.map(original.retention_worker, jobs):
            cases.append(result)
            original.write_json(OUTPUT/'validation/core_retention_progress.json', dict(
                completed=len(cases), requested=len(jobs), failures=sum(not c['passed'] for c in cases)))
    passed = bool(cases) and all(c['passed'] for c in cases)
    report = dict(passed=passed, cases=cases, wall_s=time.perf_counter()-began,
        selection='outermost captured impact parameter in every disc with capture; highest and median captured speed',
        criterion='both 2.5 and 1.25 microsecond steps bounded in final 25 ms central core; endpoints agree within 10 micrometers',
        classification_source='known central captures only; censored nodes excluded from this retention audit, not relabeled')
    original.write_json(path, report)
    print(json.dumps(dict(passed=passed, cases=len(cases), failures=sum(not c['passed'] for c in cases),
                          wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    main()
