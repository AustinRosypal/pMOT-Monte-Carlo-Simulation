"""Complete pilot masks while preserving censoring; never publish a final rate.

The original scientific gate and physical source remain immutable. This pass
retains an explicit -1 for every indeterminate node and integrates lower/upper
capture masks. New timeouts get two adaptive checks through 2 s; unperformed
fixed checks remain pending, not silently certified as converged.
"""
import run_corrected_mot_strange_loading as original
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import os
import shutil
import time
import traceback

import numpy as np
from scipy.stats import t
from numba.core.caching import NullCache
from pmot.mot_multilevel import adaptive as adaptive_core

# Numba disk-cache keys containing callback dispatchers can retain dead weak
# references after loading signatures produced by another process. These
# callback-specialized integrator helpers are compiled in each process instead;
# the mathematical kernel and all physical inputs remain unchanged.
for dispatcher in (adaptive_core.derivative, adaptive_core.embedded_step,
                   adaptive_core.integrate_local_environment, original.fast.integrate):
    dispatcher._cache = NullCache()

ROOT = original.ROOT
SOURCE = original.OUTPUT
PRIOR_DIAGNOSTIC = SOURCE.with_name(SOURCE.name+'_diagnostic_completion_v2')
OUTPUT = SOURCE.with_name(SOURCE.name+'_diagnostic_completion_v3')
ce = original.ce
fast = original.fast
write_json, read = original.write_json, original.read
VELOCITIES = np.r_[ce.VELOCITIES, ce.TAIL_VELOCITIES]


def resolve_adaptive(r, direction, speed, data, records, methods):
    for duration in (.05, .1, .2, .4, 1., 2.):
        pair = [fast.integrate(r, speed*direction, duration, data, f) for f in (1., 2.)]
        for refinement, result in zip((1., 2.), pair):
            records.append([speed, .0005/refinement, duration, float(result[0]),
                            result[1], *result[2], *result[3], result[4]])
            methods.append(1)
        if pair[0][0] >= 0 and pair[0][0] == pair[1][0]:
            return int(pair[0][0])
        if pair[0][0] >= 0 and pair[1][0] >= 0:
            break
    return -1


def resolve_fixed_diagnostic(r, direction, speed, data, records, methods):
    for coarse, fine, duration in ce.LADDER[:4]:
        pair = [fast.classify(r, speed*direction, dt, duration, data) for dt in (coarse, fine)]
        for dt, result in zip((coarse, fine), pair):
            records.append([speed, dt, duration, float(result[0]), result[1],
                            *result[2], *result[3], result[4]])
            methods.append(0)
        if pair[0][0] >= 0 and pair[0][0] == pair[1][0]:
            return int(pair[0][0])
    return -1


def collect_ray(payload):
    d, p, r, direction, data = payload
    path = OUTPUT/'rays'/f'd{d:03d}_p{p:03d}.npz'
    prior = SOURCE/'pilot/rays'/path.name
    records, methods, existing = [], [], {}
    previous_diagnostic = PRIOR_DIAGNOSTIC/'rays'/path.name
    source = (path if path.exists() else previous_diagnostic if previous_diagnostic.exists() else prior)
    if source.exists():
        with np.load(source) as old:
            if str(old['status']) == 'resolved':
                if source != path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, path)
                return dict(disc=d, point=p, unknown_nodes=0, reused=True)
            if str(old['status']) == 'diagnostic_complete':
                if source != path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, path)
                return dict(disc=d, point=p, unknown_nodes=int(np.count_nonzero(old['codes'] < 0)), reused=True)
            records = old['evaluations'].tolist()
            methods = old['evaluation_methods'].tolist()
            existing = dict(zip(old['speeds'].tolist(), old['codes'].tolist()))
    began = time.perf_counter()
    codes = []
    fixed = set()
    for speed in existing:
        ev = [row for row, method in zip(records, methods) if row[0] == speed and method == 0]
        if len(ev) >= 2 and ev[-1][3] >= 0 and ev[-1][3] == ev[-2][3]:
            fixed.add(int(np.flatnonzero(VELOCITIES == speed)[0]))
    def save(status):
        ce._atomic_npz(path, speeds=VELOCITIES[:len(codes)], codes=np.asarray(codes, dtype=np.int8),
            evaluations=np.asarray(records, dtype=float).reshape((-1, 12)),
            evaluation_methods=np.asarray(methods, dtype=np.int8),
            position_m=r, direction=direction, status=np.array(status),
            source_status=np.array('derived diagnostic; final gate remains pending'),
            elapsed_wall_s=np.array(time.perf_counter()-began))
    try:
        for i, speed in enumerate(VELOCITIES):
            # Retain already censored nodes instead of repeatedly spending the
            # full final fixed ladder on a known nonterminal trajectory.
            code = int(existing[speed]) if speed in existing else resolve_adaptive(
                r, direction, speed, data, records, methods)
            codes.append(code)
            if i % 16 == 15 or code < 0:
                save('diagnostic_running')
        while True:
            needed = ce.boundary_node_indices(codes)-fixed
            if not needed:
                break
            for i in sorted(needed):
                if codes[i] >= 0:
                    codes[i] = resolve_fixed_diagnostic(r, direction, VELOCITIES[i], data, records, methods)
                fixed.add(i)
                save('diagnostic_running')
        save('diagnostic_complete')
        return dict(disc=d, point=p, unknown_nodes=int(np.count_nonzero(np.asarray(codes) < 0)), reused=False)
    except Exception:
        save('diagnostic_error')
        raise


def calculate_bounds(masks, areas, velocities):
    """Integrate C=1 and C!=0 separately; no imputed central capture label."""
    masks = np.asarray(masks)
    if masks.ndim != 3 or len(areas) != len(masks) or masks.shape[2] != len(velocities):
        raise ValueError('mask, area, and velocity dimensions disagree')
    if not np.all(np.isin(masks, (-1, 0, 1))):
        raise ValueError('unknown mask code')
    area = np.asarray(areas)[:, None]
    weight = velocities**3*np.exp(-velocities**2/original.THERMAL_SCALE_M2_PER_S2)
    values = []
    for capture in (masks == 1, masks != 0):
        spectrum = area*capture.mean(axis=1)
        values.append(original.LOADING_RATE_PREFACTOR*np.trapezoid(
            np.column_stack((np.zeros(len(masks)), spectrum*weight)), np.r_[0., velocities], axis=1))
    lower, upper = values
    n = len(masks)
    if n < 2:
        raise ValueError('at least two independent discs are needed')
    scale = t.ppf(.975, n-1)/np.sqrt(n)
    lo = max(0., float(lower.mean()-scale*lower.std(ddof=1)))
    hi = float(upper.mean()+scale*upper.std(ddof=1))
    if np.all(upper == 0):
        maximum = original.LOADING_RATE_PREFACTOR*np.pi*.015**2*np.trapezoid(
            np.r_[0., weight], np.r_[0., velocities])
        hi = float(maximum*(1-.025**(1/n)))
    return lower, upper, (lo, hi)


def report(discs):
    with np.load(SOURCE/'pilot/geometry.npz') as geom:
        areas = geom['areas_m2'][:discs]
    masks = np.empty((discs, original.POINTS, len(ce.VELOCITIES)), dtype=np.int8)
    tail = []
    for d in range(discs):
        for p in range(original.POINTS):
            with np.load(OUTPUT/'rays'/f'd{d:03d}_p{p:03d}.npz') as saved:
                assert str(saved['status']) in ('resolved', 'diagnostic_complete')
                np.testing.assert_array_equal(saved['speeds'], VELOCITIES)
                masks[d, p] = saved['codes'][:len(ce.VELOCITIES)]
                tail.extend(saved['codes'][len(ce.VELOCITIES):].tolist())
    lower, upper, envelope = calculate_bounds(masks, areas, ce.VELOCITIES)
    manifest = read(SOURCE/'campaign_manifest.json')
    summary = dict(at_utc=original.utc(), status='FINAL LOADING RATE WITHHELD; indeterminate nodes preserved',
        stage='diagnostic pilot completion', complete_direction_discs=discs,
        points_per_disc=original.POINTS, unknown_positive_speed_nodes=int(np.count_nonzero(masks < 0)),
        unknown_rays=int(np.count_nonzero(np.any(masks < 0, axis=2))),
        lower_capture_integral_atoms_per_s=float(lower.mean()),
        upper_capture_integral_atoms_per_s=float(upper.mean()),
        capture_ambiguity_width_atoms_per_s=float(upper.mean()-lower.mean()),
        approximate_sampling_95_envelope_atoms_per_s=list(envelope),
        lower_coefficient_m3_per_s=float(lower.mean()/original.N87),
        upper_coefficient_m3_per_s=float(upper.mean()/original.N87),
        thermal_conditions=manifest['thermal_conditions'],
        velocity_range_m_per_s=[0., 80.], tail_captured_nodes=tail.count(1),
        tail_indeterminate_nodes=tail.count(-1),
        limitations='bounds of the directly sampled finite velocity-grid integral, not a final physical loading prediction; sampling envelope uses two one-sided disc-cluster t bounds; fixed final-ladder and retention/precision/aperture gates remain pending')
    write_json(OUTPUT/'latest_capture_bounds.json', summary)
    write_json(OUTPUT/'estimates'/f'discs_{discs:03d}.json', summary)
    ce._atomic_npz(OUTPUT/'disc_capture_bounds.npz', masks=masks, velocity_m_per_s=ce.VELOCITIES,
        disc_areas_m2=areas, lower_loading_atoms_per_s=lower, upper_loading_atoms_per_s=upper)
    print('CAPTURE BOUNDS '+json.dumps(summary), flush=True)
    return summary


def run(workers):
    manifest = read(SOURCE/'campaign_manifest.json')
    assert manifest['source_sha256'] == original.SOURCE_SHA
    assert read(SOURCE/'validation/test_status.json')['passed']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specification = dict(parent_source_sha256=original.SOURCE_SHA,
        diagnostic_runner_sha256=original.sha(__file__), parent_campaign=str(SOURCE),
        purpose='complete censored pilot masks for lower/upper capture integrals; final rate remains withheld',
        cache_policy='callback-bearing adaptive dispatchers use process-local compilation; physical solver unchanged',
        prior_diagnostic_output=str(PRIOR_DIAGNOSTIC),
        thermal_conditions=manifest['thermal_conditions'],
        timeout_policy='preserve -1; dual-adaptive duration ladder through 2 s; short fixed edge checks; final fixed ladder not certified for new censored nodes')
    path = OUTPUT/'diagnostic_manifest.json'
    if path.exists():
        assert read(path) == specification
    else:
        write_json(path, specification)
        shutil.copy2(__file__, OUTPUT/Path(__file__).name)
        shutil.copy2(SOURCE/'pilot/geometry.npz', OUTPUT/'geometry.npz')
    write_json(OUTPUT/'process.json', dict(pid=os.getpid(), started_at_utc=original.utc(), workers=workers))
    with np.load(SOURCE/'compiled_inputs.npz') as saved:
        data = tuple(saved[f'array_{i}'] for i in range(12))
    with np.load(SOURCE/'pilot/geometry.npz') as saved:
        positions, directions = saved['positions_m'], saved['directions']
    completed = set()
    jobs = []
    for d in range(len(positions)):
        for p in range(original.POINTS):
            file = OUTPUT/'rays'/f'd{d:03d}_p{p:03d}.npz'
            if file.exists():
                with np.load(file) as saved:
                    if str(saved['status']) in ('resolved', 'diagnostic_complete'):
                        completed.add((d, p)); continue
            jobs.append((d, p, positions[d, p], directions[d], data))
    began = time.perf_counter(); last = 0.; reported = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = iter(jobs); pending = {}
        def submit():
            job = next(iterator, None)
            if job is not None:
                pending[pool.submit(collect_ray, job)] = job[:2]
                return True
            return False
        for _ in range(workers*2):
            if not submit(): break
        while pending:
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                identity = pending.pop(future)
                future.result()
                completed.add(identity); submit()
            prefix = 0
            while prefix < len(positions) and all((prefix, p) in completed for p in range(original.POINTS)):
                prefix += 1
            ready = [n for n in (4, 8, 16, 32) if n <= prefix and n not in reported]
            if ready:
                report(max(ready)); reported.update(n for n in (4, 8, 16, 32) if n <= max(ready))
            elapsed = time.perf_counter()-began
            if elapsed-last >= 30 or not pending:
                write_json(OUTPUT/'progress.json', dict(at_utc=original.utc(), completed_rays=len(completed),
                    requested_rays=len(positions)*original.POINTS, complete_prefix_discs=prefix,
                    elapsed_s=elapsed, status='diagnostic collection, not production'))
                last = elapsed
    summary = report(len(positions))
    write_json(OUTPUT/'diagnostic_completion.json', dict(completed_at_utc=original.utc(),
        masks_complete=True, final_loading_rate_available=False,
        unknown_nodes=summary['unknown_positive_speed_nodes']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    with original.campaign_lock():
        try:
            run(args.workers)
        except Exception as error:
            write_json(OUTPUT/'failure.json', dict(at_utc=original.utc(), error=repr(error), traceback=traceback.format_exc()))
            raise
