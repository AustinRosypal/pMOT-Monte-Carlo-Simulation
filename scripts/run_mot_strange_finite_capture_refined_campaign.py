"""Central-core capture by 2 s, with retained atoms recorded separately."""
import complete_mot_strange_diagnostic_masks as diagnostic
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from importlib.metadata import version
import json
import os
import shutil
import time
import traceback

import numpy as np
from scipy.stats import t

original = diagnostic.original
BASE_OUTPUT = original.OUTPUT
PILOT_SOURCE = diagnostic.OUTPUT
COARSE_OUTPUT = BASE_OUTPUT.with_name(BASE_OUTPUT.name+'_core_by2s_v1')
OUTPUT = BASE_OUTPUT.with_name(BASE_OUTPUT.name+'_core_by2s_v2_dv0p125')
original.OUTPUT = OUTPUT
fast = diagnostic.fast
ce = diagnostic.ce
HORIZON = 2.
WALL, CORE, RETAINED, UNKNOWN = 0, 1, 2, -1
ce.VELOCITIES = np.arange(.125, 80.00001, .125)
SPEEDS = np.r_[ce.VELOCITIES, ce.TAIL_VELOCITIES]
STAT_CACHE = {}


def finite_code(result, duration):
    """A horizon observation is distinct from a physical wall-loss event."""
    raw = int(result[0])
    if raw in (WALL, CORE):
        if not 0 <= result[1] <= HORIZON+1e-9:
            raise ValueError('terminal event outside observation window')
        return raw
    if (raw == UNKNOWN and duration == HORIZON and abs(result[1]-HORIZON) <= 1e-9
        and result[4] < 0 and np.all(np.isfinite(result[2])) and np.all(np.isfinite(result[3]))):
        return RETAINED
    return UNKNOWN


def pair_code(a, b, duration):
    left, right = finite_code(a, duration), finite_code(b, duration)
    if left < 0 or left != right:
        return UNKNOWN
    if left == RETAINED and (np.linalg.norm(a[2]) > .002) != (np.linalg.norm(b[2]) > .002):
        return UNKNOWN
    return left


def append_result(records, methods, speed, parameter, duration, result, method):
    records.append([speed, parameter, duration, float(result[0]), result[1],
                    *result[2], *result[3], result[4]])
    methods.append(method)


def result_from_row(row):
    return int(row[3]), row[4], np.asarray(row[5:8]), np.asarray(row[8:11]), row[11]


def existing_proof(records, methods, speed, *, fixed_only=False):
    indices = [i for i, row in enumerate(records) if row[0] == speed
               and (not fixed_only or methods[i] == 0)]
    if len(indices) < 2:
        return UNKNOWN
    i, j = indices[-2:]
    if methods[i] != methods[j] or records[i][2] != records[j][2] or records[i][1] == records[j][1]:
        return UNKNOWN
    return pair_code(result_from_row(records[i]), result_from_row(records[j]), records[j][2])


def retained_location(code, records, speed):
    if code != RETAINED:
        return -1
    row = next(row for row in reversed(records) if row[0] == speed)
    return int(np.linalg.norm(row[5:8]) <= .002)  # 0 off-core; 1 inside without meeting core rule


def resolve_fixed(r, direction, speed, data, records, methods):
    for coarse, fine, duration in ce.LADDER:
        a = fast.classify(r, speed*direction, coarse, duration, data)
        b = fast.classify(r, speed*direction, fine, duration, data)
        for dt, result in ((coarse, a), (fine, b)):
            append_result(records, methods, speed, dt, duration, result, 0)
        code = pair_code(a, b, duration)
        if code >= 0:
            return code
    return UNKNOWN


def resolve_node(r, direction, speed, data, records, methods):
    for duration in (.05, .1, .2, .4, 1., HORIZON):
        a = fast.integrate(r, speed*direction, duration, data, 1.)
        b = fast.integrate(r, speed*direction, duration, data, 2.)
        append_result(records, methods, speed, .0005, duration, a, 1)
        append_result(records, methods, speed, .00025, duration, b, 1)
        code = pair_code(a, b, duration)
        if code >= 0:
            return code
        if a[0] >= 0 and b[0] >= 0:
            break
    return resolve_fixed(r, direction, speed, data, records, methods)


def worker(payload):
    _, d, p, r, direction, path, mode, data = payload
    path = Path(path)
    records, methods, existing = [], [], {}
    if path.exists():
        with np.load(path) as old:
            if str(old['status']) == 'resolved':
                np.testing.assert_array_equal(old['speeds'], SPEEDS)
                return dict(disc=d, point=p, status='resolved', reused=True)
            records = old['evaluations'].tolist(); methods = old['evaluation_methods'].tolist()
            existing = dict(zip(old['speeds'].tolist(), old['codes'].tolist()))
    codes = []
    began = time.perf_counter()
    def save(status, error=''):
        locations = [retained_location(c, records, speed) for c, speed in zip(codes, SPEEDS)]
        ce._atomic_npz(path, speeds=SPEEDS[:len(codes)], codes=np.asarray(codes, dtype=np.int8),
            retained_locations=np.asarray(locations, dtype=np.int8),
            evaluations=np.asarray(records, dtype=float).reshape((-1, 12)),
            evaluation_methods=np.asarray(methods, dtype=np.int8),
            position_m=r, direction=direction, status=np.array(status), error=np.array(error),
            observation_horizon_s=np.array(HORIZON), elapsed_wall_s=np.array(time.perf_counter()-began))
    try:
        for i, speed in enumerate(SPEEDS):
            code = int(existing.get(speed, UNKNOWN))
            if code < 0:
                code = resolve_node(r, direction, speed, data, records, methods)
            codes.append(code)
            if i % 64 == 63 or code == RETAINED:
                save('running')
            if code < 0:
                save('numerical_indeterminate')
                return dict(disc=d, point=p, status='numerical_indeterminate', speed=float(speed))
        fixed = {i for i, speed in enumerate(SPEEDS)
                 if existing_proof(records, methods, speed, fixed_only=True) >= 0}
        while True:
            needed = ce.boundary_node_indices(codes)-fixed
            if not needed:
                break
            for i in sorted(needed):
                codes[i] = resolve_fixed(r, direction, SPEEDS[i], data, records, methods)
                fixed.add(i)
                save('running')
                if codes[i] < 0:
                    save('numerical_indeterminate')
                    return dict(disc=d, point=p, status='numerical_indeterminate', speed=float(SPEEDS[i]))
        if any(c == CORE for c in codes[len(ce.VELOCITIES)-20:]):
            save('tail_censored')
            return dict(disc=d, point=p, status='tail_censored')
        save('resolved')
        return dict(disc=d, point=p, status='resolved', wall_s=time.perf_counter()-began)
    except Exception as error:
        save('error', repr(error))
        raise


def load_data():
    with np.load(BASE_OUTPUT/'compiled_inputs.npz') as saved:
        return tuple(saved[f'array_{i}'] for i in range(12))


def convert_pilot():
    folder = OUTPUT/'pilot'; (folder/'rays').mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_OUTPUT/'pilot/geometry.npz', folder/'geometry.npz')
    shutil.copy2(PILOT_SOURCE/'validation/core_retention_audit.json', OUTPUT/'validation/previous_core_retention_audit.json')
    parent = original.read(COARSE_OUTPUT/'campaign_manifest.json')
    assert parent['source_sha256'] == original.SOURCE_SHA
    assert parent['finite_runner_sha256'] == original.sha(Path(__file__).with_name('run_mot_strange_finite_capture_campaign.py'))
    assert original.read(COARSE_OUTPUT/'validation/pilot_conversion.json')['passed']
    converted, retained = 0, 0
    for d in range(original.PILOT_DISCS):
        for p in range(original.POINTS):
            target = folder/'rays'/f'd{d:03d}_p{p:03d}.npz'
            if target.exists():
                with np.load(target) as old:
                    retained += int(np.count_nonzero(old['codes'] == RETAINED))
                converted += 1; continue
            with np.load(COARSE_OUTPUT/'pilot/rays'/target.name) as saved:
                assert str(saved['status']) == 'resolved'
                speeds = saved['speeds']; records = saved['evaluations'].tolist()
                methods = saved['evaluation_methods'].tolist()
                r, direction = saved['position_m'], saved['direction']
            assert set(speeds).issubset(set(SPEEDS))
            codes = np.asarray([existing_proof(records, methods, speed) for speed in speeds], dtype=np.int8)
            if np.any(codes < 0):
                raise ValueError(f'pilot node lacks two-setting evidence: {d}/{p}/{speeds[codes < 0]}')
            retained += int(np.count_nonzero(codes == RETAINED))
            locations = [retained_location(code, records, speed) for code, speed in zip(codes, speeds)]
            ce._atomic_npz(target, speeds=speeds, codes=codes, retained_locations=np.asarray(locations, dtype=np.int8),
                evaluations=np.asarray(records, dtype=float), evaluation_methods=np.asarray(methods, dtype=np.int8),
                position_m=r, direction=direction, observation_horizon_s=np.array(HORIZON),
                status=np.array('refinement_pending'), provenance=np.array('exact validated coarse nodes; missing interleaved nodes require new trajectories'))
            converted += 1
    original.write_json(OUTPUT/'validation/pilot_conversion.json', dict(passed=True,
        seeded_rays=converted, retained_at_horizon_nodes=retained, fine_grid_pending=True,
        core_rule_unchanged=True, old_outputs_unchanged=True, observation_horizon_s=HORIZON,
        source_sha256=original.SOURCE_SHA))


def disc_statistics(mode, d, areas):
    key = (mode, d)
    if key in STAT_CACHE:
        return STAT_CACHE[key]
    cache = OUTPUT/mode/'discs'/f'd{d:05d}.npz'
    if cache.exists():
        with np.load(cache) as saved:
            result = {k: saved[k] for k in saved.files}
        STAT_CACHE[key] = result
        return result
    codes, locations = [], []
    for p in range(original.POINTS):
        with np.load(OUTPUT/mode/'rays'/f'd{d:03d}_p{p:03d}.npz') as saved:
            assert str(saved['status']) == 'resolved'
            assert float(saved['observation_horizon_s']) == HORIZON
            np.testing.assert_array_equal(saved['speeds'], SPEEDS)
            assert not np.any(saved['codes'] < 0)
            assert not np.any(saved['codes'][len(ce.VELOCITIES):] == CORE)
            codes.append(saved['codes'][:len(ce.VELOCITIES)])
            locations.append(saved['retained_locations'][:len(ce.VELOCITIES)])
    codes, locations = np.asarray(codes), np.asarray(locations)
    spectra = np.asarray([areas[d]*(codes == CORE).mean(axis=0),
        areas[d]*((codes == RETAINED) & (locations == 0)).mean(axis=0),
        areas[d]*((codes == RETAINED) & (locations == 1)).mean(axis=0)])
    weight = ce.VELOCITIES**3*np.exp(-ce.VELOCITIES**2/original.THERMAL_SCALE_M2_PER_S2)
    rates = original.LOADING_RATE_PREFACTOR*np.trapezoid(
        np.column_stack((np.zeros(3), spectra*weight)), np.r_[0., ce.VELOCITIES], axis=1)
    coarse = original.LOADING_RATE_PREFACTOR*np.trapezoid(
        np.r_[0., (spectra[0]*weight)[1::2]], np.r_[0., ce.VELOCITIES[1::2]])
    result = dict(spectra_m2=spectra, rates_atoms_per_s=rates, coarse_core_rate=np.array(coarse),
        nonmonotone_ray_count=np.array(np.count_nonzero(np.any(np.diff((codes == CORE).astype(int), axis=1) > 0, axis=1))),
        retained_nodes=np.array(np.count_nonzero(codes == RETAINED)))
    ce._atomic_npz(cache, **result)
    STAT_CACHE[key] = result
    return result


def statistics(mode, discs, positions, directions, areas, *, provisional=True):
    stats = [disc_statistics(mode, d, areas) for d in range(discs)]
    rates = np.asarray([s['rates_atoms_per_s'] for s in stats])
    spectra = np.asarray([s['spectra_m2'] for s in stats])
    mean = rates.mean(axis=0)
    half = t.ppf(.975, discs-1)*rates.std(axis=0, ddof=1)/np.sqrt(discs)
    weight = ce.VELOCITIES**3*np.exp(-ce.VELOCITIES**2/original.THERMAL_SCALE_M2_PER_S2)
    maximum_rate = original.LOADING_RATE_PREFACTOR*np.pi*.015**2*np.trapezoid(
        np.r_[0., weight], np.r_[0., ce.VELOCITIES])
    for category in range(3):
        if np.all(rates[:, category] == 0):
            half[category] = maximum_rate*(1-.025**(1/discs))
    quadrature = abs(float(np.mean([s['coarse_core_rate'] for s in stats]))-mean[0])
    row = dict(at_utc=original.utc(), mode=mode,
        status='provisional finite-2-second central-core capture rate; final gates pending',
        observation_horizon_s=HORIZON, core_rule='2 mm core, 5 ms continuous residence OR two entries',
        complete_direction_discs=discs, points_per_disc=original.POINTS, resolved_rays=discs*original.POINTS,
        loading_rate_atoms_per_s=float(mean[0]), loading_95_half_width_atoms_per_s=float(half[0]),
        loading_ci95_atoms_per_s=[max(0., float(mean[0]-half[0])), float(mean[0]+half[0])],
        loading_relative_half_width=float(half[0]/mean[0]) if mean[0] > 0 else None,
        loading_coefficient_m3_per_s=float(mean[0]/original.N87),
        loading_coefficient_95_half_width_m3_per_s=float(half[0]/original.N87),
        off_core_retention_integral_atoms_per_s=float(mean[1]),
        off_core_retention_95_half_width_atoms_per_s=float(half[1]),
        inside_core_without_capture_rule_integral_atoms_per_s=float(mean[2]),
        retained_at_horizon_nodes=int(sum(int(s['retained_nodes']) for s in stats)),
        thermal_conditions=original.read(OUTPUT/'campaign_manifest.json')['thermal_conditions'],
        precision_pass=bool(mean[0] > 0 and half[0]/mean[0] <= original.TARGET),
        quadrature_pass=bool(mean[0] > 0 and quadrature <= .02*mean[0]),
        velocity_quadrature_difference_atoms_per_s=float(quadrature),
        nonmonotone_ray_count=int(sum(int(s['nonmonotone_ray_count']) for s in stats)),
        interval='95% Student-t direction-disc cluster interval; interim estimates do not control stopping',
        interpretation='rate of satisfying the core rule by 2 s, not eventual capture or equilibrium loading',
        off_core_interpretation='retention at 2 s without meeting the core rule; not a claim of 3D trapping',
        scope='specified interior 15 mm launch aperture; aperture/placement checks pending')
    folder = OUTPUT/mode
    original.write_json(folder/'latest_estimate.json', row)
    original.write_json(folder/'estimates'/f'discs_{discs:05d}.json', row)
    ce._atomic_npz(folder/'disc_statistics.npz', velocity_m_per_s=ce.VELOCITIES,
        disc_cross_section_m2=spectra[:, 0], disc_off_core_retention_cross_section_m2=spectra[:, 1],
        disc_loading_atoms_per_s=rates[:, 0], disc_off_core_retention_atoms_per_s=rates[:, 1],
        disc_inside_core_without_capture_rule_atoms_per_s=rates[:, 2], disc_areas_m2=areas[:discs])
    print('LOADING UPDATE '+json.dumps(row), flush=True)
    return row


def initialize():
    parent = original.read(BASE_OUTPUT/'campaign_manifest.json')
    assert parent['source_sha256'] == original.SOURCE_SHA, 'Physical source changed; revalidate under a new revision'
    for gate in ('test_status.json', 'local_force_and_events.json'):
        assert original.read(BASE_OUTPUT/'validation'/gate)['passed']
    assert original.read(PILOT_SOURCE/'validation/full_fixed_grid.json')['passed']
    assert original.read(PILOT_SOURCE/'validation/remaining_fixed_nodes.json')['classifications_agree']
    manifest = dict(parent)
    manifest.update(campaign=OUTPUT.name, finite_runner_sha256=original.sha(__file__),
        positive_velocities_m_per_s=ce.VELOCITIES.tolist(), coarse_finite_pilot=str(COARSE_OUTPUT),
        quadrature_revision='0.125 m/s dense grid, nested 0.25 m/s comparison; original 2% tolerance unchanged',
        diagnostic_helper_sha256=original.sha(diagnostic.__file__),
        observation_horizon_s=HORIZON,
        human_decision='central-core capture within 2 s; off-center retention separate',
        outcomes={'0': 'wall_escape', '1': 'core_rule_met_by_2s',
                  '2': 'retained_at_2s_without_meeting_core_rule', '-1': 'numerically_indeterminate'},
        physical_source_unchanged=True,
        parent_campaign=str(BASE_OUTPUT), pilot_evidence=str(PILOT_SOURCE),
        packages={p: version(p) for p in ('numpy', 'scipy', 'numba', 'arc-alkali-rydberg-calculator')})
    path = OUTPUT/'campaign_manifest.json'
    if path.exists():
        assert original.read(path) == manifest, 'Finite campaign manifest mismatch'
    else:
        original.write_json(path, manifest)
        shutil.copy2(__file__, OUTPUT/Path(__file__).name)
        shutil.copy2(diagnostic.__file__, OUTPUT/Path(diagnostic.__file__).name)
    return manifest


def freeze_plan(pilot):
    path = OUTPUT/'sampling_plan.json'
    if path.exists():
        return original.read(path)
    with np.load(OUTPUT/'pilot/disc_statistics.npz') as saved:
        disc_rates = saved['disc_loading_atoms_per_s']
    with np.load(OUTPUT/'pilot/geometry.npz') as saved:
        areas = saved['areas_m2']
    masks = np.empty((original.PILOT_DISCS, original.POINTS, len(ce.VELOCITIES)), dtype=np.int8)
    for d in range(original.PILOT_DISCS):
        for p in range(original.POINTS):
            with np.load(OUTPUT/'pilot/rays'/f'd{d:03d}_p{p:03d}.npz') as saved:
                masks[d, p] = saved['codes'][:len(ce.VELOCITIES)]
    velocity = ce.VELOCITIES
    weight = velocity**3*np.exp(-velocity**2/original.THERMAL_SCALE_M2_PER_S2)
    ray_rates = original.LOADING_RATE_PREFACTOR*areas[:, None]*np.trapezoid(
        np.concatenate((np.zeros((*masks.shape[:2], 1)), (masks == CORE)*weight), axis=2),
        np.r_[0., velocity], axis=2)
    observed = float(disc_rates.var(ddof=1)); within = float(ray_rates.var(axis=1, ddof=1).mean())
    between = max(0., observed-within/original.POINTS)
    conservative_variance = max(observed, between+within/original.POINTS)
    n = 200
    while 1.5*t.ppf(.975, n-1)**2*conservative_variance/(original.TARGET*disc_rates.mean())**2 > n:
        n += 25
        if n > 5000:
            raise RuntimeError('Pilot requires more than 5000 discs; review rare-event variance')
    plan = dict(frozen_at_utc=original.utc(), disc_count=n, points_per_disc=original.POINTS,
        production_seed=original.PRODUCTION_SEED, observation_horizon_s=HORIZON,
        pilot_mean_atoms_per_s=float(disc_rates.mean()), pilot_disc_variance=observed,
        conservative_variance=conservative_variance, variance_safety_factor=1.5,
        relative_95_half_width_target=original.TARGET, optional_stopping=False,
        pilot_independent_of_production=True, rays=n*original.POINTS)
    original.write_json(path, plan)
    print('FROZEN PLAN '+json.dumps(plan), flush=True)
    return plan


def probe(payload):
    r, direction, speed, expected, data = payload
    records, methods = [], []
    code = resolve_node(r, direction, speed, data, records, methods)
    return dict(speed_m_per_s=speed, code=code, expected=expected,
                passed=code == expected, evaluations=records, evaluation_methods=methods)


def full_fixed_worker(payload):
    d, p, data = payload
    target = OUTPUT/'validation/full_fixed_grid'/f'd{d:03d}_p{p:03d}.npz'
    if target.exists():
        with np.load(target) as saved:
            if str(saved['status']) == 'complete':
                return dict(disc=d, point=p, passed=bool(saved['passed']), reused=True,
                            disagreements=saved['disagreements'].tolist())
    with np.load(OUTPUT/'pilot/rays'/target.name) as saved:
        r, direction, expected = saved['position_m'], saved['direction'], saved['codes']
        records, methods = saved['evaluations'].tolist(), saved['evaluation_methods'].tolist()
    previous = PILOT_SOURCE/'validation/full_fixed_grid'/target.name
    with np.load(previous) as saved:
        records.extend(saved['evaluations'].tolist())
        methods.extend(saved['evaluation_methods'].tolist())
    codes, disagreements, prior = [], [], {}
    if target.exists():
        with np.load(target) as saved:
            records, methods = saved['evaluations'].tolist(), saved['evaluation_methods'].tolist()
            prior = dict(zip(saved['speeds'].tolist(), saved['codes'].tolist()))
    def save(status):
        ce._atomic_npz(target, status=np.array(status), speeds=SPEEDS[:len(codes)],
            codes=np.asarray(codes), evaluations=np.asarray(records).reshape((-1, 12)),
            evaluation_methods=np.asarray(methods, dtype=np.int8), passed=np.array(not disagreements),
            disagreements=np.asarray(disagreements).reshape((-1, 3)))
    for i, speed in enumerate(SPEEDS):
        code = int(prior.get(speed, existing_proof(records, methods, speed, fixed_only=True)))
        if code < 0:
            code = resolve_fixed(r, direction, speed, data, records, methods)
        codes.append(code)
        if code != expected[i]:
            disagreements.append([float(speed), int(expected[i]), int(code)])
        if i % 32 == 31:
            save('running')
    save('complete')
    return dict(disc=d, point=p, passed=not disagreements, evaluated_nodes=len(codes),
                disagreements=disagreements, original_masks_unchanged=True)


def run(args):
    certificate = original.read(OUTPUT/'validation/finite_tests.json')
    assert certificate['passed'] and certificate['runner_sha256'] == original.sha(__file__)
    initialize()
    original.write_json(OUTPUT/'process.json', dict(pid=os.getpid(), command=os.sys.argv,
        started_at_utc=original.utc(), workers=args.workers))
    convert_pilot()
    with np.load(OUTPUT/'pilot/geometry.npz') as saved:
        positions, directions, areas = saved['positions_m'], saved['directions'], saved['areas_m2']
    data = load_data()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gate = OUTPUT/'validation/fresh_finite_workers.json'
        if not gate.exists():
            checks = list(pool.map(probe, [
                (positions[0, 4], directions[0], 19.75, RETAINED, data),
                (positions[0, 4], directions[0], 16., CORE, data),
                (positions[0, 0], directions[0], 120., WALL, data)]))
            original.write_json(gate, dict(passed=all(c['passed'] for c in checks), checks=checks))
        assert original.read(gate)['passed'], 'Fresh finite-horizon worker validation failed'
        print('FINITE-HORIZON VALIDATION PASSED', flush=True)
        if args.stage == 'validate':
            return
        original.worker = worker
        original.statistics = statistics
        pilot, positions, directions, areas, impacts = original.run_phase(
            'pilot', original.PILOT_DISCS, original.PILOT_SEED, data, pool)
        assert pilot['quadrature_pass'], 'Refined pilot speed quadrature failed; preserve output and refine again'
        fixed_path = OUTPUT/'validation/full_fixed_grid.json'
        if not fixed_path.exists():
            checks = list(pool.map(full_fixed_worker, [(d, p, data) for d, p in ((0, 0), (17, 13), (0, 4))]))
            original.write_json(fixed_path, dict(passed=all(c['passed'] for c in checks),
                cases=checks, node_count=3*len(SPEEDS), source_sha256=original.SOURCE_SHA))
        assert original.read(fixed_path)['passed'], 'Refined full fixed-grid audit failed'
        original.retention('pilot', positions, directions, impacts, data, pool)
        plan = freeze_plan(pilot)
        row, positions, directions, areas, impacts = original.run_phase(
            'production', plan['disc_count'], original.PRODUCTION_SEED, data, pool)
        original.retention('production', positions, directions, impacts, data, pool)
        row.update(retention_pass=True, status='production computed; final aperture/placement and visual checks pending')
        original.write_json(OUTPUT/'production/summary.json', row)
        original.write_json(OUTPUT/'completion.json', dict(computed=True, at_utc=original.utc(),
            precision_pass=row['precision_pass'], quadrature_pass=row['quadrature_pass'],
            retention_pass=True, aperture_and_launch_placement_pending=True, visual_review_pending=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('validate', 'all'))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    # All corrected campaign collectors share the same original OS lock.
    original.OUTPUT = BASE_OUTPUT
    with original.campaign_lock():
        original.OUTPUT = OUTPUT
        try:
            run(args)
        except Exception as error:
            original.write_json(OUTPUT/'failure.json', dict(at_utc=original.utc(),
                error=repr(error), traceback=traceback.format_exc()))
            raise
