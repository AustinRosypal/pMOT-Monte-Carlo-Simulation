"""Finite-2-s central retention using the unchanged physical force kernel.

Core events never terminate these integrations. Every retained outcome needs
the actual 2 s endpoint; central acceptance additionally needs the existing
core rule and a bounded final 25 ms. Wall-event history is secondary to the
primary retained-loading quantity, and disagreements in that history remain
explicit rather than changing a proven wall loss into a numerical timeout.
"""
from pathlib import Path
from types import SimpleNamespace
import shutil
import mot_strange_retained_numerics as numerical
import mot_strange_retained_budget as budget
import time
import numpy as np
import run_mot_strange_finite_capture_refined_campaign as parent

original, fast, ce = parent.original, parent.fast, parent.ce
OLD_OUTPUT = parent.OUTPUT
OUTPUT = parent.BASE_OUTPUT.with_name(parent.BASE_OUTPUT.name+'_retained2s_v3_dv0p125')
PREVIOUS_OUTPUT = OUTPUT
OUTPUT = parent.BASE_OUTPUT.with_name(parent.BASE_OUTPUT.name+'_retained2s_v4r1_dv0p125')
frozen_fast = fast
fast = SimpleNamespace(integrate=frozen_fast.integrate, classify=numerical.classify,
                       field=frozen_fast.field)
SPEEDS = parent.SPEEDS.copy()
HORIZON, WINDOW, CORE_RADIUS = 2., .025, .002
UNKNOWN, WALL, CENTRAL, OFFCORE, TRANSIENT_WALL, OTHER, WALL_HISTORY_UNCERTAIN = -1, 0, 1, 2, 3, 4, 5
ENDPOINT_TOLERANCE_M = 1e-5
FIXED_PAIRS = ((2.5e-6, 1.25e-6), (.625e-6, .3125e-6), (.15625e-6, .078125e-6))


def primary(code):
    return WALL if code in (WALL, TRANSIENT_WALL, WALL_HISTORY_UNCERTAIN) else code


def outcome(result):
    raw, end, r, v, first, maximum = result[:6]
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.isfinite(end)):
        return UNKNOWN
    if raw == WALL and 0 <= end <= HORIZON+1e-9:
        return TRANSIENT_WALL if 0 <= first <= end else WALL
    # An early core return, short timeout, or nonfinite window is never accepted.
    if raw != UNKNOWN or abs(end-HORIZON) > 1e-9 or not np.isfinite(maximum):
        return UNKNOWN
    if maximum < np.linalg.norm(r)-1e-12 or maximum < 0:
        return UNKNOWN
    if 0 <= first <= HORIZON and maximum <= CORE_RADIUS:
        return CENTRAL
    return OFFCORE if np.linalg.norm(r) > CORE_RADIUS else OTHER


def pair_outcome(a, b):
    left, right = outcome(a), outcome(b)
    if left < 0 or right < 0 or primary(left) != primary(right):
        return UNKNOWN
    if primary(left) == WALL:
        return left if left == right else WALL_HISTORY_UNCERTAIN
    if np.linalg.norm(a[2]-b[2]) > ENDPOINT_TOLERANCE_M:
        return UNKNOWN
    if abs(a[5]-b[5]) > ENDPOINT_TOLERANCE_M:
        return UNKNOWN
    return left


def append(records, speed, method, parameter, result):
    records.append([speed, method, parameter, HORIZON, int(result[0]), float(result[1]),
                    *result[2], *result[3], float(result[4]), float(result[5])])


def unpack(row):
    return (int(row[4]), row[5], np.asarray(row[6:9]), np.asarray(row[9:12]), row[12], row[13])


def existing_fixed(records, speed):
    rows = [r for r in records if r[0] == speed and r[1] == 0]
    if len(rows) < 2:
        return UNKNOWN
    return pair_outcome(unpack(rows[-2]), unpack(rows[-1]))


def refined_consensus(r, direction, speed, data, records):
    fixed_codes = [outcome(unpack(row)) for row in records
                   if row[0] == speed and row[1] == 0]
    definite = [primary(c) for c in fixed_codes if c >= 0]
    if len(definite) < 2 or len(set(definite)) != 1:
        return UNKNOWN
    results = []
    for refinement in (128., 256., 512., 1024.):
        attempts = max(2000000, int(np.ceil(HORIZON/.0001*refinement)*1.5)+10000)
        saved = [row for row in records if row[0] == speed and row[1] == 2 and row[2] == refinement]
        if saved:
            result = unpack(saved[-1])
        else:
            result = budget.retained_adaptive(r, speed*direction, HORIZON, data,
                                             refinement, attempts)
            append(records, speed, 2, refinement, result)
        results.append(result)
    last = results[-3:]
    codes = [outcome(x) for x in last]
    if any(c not in (CENTRAL, OFFCORE, OTHER) for c in codes):
        return UNKNOWN
    if len(set(codes)) != 1 or primary(codes[0]) != definite[0]:
        return UNKNOWN
    for i in range(3):
        for j in range(i):
            if pair_outcome(last[i], last[j]) < 0:
                return UNKNOWN
    return codes[0]


def resolve_fixed(r, direction, speed, data, records):
    for pair in FIXED_PAIRS:
        results = []
        for dt in pair:
            saved = [row for row in records if row[0] == speed and row[1] == 0 and row[2] == dt]
            if saved:
                result = unpack(saved[-1])
            else:
                result = fast.classify(r, speed*direction, dt, HORIZON, data, False)
                append(records, speed, 0, dt, result)
            results.append(result)
        code = pair_outcome(*results)
        if code >= 0:
            return code
    return refined_consensus(r, direction, speed, data, records)


def resolve(r, direction, speed, data, records):
    for a, b in ((1., 2.), (4., 8.)):
        results = [fast.integrate(r, speed*direction, HORIZON, data, f, True) for f in (a, b)]
        for refinement, result in zip((a, b), results):
            append(records, speed, 1, refinement, result)
        code = pair_outcome(*results)
        if code >= 0:
            return code
    return resolve_fixed(r, direction, speed, data, records)


def reusable_parent_proof(path, r, direction):
    """Reuse only complete, proven wall losses or no-core horizon outcomes.

An early-exit core result is never evidence of final retention. Parent wall
integrations had no earlier core event, so retaining integration has exactly
the same steps until that physical wall. Their original evaluations/hashes
remain the provenance; they are not converted into fabricated window data.
"""
    if not path.exists():
        return {}, ''
    with np.load(path) as a:
        np.testing.assert_array_equal(a['position_m'], r)
        np.testing.assert_array_equal(a['direction'], direction)
        if str(a['status']) != 'resolved':
            return {}, original.sha(path)
        np.testing.assert_array_equal(a['speeds'], SPEEDS)
        records, methods = a['evaluations'].tolist(), a['evaluation_methods'].tolist()
        by_speed = {}
        for i, row in enumerate(records):
            by_speed.setdefault(float(row[0]), []).append(i)
        result = {}
        for speed, code, location in zip(a['speeds'], a['codes'], a['retained_locations']):
            if int(code) not in (parent.WALL, parent.RETAINED):
                continue
            indices = by_speed[float(speed)][-2:]
            assert parent.existing_proof([records[i] for i in indices],
                [methods[i] for i in indices], float(speed)) == int(code)
            result[float(speed)] = WALL if int(code) == parent.WALL else (OFFCORE if location == 0 else OTHER)
    return result, original.sha(path)


def worker(payload):
    _, d, p, r, direction, path, mode, data = payload
    path = Path(path)
    previous = PREVIOUS_OUTPUT/mode/'rays'/path.name
    if not path.exists() and previous.exists():
        with np.load(previous) as saved:
            np.testing.assert_array_equal(saved['position_m'], r)
            np.testing.assert_array_equal(saved['direction'], direction)
            assert float(saved['observation_horizon_s']) == HORIZON
            assert float(saved['retention_window_s']) == WINDOW
            previous_status = str(saved['status'])
            if previous_status == 'resolved':
                np.testing.assert_array_equal(saved['speeds'], SPEEDS)
                assert np.all(saved['codes'] >= 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous, path)
        original.write_json(path.with_suffix('.lineage.json'),
            dict(source=str(previous), sha256=original.sha(previous),
                 source_status=previous_status, source_campaign=PREVIOUS_OUTPUT.name,
                 physical_source_sha256=original.SOURCE_SHA,
                 policy='exact geometry and definition; resolved proof reuse or partial seed; original preserved'))
    records, codes, reused = [], [], []
    source = OLD_OUTPUT/mode/'rays'/path.name
    reuse, parent_sha = reusable_parent_proof(source, r, direction)
    parent_reused = set(reuse)
    if path.exists():
        with np.load(path) as a:
            assert float(a['retention_window_s']) == WINDOW
            if str(a['status']) == 'resolved':
                np.testing.assert_array_equal(a['speeds'], SPEEDS)
                return dict(disc=d, point=p, status='resolved', reused=True)
            records = a['evaluations'].tolist()
            reuse.update({float(s): int(c) for s, c in zip(a['speeds'], a['codes']) if c >= 0})
    began = time.perf_counter()
    def save(status, error=''):
        ce._atomic_npz(path, status=np.array(status), error=np.array(error),
            speeds=SPEEDS[:len(codes)], codes=np.asarray(codes, dtype=np.int8),
            evaluations=np.asarray(records, dtype=float).reshape((-1, 14)),
            reused_parent_speeds=np.asarray(reused), parent_ray_sha256=np.array(parent_sha),
            position_m=r, direction=direction, observation_horizon_s=np.array(HORIZON),
            retention_window_s=np.array(WINDOW), elapsed_wall_s=np.array(time.perf_counter()-began))
    try:
        for i, speed in enumerate(SPEEDS):
            code = reuse.get(float(speed), UNKNOWN)
            if code < 0:
                code = resolve(r, direction, float(speed), data, records)
            elif float(speed) in parent_reused:
                reused.append(float(speed))
            codes.append(code)
            if i % 64 == 63 or code in (CENTRAL, OFFCORE, OTHER):
                save('running')
            if code < 0:
                save('numerical_indeterminate')
                return dict(disc=d, point=p, status='numerical_indeterminate', speed=float(speed))
        fixed = {i for i, speed in enumerate(SPEEDS) if existing_fixed(records, speed) >= 0}
        while True:
            needed = ce.boundary_node_indices([primary(c) for c in codes])-fixed
            if not needed:
                break
            for i in sorted(needed):
                codes[i] = resolve_fixed(r, direction, float(SPEEDS[i]), data, records)
                fixed.add(i)
                save('running')
                if codes[i] < 0:
                    save('numerical_indeterminate')
                    return dict(disc=d, point=p, status='numerical_indeterminate', speed=float(SPEEDS[i]))
        if any(c == CENTRAL for c in codes[len(ce.VELOCITIES)-20:]):
            save('tail_censored')
            return dict(disc=d, point=p, status='tail_censored')
        save('resolved')
        return dict(disc=d, point=p, status='resolved', wall_s=time.perf_counter()-began)
    except Exception as error:
        save('error', repr(error))
        raise


