"""Resumable corrected mot_strange loading campaign with in-cell launches."""
from pathlib import Path
import os
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = sorted((ROOT/'src/pmot').rglob('*.py')) + [Path(__file__)]
digest = hashlib.sha256()
for file in SOURCE_FILES:
    digest.update(file.relative_to(ROOT).as_posix().encode())
    digest.update(file.read_bytes())
SOURCE_SHA = digest.hexdigest()
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'outputs/compiled_cache'/SOURCE_SHA)
os.environ['MPLBACKEND'] = 'Agg'
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from dataclasses import asdict
import datetime
import json
import math
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
from scipy.stats import t as student_t
from pmot.configuration import RB87_MASS_KG, HBAR_J_S
from pmot.launch_geometry import sample_incident_disc_full_sphere
from pmot.loading import LOADING_RATE_PREFACTOR, THERMAL_SCALE_M2_PER_S2
from pmot.mot_multilevel import campaign_execution as ce
from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.rate_equations import build_rate_equation_model, rate_equation_observable
from pmot.mot_multilevel.simulation import build_multilevel_mot_beams
from pmot.mot_multilevel.campaign_validation import event_occupations
from pmot.mot_multilevel.relationship_campaign import write_json, write_csv, LoadingPoint
from pmot.mot_strange import accelerated_surrogate as fast
from pmot.mot_strange.cell_launch_sampling import sample_disc_points_inside_cell
from pmot.mot_strange.surrogate_effective_field import (
    calibrated_surrogate_field_config, surrogate_effective_field_t,
    surrogate_component_intensities_w_per_m2, numerical_field_jacobian_t_per_m,
)

CAMPAIGN = 'calibrated_10_10_20Gpcm_full_sphere_r15mm_incell_20261025'
OUTPUT = ROOT/'outputs/statistics/mot_strange'/CAMPAIGN
PILOT_DISCS, POINTS = 32, 20
PILOT_SEED, PRODUCTION_SEED = 2026102501, 2026102502
TARGET = .05
POINT = LoadingPoint('surrogate', 0, 20., 15., .027, -2*np.pi*15e6)
FIELD_CONFIG = calibrated_surrogate_field_config()
ORIGINAL_WORKER = ce._ray_worker
THERMAL_T = RB87_MASS_KG*THERMAL_SCALE_M2_PER_S2/(2*Boltzmann)
N87 = LOADING_RATE_PREFACTOR/(4/(np.sqrt(np.pi)*THERMAL_SCALE_M2_PER_S2**1.5))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def campaign_lock():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'runner.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def worker(payload):
    ce.classify = fast.classify
    ce.adaptive_integrate = fast.integrate
    ce._DATA_CACHE.clear()
    return ORIGINAL_WORKER(payload)


def fixed_comparison(payload):
    file, data = payload
    ce.classify = fast.classify
    with np.load(file) as saved:
        speeds, codes = saved['speeds'], saved['codes']
        r, direction = saved['position_m'], saved['direction']
    records, methods, differences = [], [], []
    for speed, expected in zip(speeds, codes):
        actual = ce._resolve_fixed_node(r, direction, speed, data, records, methods)
        if actual != expected:
            differences.append([float(speed), int(expected), int(actual)])
    target = OUTPUT/'validation/full_fixed'/Path(file).name
    ce._atomic_npz(target, evaluations=np.asarray(records), methods=np.asarray(methods))
    return dict(file=str(file), differences=differences, passed=not differences,
                evaluation_count=len(records))


def fresh_worker_check(payload):
    r, direction, speed, data = payload
    ce.classify = fast.classify
    ce.adaptive_integrate = fast.integrate
    records, methods = [], []
    a = ce._resolve_node(r, direction, speed, data, records, methods)
    b = ce._resolve_fixed_node(r, direction, speed, data, records, methods)
    return dict(position_m=r.tolist(), speed_m_per_s=speed, adaptive=a, fixed=b,
                passed=a == b and b >= 0, evaluations=records, methods=methods)


def retention_worker(payload):
    d, p, speed, r, direction, data = payload
    attempts = []
    for duration in (.1, .2, .4):
        results = [fast.classify(r, speed*direction, dt, duration, data, False)
                   for dt in (2.5e-6, 1.25e-6)]
        delta = float(np.linalg.norm(results[0][2]-results[1][2]))
        bounded = all(a[0] != 0 and a[4] >= 0 and a[5] <= .002 for a in results)
        attempts.append(dict(duration_s=duration, time_steps_s=[2.5e-6, 1.25e-6],
            codes=[int(a[0]) for a in results], endpoint_difference_m=delta,
            final_window_maximum_radii_m=[float(a[5]) for a in results],
            endpoints_m=[a[2].tolist() for a in results],
            velocities_m_per_s=[a[3].tolist() for a in results]))
        if bounded and delta <= 1e-5:
            return dict(disc=d, point=p, speed_m_per_s=speed, passed=True, attempts=attempts)
        if any(a[0] == 0 for a in results):
            break
    return dict(disc=d, point=p, speed_m_per_s=speed, passed=False, attempts=attempts)


def geometry(mode, discs, seed):
    folder = OUTPUT/mode
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'geometry.npz'
    if path.exists():
        with np.load(path) as saved:
            assert int(saved['seed']) == seed and saved['positions_m'].shape == (discs, POINTS, 3)
            return saved['positions_m'], saved['directions'], saved['areas_m2'], saved['impacts_m']
    positions = np.empty((discs, POINTS, 3))
    directions = np.empty((discs, 3))
    areas = np.empty(discs)
    impacts = np.empty((discs, POINTS))
    proposals = np.empty(discs, dtype=int)
    for d in range(discs):
        disc = sample_incident_disc_full_sphere(d, .010,
            np.random.default_rng(np.random.SeedSequence([seed, 0, d])))
        points, areas[d], proposals[d] = sample_disc_points_inside_cell(disc, POINTS, .015,
            np.random.default_rng(np.random.SeedSequence([seed, 1, d])), FIELD_CONFIG)
        positions[d] = [p.initial_position_m for p in points]
        impacts[d] = [p.s_m for p in points]
        directions[d] = disc.incident_unit_vector
    ce._atomic_npz(path, positions_m=positions, directions=directions, areas_m2=areas,
                   impacts_m=impacts, proposal_counts=proposals, seed=np.array(seed))
    return positions, directions, areas, impacts


def initialize():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vapor = dict(temperature_k=THERMAL_T, temperature_c=THERMAL_T-273.15,
                 rb87_density_m3=N87, rb87_partial_pressure_pa=N87*Boltzmann*THERMAL_T,
                 historical_loading_prefactor=LOADING_RATE_PREFACTOR,
                 thermal_scale_m2_per_s2=THERMAL_SCALE_M2_PER_S2,
                 convention='normalized Maxwell speed distribution; retained historical loading prefactor',
                 calibration='assumed historical vapor conditions, not measured experimental density')
    manifest = json.loads(json.dumps(dict(schema=1, campaign=CAMPAIGN,
        source_sha256=SOURCE_SHA, lock_sha256=sha(ROOT/'uv.lock'),
        field_configuration=FIELD_CONFIG.metadata(), mot_configuration=asdict(POINT.config()),
        thermal_conditions=vapor, pilot_discs=PILOT_DISCS, points_per_disc=POINTS,
        pilot_seed=PILOT_SEED, production_seed=PRODUCTION_SEED,
        disc_radius_m=.015, disc_center_distance_m=.010,
        sampling='uniform full sphere; uniform-area disc proposals; retry outside-cell points; equal accepted counts; per-disc clipped-area weighting',
        core_capture='2 mm core, 5 ms continuous residence OR two entries with intervening exit',
        origin_loading_primary=True, positive_velocities_m_per_s=ce.VELOCITIES.tolist(),
        tail_probes_m_per_s=ce.TAIL_VELOCITIES.tolist(), fixed_ladder=ce.LADDER,
        convergence='dual adaptive settings at every positive node; dual fixed RK4 at all edges and disagreements; complete fixed-grid pilot comparisons; longer retention',
        precision=dict(confidence=.95, relative_half_width_target=TARGET,
            independent_units='direction discs', allocation='independent pilot, fixed production N >=200 in blocks of 25; 1.5 variance safety factor'),
        limitations='surrogate-field conventional MOT, not physical pMOT; local steady-state populations; recoil diffusion off; statistical bars exclude model, apparatus and vapor uncertainty; interior launch sections remain aperture-limited pending checks',
        python=platform.python_version())))
    file = OUTPUT/'campaign_manifest.json'
    if file.exists():
        assert read(file) == manifest, 'Source/configuration changed: use a new documented campaign revision'
    else:
        write_json(file, manifest)
        for source in SOURCE_FILES:
            target = OUTPUT/'source_snapshot'/source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        center = surrogate_component_intensities_w_per_m2((0, 0, 0), FIELD_CONFIG)
        write_json(OUTPUT/'field_calibration.json', dict(
            central_jacobian_g_per_cm=(100*numerical_field_jacobian_t_per_m(
                step_m=1e-8, config=FIELD_CONFIG)).tolist(),
            component_origin_intensities_w_per_m2=center.tolist(),
            configuration=FIELD_CONFIG.metadata()))
    return manifest


def validate(data):
    file = OUTPUT/'validation/test_status.json'
    if not file.exists():
        tests = ['tests/mot_multilevel/test_accelerated.py', 'tests/mot_multilevel/test_adaptive.py',
                 'tests/mot_multilevel/test_population_rate_equations.py',
                 'tests/mot_strange/test_accelerated_surrogate.py',
                 'tests/mot_strange/test_cell_launch_sampling.py',
                 'tests/mot_strange/test_mot_strange_surrogate_effective_field.py',
                 'tests/test_mot_strange_surrogate_capture_campaign.py']
        result = subprocess.run(['uv', 'run', 'python', '-m', 'pytest', *tests, '-q'],
                                cwd=ROOT, capture_output=True, text=True)
        file.parent.mkdir(parents=True, exist_ok=True)
        (file.parent/'pytest.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        write_json(file, dict(passed=result.returncode == 0, source_sha256=SOURCE_SHA, tests=tests))
    assert read(file)['passed'] and read(file)['source_sha256'] == SOURCE_SHA
    file = OUTPUT/'validation/local_force_and_events.json'
    if file.exists():
        assert read(file)['passed']; return
    model = build_rate_equation_model()
    cfg = POINT.config()
    beams = build_multilevel_mot_beams(config=cfg)
    rng = np.random.default_rng(2026102503)
    checks, events = [], []
    for r in np.vstack([np.zeros(3), np.eye(3)*.001, rng.uniform(-.014, .014, (32, 3))]):
        v = rng.uniform(-40, 40, 3)
        actual = fast.observable(r, v, np.array([0., 0., 1.]), data)
        ref = rate_equation_observable(model, beams, tuple(r), tuple(v), None, cfg,
            magnetic_field_function=lambda p: surrogate_effective_field_t(p, FIELD_CONFIG))
        np.testing.assert_allclose(actual[0], ref.force_n, rtol=4e-6, atol=3e-29)
        np.testing.assert_allclose(actual[1], ref.populations, rtol=4e-6, atol=5e-10)
        np.testing.assert_allclose(fast.field(r, data), surrogate_effective_field_t(r, FIELD_CONFIG),
                                   rtol=1e-11, atol=1e-16)
        checks.append(dict(position_m=r.tolist(), velocity_m_per_s=v.tolist(), passed=True))
    zero = np.zeros(3); axis = np.array([0., 0., 1.])
    np.testing.assert_allclose(fast.observable(zero, zero, axis, data)[0], zero, atol=1e-28)
    for i in range(3):
        r = np.eye(3)[i]*1e-5; v = np.eye(3)[i]*.01
        assert fast.observable(r, zero, axis, data)[0][i] < 0
        assert fast.observable(zero, v, axis, data)[0][i] < 0
    for index, (r, v) in enumerate([((.001, 0., 0.), (0., 0., 0.)),
                                    ((.006, .002, -.001), (-8., .2, 0.))]):
        ref = rate_equation_observable(model, beams, r, v, None, cfg,
            magnetic_field_function=lambda p: surrogate_effective_field_t(p, FIELD_CONFIG),
            store_rate_matrix=True, store_beam_transition_quantities=True)
        samples = event_occupations(ref.rate_matrix_per_s, ref.populations, .002, 64, 2026102510+index)
        sem = samples.std(axis=0, ddof=1)/8
        population_pass = bool(np.all(abs(samples.mean(axis=0)-ref.populations) <= 6*sem+1e-6))
        reward = np.zeros((24, 3))
        for b, beam in enumerate(beams):
            momentum = HBAR_J_S*2*np.pi/beam.wavelength_m*np.asarray(beam.direction)
            w = ref.beam_transition_quantities.stimulated_coefficients_per_s[b]
            reward[:8] += w.sum(axis=0)[:, None]*momentum
            reward[8:] -= w.sum(axis=1)[:, None]*momentum
        fsamples = samples@reward; fsem = fsamples.std(axis=0, ddof=1)/8
        force_pass = bool(np.all(abs(fsamples.mean(axis=0)-ref.force_n) <= 6*fsem+1e-29))
        events.append(dict(position_m=r, velocity_m_per_s=v, population_pass=population_pass,
                           force_pass=force_pass, reference_force_n=list(ref.force_n),
                           sampled_force_n=fsamples.mean(axis=0).tolist()))
    passed = all(e['population_pass'] and e['force_pass'] for e in events)
    # Warm both compiled trajectory methods before worker startup.
    fast.classify(np.array([.01, 0., 0.]), np.array([-1., 0., 0.]), 2.5e-6, 1e-6, data)
    fast.integrate(np.array([.01, 0., 0.]), np.array([-1., 0., 0.]), 1e-6, data)
    write_json(file, dict(passed=passed, local_checks=checks, events=events,
        limits='Frozen-environment event comparison; moving-atom population adiabaticity remains a model limitation'))
    assert passed, 'Local physical-kernel/event comparison failed'


def statistics(mode, discs, positions, directions, areas, *, provisional=True):
    folder = OUTPUT/mode
    masks = np.empty((discs, POINTS, len(ce.VELOCITIES)), dtype=np.int8)
    for d in range(discs):
        for p in range(POINTS):
            with np.load(folder/'rays'/f'd{d:03d}_p{p:03d}.npz') as saved:
                assert str(saved['status']) == 'resolved'
                np.testing.assert_array_equal(saved['speeds'][:len(ce.VELOCITIES)], ce.VELOCITIES)
                masks[d, p] = saved['codes'][:len(ce.VELOCITIES)]
                assert np.all(saved['codes'][len(ce.VELOCITIES):] == 0)
                assert np.all((masks[d, p] == 0) | (masks[d, p] == 1))
    spectrum = areas[:discs, None]*masks.mean(axis=1)
    velocity = ce.VELOCITIES
    weighted = spectrum*velocity**3*np.exp(-velocity**2/THERMAL_SCALE_M2_PER_S2)
    rate_by_disc = LOADING_RATE_PREFACTOR*np.trapezoid(
        np.column_stack((np.zeros(discs), weighted)), np.r_[0., velocity], axis=1)
    mean = float(rate_by_disc.mean())
    half = float(student_t.ppf(.975, discs-1)*rate_by_disc.std(ddof=1)/np.sqrt(discs))
    if np.all(rate_by_disc == 0):
        # All-empty samples need a finite bound, not a spurious zero-width CI.
        maximum_rate = LOADING_RATE_PREFACTOR*np.pi*.015**2*np.trapezoid(
            np.r_[0., velocity**3*np.exp(-velocity**2/THERMAL_SCALE_M2_PER_S2)],
            np.r_[0., velocity])
        half = float(maximum_rate*(1-.025**(1/discs)))
    coarse = LOADING_RATE_PREFACTOR*np.trapezoid(np.column_stack((np.zeros(discs), weighted[:, 1::2])),
                                               np.r_[0., velocity[1::2]], axis=1)
    quadrature = abs(float(coarse.mean())-mean)
    status = 'provisional; numerical and retention gates pending' if provisional else 'pending final audit'
    summary = dict(mode=mode, at_utc=utc(), source_sha256=SOURCE_SHA, status=status,
        complete_direction_discs=discs, points_per_disc=POINTS, resolved_rays=discs*POINTS,
        loading_rate_atoms_per_s=mean, loading_95_half_width_atoms_per_s=half,
        loading_ci95_atoms_per_s=[max(0., mean-half), mean+half],
        loading_relative_half_width=half/mean if mean > 0 else None,
        loading_coefficient_m3_per_s=mean/N87,
        loading_coefficient_95_half_width_m3_per_s=half/N87,
        thermal_conditions=read(OUTPUT/'campaign_manifest.json')['thermal_conditions'],
        signed_central_gradients_g_per_cm=[10., 10., -20.],
        relative_half_width_target=TARGET,
        velocity_quadrature_difference_atoms_per_s=quadrature,
        velocity_quadrature_relative_difference=quadrature/mean if mean > 0 else None,
        quadrature_pass=bool(mean > 0 and quadrature <= .02*mean),
        precision_pass=bool(mean > 0 and half/mean <= TARGET),
        nonmonotone_ray_count=int(np.count_nonzero(np.any(np.diff(masks, axis=2)>0, axis=2))),
        maximum_captured_grid_speed_m_per_s=float(np.max(velocity[np.any(masks == 1, axis=(0, 1))])) if np.any(masks) else None,
        uncertainty='95% Student-t intervals of independent direction-disc loading integrals; interim intervals are descriptive, not sequential stopping rules',
        normalization='retained historical numerical prefactor interpreted with a normalized Maxwell distribution',
        scope='central-core loading through the specified interior launch sections')
    write_json(folder/'latest_estimate.json', summary)
    snapshots = folder/'estimates'
    snapshots.mkdir(exist_ok=True)
    write_json(snapshots/f'discs_{discs:05d}.json', summary)
    ce._atomic_npz(folder/'disc_statistics.npz', velocity_m_per_s=velocity,
        disc_cross_section_m2=spectrum, disc_loading_atoms_per_s=rate_by_disc,
        disc_areas_m2=areas[:discs], capture_masks=masks)
    write_csv(folder/'loading_by_disc.csv', [dict(disc_index=d, loading_rate_atoms_per_s=float(x),
                                                 loading_coefficient_m3_per_s=float(x/N87))
                                           for d, x in enumerate(rate_by_disc)])
    print('LOADING UPDATE '+json.dumps(summary), flush=True)
    return summary


def run_phase(mode, discs, seed, data, pool):
    positions, directions, areas, impacts = geometry(mode, discs, seed)
    folder = OUTPUT/mode
    resolved = set()
    jobs = []
    for d in range(discs):
        for p in range(POINTS):
            file = folder/'rays'/f'd{d:03d}_p{p:03d}.npz'
            if file.exists():
                with np.load(file) as saved:
                    if str(saved['status']) == 'resolved':
                        resolved.add((d, p)); continue
            jobs.append((POINT, d, p, positions[d, p], directions[d], file, mode, data))
    iterator = iter(jobs)
    pending = {}
    checkpoints = {4, 8, 16, 32, 64, 100, discs} | set(range(125, discs+1, 25))
    reported = set()
    began = time.perf_counter()
    before = len(resolved)
    def submit():
        item = next(iterator, None)
        if item is not None:
            pending[pool.submit(worker, item)] = (item[1], item[2])
            return True
        return False
    for _ in range(pool._max_workers*2):
        if not submit(): break
    last = 0.
    while pending:
        done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
        for future in done:
            identity = pending.pop(future)
            result = future.result()
            if result['status'] != 'resolved':
                write_json(OUTPUT/'unresolved_ray.json', dict(mode=mode, result=result, at_utc=utc()))
                raise RuntimeError('Direct capture node unresolved or tail-censored: '+str(result))
            resolved.add(identity)
            submit()
        elapsed = time.perf_counter()-began
        prefix = 0
        while prefix < discs and all((prefix, p) in resolved for p in range(POINTS)):
            prefix += 1
        ready = sorted(c for c in checkpoints if c <= prefix and c not in reported)
        if ready:
            chosen = ready[-1]
            statistics(mode, chosen, positions, directions, areas)
            reported.update(c for c in checkpoints if c <= chosen)
        if elapsed-last >= 30 or not pending:
            new = len(resolved)-before
            eta = (discs*POINTS-len(resolved))*elapsed/new if new else None
            progress = dict(mode=mode, at_utc=utc(), pid=os.getpid(),
                completed_rays=len(resolved), requested_rays=discs*POINTS,
                complete_prefix_discs=prefix, elapsed_this_invocation_s=elapsed,
                estimated_remaining_capture_hours=eta/3600 if eta is not None else None,
                latest_estimate_file=str(folder/'latest_estimate.json'))
            write_json(OUTPUT/'progress.json', progress)
            print('PROGRESS '+json.dumps(progress), flush=True)
            last = elapsed
    assert len(resolved) == discs*POINTS
    row = statistics(mode, discs, positions, directions, areas)
    timing_path = folder/'capture_timing.json'
    previous = read(timing_path)['total_capture_wall_s'] if timing_path.exists() else 0.
    write_json(timing_path, dict(total_capture_wall_s=previous+time.perf_counter()-began))
    return row, positions, directions, areas, impacts


def retention(mode, positions, directions, impacts, data, pool):
    path = OUTPUT/mode/'retention_audit.json'
    if path.exists():
        report = read(path); assert report['passed']; return report
    jobs = []
    for d in range(len(positions)):
        candidates = []
        for p in range(POINTS):
            with np.load(OUTPUT/mode/'rays'/f'd{d:03d}_p{p:03d}.npz') as saved:
                captured = saved['speeds'][saved['codes'] == 1]
                if len(captured):
                    candidates.append((impacts[d, p], p, float(captured.max()),
                                       float(captured[len(captured)//2])))
        if candidates:
            _, p, high, middle = max(candidates)
            for speed in sorted(set((high, middle))):
                jobs.append((d, p, speed, positions[d, p], directions[d], data))
    cases = []
    for result in pool.map(retention_worker, jobs):
        cases.append(result)
        write_json(path.with_name('retention_progress.json'), dict(
            completed=len(cases), requested=len(jobs), failures=sum(not c['passed'] for c in cases)))
    report = dict(passed=bool(cases) and all(c['passed'] for c in cases), cases=cases,
        criterion='both timesteps bounded in final 25 ms core; endpoints within 10 micrometers',
        selection='outermost captured impact parameter on every disc, highest and median captured speeds')
    write_json(path, report)
    assert report['passed'], 'Longer-duration retention gate failed'
    return report


def finalize(mode, row, areas):
    folder = OUTPUT/mode
    with np.load(folder/'disc_statistics.npz') as saved:
        spectrum = saved['disc_cross_section_m2']; velocity = saved['velocity_m_per_s']
        values = saved['disc_loading_atoms_per_s']; masks = saved['capture_masks']
    mean = spectrum.mean(axis=0)
    half = student_t.ppf(.975, len(spectrum)-1)*spectrum.std(axis=0, ddof=1)/np.sqrt(len(spectrum))
    lower, upper = np.maximum(0., mean-half), np.minimum(np.pi*.015**2, mean+half)
    boundary_fraction = 1-.025**(1/len(spectrum))
    empty = np.all(masks == 0, axis=(0, 1)); full = np.all(masks == 1, axis=(0, 1))
    upper[empty] = np.pi*.015**2*boundary_fraction
    lower[full] = np.maximum(0., lower[full]-np.pi*.015**2*boundary_fraction)
    write_csv(folder/'cross_section.csv', [dict(velocity_m_per_s=float(v), cross_section_m2=float(s),
        lower_95_m2=float(lo), upper_95_m2=float(hi)) for v, s, lo, hi in zip(velocity, mean, lower, upper)])
    rng = np.random.default_rng(2026102599)
    bootstrap = np.asarray([np.mean(rng.choice(values, len(values), replace=True)) for _ in range(4000)])
    row.update(cluster_bootstrap_percentile_ci95_atoms_per_s=np.quantile(bootstrap, [.025, .975]).tolist(),
               retention_pass=True, full_fixed_pilot_pass=True, numerically_complete=True,
               status='numerically checked; interior-aperture loading; launch-aperture audit pending')
    write_json(folder/'summary.json', row)
    figures = OUTPUT/'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axes[0].plot(velocity, mean*1e6)
    axes[0].fill_between(velocity, lower*1e6, upper*1e6, alpha=.25)
    axes[0].set(xlabel='Incident speed [m/s]', ylabel='Capture cross section [mm²]',
                title='Central-core capture · pointwise 95% disc-cluster intervals')
    axes[1].plot(velocity, LOADING_RATE_PREFACTOR*mean*velocity**3*np.exp(-velocity**2/THERMAL_SCALE_M2_PER_S2))
    axes[1].set(xlabel='Incident speed [m/s]', ylabel='Loading integrand [atoms/s per (m/s)]',
                title=f"R = {row['loading_rate_atoms_per_s']:.5g} ± {row['loading_95_half_width_atoms_per_s']:.3g} atoms/s (95%)")
    fig.suptitle('mot_strange · gradients (+10,+10,−20) G/cm · 15 mm in-cell launch discs\n'
                 f'T = {THERMAL_T:.3f} K · n87 = {N87:.6g} m⁻³')
    for ext in ('png', 'pdf'):
        fig.savefig(figures/f'capture_and_loading_spectrum.{ext}', dpi=180)
    plt.close(fig)
    write_json(OUTPUT/'completion.json', dict(at_utc=utc(), computed=True, mode=mode,
        relative_precision_pass=row['precision_pass'], numerical_gates_passed=True,
        aperture_gate_pending=True, visual_review_pending=True))


def run(args):
    initialize()
    write_json(OUTPUT/'process.json', dict(pid=os.getpid(), command=sys.argv, workers=args.workers,
                                          started_at_utc=utc(), source_sha256=SOURCE_SHA))
    print('VALIDATING '+SOURCE_SHA, flush=True)
    data = fast.prepare(POINT.config(), FIELD_CONFIG)
    ce._atomic_npz(OUTPUT/'compiled_inputs.npz', **{f'array_{i}': x for i, x in enumerate(data)})
    validate(data)
    print('LOCAL VALIDATION PASSED', flush=True)
    if args.stage == 'validate': return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        positions, directions, areas, impacts = geometry('pilot', PILOT_DISCS, PILOT_SEED)
        gate = OUTPUT/'validation/fresh_workers.json'
        if not gate.exists():
            jobs = [(positions[d, p], directions[d], speed, data)
                    for d, p in ((0, 0), (17, 13)) for speed in (.25, 10., 40., 120.)]
            checks = list(pool.map(fresh_worker_check, jobs))
            write_json(gate, dict(passed=all(c['passed'] for c in checks), checks=checks))
        assert read(gate)['passed'], 'Spawned adaptive/fixed comparison failed'
        pilot, positions, directions, areas, impacts = run_phase('pilot', PILOT_DISCS, PILOT_SEED, data, pool)
        gate = OUTPUT/'validation/full_fixed_comparison.json'
        if not gate.exists():
            checks = list(pool.map(fixed_comparison, [
                (OUTPUT/'pilot/rays'/f'd{d:03d}_p{p:03d}.npz', data) for d, p in ((0, 0), (17, 13))]))
            write_json(gate, dict(passed=all(c['passed'] for c in checks), checks=checks))
        assert read(gate)['passed'], 'Full-grid fixed-RK4 pilot comparison failed'
        retention('pilot', positions, directions, impacts, data, pool)
        assert pilot['quadrature_pass'], 'Pilot velocity quadrature failed'
        planfile = OUTPUT/'sampling_plan.json'
        if not planfile.exists():
            with np.load(OUTPUT/'pilot/disc_statistics.npz') as saved:
                values = saved['disc_loading_atoms_per_s']
            mean, variance = float(values.mean()), float(values.var(ddof=1))
            assert mean > 0, 'Zero pilot loading requires an independent expanded pilot'
            count = 200
            while 1.5*student_t.ppf(.975, count-1)**2*variance/(TARGET*mean)**2 > count:
                count += 25
                if count > 5000:
                    raise RuntimeError('Pilot implies >5000 discs; inspect variance and allocation')
            elapsed = read(OUTPUT/'pilot/capture_timing.json')['total_capture_wall_s']
            write_json(planfile, dict(frozen_at_utc=utc(), disc_count=count, points_per_disc=POINTS,
                production_seed=PRODUCTION_SEED, pilot_mean_atoms_per_s=mean,
                pilot_disc_variance=variance, relative_95_half_width_target=TARGET,
                estimated_production_capture_hours=[elapsed*count/PILOT_DISCS/3600,
                                                    1.5*elapsed*count/PILOT_DISCS/3600],
                optional_stopping=False))
            print('FROZEN PLAN '+json.dumps(read(planfile)), flush=True)
        if args.stage == 'pilot': return
        count = read(planfile)['disc_count']
        row, positions, directions, areas, impacts = run_phase('production', count, PRODUCTION_SEED, data, pool)
        retention('production', positions, directions, impacts, data, pool)
        assert row['quadrature_pass'], 'Production velocity quadrature failed'
        finalize('production', row, areas)
        print('PRODUCTION COMPLETE '+json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('validate', 'pilot', 'all'))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1: parser.error('workers must be positive')
    with campaign_lock():
        try:
            run(args)
        except Exception as error:
            write_json(OUTPUT/'failure.json', dict(at_utc=utc(), error=repr(error), traceback=traceback.format_exc()))
            raise
