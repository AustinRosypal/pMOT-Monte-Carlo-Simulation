"""Run resumable disc batches for the 240-setting effective two-level pMOT scan."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta
import hashlib
import json
from math import ceil, pi
import os
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import t

from pmot.pMOT_2Level import loading_dynamics as dynamics
from pmot.pMOT_2Level import loading_retention as retention
from pmot.pMOT_2Level import loading_sampling as sampling
from pmot.pMOT_2Level import loading_scan as scan
from validate_pmot_2level_loading import OUTPUT, ROOT, write_json


CLASSES = {'core': dynamics.CORE, 'off_core': retention.VERIFIED_OFF_CORE,
           'localized_unverified': dynamics.LOCALIZED_OFF_CORE, 'unresolved': dynamics.UNRESOLVED}
CHICAGO = ZoneInfo('America/Chicago')
PILOT_DISCS = 8
PILOT_POINTS = 8
PRODUCTION_POINTS = 16
PRODUCTION_DRAWS = 1
_settings_data = _discs_data = None


def fingerprint():
    files = [Path(__file__), Path(scan.__file__), Path(dynamics.__file__), Path(retention.__file__),
             Path(sampling.__file__), Path(scan.__file__).with_name('simulation.py'),
             Path(scan.__file__).with_name('ac_stark.py'), Path(scan.__file__).with_name('trapping_geometry.py'),
             ROOT / 'data/raw/pmot/Arora_CCSD_Differential_Polarizabilities.csv',
             ROOT / 'uv.lock']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def start_worker(settings_data, discs_data):
    global _settings_data, _discs_data
    _settings_data, _discs_data = settings_data, discs_data


def run_disc(setting_id, disc_id, destination, fingerprint_digest):
    start = perf_counter()
    data = _settings_data[setting_id]
    positions, direction, area, speeds, masses = _discs_data[disc_id]
    labels = np.empty(speeds.shape, dtype=np.uint8)
    attempts = np.full((*speeds.shape, 4), 255, dtype=np.uint8)
    witnesses = {}
    for idx in np.ndindex(speeds.shape):
        p, _, _ = idx
        label, trail, witness = retention.classify(positions[p], direction*speeds[idx], data)
        labels[idx] = label
        attempts[idx][:len(trail)] = trail
        if witness is not None:
            witnesses[','.join(map(str, idx))] = witness
    rates = {name: sampling.disc_rate(labels == code, area, masses) for name, code in CLASSES.items()}
    counts = np.bincount(labels.ravel(), minlength=6)
    counts_by_stratum = np.array([np.bincount(labels[:, group, :].ravel(), minlength=6)
                                  for group in range(len(masses))])
    tmp = destination.with_suffix('.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, labels=labels, attempts=attempts,
                            witnesses=json.dumps(witnesses, allow_nan=False),
                            speeds=speeds, positions=positions, direction=direction, area=area, masses=masses,
                            fingerprint=fingerprint_digest, setting=setting_id, disc=disc_id,
                            wall_s=perf_counter()-start)
    tmp.replace(destination)
    return rates, counts, counts_by_stratum


def checkpoint(folder, setting, disc):
    return folder / f's{setting:03}_d{disc:04}.npz'


def audit_disc(path, setting, disc, expected, digest):
    with np.load(path, allow_pickle=False) as row:
        if row['fingerprint'].item() != digest or int(row['setting']) != setting or int(row['disc']) != disc:
            raise ValueError(f'Checkpoint identity mismatch: {path}')
        for name, value in zip(('positions', 'direction', 'area', 'speeds', 'masses'), expected, strict=True):
            if not np.array_equal(row[name], value):
                raise ValueError(f'Checkpoint {name} differs from the frozen sample: {path}')
        labels = row['labels']
        attempts = row['attempts']
        if labels.shape != expected[3].shape or attempts.shape != (*labels.shape, 4):
            raise ValueError(f'Malformed checkpoint: {path}')
        if (np.any(labels > 5) or np.any(labels == dynamics.NUMERICAL)
                or np.any(attempts[..., :2] > dynamics.NUMERICAL)
                or np.any((attempts[..., 2:] > dynamics.NUMERICAL) & (attempts[..., 2:] != 255))):
            raise ValueError(f'Malformed trajectory classification: {path}')
        last_pair = np.where((attempts[..., 2] == 255)[..., None], attempts[..., :2], attempts[..., 2:])
        definitive = labels != dynamics.UNRESOLVED
        base = np.where(labels == retention.VERIFIED_OFF_CORE, dynamics.LOCALIZED_OFF_CORE, labels)
        if np.any((last_pair != base[..., None]) & definitive[..., None]):
            raise ValueError(f'Unconverged definitive classification: {path}')
        witnesses = json.loads(row['witnesses'].item())
        verified = {','.join(map(str, i)) for i in np.argwhere(labels == retention.VERIFIED_OFF_CORE)}
        if verified != {i for i, w in witnesses.items() if w['status'] == 'verified'}:
            raise ValueError(f'Verified pocket witness mismatch: {path}')
        area, masses = expected[2], expected[4]
        return ({name: sampling.disc_rate(labels == code, area, masses) for name, code in CLASSES.items()},
                np.bincount(labels.ravel(), minlength=6),
                np.array([np.bincount(labels[:, group, :].ravel(), minlength=6) for group in range(len(masses))]))


def summarize(records, count):
    output = []
    for index, n in enumerate(count):
        values = [records[index, disc] for disc in range(n)]
        output.append({
            'setting_id': index, 'discs': n,
            'statistics': {name: sampling.interval([v[0][name] for v in values]) for name in CLASSES},
            'counts': np.sum([v[1] for v in values], axis=0).tolist(),
            'counts_by_speed_stratum': np.sum([v[2] for v in values], axis=0).tolist(),
        })
    return output


def freeze_counts(pilot, max_discs):
    counts = []
    for row in pilot['settings']:
        candidates = []
        for name in ('core', 'off_core'):
            result = row['statistics'][name]
            if result['mean'] > 0:
                ratio = result['sd'] / result['mean']
                demand = next((n for n in range(24, max_discs+1)
                               if t.ppf(.975, n-1)*ratio/np.sqrt(n) < .10), max_discs)
                candidates.append(demand)
        counts.append(max(candidates, default=24))
    return counts


def run(phase, workers, max_discs):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if phase == 'plan':
        pilot_path = OUTPUT / 'pilot_summary.json'
        pilot = json.loads(pilot_path.read_text(encoding='utf-8'))
        plan = {'seed': sampling.PRODUCTION_SEED, 'pilot_seed': sampling.PILOT_SEED,
                'pilot_sha256': hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
                'points_per_disc': PRODUCTION_POINTS, 'draws_per_stratum': PRODUCTION_DRAWS,
                'max_discs': max_discs, 'counts': freeze_counts(pilot, max_discs)}
        destination = OUTPUT / 'production_plan.json'
        if destination.exists() and json.loads(destination.read_text(encoding='utf-8')) != plan:
            raise ValueError('A different production plan is already frozen')
        write_json(destination, plan)
        print(f'Frozen {sum(plan["counts"]):,} direction discs across {len(plan["counts"])} settings')
        return

    path = OUTPUT / phase
    path.mkdir(exist_ok=True)
    if phase == 'pilot':
        design = {'seed': sampling.PILOT_SEED, 'points_per_disc': PILOT_POINTS, 'draws_per_stratum': 1,
                  'counts': [PILOT_DISCS] * len(scan.settings())}
    else:
        design = json.loads((OUTPUT / 'production_plan.json').read_text(encoding='utf-8'))
        if design['seed'] != sampling.PRODUCTION_SEED or design['seed'] == design['pilot_seed']:
            raise ValueError('Production must be independent of the pilot')
    manifest = {'phase': phase, **design, 'speed_edges_m_s': sampling.SPEED_EDGES, 'sources': fingerprint(),
                'step_cap_s': retention.STEP_CAP_S, 'retention_horizon_s': retention.HORIZON_S}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = path / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))['sha256'] != digest:
            raise ValueError('Checkpoint phase, source or sampling plan changed')
    else:
        write_json(manifest_path, {'sha256': digest, **manifest})

    inputs = []
    for setting in scan.settings():
        trap, table, power, _, _ = scan.prepare_reference(setting)
        params, beams, waves, shifts = dynamics.prepare(trap, table, power)
        lo, hi = np.searchsorted(waves, [setting.wavelength_nm-.003, setting.wavelength_nm+.003])
        if lo == 0 or hi == len(waves):
            raise ValueError('Required wavelength support not in polarizability table')
        inputs.append((params, beams, waves[lo-1:hi+1].copy(), shifts[lo-1:hi+1].copy()))
    count = design['counts']
    discs = [sampling.sampled_disc(design['seed'], disc, design['points_per_disc'], design['draws_per_stratum'])
             for disc in range(max(count))]
    records, pending = {}, []
    for disc in range(max(count)):
        for setting, n in enumerate(count):
            if disc >= n:
                continue
            destination = checkpoint(path, setting, disc)
            if destination.exists():
                records[setting, disc] = audit_disc(destination, setting, disc, discs[disc], digest)
            else:
                pending.append((setting, disc, destination))

    completed_at_start = len(records)
    started = perf_counter()
    total = sum(count)

    def progress():
        now = datetime.now(CHICAGO)
        elapsed = perf_counter() - started
        fresh = len(records)-completed_at_start
        eta = now + timedelta(seconds=elapsed*(total-len(records))/fresh) if fresh >= 40 else None
        value = {'phase': phase, 'pid': os.getpid(), 'workers': workers, 'updated_chicago': now.isoformat(),
                 'disc_batches_done': len(records), 'disc_batches_total': total, 'elapsed_this_process_s': elapsed,
                 'settings_done': sum(sum((i, j) in records for j in range(n)) == n for i, n in enumerate(count)),
                 'eta_chicago': eta.isoformat() if eta else None}
        write_json(OUTPUT / f'{phase}_progress.json', value)
        print(json.dumps(value), flush=True)

    progress()
    last_progress = started
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=start_worker, initargs=(inputs, discs)) as pool:
            queue = iter(pending)
            active = {}

            def schedule():
                job = next(queue, None)
                if job:
                    active[pool.submit(run_disc, *job, digest)] = job[:2]

            for _ in range(2*workers):
                schedule()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    records[active.pop(future)] = future.result()
                    schedule()
                if perf_counter()-last_progress > 30:
                    progress()
                    last_progress = perf_counter()
    progress()
    # Reread all saved batches and independently reconstruct their statistics.
    for setting, disc in records:
        records[setting, disc] = audit_disc(checkpoint(path, setting, disc), setting, disc, discs[disc], digest)
    write_json(OUTPUT / f'{phase}_summary.json', {'manifest_sha256': digest, 'phase': phase,
               'workers': workers, 'speed_edges_m_s': sampling.SPEED_EDGES, **design,
               'settings': summarize(records, count)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pilot', 'plan', 'production'))
    parser.add_argument('--workers', type=int, default=10, choices=range(1, 11))
    parser.add_argument('--max-discs', type=int, default=128, choices=range(24, 513), metavar='24..512')
    args = parser.parse_args()
    run(args.phase, args.workers, args.max_discs)


if __name__ == '__main__':
    main()
