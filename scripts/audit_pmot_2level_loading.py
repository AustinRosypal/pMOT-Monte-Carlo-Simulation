"""Replay saved two-level pMOT shots and independently aggregate paired comparisons."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.stats import t

from pmot.pMOT_2Level import loading_dynamics as dyn
from pmot.pMOT_2Level import loading_retention as ret
from pmot.pMOT_2Level import loading_sampling as samp
from pmot.pMOT_2Level import loading_scan as scan
from run_pmot_2level_loading import CLASSES, audit_disc, checkpoint, fingerprint, summarize
from validate_pmot_2level_loading import OUTPUT, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay(job):
    index, disc, node, label, r, v = job
    trap, table, power, _, _ = scan.prepare_reference(scan.settings()[index])
    data = dyn.prepare(trap, table, power)
    exact, attempts, witness = ret.classify(r, v, data)
    convergence = []
    for step, horizon in ((1.25e-6, 2.), (.625e-6, 2.), (1.25e-6, 3.)):
        code, end, speed, elapsed = dyn.shoot(r, v, step, horizon, data)
        convergence.append({'code': int(code), 'step_cap_s': step, 'duration_s': horizon,
                            'elapsed_s': elapsed, 'end_m': end.tolist() if np.isfinite(end).all() else None,
                            'end_m_s': speed.tolist() if np.isfinite(speed).all() else None})
    base = dyn.LOCALIZED_OFF_CORE if label == ret.VERIFIED_OFF_CORE else label
    exact_pass = exact == label
    # A change among ambiguous labels is useful diagnostic evidence. Only a
    # contradiction of a definitive classification blocks the report.
    stable = all(x['code'] == base for x in convergence)
    passed = exact_pass and (stable or label in (dyn.UNRESOLVED, dyn.LOCALIZED_OFF_CORE))
    return {'setting': index, 'disc': disc, 'node': node, 'saved_code': int(label),
            'replay_code': int(exact), 'exact_attempts': attempts, 'witness': witness,
            'replays': convergence, 'passed': passed}


def paired_comparisons(records, counts):
    ids = {setting: i for i, setting in enumerate(scan.settings())}
    results = []
    for first, i in ids.items():
        if first.wavelength_nm != scan.WAVELENGTHS_NM[0]:
            continue
        second = scan.Setting(first.ranks, scan.WAVELENGTHS_NM[1], first.geometry, first.axial_magnitude_g_cm)
        j = ids[second]
        n = min(counts[i], counts[j])
        row = {'ranks': first.ranks, 'geometry': first.geometry, 'axial_magnitude_g_cm': first.axial_magnitude_g_cm,
               'paired_discs': n}
        for category in ('core', 'off_core'):
            differences = np.asarray([records[j, d][0][category]-records[i, d][0][category] for d in range(n)])
            mean = float(np.mean(differences))
            half = float(t.ppf(.975, n-1) * differences.std(ddof=1)/np.sqrt(n))
            row[category] = {'delta_atoms_s': mean, 'half_width': half if np.any(differences) else None,
                             'outcome': 'scalar_zero_higher' if mean > half else
                             'cancellation_higher' if mean < -half else 'no_resolved_difference'}
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pilot', 'production'))
    parser.add_argument('--workers', type=int, default=10, choices=range(1, 17))
    args = parser.parse_args()
    phase = args.phase
    path = OUTPUT / phase
    summary = read(OUTPUT / f'{phase}_summary.json')
    manifest = read(path / 'manifest.json')
    digest = manifest['sha256']
    if summary['manifest_sha256'] != digest or manifest['sources'] != fingerprint():
        raise ValueError('Frozen reference inputs changed')
    expected_seed = samp.PILOT_SEED if phase == 'pilot' else samp.PRODUCTION_SEED
    if manifest['seed'] != expected_seed or summary['seed'] != expected_seed:
        raise ValueError('Phase seeds do not match')
    counts = manifest['counts']
    if len(counts) != len(scan.settings()) or counts != summary['counts']:
        raise ValueError('Incomplete or changed counts')
    samples = [samp.sampled_disc(expected_seed, d, manifest['points_per_disc'], manifest['draws_per_stratum'])
               for d in range(max(counts))]
    # Reconstruct every disc from its original independent seed, then pick
    # random phase-space points per outcome and scan family for numerical replay.
    rng = np.random.default_rng(2026110911 if phase == 'pilot' else 2026110912)
    records, candidates = {}, {}
    seen = defaultdict(int)
    for i, (setting, count) in enumerate(zip(scan.settings(), counts, strict=True)):
        family = (setting.ranks, setting.wavelength_nm, setting.geometry)
        for d in range(count):
            batch = checkpoint(path, i, d)
            sample = samples[d]
            records[i, d] = audit_disc(batch, i, d, sample, digest)
            with np.load(batch, allow_pickle=False) as archive:
                labels = archive['labels']
                for label in (dyn.ESCAPED, *CLASSES.values()):
                    choices = np.argwhere(labels == label)
                    if not len(choices):
                        continue
                    key = (*family, label)
                    seen[key] += len(choices)
                    if key in candidates and rng.random() >= len(choices) / seen[key]:
                        continue
                    p, band, draw = map(int, choices[rng.integers(len(choices))])
                    candidates[key] = (i, d, [p, band, draw], label, sample[0][p],
                                       sample[1]*sample[3][p, band, draw])
    reconstructed = summarize(records, counts)
    if reconstructed != summary['settings']:
        raise ValueError('Summary does not equal independent checkpoint aggregation')
    if phase == 'production':
        write_json(OUTPUT / 'production_paired_wavelengths.json', {
            'manifest_sha256': digest, 'method': 'Matched independent direction discs, scalar-zero minus cancellation',
            'rows': paired_comparisons(records, counts),
        })
    print(f'Reconstructed {len(records):,} discs; replaying {len(candidates)} shots', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        replays = list(pool.map(replay, candidates.values()))
    audit = {'phase': phase, 'manifest_sha256': digest, 'disc_batches': len(records),
             'passed': all(row['passed'] for row in replays), 'replays': replays,
             'unresolved_settings': [row['setting_id'] for row in reconstructed if row['counts'][dyn.UNRESOLVED]],
             'localized_unverified_settings': [row['setting_id'] for row in reconstructed
                                                if row['counts'][dyn.LOCALIZED_OFF_CORE]]}
    write_json(OUTPUT / f'{phase}_replay_audit.json', audit)
    print(f'Replay audit passed: {audit["passed"]}', flush=True)
    if not audit['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
