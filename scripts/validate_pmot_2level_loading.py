"""Preflight the effective two-level loading settings and measure trajectory costs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pmot.pMOT_2Level.loading_dynamics import local, prepare, shoot
from pmot.pMOT_2Level.loading_scan import (
    cooling_saturation, prepare_reference, settings, speed_support_upper_m_s,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/statistics/pMOT_2Level/loading_20261109'
AXIS = np.array([0., 0., 1.])
ZERO = np.zeros(3)


def controls(setting, data, power, intensity, signed):
    def derivative(index, h, velocity=False):
        dr = h*np.eye(3)[index]
        plus = local(ZERO, dr, AXIS, *data) if velocity else local(dr, ZERO, AXIS, *data)
        minus = local(ZERO, -dr, AXIS, *data) if velocity else local(-dr, ZERO, AXIS, *data)
        return (plus[0][index] - minus[0][index]) / (2*h)

    return {
        **asdict(setting), 'power_w_xyz': power.tolist(),
        'center_intensity_w_m2_six_components': intensity.tolist(),
        'signed_gradient_g_cm_xyz': signed.tolist(),
        'position_slopes_s2_inv_xyz': [derivative(j, 1e-7) for j in range(3)],
        'velocity_slopes_s_inv_xyz': [derivative(j, 1e-4, True) for j in range(3)],
    }


def timed_cases(task):
    index, data = task
    # Include clean shots, deliberately waist-skimming shots, an off-axis ray,
    # a thermal tail check, and the preloaded core (diagnostics only).
    shots = [
        ((.01, 0., 0.), (-18., 0., 0.)),
        ((.000001, .000002, -.01001), (.001, .003, 30.)),
        ((.006, -.001, .010), (0., 0., -22.)),
        ((.009, -.003, .006), (-80., 11., -40.)),
        ((0., 0., 0.), (0., 0., 0.)),
    ]
    rows = []
    for r, v in shots:
        attempts = []
        for dt, duration in ((5e-6, .5), (2.5e-6, .5), (2.5e-6, 2.), (1.25e-6, 2.)):
            start = perf_counter()
            code, end, speed, elapsed = shoot(np.array(r), np.array(v), dt, duration, data)
            attempts.append({'step_cap_s': dt, 'horizon_s': duration,
                             'code': int(code), 'end_m': end.tolist(),
                             'end_m_s': speed.tolist(), 'elapsed_s': elapsed,
                             'wall_s': perf_counter() - start})
        rows.append({'r_m': r, 'v_m_s': v, 'attempts': attempts})
    return index, rows


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=10, choices=range(1, 11))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for index, setting in enumerate(settings()):
        trap, table, power, intensity, signed = prepare_reference(setting)
        data = prepare(trap, table, power)
        rows.append(controls(setting, data, power, intensity, signed))
        if setting.axial_magnitude_g_cm in (2, 20, 40):
            # These trajectories are well within the original table. Crop the
            # immutable parent input for spawn; never extrapolate at runtime.
            waves = data[2]
            lo, hi = np.searchsorted(waves, [setting.wavelength_nm-.003, setting.wavelength_nm+.003])
            jobs.append((index, (data[0], data[1], waves[lo-1:hi+1].copy(), data[3][lo-1:hi+1].copy())))
    write_json(OUTPUT / 'controls.json', {'settings': rows, 'cooling_s0_seff': cooling_saturation(),
                                          'rigorous_speed_support_m_s': speed_support_upper_m_s()})
    print(f'Calibrated {len(rows)} settings; timing {len(jobs)} control settings', flush=True)
    start = perf_counter()
    complete = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, probes in pool.map(timed_cases, jobs):
            complete[index] = probes
            write_json(OUTPUT / 'trajectory_preflight.json', {
                'settings_done': len(complete), 'settings_total': len(jobs),
                'wall_s': perf_counter()-start, 'probes': complete,
            })
            print(f'Preflight {len(complete)}/{len(jobs)}: setting {index}', flush=True)


if __name__ == '__main__':
    main()
