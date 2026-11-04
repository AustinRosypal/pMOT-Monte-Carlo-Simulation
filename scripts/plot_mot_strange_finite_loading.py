"""Render saved finite-horizon spectra and direction-cluster loading intervals."""
from pathlib import Path
import argparse
import datetime
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def render(root, mode, estimate=None):
    folder = root/mode
    estimate_path = estimate or folder/'latest_estimate.json'
    row = json.loads(estimate_path.read_text())
    assert row['mode'] == mode
    retained = row.get('final_retention_window_s') == .025
    core_label = ('Central: final 25 ms inside 2 mm at 2 s' if retained
                  else 'Central-core capture by 2 s')
    statistics_path = estimate_path.with_suffix('.npz') if estimate else folder/'disc_statistics.npz'
    with np.load(statistics_path) as saved:
        v = saved['velocity_m_per_s']
        core = saved['disc_cross_section_m2']
        off = saved['disc_off_core_retention_cross_section_m2']
        areas = saved['disc_areas_m2']
    n = len(core)
    assert n == row['complete_direction_discs'] and n > 1
    factor = t.ppf(.975, n-1)/np.sqrt(n)
    figure, grid = plt.subplots(2, 2, figsize=(12.5, 9), layout='constrained')
    axes = grid.ravel()
    for values, label, color in ((core, core_label, 'tab:blue'),
                                 (off, 'Off-core endpoint retention (not validated trapping)', 'tab:orange')):
        mean, half = values.mean(axis=0), factor*values.std(axis=0, ddof=1)
        low, high = np.maximum(0., mean-half), np.minimum(np.pi*.015**2, mean+half)
        empty = np.all(values == 0, axis=0)
        boundary = np.full(len(v), np.nan)
        boundary[empty] = np.pi*.015**2*(1-.025**(1/n))
        axes[0].plot(v, mean*1e6, label=label, color=color)
        axes[0].fill_between(v, low*1e6, high*1e6, where=~empty, color=color, alpha=.16)
        axes[1].plot(v, boundary*1e6, ':', color=color, label=label)
    axes[0].set(xlabel='Incident speed (m/s)', ylabel='Direction-averaged cross section (mm²)',
                title='Pointwise 95% disc-cluster intervals', xlim=(0, 80))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Incident speed (m/s)', ylabel='Conservative 95% upper bound (mm²)',
                title='Finite-sample bounds at all-empty speeds', xlim=(0, 80), ylim=(0, None))
    axes[1].legend(fontsize=8)
    history = [json.loads(p.read_text()) for p in sorted((folder/'estimates').glob('discs_*.json'))]
    history = [item for item in history if item['complete_direction_discs'] < n] + [row]
    count = [item['complete_direction_discs'] for item in history]
    rate = np.asarray([item['loading_rate_atoms_per_s'] for item in history])
    half = np.asarray([item['loading_95_half_width_atoms_per_s'] for item in history])
    lower = np.maximum(0., rate-half)
    axes[2].errorbar(count, rate/1e6, yerr=np.vstack((rate-lower, half))/1e6, fmt='.-', capsize=3)
    axes[2].set(xlabel='Complete independent direction discs', ylabel='Central rate (million atoms/s)',
                title='Descriptive interim estimates; no optional stopping')
    axes[3].hist(areas*1e6, bins=min(30, max(5, n//4)), color='tab:green', alpha=.8)
    axes[3].set(xlabel='Clipped disc/cell intersection area (mm²)', ylabel='Direction-disc count',
                title='15 mm radius; inside-cell launches')
    thermal = row['thermal_conditions']
    criterion = ('Core rule plus continuous final 25 ms central retention at 2 s'
                 if retained else 'Core rule met by 2 s; final retention not established')
    figure.suptitle(f'{mode.capitalize()}: {n} discs × {row["points_per_disc"]} points; '
                   'surrogate-field conventional MOT\n'
                   f'{criterion}\n'
                   f'T = {thermal["temperature_k"]:.8f} K; n₈₇ = {thermal["rb87_density_m3"]/1e12:.8f} × 10¹² m⁻³; '
                   f'velocity quadrature gate: {"PASS" if row["quadrature_pass"] else "FAIL"}\n'
                   'Aperture and final acceptance gates remain separate', fontsize=11)
    out = root/'figures'; out.mkdir(exist_ok=True)
    target = out/f'{mode}_finite_loading.png'
    figure.savefig(target, dpi=180)
    plt.close(figure)
    record = dict(rendered_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        figure=str(target), complete_direction_discs=n, source_estimate=row,
        thermal_conditions=thermal, visual_review_pending=True,
        acceptance_criterion=criterion,
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        estimate_file=str(estimate_path.resolve()), statistics_file=str(statistics_path.resolve()),
        interval='pointwise Student-t direction-cluster; all-empty points use conservative cluster bound')
    (out/f'{mode}_plot_record.json').write_text(json.dumps(record, indent=2))
    print(target.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--mode', choices=('pilot', 'production'), default='pilot')
    parser.add_argument('--estimate', type=Path, help='Separate monitor JSON with matching NPZ, if desired')
    arguments = parser.parse_args()
    render(arguments.root, arguments.mode, arguments.estimate)
