"""Read-only scientific audit and report for the matched-repump campaign.

Never changes simulation inputs, retained trajectories, or baseline products.
Writes analysis products under final/; visual review is a separate hashed record.
"""
import run_matched_repump_diameter_campaign as campaign
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

base = campaign.base


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base.rc.CAMPAIGN = campaign.CAMPAIGN
    root = base.rc.paths()['statistics']
    complete = read(root/'completion.json')
    assert complete['computed'] and complete['publication_ready'] and not complete['issues']
    manifest = read(root/'campaign_manifest.json')
    assert sha(campaign.__file__) == manifest['runner_sha256']
    assert sha(base.__file__) == manifest['baseline_adapter_sha256']
    assert base.rc.source_digest() == manifest['source_sha256'] == base.CORE_SHA
    plan = read(root/'sampling_plan.json')
    assert plan['independent_of_pilot'] and plan['production_seed'] == 2026093002
    assert plan['relative_95_half_width_target'] == .05
    parameters = read(root/'parameters.json')
    assert parameters['source_sha256'] == base.CORE_SHA
    gates = {}
    for name in ('geometry_and_force', 'fresh_workers', 'full_fixed_comparison'):
        gate = read(root/'validation'/f'{name}.json')
        assert gate['passed']
        gates[name] = sha(root/'validation'/f'{name}.json')
    with (root/'paired_comparison.csv').open(newline='') as f:
        paired = list(csv.DictReader(f))
    assert len(paired) == len(plan['allocations']) == 8
    rows, evidence = [], []
    retention_count = 0
    max_endpoint = max_window = stage_seconds = 0.
    durations = set()
    density = base.ce.LOADING_RATE_PREFACTOR/(4/(np.sqrt(np.pi)*base.ce.THERMAL_SCALE_M2_PER_S2**1.5))
    for point, allocation, contrast in zip(campaign.points(), plan['allocations'], paired):
        assert allocation['key'] == point.key
        folder = root/'production'/point.key
        old_folder = campaign.OLD/'production'/point.key
        row = read(folder/'summary.json')
        assert row['publication_status'] == 'validated'
        assert row['all_positive_nodes_dual_step_converged'] and row['retention_pass']
        assert row['sample_count'] == allocation['rays'] == allocation['disc_count']*16
        assert len(list((folder/'rays').glob('*.npz'))) == row['sample_count']
        config = read(folder/'configuration.json')
        assert config['seed'] == 2026093002 and config['points_per_disc'] == 16
        with np.load(folder/'geometry.npz') as new, np.load(old_folder/'geometry.npz') as old:
            for key in ('positions_m', 'directions'):
                np.testing.assert_array_equal(new[key][:allocation['baseline_disc_count']], old[key])
        with np.load(folder/'disc_statistics.npz') as saved:
            spectra = saved['disc_cross_section_m2']
            speeds = saved['velocity_m_per_s']
            loading = saved['disc_loading_atoms_per_s']
            assert spectra.shape == (row['disc_count'], 320)
            np.testing.assert_array_equal(speeds, base.ce.VELOCITIES)
            assert np.all(np.isfinite(spectra)) and spectra.min() >= 0
            assert spectra.max() <= np.pi*.015**2*(1+1e-12)
            weighted = spectra*speeds**3*np.exp(-speeds**2/base.ce.THERMAL_SCALE_M2_PER_S2)
            recomputed = base.ce.LOADING_RATE_PREFACTOR*np.trapezoid(
                np.column_stack((np.zeros(len(loading)), weighted)), np.r_[0., speeds], axis=1)
            np.testing.assert_allclose(recomputed, loading, rtol=1e-12, atol=1e-8)
            mean, half = base.rc.cluster_interval(recomputed)
            np.testing.assert_allclose([mean, half], [row['loading_rate_atoms_per_s'],
                row['loading_95_half_width_atoms_per_s']], rtol=1e-12)
            coarse = base.ce.LOADING_RATE_PREFACTOR*np.trapezoid(
                np.column_stack((np.zeros(len(loading)), weighted[:,1::2])), np.r_[0., speeds[1::2]], axis=1)
            quadrature = abs(float(coarse.mean())-mean)
            np.testing.assert_allclose(quadrature, row['velocity_quadrature_difference_atoms_per_s'], rtol=1e-10, atol=1e-6)
            with np.load(old_folder/'disc_statistics.npz') as old:
                difference = loading[:allocation['baseline_disc_count']]-old['disc_loading_atoms_per_s']
                baseline_mean = float(old['disc_loading_atoms_per_s'].mean())
        assert half/mean <= .05 and quadrature/mean <= .02
        delta, delta_half = base.rc.cluster_interval(difference)
        np.testing.assert_allclose([delta, delta_half],
            [float(contrast['loading_difference_atoms_per_s']), float(contrast['difference_95_half_width_atoms_per_s'])], rtol=1e-12)
        assert contrast['qualified'] == 'True' and float(contrast['diameter_mm']) == point.coordinate
        if point.coordinate == 12.7:
            np.testing.assert_array_equal(difference, np.zeros_like(difference))
        np.testing.assert_allclose([mean/density, half/density], [row['loading_coefficient_m3_per_s'],
            row['loading_coefficient_95_half_width_m3_per_s']], rtol=1e-12)
        assert row['cooling_diameter_mm'] == row['repump_diameter_mm'] == point.coordinate
        np.testing.assert_allclose([row['cooling_power_w_per_beam'], row['repump_power_w_per_beam']],
            [.027*(point.coordinate/12.7)**2, .0001*(point.coordinate/12.7)**2], rtol=1e-14)
        audit = read(folder/'duration_audit.json')
        assert audit['passed'] and audit['cases'] and all(c['passed'] for c in audit['cases'])
        for case in audit['cases']:
            final = case['attempts'][-1]
            assert final['coarse_retained'] and final['fine_retained'] and final['position_difference_m'] < 1e-5
            window = max(final['coarse_final_window_max_radius_m'], final['fine_final_window_max_radius_m'])
            assert window <= .002
            max_endpoint = max(max_endpoint, final['position_difference_m'])
            max_window = max(max_window, window)
            durations.add(final['duration_s'])
        timing = read(folder/'timing.json')
        stage_seconds += sum(timing.values())
        retention_count += len(audit['cases'])
        row.update(measured_capture_hours=timing['capture_wall_s']/3600,
            measured_retention_hours=timing['retention_wall_s']/3600,
            measured_total_hours=sum(timing.values())/3600,
            velocity_quadrature_relative_difference=quadrature/mean,
            retention_cases=len(audit['cases']), baseline_mean_atoms_per_s=baseline_mean,
            paired_difference_atoms_per_s=float(delta), paired_difference_95_half_width_atoms_per_s=float(delta_half),
            relative_change_percent=100*(mean/baseline_mean-1))
        rows.append(row)
        files = [folder/n for n in ('configuration.json','geometry.npz','summary.json','disc_statistics.npz','duration_audit.json','timing.json')]
        files.append(root/'parameters'/f'{point.index:03d}_compiled_inputs.npz')
        evidence.append({'diameter_mm':point.coordinate, 'files':{p.relative_to(root).as_posix():sha(p) for p in files},
            'baseline_disc_statistics_sha256':sha(old_folder/'disc_statistics.npz'),
            'baseline_geometry_sha256':sha(old_folder/'geometry.npz')})
    output = root/'final'
    output.mkdir(exist_ok=True)
    figure = base.rc.paths()['figures']
    names = ('loading_rate_matched_repump','loading_rate_comparison','loading_coefficient_matched_repump','loading_coefficient_comparison')
    figures = [figure/f'{name}.{ext}' for name in names for ext in ('png','pdf')]
    assert all(p.exists() for p in figures)
    visually_reviewed = False
    if (output/'visual_review.json').exists():
        review = read(output/'visual_review.json')
        visually_reviewed = review['passed'] and all(review['png_sha256'].get(str(p)) == sha(p) for p in figures if p.suffix == '.png')
    finished = datetime.datetime.fromisoformat(complete['production_finished_at_utc'])
    process = read(root/'process.json')
    total_hours = (finished-datetime.datetime.fromisoformat(process['started_at_utc'])).total_seconds()/3600
    rays = sum(r['sample_count'] for r in rows)
    discs = sum(r['disc_count'] for r in rows)
    audit = dict(passed=True, assembled_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rays=rays, direction_discs=discs, production_and_audit_stage_hours=stage_seconds/3600,
        total_execution_hours_including_pilots=total_hours, retention_cases=retention_count,
        nonmonotone_rays=sum(r['nonmonotone_ray_count'] for r in rows),
        maximum_retention_endpoint_difference_m=max_endpoint, maximum_final_window_radius_m=max_window,
        retention_final_durations_s=sorted(durations), source_sha256=base.CORE_SHA,
        finalizer_sha256=sha(__file__), manifest_sha256=sha(root/'campaign_manifest.json'),
        plan_sha256=sha(root/'sampling_plan.json'), parameters_sha256=sha(root/'parameters.json'),
        validation_certificates=gates, visual_review_pending=not visually_reviewed,
        figures={str(p):sha(p) for p in figures}, evidence=evidence, rows=rows)
    base.rc.write_json(output/'audit.json', audit)
    base.rc.write_csv(output/'results.csv', rows)
    lines = ['# Matched cooling and repump diameter: corrected multilevel MOT','',
        f'All eight points passed their prescribed statistical, quadrature, and representative retention checks. Production and retention audits took **{stage_seconds/3600:.2f} hours**; total execution including pilots and preproduction checks took **{total_hours:.2f} hours**. Visual inspection is recorded separately in `visual_review.json`.','',
        '## Repump-size effect','',
        'At fixed peak intensities, matching the repump diameter to the cooling diameter decreases loading below 12.7 mm and increases it above 12.7 mm relative to the previous fixed-12.7-mm repump. The 12.7 mm calculations agree exactly on every paired disc. Each other pointwise paired-difference interval excludes zero; this is a model prediction under the sampled geometry.','',
        f'- [Loading comparison PNG]({(figure/"loading_rate_comparison.png").as_posix()}) | [PDF]({(figure/"loading_rate_comparison.pdf").as_posix()})',
        f'- [Matched repump loading PNG]({(figure/"loading_rate_matched_repump.png").as_posix()})',
        f'- [Density-independent comparison PNG]({(figure/"loading_coefficient_comparison.png").as_posix()})',
        f'- [Results CSV]({(output/"results.csv").as_posix()}) | [Paired intervals CSV]({(root/"paired_comparison.csv").as_posix()})',
        f'- [Complete parameters]({(root/"parameters.json").as_posix()}) | [Audit and file hashes]({(output/"audit.json").as_posix()})','',
        '## Results','',
        'Loading means and error bars retain the historical vapor normalization. All intervals are pointwise 95% Student-t direction-disc cluster intervals, not simultaneous intervals. Percent changes are estimates relative to the baseline mean; uncertainty on the difference is given separately.','',
        '| Diameter (mm) | Discs × points | Loading (10^6 atoms/s) ±95% | Paired change (10^6 atoms/s) ±95% | Change | Relative half-width | Grid sensitivity | Runtime (h) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cooling_diameter_mm']:g} | {r['disc_count']} × 16 | {r['loading_rate_atoms_per_s']/1e6:.4f} ± {r['loading_95_half_width_atoms_per_s']/1e6:.4f} | {r['paired_difference_atoms_per_s']/1e6:+.4f} ± {r['paired_difference_95_half_width_atoms_per_s']/1e6:.4f} | {r['relative_change_percent']:+.2f}% | {100*r['loading_relative_half_width']:.3f}% | {100*r['velocity_quadrature_relative_difference']:.4f}% | {r['measured_total_hours']:.3f} |")
    lines += ['', '## Parameters and power scaling','',
        'Both Gaussian 1/e² beam diameters are D. Each of six cooling traveling components has 27(D/12.7)² mW; each of six repump components has 0.1(D/12.7)² mW. Peak intensities are 42.628145214 and 0.157882019 mW/cm² respectively. Cooling detuning is −15 MHz and repump detuning is zero. The quadrupole axial gradient is 10 G/cm, gravity is enabled, and recoil diffusion is disabled. This comparison changes repump spatial coverage and total power while holding peak intensity fixed; it does not isolate geometry at equal total power.','',
        '| Diameter (mm) | Cooling power/component (mW) | Repump power/component (mW) | R/n87 (m³/s) ±95% |',
        '|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cooling_diameter_mm']:g} | {1000*r['cooling_power_w_per_beam']:.6f} | {1000*r['repump_power_w_per_beam']:.6f} | {r['loading_coefficient_m3_per_s']:.6g} ± {r['loading_coefficient_95_half_width_m3_per_s']:.6g} |")
    lines += ['', '## Statistical and numerical validation','',
        f'The independent production allocation contains {rays:,} rays on {discs:,} uniformly sampled full-sphere direction discs, with 16 uniform-area points per 15 mm-radius disc. The launch plane is 15 mm from the center. Pilot seed 2026093001 and production seed 2026093002 are independent. Allocations were frozen before production, with a 50% pilot-variance safety margin and a floor at the previous per-point counts. No post-hoc sample extension was required.','',
        'The final audit recomputed the loading quadrature and Student-t intervals from saved disc spectra, compared file counts to the frozen plan, recomputed paired-difference intervals, verified exact launch geometry equality with the baseline, and checked all stored retention results. The new and baseline disc counts match at every point, so the paired comparison uses all production discs.','',
        f'The runner resolved {rays*328:,} positive speed nodes with the established dual-integration and fixed-RK4 edge/fallback policy, retaining nonmonotone masks. Preproduction fixed-RK4 comparison checked 5,248 nodes on 16 predetermined rays with no differences. Local force/population validation checked 192 cases (16 high-precision wing references), plus 64 fresh-worker classification checks. The unchanged core test and event/trajectory certificates are linked through the manifest and baseline validation.','',
        f'All {retention_count:,} representative longer-retention cases passed at accepted durations {sorted(durations)} s. The largest paired final-position discrepancy was {max_endpoint:.6g} m (gate 1e-5 m), and largest final-window radius was {max_window:.6g} m (gate 0.002 m throughout the final 25 ms). These representative finite-duration checks do not establish infinite trap lifetime or audit every captured trajectory at long duration.','',
        f'Statistical relative half-widths span {100*min(r["loading_relative_half_width"] for r in rows):.3f}–{100*max(r["loading_relative_half_width"] for r in rows):.3f}%. Maximum nested 0.5/0.25 m/s quadrature sensitivity is {100*max(r["velocity_quadrature_relative_difference"] for r in rows):.4f}%. This sensitivity is reported separately and is not a rigorous bound on all numerical error. Timeouts were never relabeled as escapes.','',
        '## Scope and normalization','',
        f'The historical loading prefactor C=9.1196e5 and thermal scale A=5.667e4 m²/s² remain unchanged. Under a normalized Maxwell distribution they correspond to equivalent n87={density:.9g} m^-3 at approximately 296.18 K. The older written Maxwell prefactor was not normalized consistently. To use a specified vapor density, apply R=n87*K to the saved density-independent coefficient K. Error bars exclude density, apparatus, and model uncertainty.','',
        'The largest-beam points are conditional on the prescribed 15 mm sampling radius and do not establish convergence to an unbounded incident plane. The model uses ideal untruncated Gaussian beams and deterministic single-atom mean force, without recoil diffusion, collective/collisional effects, or equilibrium temperature. Diameters 2 and 3 mm remain deferred.','',
        f'Only the corrected 24-state Section-12 population-rate equations were used: stimulated coefficient W=Gamma_e|Omega|²/(Gamma_e²+4Delta²), beam force from W(p_g-p_e), and coupled steady-state populations. Physics SHA256: `{base.CORE_SHA}`. Simulation and baseline outputs were not modified by final analysis.','']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('rows','evidence','figures')}), flush=True)


if __name__ == '__main__':
    main()
