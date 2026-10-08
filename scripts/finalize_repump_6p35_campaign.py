"""Audit saved results and document the completed two-case and prior-sweep review.

No simulation inputs, trajectories, source manifests or previous plots are changed.
Visual acceptance here records the four PNGs inspected by the agent on 2026-10-02.
"""
import run_repump_6p35_campaign as campaign
import csv
import datetime
import json
from pathlib import Path
import numpy as np

base = campaign.base
read, sha = campaign.read, campaign.sha


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    base.rc.CAMPAIGN = campaign.CAMPAIGN
    root = base.rc.paths()['statistics']
    figures = base.rc.paths()['figures']
    final = root/'final'
    final.mkdir(exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    complete = read(root/'completion.json')
    assert complete['computed'] and complete['publication_ready'] and not complete['issues']
    manifest = read(root/'campaign_manifest.json')
    assert base.rc.source_digest() == manifest['source_sha256'] == base.CORE_SHA
    for key, path in [('runner_sha256', campaign.__file__),
                      ('helper_sha256', campaign.previous.__file__),
                      ('baseline_adapter_sha256', base.__file__)]:
        assert sha(path) == manifest[key]
    plan = read(root/'sampling_plan.json')
    assert plan['independent_of_pilot'] and plan['independent_case_seeds']
    assert plan['relative_95_half_width_target'] == .05
    for name in ('geometry_and_force', 'fresh_workers', 'full_fixed_comparison'):
        assert read(root/'validation'/f'{name}.json')['passed']
    evidence, results = [], []
    for point, allocation in zip(campaign.points(), plan['allocations']):
        folder = root/'production'/point.key
        row = read(folder/'summary.json')
        assert row['publication_status'] == 'validated'
        assert row['all_positive_nodes_dual_step_converged'] and row['retention_pass']
        assert row['sample_count'] == allocation['rays'] == allocation['disc_count']*16
        assert read(folder/'configuration.json')['seed'] == allocation['production_seed']
        ev = campaign.audit_aggregate(folder, row)
        with np.load(folder/'disc_statistics.npz') as data:
            v = data['velocity_m_per_s']
            sigma = data['disc_cross_section_m2']
            assert sigma.min() >= 0 and sigma.max() <= np.pi*.015**2*(1+1e-12)
            weighted = sigma*v**3*np.exp(-v**2/base.ce.THERMAL_SCALE_M2_PER_S2)
            coarse = base.ce.LOADING_RATE_PREFACTOR*np.trapezoid(
                np.column_stack((np.zeros(len(sigma)), weighted[:,1::2])), np.r_[0.,v[1::2]], axis=1)
            delta = abs(float(coarse.mean())-row['loading_rate_atoms_per_s'])
            np.testing.assert_allclose(delta, row['velocity_quadrature_difference_atoms_per_s'], atol=1e-6)
        assert row['loading_relative_half_width'] <= .05
        assert delta/row['loading_rate_atoms_per_s'] <= .02
        row.update(read(folder/'timing.json'))
        row['production_and_retention_hours'] = (row['capture_wall_s']+row['retention_wall_s'])/3600
        results.append(row)
        evidence.append(ev)
    visual = dict(passed=True, reviewed_at_utc=now,
        checks=['All four PNGs visually inspected; axes, legends and units legible',
                'Matched curve has nine points including 6.35 mm',
                '12.7 cooling / 6.35 repump is a separately labelled green marker',
                '95% bars present; small bars can be hidden by marker size',
                'Prior campaign plots preserved'],
        figures={p.name:sha(p) for p in sorted(figures.glob('*.png'))})
    assert len(visual['figures']) == 4
    base.rc.write_json(final/'visual_review.json', visual)

    # Review authoritative selected data, not superseded failed-precision originals.
    old = root.parent/'relationships_20260919'
    assembly = read(old/'final/assembly_manifest.json')
    assert assembly['all_seven_plot_types_ready'] and not assembly['pending_points']
    selected = rows(old/'final/validated_loading_points.csv')
    assert len(selected) == 75
    best = []
    for study in ('02_raw_saturation', '03_effective_saturation', '04_detuning'):
        best.append(max((r for r in selected if r['study']==study), key=lambda r:float(r['loading_rate_atoms_per_s'])))
    baseline = next(r for r in selected if r['study']=='01_radius' and float(r['coordinate'])==15.)
    review = []
    for r in [baseline]+best:
        key = f"{r['study']}/{int(r['index']):03d}"
        record = next(p for p in assembly['point_provenance'] if p['key']==key)
        source = root.parent/r['source_campaign']/'production'/key
        for filename, hashkey in [('summary.json','summary_sha256'),
                                  ('disc_statistics.npz','disc_statistics_sha256'),
                                  ('duration_audit.json','retention_audit_sha256')]:
            assert sha(source/filename) == record[hashkey]
        cfg = read(source/'configuration.json')['point']
        assert cfg['radius_mm'] == 15.
        review.append(dict(r, cooling_power_mw=cfg['cooling_power_w']*1000,
            cooling_detuning_mhz=cfg['detuning_rad_s']/(2*np.pi*1e6),
            repump_power_mw=.1, repump_detuning_mhz=0.,
            configuration_sha256=sha(source/'configuration.json')))
    force = rows(old/'final/force_vs_detuning.csv')
    slope_best = min(force, key=lambda r:float(r['slope_x_n_per_m']))
    f375 = min(force, key=lambda r:abs(float(r['detuning_over_gamma'])+3.75))
    f250 = min(force, key=lambda r:abs(float(r['detuning_over_gamma'])+2.5))
    source_files = [old/'final'/name for name in ('REPORT.md','assembly_manifest.json',
        'validated_loading_points.csv','force_vs_detuning.csv')]
    base.rc.write_json(final/'operating_point_evidence.json', dict(
        reviewed_at_utc=now, selected_points=review,
        source_files={str(p):sha(p) for p in source_files},
        repump_swept=False, joint_optimization_performed=False,
        selection='Largest observed means on separate one-dimensional slices; pointwise intervals are not selection-adjusted',
        strongest_local_x_restoring= slope_best,
        force_near_baseline=f250, force_at_best_loading=f375))
    b, raw, eff, det = review
    baseline_rate=float(b['loading_rate_atoms_per_s'])
    gain=100*(float(det['loading_rate_atoms_per_s'])/baseline_rate-1)
    op = ['# Operating-point review: both beam diameters 12.7 mm', '',
        'The best measured loading setting in the prior corrected relationship campaign is '
        '**27 mW per cooling component, -22.7625 MHz cooling detuning, 0.1 mW per repump '
        'component, and zero repump detuning**. This is the best tested mean, not a joint four-parameter optimum.', '',
        '| Setting | Cooling power/component (mW) | Cooling detuning (MHz) | Loading (million atoms/s, pointwise 95%) | Discs x points |',
        '|---|---:|---:|---:|---:|']
    for label,r in zip(['Traditional baseline, radius study at 15 mm','Raw-saturation maximum, s0=90',
                         'Effective-saturation maximum, s_eff=3.25','Detuning maximum, -3.75 linewidths'],review):
        op.append(f"| {label} | {r['cooling_power_mw']:.6g} | {r['cooling_detuning_mhz']:.6g} | "
                  f"{float(r['loading_rate_atoms_per_s'])/1e6:.4f} +/- {float(r['loading_95_half_width_atoms_per_s'])/1e6:.4f} | {r['disc_count']} x {r['points_per_disc']} |")
    op += ['', f'The detuning change raises the mean by {gain:.2f}% relative to the exact -15 MHz baseline '
        'from the same historical relationship campaign. That baseline is 61.6116 +/- 1.3398 million atoms/s. '
        'The later independent diameter campaign gave 59.9099 +/- 2.2806 million atoms/s at the same parameters; '
        'the estimates agree within their statistical uncertainty. Against that later estimate the gain is 53.50%. '
        'These percentages are comparisons of means, not confidence intervals on the gain.', '',
        'At fixed -15 MHz, increasing cooling power gives diminishing returns: both power coordinates '
        'flatten near 72 million atoms/s, with their largest means near 87-95 mW/component. '
        'Raw and effective saturation at fixed detuning are two coordinates for cooling intensity, '
        'not two independent physical controls. Those high-power results do not establish that '
        'combining 95 mW with -22.7625 MHz improves loading: that combination was never simulated.', '',
        'The adjacent -3.5-linewidth point (-21.245 MHz) gives 90.4652 +/- 6.3237 million atoms/s. '
        'The broad peak and pointwise intervals do not establish a unique optimal detuning; selection '
        'of the largest measured mean also is not accounted for in its displayed pointwise interval. '
        'The old loading target was 10% half-width, unlike the 5% target of the new diameter points.', '',
        f"The force diagnostics have strongest local restoring slope near {float(slope_best['detuning_over_gamma']):.2f} linewidths, "
        'rather than at the loading maximum. The damping-force minimum moves to higher velocity with red detuning '
        f"(x-axis {float(f250['turnaround_x_m_per_s']):.3f} m/s at -2.5 linewidths versus {float(f375['turnaround_x_m_per_s']):.3f} m/s at -3.75). "
        'This is consistent with improving capture of faster incident atoms, but local force extrema are not '
        'capture thresholds and do not themselves prove the cause of the integrated loading maximum.', '',
        'Repump power and detuning were held at 0.1 mW/component and zero throughout these relationship sweeps. '
        'They are therefore the supported reference settings, not measured repump optima. The diameter sweeps '
        'change the spatial envelope and sometimes total power at fixed peak intensity; they do not substitute '
        'for a repump-power or repump-detuning scan at fixed 12.7 mm. A joint optimum and its loading rate remain unmeasured.', '',
        'All comparisons use full-sphere uniform-area 15 mm sampling discs, the same historical vapor normalization, '
        'and corrected 24-state local-steady-state population-rate mean force. Numerical validation does not '
        'validate instantaneous internal-population equilibration in weak repump wings. Finite-rate population '
        'dynamics, vapor-density uncertainty and apparatus/model discrepancy are not included in these error bars.', '']
    (final/'OPERATING_POINT_REVIEW.md').write_text('\n'.join(op), encoding='utf-8')

    elapsed = (datetime.datetime.fromisoformat(complete['production_finished_at_utc'])-
               datetime.datetime.fromisoformat(read(root/'process.json')['started_at_utc'])).total_seconds()/3600
    report = ['# Completed 6.35 mm repump loading cases', '',
        'Both independent cases passed the prescribed statistical and numerical gates. All four updated PNGs were visually inspected.', '',
        '| Cooling / repump diameter (mm) | Cooling / repump power per component (mW) | Loading (million atoms/s, 95%) | Discs x points | Production + retention (hours) |',
        '|---|---|---:|---:|---:|']
    for r in results:
        report.append(f"| {r['cooling_diameter_mm']} / 6.35 | {r['cooling_power_w_per_beam']*1000:g} / 0.025 | "
            f"{r['loading_rate_atoms_per_s']/1e6:.4f} +/- {r['loading_95_half_width_atoms_per_s']/1e6:.4f} | "
            f"{r['disc_count']} x 16 | {r['production_and_retention_hours']:.3f} |")
    report += ['', f'Total simulation elapsed time including startup, validation and pilots: {elapsed:.3f} hours. '
        'Production used 40,704 rays from 2,544 independent direction discs. Independent pilot data only determined '
        'the frozen production allocations and are not pooled into these estimates.', '',
        'The respective pointwise 95% Student-t disc-cluster half-widths are 4.426% and 4.017%, both below 5%. '
        'Density-independent loading coefficients R/n87 are (1.16677 +/- 0.05164) x 10^-6 and '
        '(2.80233 +/- 0.11257) x 10^-6 m^3/s respectively. Equivalent historical density is '
        '5.451556957 x 10^12 Rb-87 atoms/m^3. Absolute rates retain the historical prefactor '
        '9.1196e5 and thermal scale 5.667e4 (m/s)^2 without renormalization.', '',
        'Cooling detuning is -15 MHz; repump detuning is zero. Diameters are Gaussian 1/e^2. '
        'There are six traveling cooling and six traveling repump components along +/-x, +/-y, +/-z. '
        'Both families keep their peak intensities referenced to 27 mW and 0.1 mW respectively at 12.7 mm. '
        'Propagation-frame helicities are sigma+ on x/y and sigma- on z for both directions and both families. '
        'The axial gradient is 10 G/cm; gravity is -9.80665 m/s^2 along z. Disc radius and launch-plane '
        'distance are 15 mm; directions are uniform in solid angle over the full sphere and offsets uniform '
        'in area. All velocities on a disc are parallel to its normal. Full beam, atomic, field, integration '
        'and capture parameters are retained in [parameters.json](../parameters.json).', '',
        'The corrected Section-12 24-state steady-state population solver supplies the deterministic mean force '
        'with recoil diffusion disabled. Capture requires 5 ms continuous residence within the 2 mm core or '
        'two core entries separated by an exit. Every ray retains 328 positive-speed classifications; '
        'nonmonotone masks remain authoritative, and timeouts were never recast as escapes. Zero speed enters '
        'only through the exact zero weighted-flux integrand.', '',
        'All 4,838 representative longer-retention cases passed paired fixed-RK4 checks. The fresh validation '
        'includes 64 local-force comparisons (22 high-precision), 32 worker checks and 1,968 full-grid '
        'fixed-RK4 comparison nodes with no mismatches. All positive nodes pass the required dual-setting '
        'classification gate and velocity-tail checks. Nested 0.5 versus 0.25 m/s loading quadratures differ '
        'by 52.949 and 7,830.830 atoms/s (0.000832% and 0.05126%). These differences are discretization '
        'sensitivity estimates, not rigorous total numerical error bounds; statistical bars exclude them.', '',
        'The matched 6.35 mm point is part of the nine-point matched-diameter curve. The 12.7 mm cooling / '
        '6.35 mm repump case is a separate green marker on the comparison plot. All prior campaign figures '
        'remain unchanged. Very small error bars may be covered by marker symbols.', '']
    for name in visual['figures']:
        report.append(f'- [{name}]({(figures/name).as_posix()})')
    report += ['', 'These are numerical results conditional on instantaneous local steady state. The known weak-wing '
        'population-relaxation limitation remains unresolved: internal populations are not evolved along '
        'trajectories. The error bars omit this model discrepancy, vapor-density uncertainty and experimental '
        'apparatus uncertainty. Temperature remains deferred.', '',
        'The requested subsequent review is complete: [operating-point review](OPERATING_POINT_REVIEW.md). '
        'Its best tested condition is 27 mW cooling at -22.7625 MHz with 0.1 mW resonant repump, both 12.7 mm, '
        'giving (91.9605 +/- 6.8593) million atoms/s. Repump and joint optima were not measured.', '',
        f"Core source SHA-256: `{manifest['source_sha256']}`. Simulation runner and manifest remain unchanged. "
        'The separate finalizer records independent aggregate/quadrature checks and source hashes in '
        '[audit.json](audit.json); inspected figure hashes are in [visual_review.json](visual_review.json).', '']
    (final/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    base.rc.write_json(final/'audit.json', dict(passed=True, reviewed_at_utc=now,
        finalizer_sha256=sha(__file__), source_sha256=manifest['source_sha256'],
        manifest_sha256=sha(root/'campaign_manifest.json'), evidence=evidence,
        total_elapsed_hours=elapsed, results=results,
        operating_review_sha256=sha(final/'OPERATING_POINT_REVIEW.md'),
        report_sha256=sha(final/'REPORT.md')))
    complete.update(visual_review_pending=False, optimal_settings_review_pending=False,
        final_audit_passed=True, finalized_at_utc=now, report=str(final/'REPORT.md'),
        operating_point_review=str(final/'OPERATING_POINT_REVIEW.md'))
    base.rc.write_json(root/'completion.json', complete)
    print(json.dumps(dict(passed=True, report=str(final/'REPORT.md'), elapsed_hours=elapsed,
        baseline_gain_percent=gain, selected_operating_points=review), indent=2))


if __name__ == '__main__':
    main()
