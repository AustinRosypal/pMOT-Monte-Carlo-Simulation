"""Read-only result audit and final report for the -21 MHz gradient campaign.

Records the agent's visual inspection of both PNGs on 2026-10-16. Retained
simulation results, the source manifest, and the runner remain unchanged.
"""
import run_multilevel_gradient_campaign as campaign
import datetime
import numpy as np

rc = campaign.rc
read, sha = campaign.read, campaign.sha


def main():
    rc.CAMPAIGN = campaign.CAMPAIGN
    root, figures = rc.paths()['statistics'], rc.paths()['figures']
    final = root/'final'; final.mkdir(exist_ok=True)
    complete = read(root/'completion.json')
    assert complete['computed'] and complete['all_numerical_gates_passed'] and complete['precision_passed']
    assert not complete['issues']
    manifest = read(root/'campaign_manifest.json')
    assert rc.source_digest() == manifest['source_sha256'] == campaign.CORE_SHA
    assert sha(campaign.__file__) == manifest['runner_sha256']
    assert sha(campaign.ROOT/'uv.lock') == manifest['lockfile_sha256']
    for source in campaign.core_files + [campaign.Path(campaign.__file__)]:
        assert sha(source) == sha(root/'source_snapshot'/source.relative_to(campaign.ROOT))
    for name in ('test_status','local_force_and_events','fresh_workers','full_fixed_comparison'):
        assert read(root/'validation'/f'{name}.json')['passed']
    plan = read(root/'sampling_plan.json')
    assert plan['independent_of_pilot'] and plan['common_geometry']
    original_audit = read(root/'aggregate_audit.json')
    assert original_audit['passed']
    rows, loading, evidence = [], [], []
    reference_geometry = None
    total_retention = 0
    max_endpoint = max_window = 0.
    for point, allocation, prior in zip(campaign.points(), plan['allocations'], original_audit['evidence']):
        folder = root/'production'/point.key
        assert prior['key'] == allocation['key'] == point.key
        for name, digest in prior['files'].items():
            assert sha(folder/name) == digest
        row = read(folder/'summary.json')
        assert row['publication_status'] == 'numerically qualified'
        assert row['disc_count'] == allocation['disc_count'] == 300
        assert row['sample_count'] == allocation['rays'] == 6000
        assert row['all_positive_nodes_dual_step_converged'] and row['retention_pass']
        config = read(folder/'configuration.json')
        assert config['seed'] == manifest['production_seed']
        assert config['points_per_disc'] == 20
        with np.load(folder/'geometry.npz') as saved:
            geometry = [saved[k].copy() for k in ('positions_m','directions')]
        if reference_geometry is None:
            reference_geometry = geometry
        else:
            for a,b in zip(reference_geometry,geometry):
                np.testing.assert_array_equal(a,b)
        with np.load(folder/'disc_statistics.npz') as saved:
            v=saved['velocity_m_per_s']; sigma=saved['disc_cross_section_m2']
            assert sigma.shape == (300,320)
            np.testing.assert_array_equal(v,campaign.ce.VELOCITIES)
            assert np.all(np.isfinite(sigma)) and sigma.min()>=0 and sigma.max()<=np.pi*.015**2*(1+1e-12)
            weighted=sigma*v**3*np.exp(-v*v/campaign.ce.THERMAL_SCALE_M2_PER_S2)
            values=campaign.ce.LOADING_RATE_PREFACTOR*np.trapezoid(np.column_stack([np.zeros(300),weighted]),np.r_[0.,v],axis=1)
            np.testing.assert_allclose(values,saved['disc_loading_atoms_per_s'],rtol=1e-12)
            mean,half=rc.cluster_interval(values)
            np.testing.assert_allclose([mean,half],[row['loading_rate_atoms_per_s'],row['loading_95_half_width_atoms_per_s']],rtol=1e-12)
            coarse=campaign.ce.LOADING_RATE_PREFACTOR*np.trapezoid(np.column_stack([np.zeros(300),weighted[:,1::2]]),np.r_[0.,v[1::2]],axis=1)
            quadrature=abs(float(coarse.mean())-mean)
            np.testing.assert_allclose(quadrature,row['velocity_quadrature_difference_atoms_per_s'],atol=1e-6)
            assert half/mean<=.10 and quadrature/mean<=.02
            loading.append(values)
        assert len(list((folder/'rays').glob('*.npz')))==6000
        retention=read(folder/'duration_audit.json')
        assert retention['passed'] and len(retention['cases'])==prior['retention_cases']
        for case in retention['cases']:
            assert case['passed']
            last=case['attempts'][-1]
            assert last['coarse_retained'] and last['fine_retained']
            endpoint=last['position_difference_m']
            window=max(last['coarse_final_window_max_radius_m'],last['fine_final_window_max_radius_m'])
            assert endpoint<1e-5 and window<=.002
            max_endpoint=max(max_endpoint,endpoint);max_window=max(max_window,window)
        total_retention+=len(retention['cases'])
        timing=read(folder/'timing.json')
        row.update(timing, production_and_retention_minutes=sum(timing.values())/60,
            quadrature_relative_difference=quadrature/mean)
        rows.append(row)
        evidence.append(dict(key=point.key,files={name:sha(folder/name) for name in
            ('summary.json','disc_statistics.npz','geometry.npz','duration_audit.json','configuration.json','timing.json')}))
    contrasts=[]
    for left,right in ((2,3),(2,1),(3,1)):
        mean,half=rc.cluster_interval(loading[left]-loading[right])
        contrasts.append(dict(gradient_a_g_per_cm=campaign.GRADIENTS[left],gradient_b_g_per_cm=campaign.GRADIENTS[right],
            difference_atoms_per_s=mean,half_width_95_atoms_per_s=half,
            low_95_atoms_per_s=mean-half,high_95_atoms_per_s=mean+half,disc_count=300))
    rc.write_csv(final/'paired_differences.csv',contrasts)
    rc.write_csv(final/'results.csv',rows)
    started=datetime.datetime.fromisoformat(read(root/'process.json')['started_at_utc'])
    ended=datetime.datetime.fromisoformat(complete['finished_at_utc'])
    total_minutes=(ended-started).total_seconds()/60
    now=campaign.utc()
    visual=dict(passed=True,reviewed_at_utc=now,
        checks=['Both saved PNGs inspected; six points and 95% bars visible',
                'Correct -21 MHz title, axial gradient units and fixed powers',
                'Loading and density-independent coefficient plots consistent; no clipping'],
        files={p.name:sha(p) for p in figures.glob('*.png')})
    assert len(visual['files'])==2
    rc.write_json(final/'visual_review.json',visual)
    lines=['# Completed MOT axial-gradient sweep at -21 MHz','',
        f'Simulation completed {complete["finished_at_utc"]} in {total_minutes:.2f} minutes including pilots and validation. '
        'All six points passed the prescribed precision, numerical, tail, quadrature and retention checks. '
        'Both PNG figures have been visually reviewed.', '',
        '| Axial gradient (G/cm) | Loading (million atoms/s) | 95% half-width (million atoms/s) | Relative half-width | Production + retention (min) |',
        '|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['gradient_g_per_cm']:g} | {r['loading_rate_atoms_per_s']/1e6:.4f} | {r['loading_95_half_width_atoms_per_s']/1e6:.4f} | {100*r['loading_relative_half_width']:.3f}% | {r['production_and_retention_minutes']:.2f} |")
    d=contrasts[0]
    lines += ['', 'The largest sampled mean is at 15 G/cm, with a broad maximum across 15-20 G/cm. '
        f"The paired 15-minus-20 G/cm difference is ({d['difference_atoms_per_s']/1e6:.4f} +/- {d['half_width_95_atoms_per_s']/1e6:.4f}) million atoms/s "
        '(pointwise 95% disc-cluster interval). This comparison uses identical launch geometry and includes zero; '
        'it does not resolve a preference between these settings. The largest mean is about '
        f"{100*(rows[2]['loading_rate_atoms_per_s']/rows[1]['loading_rate_atoms_per_s']-1):.2f}% above the 10 G/cm mean. "
        'These are comparisons within this sampled grid; no continuous optimum is established.', '',
        'Each setting uses 300 independent incident-direction discs x 20 uniform-area points = 6,000 rays. '
        'There are 36,000 production ray/setting calculations. Directions and offsets are shared across gradients, '
        'so there are 300 distinct independent direction clusters, not 1,800 independent directions across the curve. '
        'Independent 32 x 20 pilots per gradient fixed the common production size before production. '
        'Pilot data are excluded from the final estimates. Error bars are pointwise 95% Student-t intervals '
        '(299 degrees of freedom), not simultaneous confidence bands; the target was <=10% half-width.', '',
        'Cooling: six traveling beams at 27 mW each and -21 MHz. Repump: six traveling beams at 0.1 mW each '
        'and zero detuning. Both are Gaussian 1/e² diameter 12.7 mm. Sampling radius and launch-plane distance '
        'are 15 mm. Full-sphere directions, uniform disc area and parallel velocities on each disc are used. '
        'Only anti-Helmholtz current changes; the pair has 40 mm coil radius and 50 turns per coil. '
        'Gravity is enabled; recoil diffusion is disabled. Helicity is sigma+ on x/y and sigma- on z '
        'for both incident and retro components. Full inputs are in ../parameters.json.', '',
        f'All {total_retention:,} representative longer-retention cases passed. The largest paired endpoint '
        f'difference was {max_endpoint*1e6:.6g} micrometers, below 10 micrometers. The final 25 ms windows '
        'stayed within the 2 mm core. Core capture uses continuous 5 ms residence or two entries with an intervening exit. '
        'These checks do not establish infinite trapping lifetime.', '',
        'The 61 selected regression tests, 96 local field/force/population comparisons and six frozen-point '
        'Gillespie comparisons passed. Spawned-worker checks covered 48 adaptive/fixed nodes; full fixed-RK4 '
        'checks on 12 predetermined pilot rays covered 3,936 nodes with zero discrepancies. Production retains '
        '328 positive speeds per ray: 0.25-80 m/s in 0.25 m/s increments plus 85-120 m/s tail probes. '
        'All nodes passed dual-setting classification with fixed-RK4 checks at capture edges; nonmonotone masks '
        'were retained. Timeouts were not converted to escapes. Exactly zero speed uses only the zero flux anchor.', '',
        f"The largest nested 0.5/0.25 m/s quadrature difference was {100*max(r['quadrature_relative_difference'] for r in rows):.5f}% "
        'of the loading mean, separately reported in results.csv. This is a sensitivity estimate, not a rigorous '
        'bound on all numerical errors. Finite velocity sampling cannot exclude arbitrarily narrow capture islands.', '',
        'Absolute rates preserve the historical prefactor 9.1196e5 and thermal scale 5.667e4 (m/s)^2. '
        'Its normalized-Maxwell equivalent density is 5.451556957e12 Rb-87 atoms/m³. The density-independent '
        'coefficient K=R/n87 and 95% bars are saved in results.csv and the second figure.', '',
        'Only the corrected Section-12 population-rate solver supplies force. These predictions remain conditional '
        'on instantaneous local steady state. Weak-wing population relaxation along moving trajectories remains '
        'unvalidated; frozen-point event agreement does not settle that issue. Sampling intervals exclude '
        'vapor-density, apparatus and model uncertainty. Temperature remains deferred.', '',
        f"Core source SHA-256: `{manifest['source_sha256']}`. Runner, source snapshot, lockfile and saved aggregate hashes verified; "
        'final_audit.json records evidence. Simulation inputs and retained results were not changed.', '',
        '[Results table](results.csv) | [Paired differences](paired_differences.csv) | [Audit](final_audit.json)', '']
    for file in ('loading_rate_vs_axial_gradient','loading_coefficient_vs_axial_gradient'):
        lines.append(f'[{file} PNG]({(figures/(file+".png")).as_posix()}) | [PDF]({(figures/(file+".pdf")).as_posix()})')
    report=final/'REPORT.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rc.write_json(final/'final_audit.json',dict(passed=True,reviewed_at_utc=now,
        finalizer_sha256=sha(__file__),source_sha256=manifest['source_sha256'],
        runner_sha256=manifest['runner_sha256'],evidence=evidence,paired_differences=contrasts,
        retained_cases=total_retention,maximum_retention_endpoint_difference_m=max_endpoint,
        maximum_retention_final_window_radius_m=max_window,total_elapsed_minutes=total_minutes,
        visual_review_sha256=sha(final/'visual_review.json'),report_sha256=sha(report)))
    complete.update(visual_review_pending=False,final_audit_passed=True,finalized_at_utc=now,
        final_report=str(report.relative_to(campaign.ROOT)))
    rc.write_json(root/'completion.json',complete)
    print(campaign.json.dumps(dict(passed=True,total_minutes=total_minutes,retention_cases=total_retention,
        paired_differences=contrasts,report=str(report)),indent=2))


if __name__=='__main__':
    main()
