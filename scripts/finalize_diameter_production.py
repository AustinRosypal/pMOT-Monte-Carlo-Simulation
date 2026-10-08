"""Audit saved independent-cluster results and assemble the 5--25 mm report."""
import run_diameter_production_5mm as production
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

runner=production.runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runner.rc.CAMPAIGN='cooling_diameter_20260930_fixed_intensity_baseline'
    root=runner.rc.paths()['statistics']
    complete=json.loads((root/'production_5mm_completion.json').read_text())
    assert complete['computed'] and complete['publication_ready'] and not complete['issues']
    plan=json.loads((root/'production_5mm_plan.json').read_text())
    assert runner.rc.source_digest()==plan['source_sha256']==runner.CORE_SHA
    assert sha(production.__file__)==plan['production_runner_sha256']
    assert sha(runner.__file__)==plan['adapter_sha256']
    output=root/'final_from_5mm';output.mkdir(parents=True,exist_ok=True)
    rows=[];evidence=[];audits=0;nonmonotone=0;stage_seconds=0.
    max_endpoint=0.;max_window=0.;durations=set()
    for index,allocation in zip(production.INDICES,plan['allocations']):
        folder=root/'production'/f'01_cooling_diameter/{index:03d}'
        row=json.loads((folder/'summary.json').read_text())
        assert row['publication_status']=='validated'
        assert row['all_positive_nodes_dual_step_converged'] and row['retention_pass']
        assert row['sample_count']==allocation['rays']==allocation['disc_count']*allocation['points_per_disc']
        assert len(list((folder/'rays').glob('*.npz')))==row['sample_count']
        configuration=json.loads((folder/'configuration.json').read_text())
        assert configuration['seed']==2026093002
        with np.load(folder/'disc_statistics.npz') as z:
            spectra=z['disc_cross_section_m2']
            speeds=z['velocity_m_per_s']
            loading=z['disc_loading_atoms_per_s']
            assert spectra.shape==(row['disc_count'],320)
            np.testing.assert_array_equal(speeds,runner.ce.VELOCITIES)
            assert np.all(np.isfinite(spectra)) and np.min(spectra)>=0 and np.max(spectra)<=np.pi*.015**2*(1+1e-12)
            weighted=spectra*speeds**3*np.exp(-speeds**2/runner.ce.THERMAL_SCALE_M2_PER_S2)
            recomputed=runner.ce.LOADING_RATE_PREFACTOR*np.trapezoid(np.column_stack((np.zeros(len(loading)),weighted)),np.r_[0.,speeds],axis=1)
            np.testing.assert_allclose(recomputed,loading,rtol=1e-12,atol=1e-8)
            mean,half=runner.rc.cluster_interval(recomputed)
            np.testing.assert_allclose([mean,half],[row['loading_rate_atoms_per_s'],row['loading_95_half_width_atoms_per_s']],rtol=1e-12)
            coarse=runner.ce.LOADING_RATE_PREFACTOR*np.trapezoid(np.column_stack((np.zeros(len(loading)),weighted[:,1::2])),np.r_[0.,speeds[1::2]],axis=1)
            quadrature=abs(float(coarse.mean())-mean)
            np.testing.assert_allclose(quadrature,row['velocity_quadrature_difference_atoms_per_s'],rtol=1e-10,atol=1e-6)
        assert half/mean<=.05 and quadrature/mean<=.02
        audit=json.loads((folder/'duration_audit.json').read_text())
        assert audit['passed'] and audit['cases'] and all(c['passed'] for c in audit['cases'])
        for case in audit['cases']:
            final=case['attempts'][-1]
            assert final['coarse_retained'] and final['fine_retained']
            assert final['position_difference_m']<1e-5
            assert max(final['coarse_final_window_max_radius_m'],final['fine_final_window_max_radius_m'])<=.002
            max_endpoint=max(max_endpoint,final['position_difference_m'])
            max_window=max(max_window,final['coarse_final_window_max_radius_m'],final['fine_final_window_max_radius_m'])
            durations.add(final['duration_s'])
        timing=json.loads((folder/'timing.json').read_text())
        row['measured_capture_hours']=timing['capture_wall_s']/3600
        row['measured_retention_hours']=timing['retention_wall_s']/3600
        row['measured_total_hours']=sum(timing.values())/3600
        row['velocity_quadrature_relative_difference']=quadrature/mean
        row['retention_cases']=len(audit['cases'])
        rows.append(row);audits+=len(audit['cases']);nonmonotone+=row['nonmonotone_ray_count']
        stage_seconds+=sum(timing.values())
        files=[folder/n for n in ('configuration.json','geometry.npz','summary.json','disc_statistics.npz','duration_audit.json','timing.json')]
        files.append(root/'production_parameters'/f'{index:03d}_compiled_inputs.npz')
        evidence.append({'diameter_mm':row['cooling_diameter_mm'],'files':{str(p.relative_to(root)).replace('\\','/'):sha(p) for p in files}})
    assert sum(r['sample_count'] for r in rows)==110336
    figure=runner.rc.paths()['figures']/'from_5mm'
    figures=[figure/f'{name}.{ext}' for name in ('loading_vs_cooling_diameter','loading_coefficient_vs_cooling_diameter') for ext in ('png','pdf')]
    assert all(p.exists() for p in figures)
    process=json.loads((root/'production_5mm_process.json').read_text())
    finished=(datetime.datetime.fromisoformat(complete['production_finished_at_utc'])
              if 'production_finished_at_utc' in complete else
              datetime.datetime.fromtimestamp((root/'production_5mm_completion.json').stat().st_mtime,datetime.timezone.utc))
    review_file=output/'visual_review.json'
    visually_reviewed=False
    if review_file.exists():
        review=json.loads(review_file.read_text())
        visually_reviewed=review['passed'] and all(review['png_sha256'].get(str(p))==sha(p) for p in figures if p.suffix=='.png')
    runtime=(finished-datetime.datetime.fromisoformat(process['started_at_utc'])).total_seconds()/3600
    report={'passed':True,'assembled_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'production_wall_hours':runtime,'summed_stage_hours':stage_seconds/3600,'rays':110336,
        'direction_discs':sum(r['disc_count'] for r in rows),'retention_cases':audits,'nonmonotone_rays':nonmonotone,
        'maximum_retention_endpoint_difference_m':max_endpoint,'maximum_final_window_radius_m':max_window,
        'retention_final_durations_s':sorted(durations),'source_sha256':runner.CORE_SHA,
        'analysis_script_sha256':sha(__file__),'visual_review_pending':not visually_reviewed,
        'plan_sha256':sha(root/'production_5mm_plan.json'),'parameters_sha256':sha(root/'production_parameters.json'),
        'figures':{str(p):sha(p) for p in figures},'evidence':evidence,'rows':rows}
    runner.rc.write_json(output/'audit.json',report)
    runner.rc.write_csv(output/'results.csv',rows)
    lines=['# Corrected multilevel MOT: loading versus cooling-beam diameter','',
        f'All eight requested 5--25 mm settings passed their prescribed statistical, velocity-grid and representative retention gates. Production completed in **{runtime:.2f} hours**, including its audits; this excludes earlier development and pilots. Visual review is recorded separately in `visual_review.json`.','',
        'The calculation used 110,336 launch rays on 6,896 independent full-sphere direction discs, with 16 uniform-area points per 15 mm-radius disc. Peak cooling intensity was fixed at 42.628145 mW/cm² per traveling beam, cooling detuning at −15 MHz, and the six repump components at 0.1 mW each with 12.7 mm diameter and zero detuning. The 2 and 3 mm points were explicitly deferred.','',
        '## Figures and data','',
        f'- [Loading rate PNG]({figures[0].as_posix()}) | [PDF]({figures[1].as_posix()})',
        f'- [Density-independent loading coefficient PNG]({figures[2].as_posix()}) | [PDF]({figures[3].as_posix()})',
        f'- [Results CSV]({(output/"results.csv").as_posix()}) | [Audit and file hashes]({(output/"audit.json").as_posix()})',
        f'- [Complete parameter record]({(root/"PRODUCTION_PARAMETERS.md").as_posix()}) | [Machine-readable parameters]({(root/"production_parameters.json").as_posix()})','',
        '## Results','',
        'Loading-rate bars below are pointwise 95% Student-t direction-cluster intervals. Atoms/s values retain the historical vapor coefficient; its absolute normalization remains unresolved.','',
        '| Diameter (mm) | Power/beam (mW) | Discs × points | Loading (10^6 atoms/s) ±95% | Relative half-width | Grid sensitivity | Runtime (h) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cooling_diameter_mm']:g} | {r['cooling_power_w_per_beam']*1000:.3f} | {r['disc_count']} × 16 | {r['loading_rate_atoms_per_s']/1e6:.4f} ± {r['loading_95_half_width_atoms_per_s']/1e6:.4f} | {100*r['loading_relative_half_width']:.3f}% | {100*r['velocity_quadrature_relative_difference']:.4f}% | {r['measured_total_hours']:.3f} |")
    lines+=['','The loading mean increases with beam diameter across this sampled range and flattens toward the large-diameter end. This curve holds peak intensity fixed, so cooling power rises from 4.185 to 104.625 mW per traveling beam. The repump remains narrower than the largest cooling beams.','',
        '## Statistical and numerical evidence','',
        'The fixed production allocations were selected from independent pilots with a 50% variance safety margin. Production used seed 2026093002; pilot seed was 2026093001. No statistical confirmation or post-hoc sample extension was needed. The final audit independently recomputed loading integrals and Student-t intervals from the saved direction-disc spectra and checked sample-file counts against the frozen plan.','',
        f'All {110336*328:,} positive-speed classification nodes were resolved using dual numerical settings, with fixed RK4 checks around every capture edge. The preproduction full fixed-RK4 comparisons covered 6,560 nodes on 20 predetermined rays with no differences. The production data retain {nonmonotone:,} nonmonotone ray masks rather than replacing them with scalar capture thresholds.','',
        f'All {audits:,} representative longer-duration retention cases passed. The largest final paired endpoint discrepancy was {max_endpoint:.3g} m; the gate was 1e-5 m. Accepted final audit durations were {sorted(durations)} s, and the final 25 ms remained inside the 2 mm core. This is representative finite-duration validation, not an infinite-lifetime guarantee.','',
        'Statistical half-widths are below 5% at all points. The nested 0.5/0.25 m/s loading-quadrature differences are below 2% at all points and are listed separately above. They measure grid sensitivity rather than a rigorous bound on every numerical error. No timeout was reclassified as an escape.','',
        '## Normalization and limits','',
        'The historical C=9.1196e5 loading coefficient and thermal scale 5.667e4 m²/s² are retained. For a normalized Maxwell distribution this coefficient corresponds to an equivalent n87≈5.45156e12 m^-3 (T≈296.18 K); the older written Maxwell prefactor was not normalized consistently. Use the saved coefficient K=R/n87 in m³/s to apply a specified vapor density: R=n87*K. The statistical bars do not include vapor-density uncertainty, apparatus uncertainty or model discrepancy.','',
        'The launch-disc radius is fixed at 15 mm. The largest-beam points do not establish convergence to an unlimited incident plane; the curve also includes the effect of keeping the repump diameter fixed. The force assumes ideal untruncated Gaussian beams. There is no recoil diffusion, equilibrium-temperature prediction, or collective/collisional MOT model.','',
        f'Only the corrected Section-12 population-rate equations were used. Source SHA-256: `{runner.CORE_SHA}`. The isolated compiled cache, tests, local high-precision checks and event/trajectory comparisons are retained in the campaign validation directory. Historical failed narrow-beam pilots and all revision evidence remain intact.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','evidence','figures')}),flush=True)


if __name__=='__main__':
    main()
