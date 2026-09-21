"""Audit and assemble validated original/independent-confirmation results.

Postprocessing only: never changes simulation inputs or launches trajectories.
--available emits complete validated studies while confirmation is in progress.
"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.venv_pMOT_MC/Lib/site-packages'),str(ROOT/'src')]
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLBACKEND']='Agg'
import argparse,hashlib,json,shutil,datetime
import numpy as np
import matplotlib.pyplot as plt
from pmot.mot_multilevel.relationship_campaign import (
    paths,loading_points,source_digest,cluster_interval,write_json,write_csv)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(available=False):
    original=paths()['statistics'];figures=paths()['figures']/'final'
    destination=original/'final';plan=json.loads((original/'precision_followup_plan.json').read_text())
    confirmation=original.parent/plan['confirmation_campaign']
    replacements={p['key'] for p in plan['points']}
    digest=source_digest()
    for source in (original,confirmation):
        manifest=json.loads((source/'campaign_manifest.json').read_text())
        assert manifest['source_sha256']==digest
        assert manifest['model']=='pmot.mot_multilevel Section-12 population rate equations'
        tests=json.loads((source/'validation/test_status.json').read_text())
        assert tests=={'exit_code':0,'source_sha256':digest}
    reference_seed=json.loads((original/'campaign_manifest.json').read_text())['production_seed']
    confirmation_manifest=json.loads((confirmation/'campaign_manifest.json').read_text())
    assert confirmation_manifest['production_seed']==plan['confirmation_seed']!=reference_seed
    frozen=json.loads((confirmation/'confirmation_manifest.json').read_text())
    assert frozen['allocation_plan_sha256']==sha(original/'precision_followup_plan.json')
    assert frozen['source_sha256']==digest
    rows=[];pending=[];provenance=[];retention_cases=0
    for point in loading_points():
        source=confirmation if point.key in replacements else original
        folder=source/'production'/point.key
        summary=folder/'summary.json'
        if not summary.exists():pending.append(point.key);continue
        row=json.loads(summary.read_text())
        if row['publication_status']!='validated':pending.append(point.key);continue
        assert row['coordinate']==point.coordinate and row['radius_mm']==point.radius_mm
        assert row['loading_precision_pass'] and row['velocity_quadrature_pass'] and row['all_positive_nodes_dual_step_converged']
        if point.study=='01_radius':assert row['cross_section_precision_pass']
        audit=json.loads((folder/'duration_audit.json').read_text())
        assert audit['passed'] and all(r['passed'] for r in audit['cases'])
        retention_cases+=len(audit['cases'])
        with np.load(folder/'disc_statistics.npz') as saved:
            rates=saved['disc_loading_atoms_per_s']
            assert len(rates)==row['disc_count']
            mean,half=cluster_interval(rates)
            np.testing.assert_allclose([mean,half],[row['loading_rate_atoms_per_s'],row['loading_95_half_width_atoms_per_s']],rtol=1e-12)
        configuration=json.loads((folder/'configuration.json').read_text())
        assert configuration['discs']==row['disc_count'] and configuration['points_per_disc']==row['points_per_disc']
        assert configuration['seed']==(plan['confirmation_seed'] if point.key in replacements else reference_seed)
        csvrow={k:row[k] for k in ('study','index','coordinate','disc_count','points_per_disc','sample_count','radius_mm',
            'loading_rate_atoms_per_s','loading_95_half_width_atoms_per_s','loading_relative_half_width',
            'velocity_quadrature_difference_atoms_per_s','maximum_capture_speed_on_grid_m_per_s','nonmonotone_ray_count')}
        csvrow['source_campaign']=source.name
        rows.append(csvrow)
        provenance.append({'key':point.key,'source_root':str(source),'seed':configuration['seed'],
            'inference':'independent precision confirmation' if point.key in replacements else 'original independent production',
            'summary_sha256':sha(summary),'disc_statistics_sha256':sha(folder/'disc_statistics.npz'),
            'retention_audit_sha256':sha(folder/'duration_audit.json')})
    if pending and not available:raise RuntimeError(f'Final assembly withheld: {pending}')
    figures.mkdir(parents=True,exist_ok=True);destination.mkdir(parents=True,exist_ok=True)
    labels={'01_radius':('Sampling-disc radius [mm]','loading_vs_radius',8),
        '02_raw_saturation':('Single-beam center saturation s₀','loading_vs_saturation',24),
        '03_effective_saturation':('Single-beam center effective saturation','loading_vs_effective_saturation',20),
        '04_detuning':('Cooling detuning Δ/Γ','loading_vs_detuning',23)}
    emitted=[]
    for study,(xlabel,stem,count) in labels.items():
        group=[r for r in rows if r['study']==study]
        if len(group)!=count:continue
        group.sort(key=lambda r:r['coordinate'])
        fig,ax=plt.subplots(figsize=(8.8,5.6),layout='constrained')
        ax.errorbar([r['coordinate'] for r in group],[r['loading_rate_atoms_per_s'] for r in group],
            yerr=[r['loading_95_half_width_atoms_per_s'] for r in group],fmt='o-',capsize=3)
        ax.set(xlabel=xlabel,ylabel='Loading rate [atoms/s]',title='Corrected population-rate MOT · 95% direction-cluster intervals')
        ax.grid(alpha=.25)
        for ext in ('png','pdf'):fig.savefig(figures/f'{stem}.{ext}',dpi=180)
        plt.close(fig);emitted.append(stem)
        write_csv(destination/f'{stem}.csv',group)
    radius=[r for r in rows if r['study']=='01_radius']
    if len(radius)==8:
        for full in (False,True):
            fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
            for row in radius:
                source=original/'production/01_radius'/f"{row['index']:03d}"/'cross_section.csv'
                table=np.genfromtxt(source,delimiter=',',names=True)
                mask=np.ones(len(table),dtype=bool) if full else (table['velocity_m_per_s']>=1)&(table['velocity_m_per_s']<=30)
                table=table[mask];mean=table['cross_section_m2']*1e6
                error=np.array([mean-table['lower_95_m2']*1e6,table['upper_95_m2']*1e6-mean])
                assert np.all(error>=0)
                ax.errorbar(table['velocity_m_per_s'],mean,yerr=error,label=f"{row['radius_mm']:g} mm",lw=1,marker='.',ms=2,capsize=1)
                shutil.copy2(source,destination/f"cross_section_radius_{row['radius_mm']:g}mm.csv")
            ax.set(xlabel='Launch speed [m/s]',ylabel='Capture cross section [mm²]',
                title='Corrected population-rate MOT · 95% direction-cluster intervals',ylim=(0,None))
            ax.grid(alpha=.25);ax.legend(ncol=2)
            stem='cross_section_by_radius_full_grid' if full else 'cross_section_by_radius'
            for ext in ('png','pdf'):fig.savefig(figures/f'{stem}.{ext}',dpi=180)
            plt.close(fig)
            if not full:emitted.append(stem)
    force_summary=json.loads((original/'05_force/summary.json').read_text())
    assert force_summary['converged'] and force_summary['points']==111 and not force_summary['gravity_in_force']
    for stem in ('restoring_force_slope_vs_detuning','damping_turnaround_vs_detuning'):
        for ext in ('png','pdf'):shutil.copy2(paths()['figures']/'05_force'/f'{stem}.{ext}',figures/f'{stem}.{ext}')
        emitted.append(stem)
    write_csv(destination/'validated_loading_points.csv',rows)
    report={'source_sha256':digest,'assembler_sha256':sha(__file__),
        'assembled_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'all_seven_plot_types_ready':not pending and len(emitted)==7,
        'pending_points':pending,'validated_loading_points':len(rows),'selected_rays':sum(r['sample_count'] for r in rows),
        'retention_audit_cases':retention_cases,'primary_figures':emitted,'figures_directory':str(figures),
        'point_provenance':provenance,'force_source_root':str(original/'05_force'),
        'force_summary_sha256':sha(original/'05_force/summary.json'),
        'loading_relative_95_half_width_range':[min(r['loading_relative_half_width'] for r in rows),max(r['loading_relative_half_width'] for r in rows)],
        'maximum_relative_velocity_quadrature_difference':max(r['velocity_quadrature_difference_atoms_per_s']/r['loading_rate_atoms_per_s'] for r in rows),
        'temperature':'deferred by user',
        'limits':'Pointwise direction-cluster Student-t intervals; force bars are numerical refinement differences. Mean-force capture only. Finite speed-grid and representative retention audits do not prove absence of sub-grid islands or indefinite trapping.'}
    write_json(destination/'assembly_manifest.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='point_provenance'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--available',action='store_true')
    assemble(parser.parse_args().available)
