"""Independent, fixed-size confirmation of failed loading precision targets.

Run only after the original pool exits. --check performs read-only preflight.
The original failed-precision samples determine size but never enter these
independent estimates. Existing physics, velocity masks, and audits are reused.
"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.venv_pMOT_MC/Lib/site-packages'),str(ROOT/'src')]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import argparse,hashlib,json,shutil
from concurrent.futures import ProcessPoolExecutor
from pmot.mot_multilevel import relationship_campaign as relationships
from pmot.mot_multilevel import campaign_execution as execution


def run(workers,check):
    original=relationships.paths()['statistics']
    plan_file=original/'precision_followup_plan.json'
    plan=json.loads(plan_file.read_text())
    digest=relationships.source_digest()
    if plan['current_source_sha256']!=digest:raise RuntimeError('confirmation source mismatch')
    points={p.key:p for p in relationships.loading_points()}
    selected=plan['points']
    if not selected or len({p['key'] for p in selected})!=len(selected):raise ValueError('empty or duplicate confirmation points')
    for allocation in selected:
        key=allocation['key']
        if key not in points:raise ValueError('unknown historical setting')
        if allocation['confirmation_discs']<64 or allocation['points_per_disc']!=16:raise ValueError('invalid allocation')
        summary=original/'production'/key/'summary.json'
        if hashlib.sha256(summary.read_bytes()).hexdigest()!=allocation['original_summary_sha256']:
            raise RuntimeError('allocation evidence changed; review the confirmation plan')
    for name in ('test_status.json','event_and_trajectory_checks.json','hybrid_pilot_comparison.json'):
        report=json.loads((original/'validation'/name).read_text())
        if name=='test_status.json' and report!={'exit_code':0,'source_sha256':digest}:raise RuntimeError('tests stale or failed')
        if name=='event_and_trajectory_checks.json' and not report['passed']:raise RuntimeError('event/trajectory gate failed')
        if name=='hybrid_pilot_comparison.json' and (report['source_sha256']!=digest or report['mismatching_nodes'] or any(r['status']!='resolved' for r in report['rows'])):
            raise RuntimeError('hybrid validation stale or failed')
    completion=original/'completion.json'
    ready=completion.exists() and json.loads(completion.read_text())['all_75_loading_points_computed']
    if check:
        print(json.dumps({'source_sha256':digest,'selected_points':len(selected),
            'total_confirmation_rays':sum(p['confirmation_discs']*p['points_per_disc'] for p in selected),
            'original_campaign_finished':ready,'status':'preflight passed; wait for original pool exit before launch'},indent=2))
        return
    if not ready:raise RuntimeError('original campaign has not finished; do not start a duplicate pool')
    # These controls are explicitly saved in the new campaign manifest. No
    # mutation of the original campaign or its numerical/physical controls.
    relationships.CAMPAIGN=plan['confirmation_campaign']
    execution.PRODUCTION_SEED=plan['confirmation_seed']
    output=relationships.paths()['statistics']
    execution.initialize_manifest()
    certificate={'schema':1,'source_sha256':digest,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_campaign':str(original),'allocation_plan_sha256':hashlib.sha256(plan_file.read_bytes()).hexdigest(),
        'plan':plan,'inference':'independent confirmation only; original failed-precision estimates excluded; fixed counts without early precision stopping'}
    provenance=output/'confirmation_manifest.json'
    if provenance.exists() and json.loads(provenance.read_text())!=certificate:raise RuntimeError('confirmation manifest mismatch')
    relationships.write_json(provenance,certificate)
    (output/'validation').mkdir(parents=True,exist_ok=True)
    for name in ('test_status.json','event_and_trajectory_checks.json','hybrid_pilot_comparison.json'):
        shutil.copy2(original/'validation'/name,output/'validation'/name)
    relationships.write_json(output/'process.json',{'pid':os.getpid(),'stage':'precision_confirmation','workers':workers})
    try:
        rows=[]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for allocation in selected:
                point=points[allocation['key']];n=allocation['confirmation_discs'];p=allocation['points_per_disc']
                row=execution.run_loading_point(point,'production',n,p,executor)
                execution.duration_audits(point,'production',n,p,executor)
                passed=row['loading_precision_pass'] and row['velocity_quadrature_pass'] and (point.study!='01_radius' or row['cross_section_precision_pass'])
                row['publication_status']='validated' if passed else 'precision target not met'
                row['inference_sample']='independent precision confirmation'
                relationships.write_json(output/'production'/point.key/'summary.json',row)
                rows.append(row);relationships.write_csv(output/'production_summary.csv',rows)
        relationships.write_json(output/'completion.json',{'all_confirmation_points_computed':True,
            'publication_ready':all(r['publication_status']=='validated' for r in rows),
            'remaining':[f"{r['study']}/{r['index']:03d}" for r in rows if r['publication_status']!='validated'],
            'next':'Assemble seven final figures from original validated settings and independent confirmation replacements; preserve both data roots.'})
    except Exception as error:
        import traceback
        relationships.write_json(output/'run_failure.json',{'error':repr(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();run(args.workers,args.check)
