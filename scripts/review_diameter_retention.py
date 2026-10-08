"""Derive separate retention-qualified pilot masks; never publish loading."""
import run_population_diameter_campaign as runner
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def main():
    root=runner.ROOT/'outputs/statistics/mot_multilevel_population_rate_v1/cooling_diameter_20260930_fixed_intensity_baseline'
    pilot=root/'pilot/01_cooling_diameter/000'
    audits=root/'retention_recheck/000'
    output=root/'retention_qualified_pilot/000'
    assert runner.rc.source_digest()==runner.CORE_SHA
    completion=json.loads((audits/'completion.json').read_text())
    assert completion['complete'] and completion['all_nodes_definitive']
    masks=np.zeros((32,64,len(runner.ce.VELOCITIES)),dtype=np.int8)
    sources=[]
    changes=[]
    last_save=0.
    for disc in range(32):
        for point in range(64):
            filename=pilot/'rays'/f'd{disc:03d}_p{point:03d}.npz'
            digest=hashlib.sha256(filename.read_bytes()).hexdigest()
            last_save=max(last_save,filename.stat().st_mtime)
            with np.load(filename) as saved:
                assert str(saved['status'])=='resolved'
                speeds=saved['speeds'].copy()
                codes=saved['codes'].copy()
            auditfile=audits/'rays'/f'{filename.stem}.json'
            source={'ray':filename.name,'capture_sha256':digest}
            if np.any(codes==1):
                audit=json.loads(auditfile.read_text())
                assert audit['source_sha256']==runner.CORE_SHA and audit['input_sha256']==digest
                source['audit_sha256']=hashlib.sha256(auditfile.read_bytes()).hexdigest()
                checked={n['speed_m_per_s']:n for n in audit['nodes']}
                for index in np.flatnonzero(codes==1):
                    result=checked[float(speeds[index])]
                    assert result['code'] in (0,1)
                    if result['code']==0:
                        assert all(o['escaped'] for o in result['attempts'][-1]['observations'])
                        changes.append({'disc':disc,'point':point,'speed_m_per_s':float(speeds[index]),
                                        'old_early_code':1,'qualified_code':0})
                    codes[index]=result['code']
            sources.append(source)
            np.testing.assert_array_equal(speeds[:len(runner.ce.VELOCITIES)],runner.ce.VELOCITIES)
            masks[disc,point]=codes[:len(runner.ce.VELOCITIES)]
    velocities=runner.ce.VELOCITIES
    weight=velocities**3*np.exp(-velocities**2/runner.ce.THERMAL_SCALE_M2_PER_S2)
    integrands=np.concatenate((np.zeros((32,64,1)),masks*weight),axis=2)
    values=np.pi*.015**2*runner.ce.LOADING_RATE_PREFACTOR*np.trapezoid(integrands,np.r_[0.,velocities],axis=2)
    mean,half=runner.rc.cluster_interval(values.mean(axis=1))
    coarse=np.pi*.015**2*runner.ce.LOADING_RATE_PREFACTOR*np.trapezoid(
        np.concatenate((np.zeros((32,64,1)),(masks*weight)[:,:,1::2]),axis=2),np.r_[0.,velocities[1::2]],axis=2)
    between,within=runner.ce.variance_components(values)
    choices=[]
    for points in (16,32,64,128):
        score=float(between+within/points)/(.05*mean)**2
        discs=max(64,int(np.ceil(1.5*1.96**2*score/16))*16)
        while 1.5*t.ppf(.975,discs-1)**2*score>discs:
            discs+=16
        choices.append({'rays':discs*points,'discs':discs,'points_per_disc':points})
    best=min(choices,key=lambda r:r['rays'])
    process=json.loads((root/'process.json').read_text())
    elapsed=last_save-datetime.datetime.fromisoformat(process['started_at_utc']).timestamp()
    raw_hours=(elapsed+completion['elapsed_wall_s'])/2048*best['rays']/3600
    report={'source_sha256':runner.CORE_SHA,'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'diameter_mm':2.,'discs':32,'points_per_disc':64,'changed_positive_nodes':len(changes),
        'all_early_positive_nodes_audited':True,'indeterminate_nodes':0,
        'candidate_loading_atoms_per_s':mean,'candidate_95_half_width_atoms_per_s':half,
        'relative_95_half_width':half/mean,'relative_nested_quadrature_difference':abs(float(coarse.mean())-mean)/mean,
        'publication_ready':False,'reason':'pilot only; corrected pilot nested quadrature passes 2%, but statistical precision and full fixed-RK4 comparison remain unqualified',
        'provisional_uniform_sampling_allocation':best,'allocation_candidates':choices,
        'observed_capture_wall_s':elapsed,'observed_all_positive_audit_wall_s':completion['elapsed_wall_s'],
        'provisional_uniform_sampling_hours':[1.25*raw_hours,1.75*raw_hours],
        'timing_scope':'capture plus every-positive retention at measured pilot throughput; excludes finer velocity grid, validation and pilot costs',
        'changes':changes,'sources':sources}
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'qualified_masks.npz',velocity_m_per_s=velocities,
        codes=masks,disc_loading_atoms_per_s=values.mean(axis=1))
    runner.rc.write_json(output/'review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','changes','allocation_candidates')}),flush=True)


if __name__=='__main__':
    main()
