"""Independent raw-ray reduction of the separate 1s continuation; read-only inputs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import numpy as np
from scipy.stats import t


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def reduce(root,count):
    manifest=read(root/'campaign_manifest.json');assert manifest['observation_horizon_s']==1.
    start=read(root/'predecessor_inventory.json')['new_disc_start']
    thermal=manifest['thermal_conditions'];v=np.array(manifest['positive_velocities_m_per_s'])
    assert count>=2
    with np.load(root/'production/geometry.npz') as a:areas=a['areas_m2'][start:start+count].copy()
    spectrum=np.zeros((count,5,len(v)));counts=np.zeros(6,dtype=np.int64)
    for local,d in enumerate(range(start,start+count)):
        for p in range(20):
            with np.load(root/'production/rays'/f'd{d:03d}_p{p:03d}.npz') as a:
                assert str(a['status'])=='resolved'
                assert float(a['observation_horizon_s'])==1. and float(a['retention_window_s'])==.025
                np.testing.assert_array_equal(a['speeds'][:len(v)],v)
                codes=a['codes'][:len(v)]
                assert np.all(np.isin(a['codes'],np.arange(6)))
                assert not np.any(a['codes'][len(v):]==1)
                for k in range(5):spectrum[local,k]+=(codes==k+1)*areas[local]/20
                counts+=np.bincount(codes,minlength=6)
    weight=v**3*np.exp(-v*v/thermal['thermal_scale_m2_per_s2'])
    integrands=spectrum*weight
    rates=thermal['historical_loading_prefactor']*np.trapezoid(np.concatenate([np.zeros((count,5,1)),integrands],axis=-1),np.r_[0.,v],axis=-1)
    mean=rates.mean(axis=0);half=t.ppf(.975,count-1)*rates.std(axis=0,ddof=1)/np.sqrt(count)
    maximum=thermal['historical_loading_prefactor']*np.pi*.015**2*np.trapezoid(np.r_[0.,weight],np.r_[0.,v])
    half[np.all(rates==0,axis=0)]=maximum*(1-.025**(1/count))
    coarse=thermal['historical_loading_prefactor']*np.trapezoid(np.c_[np.zeros(count),integrands[:,0,1::2]],np.r_[0.,v[1::2]],axis=1)
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),status='independent descriptive1s complete-disc estimate; final gates pending',
        observation_horizon_s=1.,final_retention_window_s=.025,complete_direction_discs=count,resolved_rays=count*20,
        disc_start=start,disc_end_exclusive=start+count,loading_rate_atoms_per_s=float(mean[0]),
        loading_95_half_width_atoms_per_s=float(half[0]),loading_ci95_atoms_per_s=[max(0.,float(mean[0]-half[0])),float(mean[0]+half[0])],
        loading_coefficient_m3_per_s=float(mean[0]/thermal['rb87_density_m3']),
        off_core_retention_integral_atoms_per_s=float(mean[1]),off_core_retention_95_half_width_atoms_per_s=float(half[1]),
        observed_transient_core_then_wall_integral_atoms_per_s=float(mean[2]),observed_transient_core_then_wall_95_half_width_atoms_per_s=float(half[2]),
        velocity_quadrature_difference_atoms_per_s=float(abs(coarse.mean()-mean[0])),outcome_node_counts=counts.tolist(),
        thermal_conditions=thermal,source_sha256=manifest['source_sha256'],combined_with_2s_results=False,
        interval='Nominal95% Student-t independent-disc interval; time-truncated complete cases can select by computational cost; coverage not certified.',
        pending_gates=['final numerical convergence','production longer retention','precision','aperture/placement','final visual review'],
        off_core_interpretation='Off-core endpoint retention at1s; not validated3D trapping',
        reducer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    arrays=dict(velocity_m_per_s=v,disc_ids=np.arange(start,start+count),disc_all_outcome_spectra_m2=spectrum,
        disc_cross_section_m2=spectrum[:,0],disc_all_outcome_rates_atoms_per_s=rates,disc_loading_atoms_per_s=rates[:,0],disc_areas_m2=areas)
    return row,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root;published=read(root/'latest_estimate.json');n=published['complete_direction_discs']
    check,_=reduce(root,n)
    keys=['loading_rate_atoms_per_s','loading_95_half_width_atoms_per_s','off_core_retention_integral_atoms_per_s',
          'off_core_retention_95_half_width_atoms_per_s','observed_transient_core_then_wall_integral_atoms_per_s','velocity_quadrature_difference_atoms_per_s']
    for key in keys:np.testing.assert_allclose(check[key],published[key],rtol=1e-11,atol=1e-7)
    count=read(root/'progress.json')['complete_prefix_discs'];assert count>=n
    row,arrays=(check,reduce(root,n)[1]) if count==n else reduce(root,count)
    folder=root/'monitor_estimates';folder.mkdir(exist_ok=True)
    (folder/f'validation_{n:05d}.json').write_text(json.dumps(dict(passed=True,matched_published_discs=n,keys=keys,reducer_sha256=row['reducer_sha256']),indent=2)+'\n')
    stem=folder/f'production_discs_{count:05d}';row['estimate_file']=str(stem.with_suffix('.json').resolve())
    stem.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n');np.savez_compressed(stem.with_suffix('.npz'),**arrays)
    snapshot=folder/f'reducer_{row["reducer_sha256"]}.py'
    if not snapshot.exists():shutil.copy2(__file__,snapshot)
    print(json.dumps(row))


if __name__=='__main__':main()
