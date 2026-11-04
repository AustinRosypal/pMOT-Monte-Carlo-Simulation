"""Read-only, independent reduction of complete final-retention discs.

Only separate monitor_estimates are written. Collector masks, caches,
checkpoints, latest aliases, source and fixed allocation are untouched.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import numpy as np
from scipy.stats import t


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def reduce_discs(root,count,mode='production'):
    manifest=read(root/'campaign_manifest.json')
    assert manifest['observation_horizon_s']==2. and manifest['final_retention_window_s']==.025
    assert manifest['outcomes']['1']=='central final-window retention after core rule'
    points=manifest['points_per_disc']; velocity=np.asarray(manifest['positive_velocities_m_per_s'])
    thermal=manifest['thermal_conditions']; prefactor=thermal['historical_loading_prefactor']
    weight=velocity**3*np.exp(-velocity**2/thermal['thermal_scale_m2_per_s2'])
    with np.load(root/mode/'geometry.npz') as a: areas=a['areas_m2'][:count]
    assert 1<count<=len(areas)
    spectra=np.zeros((count,5,len(velocity))); counts=np.zeros(6,dtype=np.int64)
    for d in range(count):
        for p in range(points):
            with np.load(root/mode/'rays'/f'd{d:03d}_p{p:03d}.npz') as a:
                assert str(a['status'])=='resolved', f'incomplete disc {d} point {p}'
                assert float(a['observation_horizon_s'])==2. and float(a['retention_window_s'])==.025
                np.testing.assert_array_equal(a['speeds'][:len(velocity)],velocity)
                assert np.all(np.isin(a['codes'],np.arange(6)))
                assert not np.any(a['codes'][len(velocity):]==1)
                codes=a['codes'][:len(velocity)]
                for i,category in enumerate((1,2,3,4,5)):
                    spectra[d,i]+=(codes==category)*areas[d]/points
                counts+=np.bincount(codes,minlength=6)
    integrands=spectra*weight
    rates=prefactor*np.trapezoid(np.concatenate((np.zeros((count,5,1)),integrands),axis=2),
                                np.r_[0.,velocity],axis=2)
    mean=rates.mean(axis=0); half=t.ppf(.975,count-1)*rates.std(axis=0,ddof=1)/np.sqrt(count)
    maximum=prefactor*np.pi*manifest['disc_radius_m']**2*np.trapezoid(np.r_[0.,weight],np.r_[0.,velocity])
    half[np.all(rates==0,axis=0)]=maximum*(1-.025**(1/count))
    coarse=prefactor*np.trapezoid(np.column_stack((np.zeros(count),integrands[:,0,1::2])),
                                np.r_[0.,velocity[1::2]],axis=1)
    quadrature=float(abs(coarse.mean()-mean[0])); density=thermal['rb87_density_m3']
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),mode=mode,
        status='descriptive final-central-retention monitor estimate; final gates pending',
        observation_horizon_s=2.,final_retention_window_s=.025,
        complete_direction_discs=count,points_per_disc=points,resolved_rays=count*points,
        loading_rate_atoms_per_s=float(mean[0]),loading_95_half_width_atoms_per_s=float(half[0]),
        loading_ci95_atoms_per_s=[max(0.,float(mean[0]-half[0])),float(mean[0]+half[0])],
        loading_relative_half_width=float(half[0]/mean[0]) if mean[0]>0 else None,
        loading_coefficient_m3_per_s=float(mean[0]/density),
        loading_coefficient_95_half_width_m3_per_s=float(half[0]/density),
        off_core_retention_integral_atoms_per_s=float(mean[1]),
        off_core_retention_95_half_width_atoms_per_s=float(half[1]),
        observed_transient_core_then_wall_integral_atoms_per_s=float(mean[2]),
        observed_transient_core_then_wall_95_half_width_atoms_per_s=float(half[2]),
        other_inside_core_endpoint_integral_atoms_per_s=float(mean[3]),
        wall_loss_ambiguous_core_history_integral_atoms_per_s=float(mean[4]),
        outcome_node_counts=counts.tolist(),velocity_quadrature_difference_atoms_per_s=quadrature,
        quadrature_pass=bool(mean[0]>0 and quadrature<=.02*mean[0]),
        precision_pass=bool(mean[0]>0 and half[0]<=.05*mean[0]),
        thermal_conditions=thermal,source_sha256=manifest['source_sha256'],
        reducer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interval='95% Student-t direction-disc cluster interval; no stopping decision',
        interpretation='core rule plus continuous final25ms central residence at2s',
        off_core_interpretation='off-core endpoint retention at2s; not validated3D trapping',
        secondary_wall_history='observed transient histories are diagnostic and never retained loading',
        collector_outputs_unchanged=True,fixed_sampling_plan_unchanged=True,
        pending_gates=['final numerical convergence','longer retention','precision','aperture/placement','visual review'])
    arrays=dict(velocity_m_per_s=velocity,disc_cross_section_m2=spectra[:,0],
        disc_off_core_retention_cross_section_m2=spectra[:,1],disc_all_outcome_spectra_m2=spectra,
        disc_loading_atoms_per_s=rates[:,0],disc_all_outcome_rates_atoms_per_s=rates,disc_areas_m2=areas)
    return row,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path); parser.add_argument('--mode',choices=('pilot','production'),default='production')
    parser.add_argument('--discs',type=int); parser.add_argument('--check-published',action='store_true')
    args=parser.parse_args(); folder=args.root/'monitor_estimates'; folder.mkdir(exist_ok=True)
    published=read(args.root/args.mode/'latest_estimate.json')
    if args.check_published:
        check,_=reduce_discs(args.root,published['complete_direction_discs'],args.mode)
        keys=('loading_rate_atoms_per_s','loading_95_half_width_atoms_per_s',
              'off_core_retention_integral_atoms_per_s','off_core_retention_95_half_width_atoms_per_s',
              'observed_transient_core_then_wall_integral_atoms_per_s','velocity_quadrature_difference_atoms_per_s')
        for key in keys: np.testing.assert_allclose(check[key],published[key],rtol=1e-11,atol=1e-7)
        record=dict(passed=True,at_utc=datetime.now(timezone.utc).isoformat(),mode=args.mode,
                    matched_count=published['complete_direction_discs'],keys=list(keys),reducer_sha256=check['reducer_sha256'])
        (folder/f'{args.mode}_validation_{published["complete_direction_discs"]:05d}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    if args.discs is None:
        progress=read(args.root/'progress.json'); assert progress['mode']==args.mode
        count=progress['complete_prefix_discs']
    else: count=args.discs
    row,arrays=reduce_discs(args.root,count,args.mode)
    stem=folder/f'{args.mode}_discs_{count:05d}'
    np.savez_compressed(stem.with_suffix('.npz'),**arrays)
    stem.with_suffix('.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    snapshot=folder/f'reducer_source_{row["reducer_sha256"]}.py'
    if not snapshot.exists(): shutil.copy2(__file__,snapshot)
    print(json.dumps(row),flush=True)


if __name__=='__main__': main()
