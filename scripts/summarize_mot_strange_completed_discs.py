"""Read-only reduction of complete finite-horizon production direction discs.

Writes separate monitor artifacts, never the collector's checkpoints, caches,
latest-estimate alias, geometry, masks or stopping allocation.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

import numpy as np
from scipy.stats import t


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def reduce_discs(root, count):
    manifest = read(root/'campaign_manifest.json')
    assert manifest['observation_horizon_s'] == 2.
    points = manifest['points_per_disc']
    velocity = np.asarray(manifest['positive_velocities_m_per_s'])
    thermal = manifest['thermal_conditions']
    prefactor = thermal['historical_loading_prefactor']
    density = thermal['rb87_density_m3']
    weight = velocity**3*np.exp(-velocity**2/thermal['thermal_scale_m2_per_s2'])
    with np.load(root/'production/geometry.npz') as saved:
        areas = saved['areas_m2'][:count]
    assert 1 < count <= len(areas)
    spectra = np.zeros((count, 3, len(velocity)))
    retained_nodes, nonmonotone_rays = 0, 0
    for d in range(count):
        for p in range(points):
            with np.load(root/f'production/rays/d{d:03d}_p{p:03d}.npz') as saved:
                assert str(saved['status']) == 'resolved', f'incomplete disc {d}, point {p}'
                assert float(saved['observation_horizon_s']) == 2.
                np.testing.assert_array_equal(saved['speeds'][:len(velocity)], velocity)
                assert np.all(np.isin(saved['codes'], [0, 1, 2]))
                assert not np.any(saved['codes'][len(velocity):] == 1)
                codes = saved['codes'][:len(velocity)]
                locations = saved['retained_locations'][:len(velocity)]
                assert np.all(np.isin(locations[codes == 2], [0, 1]))
            spectra[d, 0] += (codes == 1)*areas[d]/points
            spectra[d, 1] += ((codes == 2) & (locations == 0))*areas[d]/points
            spectra[d, 2] += ((codes == 2) & (locations == 1))*areas[d]/points
            retained_nodes += int(np.count_nonzero(codes == 2))
            nonmonotone_rays += int(np.any(np.diff((codes == 1).astype(int)) > 0))
    integrands = spectra*weight
    rates = prefactor*np.trapezoid(np.concatenate((np.zeros((count, 3, 1)), integrands), axis=2),
                                  np.r_[0., velocity], axis=2)
    mean = rates.mean(axis=0)
    half = t.ppf(.975, count-1)*rates.std(axis=0, ddof=1)/np.sqrt(count)
    maximum = prefactor*np.pi*manifest['disc_radius_m']**2*np.trapezoid(np.r_[0., weight], np.r_[0., velocity])
    half[np.all(rates == 0, axis=0)] = maximum*(1-.025**(1/count))
    coarse = prefactor*np.trapezoid(np.column_stack((np.zeros(count), integrands[:, 0, 1::2])),
                                   np.r_[0., velocity[1::2]], axis=1)
    difference = float(abs(coarse.mean()-mean[0]))
    row = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), mode='production',
        status='descriptive monitor estimate from complete prefix discs; final gates pending',
        observation_horizon_s=2., core_rule=manifest['core_capture'],
        complete_direction_discs=count, points_per_disc=points, resolved_rays=count*points,
        loading_rate_atoms_per_s=float(mean[0]), loading_95_half_width_atoms_per_s=float(half[0]),
        loading_ci95_atoms_per_s=[max(0., float(mean[0]-half[0])), float(mean[0]+half[0])],
        loading_relative_half_width=float(half[0]/mean[0]) if mean[0] > 0 else None,
        loading_coefficient_m3_per_s=float(mean[0]/density),
        loading_coefficient_95_half_width_m3_per_s=float(half[0]/density),
        off_core_retention_integral_atoms_per_s=float(mean[1]),
        off_core_retention_95_half_width_atoms_per_s=float(half[1]),
        inside_core_without_capture_rule_integral_atoms_per_s=float(mean[2]),
        retained_at_horizon_nodes=retained_nodes, nonmonotone_ray_count=nonmonotone_rays,
        velocity_quadrature_difference_atoms_per_s=difference,
        quadrature_pass=bool(mean[0] > 0 and difference <= .02*mean[0]),
        precision_pass=bool(mean[0] > 0 and half[0] <= .05*mean[0]),
        thermal_conditions=thermal, source_sha256=manifest['source_sha256'],
        reducer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interval='95% Student-t direction-disc cluster interval; no stopping decision',
        interpretation='central-core capture by 2 s; off-core retention is not a 3D trapping claim',
        collector_outputs_unchanged=True, fixed_sampling_plan_unchanged=True,
        pending_gates=['final numerical convergence', 'production retention', 'final precision',
                       'aperture and launch placement', 'final visual review'])
    arrays = dict(velocity_m_per_s=velocity, disc_cross_section_m2=spectra[:, 0],
        disc_off_core_retention_cross_section_m2=spectra[:, 1],
        disc_inside_core_without_capture_rule_cross_section_m2=spectra[:, 2],
        disc_loading_atoms_per_s=rates[:, 0], disc_off_core_retention_atoms_per_s=rates[:, 1],
        disc_inside_core_without_capture_rule_atoms_per_s=rates[:, 2], disc_areas_m2=areas)
    return row, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--discs', type=int)
    parser.add_argument('--check-published', action='store_true')
    args = parser.parse_args()
    published = read(args.root/'production/latest_estimate.json')
    if args.check_published:
        check, _ = reduce_discs(args.root, published['complete_direction_discs'])
        for key in ('loading_rate_atoms_per_s', 'loading_95_half_width_atoms_per_s',
                    'off_core_retention_integral_atoms_per_s', 'off_core_retention_95_half_width_atoms_per_s',
                    'velocity_quadrature_difference_atoms_per_s'):
            np.testing.assert_allclose(check[key], published[key], rtol=1e-11, atol=1e-7)
        print('Independent reduction matches published checkpoint.', flush=True)
    count = args.discs or read(args.root/'progress.json')['complete_prefix_discs']
    row, arrays = reduce_discs(args.root, count)
    folder = args.root/'monitor_estimates'; folder.mkdir(exist_ok=True)
    stem = folder/f'discs_{count:05d}'
    np.savez_compressed(stem.with_suffix('.npz'), **arrays)
    stem.with_suffix('.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
