"""Isolated cooling-diameter campaign using the validated population-rate engine.

Core source and prior outputs are immutable. This adapter constructs beam
geometry explicitly, validates it against the public 24-state reference, and
records its own source hash in a distinct resumable campaign.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE_SHA = '5ac7e8791ace6010d2311b93443a3d3f9f0601e84f5a4ab38f653827a07904e5'
# Numba does not invalidate cached callers when an imported callee changes.
# Isolate all compiled dependencies by the verified physics source digest.
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'outputs/compiled_cache'/CORE_SHA)
sys.path[:0] = [str(ROOT / '.venv_pMOT_MC/Lib/site-packages'), str(ROOT / 'src')]
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import datetime
import hashlib
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from pmot.configuration import default_mot_apparatus_config
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import relationship_campaign as rc
from pmot.mot_multilevel import campaign_execution as ce
from pmot.mot_multilevel.accelerated import prepare, observable
from pmot.mot_multilevel.configuration import MultilevelMOTConfig
from pmot.mot_multilevel.simulation import build_multilevel_mot_beams
from pmot.mot_multilevel.rate_equations import build_rate_equation_model, rate_equation_observable
from pmot.mot_multilevel.rate_equations import build_beam_transition_quantities
from pmot.magnetic_fields import anti_helmholtz_field_t
from pmot.configuration import HBAR_J_S

DIAMETERS = (2., 3., 5., 7., 10., 12.7, 15., 18., 21., 25.)
ORIGINAL_RAY_WORKER = ce._ray_worker
PILOT_N, PILOT_P = 32, 64
TARGET = .05


@dataclass(frozen=True)
class DiameterPoint:
    index: int
    coordinate: float
    cooling_power_w: float
    detuning_rad_s: float
    radius_mm: float = 15.
    study: str = '01_cooling_diameter'

    @property
    def key(self):
        return f'{self.study}/{self.index:03d}'

    def config(self):
        return DiameterConfig(cooling_power_w_per_beam=self.cooling_power_w,
                              cooling_detuning_rad_per_s=self.detuning_rad_s,
                              cooling_diameter_mm=self.coordinate)


@dataclass(frozen=True)
class DiameterConfig(MultilevelMOTConfig):
    cooling_diameter_mm: float = 12.7


def beams_for(config):
    apparatus = default_mot_apparatus_config()
    apparatus = replace(apparatus, cooling=replace(
        apparatus.cooling, beam_diameter_m=config.cooling_diameter_mm * .001))
    # The existing builder derives repump geometry from cooling geometry.
    # Explicitly restore the repump waist for this cooling-only experiment.
    return [replace(b, beam_radius_m=.00635) if b.family == 'repump' else b
            for b in build_multilevel_mot_beams(apparatus, config)]


def prepare_diameter(config):
    data = list(prepare(config))
    optical = data[1].copy()
    beams = beams_for(config)
    for i, beam in enumerate(beams):
        optical[i, 1] = beam.beam_radius_m ** 2
        optical[i, 2] = (np.pi * beam.beam_radius_m ** 2 / beam.wavelength_m) ** 2
    data[1] = optical
    return tuple(data)


def diameter_ray_worker(payload):
    # Legacy worker cache keys contain power/detuning only. Different diameters
    # can have identical keys; clear that cache before each immutable payload.
    ce._DATA_CACHE.clear()
    return ORIGINAL_RAY_WORKER(payload)


def fixed_comparison(payload):
    point, file, data = payload
    with np.load(file) as z:
        speeds, expected = z['speeds'], z['codes']
        position, direction = z['position_m'], z['direction']
    records, methods, differences = [], [], []
    for speed, code in zip(speeds, expected):
        result = ce._resolve_fixed_node(position, direction, speed, data, records, methods)
        if result != code:
            differences.append({'speed_m_per_s': float(speed), 'hybrid': int(code), 'fixed': result})
    target = file.parents[4] / 'validation' / 'fixed_comparisons' / point.key / file.name
    ce._atomic_npz(target, evaluations=np.asarray(records), evaluation_methods=np.asarray(methods),
                   position_m=position, direction=direction)
    return {'point': point.key, 'ray': file.name, 'nodes': len(speeds), 'differences': differences}


def high_precision_reference(model, beams, r, v, coil, cfg):
    """Full 24-state reference with accurate diagonals in near-dark beam wings.

    The ordinary double-precision reference is ill-conditioned when the narrow
    cooling beams are >10 waist radii away. This is validation-only; production
    still uses the unchanged compiled solver and its physical residual gates.
    """
    import mpmath as mp
    field=np.asarray(anti_helmholtz_field_t(*r,coil))
    axis=field/np.linalg.norm(field) if np.linalg.norm(field)>cfg.magnetic_field_epsilon_t else np.array([0.,0.,1.])
    w=build_beam_transition_quantities(model,beams,tuple(r),tuple(v),
        float(np.linalg.norm(field)),tuple(axis),cfg).stimulated_coefficients_per_s
    with mp.workdps(400):
        matrix=mp.matrix(24,24)
        for e in range(16):
            for g in range(8):
                stimulated=sum(mp.mpf(float(w[b,e,g])) for b in range(len(beams)))
                down=stimulated+mp.mpf(float(model.spontaneous_decay_matrix_per_s[g,e]))
                matrix[8+e,g]=stimulated
                matrix[g,8+e]=down
                matrix[g,g]-=stimulated
                matrix[8+e,8+e]-=down
        for j in range(24):matrix[23,j]=1
        rhs=mp.matrix(24,1);rhs[23]=1
        populations=np.array([float(x) for x in mp.lu_solve(matrix,rhs)])
    rates=np.sum(w*(populations[:8][None,None,:]-populations[8:][None,:,None]),axis=(1,2))
    force=sum(HBAR_J_S*2*np.pi/b.wavelength_m*np.asarray(b.direction)*rate for b,rate in zip(beams,rates))
    return force,populations


def validate(points, output):
    certificate = output / 'validation/geometry_and_force.json'
    if certificate.exists():
        saved=json.loads(certificate.read_text())
        assert saved['passed'] and saved['source_sha256']==CORE_SHA
        return
    model, coil = build_rate_equation_model(), default_anti_helmholtz_config()
    rng = np.random.default_rng(2026093003)
    count = 0
    high_precision_checks = 0
    for point in points:
        cfg = point.config()
        data, beams = prepare_diameter(cfg), beams_for(cfg)
        assert len(beams) == 12
        assert all(b.beam_radius_m == .00635 for b in beams if b.family == 'repump')
        assert all(np.isclose(b.beam_radius_m, point.coordinate * .0005)
                   for b in beams if b.family == 'cooling')
        if point.coordinate == 12.7:
            for a, b in zip(data, prepare(cfg)):
                np.testing.assert_array_equal(a, b)
        # Include origin, beam overlap, beam wings and launch-region positions.
        scale = point.coordinate * .0005
        positions = np.vstack([np.zeros(3), np.eye(3)*scale*.5,
                               rng.uniform(-scale, scale, (12, 3)),
                               rng.uniform(-.02, .02, (8, 3))])
        for r in positions:
            v = rng.uniform(-40, 40, 3)
            fast = observable(r, v, np.array([0., 0., 1.]), data)
            # Use the full high-precision matrix in the far wings of narrow
            # beams, where a normalized double-precision solve loses rank.
            if point.coordinate<=7 and np.linalg.norm(r)>3*scale:
                force,populations=high_precision_reference(model,beams,r,v,coil,cfg)
                high_precision_checks+=1
            else:
                ref = rate_equation_observable(model, beams, tuple(r), tuple(v), coil, cfg)
                force,populations=ref.force_n,ref.populations
            np.testing.assert_allclose(fast[0], force, rtol=3e-6, atol=3e-29)
            np.testing.assert_allclose(fast[1], populations, rtol=4e-6, atol=4e-10)
            count += 1
    rc.write_json(certificate, {'passed': True, 'local_checks': count,
        'source_sha256': CORE_SHA, 'baseline_arrays_identical': True,
        'repump_diameter_mm': 12.7, 'high_precision_checks':high_precision_checks,
        'reference': 'public full 24-state equations; 400-digit matrix solve in narrow-beam far wings'})


def allocation(row, output, workers):
    folder = output / 'pilot' / row['study'] / f"{row['index']:03d}"
    # For each candidate points/disc, estimate hierarchical variance from the
    # independent pilot, then freeze N before collecting production samples.
    samples = np.empty((PILOT_N, PILOT_P))
    weight = ce.VELOCITIES**3 * np.exp(-ce.VELOCITIES**2 / ce.THERMAL_SCALE_M2_PER_S2)
    ray_seconds = []
    for d in range(PILOT_N):
        for p in range(PILOT_P):
            with np.load(folder/'rays'/f'd{d:03d}_p{p:03d}.npz') as z:
                samples[d,p] = np.pi * .015**2 * ce.LOADING_RATE_PREFACTOR * np.trapezoid(
                    np.r_[0., z['codes'][:len(weight)] * weight], np.r_[0., ce.VELOCITIES])
                ray_seconds.append(float(z['elapsed_wall_s']))
    mean = float(np.mean(samples))
    if mean <= 0:
        raise RuntimeError('zero-capture pilot: extend independently before allocating relative precision')
    between, within = ce.variance_components(samples)
    choices = []
    for p in (16, 32, 64, 128):
        score = float(between + within/p) / (TARGET*mean)**2
        n = 64
        while 1.5*t.ppf(.975,n-1)**2*score > n:
            n += 16
        choices.append((n*p, n, p))
    rays, n, p = min(choices)
    timing = json.loads((folder/'timing.json').read_text())
    # Observed pool wall time includes serialization/IO; audit estimate scales
    # with direction count. Add 25--75% for production/runtime variation.
    raw_hours = ((timing['capture_wall_s']/(PILOT_N*PILOT_P))*rays
                 + timing['retention_wall_s']/PILOT_N*n)/3600
    return {'key': f"{row['study']}/{row['index']:03d}", 'diameter_mm': row['coordinate'],
        'disc_count': n, 'points_per_disc': p, 'rays': rays,
        'loading_pilot_mean_atoms_per_s': mean,
        'pilot_relative_95_half_width': row['loading_relative_half_width'],
        'between_disc_variance': float(between), 'within_disc_variance': float(within),
        'mean_ray_wall_s': float(np.mean(ray_seconds)), 'workers': workers,
        'estimated_hours': [1.25*raw_hours, 1.75*raw_hours],
        'allocation_candidates': [{'rays':x,'discs':y,'points_per_disc':z} for x,y,z in choices]}


def run(args):
    campaign = f'cooling_diameter_20260930_{args.power_mode}_{args.detuning}'
    rc.CAMPAIGN = campaign
    ce.PILOT_SEED, ce.PRODUCTION_SEED = 2026093001, 2026093002
    ce.prepare, ce._ray_worker = prepare_diameter, diameter_ray_worker
    detuning = -15e6 if args.detuning == 'baseline' else -3.75*6.07e6
    points = [DiameterPoint(i,d,.027 if args.power_mode=='fixed_power' else .027*(d/12.7)**2,
                            2*np.pi*detuning) for i,d in enumerate(DIAMETERS)]
    output = rc.paths()['statistics']
    output.mkdir(parents=True, exist_ok=True)
    assert rc.source_digest() == CORE_SHA, 'core changed: new revision and validation required'
    old = ROOT/'outputs/statistics/mot_multilevel_population_rate_v1/relationships_20260919/validation'
    assert json.loads((output/'validation/test_status.json').read_text()) == {'exit_code':0,'source_sha256':CORE_SHA}
    manifest = {'model':'pmot.mot_multilevel Section-12 population rate equations',
        'source_sha256':CORE_SHA,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'diameters_mm':DIAMETERS,'power_mode':args.power_mode,'detuning_hz':detuning,
        'sampling_disc_radius_mm':15.,'repump_diameter_mm':12.7,'repump_power_w_per_beam':.0001,
        'pilot_seed':ce.PILOT_SEED,'production_seed':ce.PRODUCTION_SEED,
        'pilot_discs':PILOT_N,'pilot_points_per_disc':PILOT_P,'loading_95_relative_half_width_target':TARGET,
        'normalization':'unchanged historical 9.1196e5; also report R/n87 using normalized Maxwell distribution; absolute vapor density unresolved',
        'old_validation_root':str(old),'numerics':'unchanged dual adaptive integration, fixed RK4 edge/fallback checks, dense speed masks, fixed RK4 retention ladder',
        'geometry_adapter':'cooling waist and Rayleigh length only; repump waist fixed; worker cache cleared to prevent reuse across diameters',
        'sampling_scope':'fixed 15 mm incident disc; no infinite-area loading claim'}
    manifest = json.loads(json.dumps(manifest))
    file = output/'campaign_manifest.json'
    if file.exists():
        assert json.loads(file.read_text()) == manifest, 'campaign manifest mismatch'
    else:
        rc.write_json(file,manifest)
    rc.write_json(output/'process.json',{'pid':os.getpid(),'stage':args.stage,'workers':args.workers,
        'command':sys.argv,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    validate(points, output)
    if args.stage == 'validate':
        return
    event=json.loads((output/'validation/event_and_trajectory_checks.json').read_text())
    assert event['passed'] and event['source_sha256']==CORE_SHA
    workers_gate=json.loads((output/'validation/fresh_worker_checks.json').read_text())
    assert workers_gate['passed'] and workers_gate['source_sha256']==CORE_SHA
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage in ('pilot','all'):
            plans = []
            for point in points:
                folder=output/'pilot'/point.key
                start=time.perf_counter()
                row=ce.run_loading_point(point,'pilot',PILOT_N,PILOT_P,pool)
                capture=time.perf_counter()-start
                start=time.perf_counter()
                ce.duration_audits(point,'pilot',PILOT_N,PILOT_P,pool)
                audit=time.perf_counter()-start
                if not (folder/'timing.json').exists():
                    rc.write_json(folder/'timing.json',{'capture_wall_s':capture,'retention_wall_s':audit})
                plans.append(allocation(row,output,args.workers))
                rc.write_json(output/'pilot_progress.json',{'completed_settings':len(plans),'estimates':plans})
                print(json.dumps(plans[-1]),flush=True)
            # Two preselected dense rays at each diameter; masks remain
            # diagnostic until these full fixed-RK4 comparisons pass.
            gate=output/'validation/diameter_fixed_comparison.json'
            if not gate.exists():
                jobs=[(point,output/'pilot'/point.key/'rays'/f'd{d:03d}_p{p:03d}.npz',prepare_diameter(point.config()))
                      for point in points for d,p in ((0,0),(17,31))]
                checks=list(pool.map(fixed_comparison,jobs))
                rc.write_json(gate,{'passed':all(not c['differences'] for c in checks),'rows':checks})
            assert json.loads(gate.read_text())['passed'], 'diameter fixed-RK4 comparison failed'
            plan={'allocations':plans,'independent_fixed_sample':True,'relative_95_half_width_target':TARGET,
                  'variance_safety_factor':1.5,'estimated_total_hours':[sum(p['estimated_hours'][i] for p in plans) for i in (0,1)]}
            target=output/'sampling_plan.json'
            if target.exists():
                assert json.loads(target.read_text()) == plan
            else:
                rc.write_json(target,plan)
        if args.stage in ('production','all'):
            assert json.loads((output/'validation/diameter_fixed_comparison.json').read_text())['passed']
            plan=json.loads((output/'sampling_plan.json').read_text())
            rows=[]
            for point,a in zip(points,plan['allocations']):
                assert point.key == a['key']
                n,p=a['disc_count'],a['points_per_disc']
                row=ce.run_loading_point(point,'production',n,p,pool)
                ce.duration_audits(point,'production',n,p,pool)
                row['loading_precision_pass']=row['loading_relative_half_width'] <= TARGET
                row['publication_status']='validated' if row['loading_precision_pass'] and row['velocity_quadrature_pass'] else 'precision target not met'
                row['cooling_diameter_mm']=point.coordinate
                row['cooling_power_w_per_beam']=point.cooling_power_w
                density=ce.LOADING_RATE_PREFACTOR/(4/(np.sqrt(np.pi)*ce.THERMAL_SCALE_M2_PER_S2**1.5))
                row['loading_coefficient_m3_per_s']=row['loading_rate_atoms_per_s']/density
                row['loading_coefficient_95_half_width_m3_per_s']=row['loading_95_half_width_atoms_per_s']/density
                rc.write_json(output/'production'/point.key/'summary.json',row)
                rows.append(row)
                rc.write_csv(output/'production_summary.csv',rows)
            ready=all(r['publication_status']=='validated' for r in rows)
            figure=rc.paths()['figures'];figure.mkdir(parents=True,exist_ok=True)
            for coefficient in (False,True):
                y='loading_coefficient_m3_per_s' if coefficient else 'loading_rate_atoms_per_s'
                err='loading_coefficient_95_half_width_m3_per_s' if coefficient else 'loading_95_half_width_atoms_per_s'
                fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
                ax.errorbar(DIAMETERS,[r[y] for r in rows],yerr=[r[err] for r in rows],fmt='o-',capsize=3)
                ax.set(xlabel='Cooling-beam Gaussian 1/e² diameter [mm]',ylabel='Loading coefficient R/n87 [m³/s]' if coefficient else 'Loading rate [atoms/s; historical vapor normalization]',
                    title=f'Population-rate MOT · 15 mm sampling radius · 95% disc-cluster intervals\n{args.power_mode.replace("_"," ")} · detuning {detuning/1e6:g} MHz · repump diameter 12.7 mm')
                ax.grid(alpha=.25)
                if not ready:ax.text(.02,.98,'Precision qualification pending',transform=ax.transAxes,va='top',color='darkred')
                for ext in ('png','pdf'):fig.savefig(figure/f'loading{"_coefficient" if coefficient else ""}_vs_cooling_diameter.{ext}',dpi=180)
                plt.close(fig)
            rc.write_json(output/'completion.json',{'computed':True,'publication_ready':ready,'visual_review_pending':True,
                'remaining_precision_points':[r['cooling_diameter_mm'] for r in rows if r['publication_status']!='validated']})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('validate','pilot','production','all'))
    parser.add_argument('--power-mode',choices=('fixed_power','fixed_intensity'),default='fixed_power')
    parser.add_argument('--detuning',choices=('baseline','peak'),default='baseline')
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as error:
        rc.write_json(rc.paths()['statistics']/'run_failure.json',{'error':repr(error),'traceback':traceback.format_exc()})
        raise
