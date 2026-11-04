"""Six axial-gradient loading points at -21 MHz using the corrected MOT kernel.

Independent pilots freeze a common production allocation. Every positive-speed
node uses the established dual-setting, fixed-RK4-edge and retention policy.
Run from the repository root: uv run python scripts/run_multilevel_gradient_campaign.py all
"""
from pathlib import Path
import os
import hashlib

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = 'axial_gradient_5_10_15_20_25_30Gpcm_minus21MHz_20261016'
core_files = sorted(list((ROOT/'src/pmot/mot_multilevel').glob('*.py')) +
    [ROOT/'src/pmot'/n for n in ('configuration.py', 'magnetic_fields.py',
      'fields.py', 'beams.py', 'launch_geometry.py', 'loading.py')])
digest = hashlib.sha256()
for file in core_files:
    digest.update(file.relative_to(ROOT).as_posix().encode())
    digest.update(file.read_bytes())
CORE_SHA = digest.hexdigest()
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'outputs/compiled_cache'/CORE_SHA)
os.environ['MPLBACKEND'] = 'Agg'
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import datetime
import json
import math
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from importlib.metadata import version

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from pmot.configuration import HBAR_J_S
from pmot.magnetic_fields import default_anti_helmholtz_config, anti_helmholtz_field_t
from pmot.mot_multilevel import relationship_campaign as rc
from pmot.mot_multilevel import campaign_execution as ce
from pmot.mot_multilevel.accelerated import prepare, observable, magnetic_field
from pmot.mot_multilevel.configuration import MultilevelMOTConfig
from pmot.mot_multilevel.rate_equations import build_rate_equation_model, rate_equation_observable
from pmot.mot_multilevel.simulation import build_multilevel_mot_beams
from pmot.mot_multilevel.campaign_validation import event_occupations

GRADIENTS = (5., 10., 15., 20., 25., 30.)
PILOT_N, POINTS = 32, 20
PILOT_SEED, PRODUCTION_SEED = 2026101601, 2026101602
TARGET = .10
ORIGINAL_RAY_WORKER = ce._ray_worker


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass(frozen=True)
class GradientConfig(MultilevelMOTConfig):
    axial_gradient_g_per_cm: float = 10.


@dataclass(frozen=True)
class GradientPoint:
    index: int
    coordinate: float
    study: str = '01_axial_gradient'
    radius_mm: float = 15.
    cooling_power_w: float = .027
    detuning_rad_s: float = -2*np.pi*21e6

    @property
    def key(self):
        return f'{self.study}/{self.index:03d}'

    def config(self):
        return GradientConfig(cooling_detuning_rad_per_s=self.detuning_rad_s,
            cooling_power_w_per_beam=self.cooling_power_w,
            repump_power_w_per_beam=.0001, repump_detuning_rad_per_s=0.,
            axial_gradient_g_per_cm=self.coordinate)


def points():
    return [GradientPoint(i, g) for i, g in enumerate(GRADIENTS)]


def prepare_gradient(cfg):
    return prepare(cfg, default_anti_helmholtz_config(target_gradient_g_per_cm=cfg.axial_gradient_g_per_cm))


def gradient_worker(payload):
    # The shared cache key predates gradient scans and contains only P and Delta.
    # Every task already carries immutable parent-prepared arrays: never reuse
    # another gradient's controls, including after spawned-worker task reuse.
    ce._DATA_CACHE.clear()
    return ORIGINAL_RAY_WORKER(payload)


@contextmanager
def campaign_lock(root):
    with (root/'runner.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def initialize():
    rc.CAMPAIGN = CAMPAIGN
    ce.PILOT_SEED, ce.PRODUCTION_SEED = PILOT_SEED, PRODUCTION_SEED
    ce.prepare, ce._ray_worker = prepare_gradient, gradient_worker
    root = rc.paths()['statistics']; root.mkdir(parents=True, exist_ok=True)
    assert rc.source_digest() == CORE_SHA
    cfg = points()[0].config()
    beams = build_multilevel_mot_beams(config=cfg)
    manifest = dict(schema=1, campaign=CAMPAIGN, source_sha256=CORE_SHA,
        runner_sha256=sha(__file__), lockfile_sha256=sha(ROOT/'uv.lock'),
        gradients_g_per_cm=GRADIENTS, cooling_detuning_mhz=-21.,
        cooling_power_mw_per_beam=27., repump_power_mw_per_beam=.1,
        repump_detuning_mhz=0., beam_diameter_mm=12.7, sampling_radius_mm=15.,
        pilot_discs=PILOT_N, points_per_disc=POINTS, pilot_seed=PILOT_SEED,
        production_seed=PRODUCTION_SEED, relative_95_half_width_target=TARGET,
        allocation='Independent pilots; fixed common N >=100 rounded to 25 discs; 1.5 variance safety factor; no optional stopping',
        velocities_m_per_s=ce.VELOCITIES.tolist(), tail_probes_m_per_s=ce.TAIL_VELOCITIES.tolist(),
        fixed_ladder=ce.LADDER, normalization=dict(prefactor=ce.LOADING_RATE_PREFACTOR,
            thermal_scale_m2_per_s2=ce.THERMAL_SCALE_M2_PER_S2),
        python=platform.python_version(), packages={p:version(p) for p in ('numpy','scipy','numba','arc-alkali-rydberg-calculator')},
        model='Corrected Section-12 local-steady-state 24-state mean force; gravity on; recoil diffusion off',
        limitations='Instantaneous populations not validated in weak wings; pointwise statistical intervals exclude physical-model and vapor uncertainty')
    manifest = json.loads(json.dumps(manifest))
    file = root/'campaign_manifest.json'
    if file.exists():
        assert read(file) == manifest, 'Manifest mismatch: recorded revision and revalidation required'
    else:
        rc.write_json(file, manifest)
        rc.write_json(root/'parameters.json', dict(configurations=[asdict(p.config()) for p in points()],
            beams=[asdict(b) for b in beams], coils=[asdict(default_anti_helmholtz_config(target_gradient_g_per_cm=g)) for g in GRADIENTS],
            launch_plane_distance_m=.015, disc_radius_m=.015,
            sampling='full-sphere uniform directions, uniform-area offsets, parallel incident velocities',
            trapping='2 mm core: continuous 5 ms residence or two entries with intervening exit',
            retention='outermost captured ray per disc at highest and median captured speed; final 25 ms inside core; dual-step endpoints within 10 micrometers',
            zero_speed='No sigma(0) inferred; only exact zero weighted-flux anchor'))
        for file in core_files + [Path(__file__)]:
            target = root/'source_snapshot'/file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
    for p in points():
        file = root/'parameters'/f'{p.index:03d}_compiled_inputs.npz'
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file, **{f'array_{i}':a for i,a in enumerate(prepare_gradient(p.config()))})
    return root


def validate_local(root):
    path = root/'validation/local_force_and_events.json'
    if path.exists():
        assert read(path)['passed']; return
    model = build_rate_equation_model()
    rng = np.random.default_rng(2026101603)
    checks, events = [], []
    for p in points():
        cfg = p.config(); data = prepare_gradient(cfg)
        beams = build_multilevel_mot_beams(config=cfg)
        coil = default_anti_helmholtz_config(target_gradient_g_per_cm=p.coordinate)
        assert len(beams)==12 and all(np.isclose(2*b.beam_radius_m,.0127) for b in beams)
        assert all(np.isclose(b.detuning_hz,-21e6) and np.isclose(b.power_w,.027) for b in beams if b.family=='cooling')
        assert all(b.detuning_hz==0. and np.isclose(b.power_w,.0001) for b in beams if b.family=='repump')
        h = 1e-6
        gradient = abs((magnetic_field(np.array([0.,0.,h]),data[-1])[2]-magnetic_field(np.array([0.,0.,-h]),data[-1])[2])/(2*h))
        np.testing.assert_allclose(gradient, p.coordinate*.01, rtol=1e-7)
        for r in np.vstack([np.zeros(3),np.eye(3)*.001,rng.uniform(-.02,.02,(12,3))]):
            v = rng.uniform(-35,35,3)
            np.testing.assert_allclose(magnetic_field(r,data[-1]),anti_helmholtz_field_t(*r,coil),rtol=1e-8,atol=2e-16)
            fast = observable(r,v,np.array([0.,0.,1.]),data)
            ref = rate_equation_observable(model,beams,tuple(r),tuple(v),coil,cfg)
            np.testing.assert_allclose(fast[0],ref.force_n,rtol=3e-6,atol=3e-29)
            np.testing.assert_allclose(fast[1],ref.populations,rtol=4e-6,atol=4e-10)
        r = (.006,.002,-.001); v = (-8.,.2,0.)
        ref = rate_equation_observable(model,beams,r,v,coil,cfg,
            store_rate_matrix=True,store_beam_transition_quantities=True)
        estimates = event_occupations(ref.rate_matrix_per_s,ref.populations,.002,64,2026101610+p.index)
        sem = estimates.std(axis=0,ddof=1)/8
        pop_pass = bool(np.all(abs(estimates.mean(axis=0)-ref.populations)<=6*sem+1e-6))
        reward = np.zeros((24,3)); matrices = ref.beam_transition_quantities.stimulated_coefficients_per_s
        for b,beam in enumerate(beams):
            momentum = HBAR_J_S*2*np.pi/beam.wavelength_m*np.asarray(beam.direction)
            reward[:8] += matrices[b].sum(axis=0)[:,None]*momentum
            reward[8:] -= matrices[b].sum(axis=1)[:,None]*momentum
        force = estimates@reward; fsem=force.std(axis=0,ddof=1)/8
        force_pass=bool(np.all(abs(force.mean(axis=0)-ref.force_n)<=6*fsem+1e-29))
        events.append(dict(gradient_g_per_cm=p.coordinate,position_m=r,velocity_m_per_s=v,
            population_pass=pop_pass,force_pass=force_pass,reference_population=ref.populations.tolist(),
            event_population=estimates.mean(axis=0).tolist(),population_se=sem.tolist(),
            reference_force_n=list(ref.force_n),event_force_n=force.mean(axis=0).tolist(),force_se_n=fsem.tolist()))
        checks.append(dict(gradient_g_per_cm=p.coordinate,local_checks=16,measured_gradient_t_per_m=gradient))
    passed=all(e['population_pass'] and e['force_pass'] for e in events)
    rc.write_json(path,dict(passed=passed,source_sha256=CORE_SHA,checks=checks,events=events,
        limits='Frozen-point equilibrium event comparisons; not validation of internal population adiabaticity along trajectories'))
    assert passed, 'Local event comparison failed'


def fresh_check(payload):
    p,d,j,r,direction,speed,data = payload
    records, methods = [], []
    a=ce._resolve_node(r,direction,speed,data,records,methods)
    b=ce._resolve_fixed_node(r,direction,speed,data,records,methods)
    return dict(gradient_g_per_cm=p.coordinate,disc=d,point=j,speed=speed,
        passed=a==b and b>=0,adaptive=a,fixed=b,evaluations=records,methods=methods)


def fixed_comparison(payload):
    p,file,data,target = payload
    with np.load(file) as saved:
        speeds,codes=saved['speeds'],saved['codes'];r,direction=saved['position_m'],saved['direction']
    records,methods,differences=[],[],[]
    for speed,expected in zip(speeds,codes):
        actual=ce._resolve_fixed_node(r,direction,speed,data,records,methods)
        if actual!=expected:
            differences.append(dict(speed=float(speed),hybrid=int(expected),fixed=actual))
    ce._atomic_npz(target,evaluations=np.asarray(records),evaluation_methods=np.asarray(methods),position_m=r,direction=direction)
    return dict(gradient_g_per_cm=p.coordinate,ray=Path(file).name,nodes=len(speeds),differences=differences)


def run_point(p,mode,n,root,pool):
    folder=root/mode/p.key
    started=time.perf_counter()
    row=ce.run_loading_point(p,mode,n,POINTS,pool)
    capture=time.perf_counter()-started; started=time.perf_counter()
    audit=ce.duration_audits(p,mode,n,POINTS,pool)
    if not (folder/'timing.json').exists():
        rc.write_json(folder/'timing.json',dict(capture_wall_s=capture,retention_wall_s=time.perf_counter()-started))
    assert row['velocity_quadrature_pass'], 'Quadrature gate failed'
    row['retention_pass']=audit['passed']
    return row


def finalize(root,plan,rows):
    figures=rc.paths()['figures']; figures.mkdir(parents=True,exist_ok=True)
    all_ready=all(r['publication_status']=='numerically qualified' for r in rows)
    for coefficient in (False,True):
        y='loading_coefficient_m3_per_s' if coefficient else 'loading_rate_atoms_per_s'
        e='loading_coefficient_95_half_width_m3_per_s' if coefficient else 'loading_95_half_width_atoms_per_s'
        fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
        ax.errorbar(GRADIENTS,[r[y] for r in rows],yerr=[r[e] for r in rows],fmt='o-',capsize=4)
        ax.set(xlabel='Axial magnetic-field gradient [G/cm]',
            ylabel='Loading coefficient R/n87 [m³/s]' if coefficient else 'Loading rate [atoms/s; historical vapor normalization]',
            title='Corrected population-rate MOT · −21 MHz · 12.7 mm beams\n27 mW cooling / 0.1 mW repump · 95% disc-cluster intervals')
        ax.set_xticks(GRADIENTS); ax.grid(alpha=.25)
        if not all_ready:
            ax.text(.02,.98,'Precision target not met at all points',transform=ax.transAxes,va='top',color='darkred')
        name='loading_coefficient_vs_axial_gradient' if coefficient else 'loading_rate_vs_axial_gradient'
        for ext in ('png','pdf'):fig.savefig(figures/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    evidence=[]
    for p,row in zip(points(),rows):
        folder=root/'production'/p.key
        with np.load(folder/'disc_statistics.npz') as saved:
            v=saved['velocity_m_per_s'];sigma=saved['disc_cross_section_m2']
            weighted=sigma*v**3*np.exp(-v*v/ce.THERMAL_SCALE_M2_PER_S2)
            values=ce.LOADING_RATE_PREFACTOR*np.trapezoid(np.column_stack([np.zeros(len(sigma)),weighted]),np.r_[0.,v],axis=1)
            np.testing.assert_allclose(values,saved['disc_loading_atoms_per_s'],rtol=1e-12)
            np.testing.assert_allclose(rc.cluster_interval(values),[row['loading_rate_atoms_per_s'],row['loading_95_half_width_atoms_per_s']],rtol=1e-12)
        assert len(list((folder/'rays').glob('*.npz')))==row['sample_count']
        audit=read(folder/'duration_audit.json')
        assert audit['passed'] and all(c['passed'] for c in audit['cases'])
        evidence.append(dict(key=p.key,retention_cases=len(audit['cases']),
            files={name:sha(folder/name) for name in ('summary.json','disc_statistics.npz','duration_audit.json','configuration.json')}))
    rc.write_json(root/'aggregate_audit.json',dict(passed=True,evidence=evidence))
    rc.write_json(root/'completion.json',dict(computed=True,all_numerical_gates_passed=True,
        precision_passed=all_ready,visual_review_pending=True,finished_at_utc=utc(),
        issues=[r['gradient_g_per_cm'] for r in rows if r['publication_status']!='numerically qualified']))
    lines=['# Axial-gradient loading sweep at -21 MHz','',
        'Corrected 24-state local-steady-state MOT. Six cooling and six repump traveling beams; '
        '27 mW and 0.1 mW per component respectively, 12.7 mm Gaussian 1/e² diameter, resonant repump, gravity enabled. '
        'Full-sphere uniform-area sampling discs have radius and launch distance 15 mm. '
        'Powers, detunings and geometry are fixed; only coil current/axial gradient changes.','',
        '| Axial gradient (G/cm) | Loading (atoms/s) | 95% half-width | Discs x points | Capture + audit hours |',
        '|---:|---:|---:|---:|---:|']
    for p,row in zip(points(),rows):
        timing=read(root/'production'/p.key/'timing.json')
        hours=(timing['capture_wall_s']+timing['retention_wall_s'])/3600
        lines.append(f"| {p.coordinate:g} | {row['loading_rate_atoms_per_s']:.8g} | {row['loading_95_half_width_atoms_per_s']:.6g} | {row['disc_count']} x {POINTS} | {hours:.3f} |")
    lines += ['', 'Intervals are pointwise Student-t direction-disc cluster intervals, not simultaneous bands. '
        'Independent pilots determined a fixed common production allocation. Production was not repeatedly '
        'extended until an interval target passed. The target is a 10% relative half-width. '
        'All numerical, retention and positive-speed checks passed; actual precision status is in completion.json.', '',
        'The historical vapor normalization is unchanged. Density-independent coefficients and their intervals '
        'are in production_summary.csv. Numerical quadrature differences are reported separately from sampling '
        'uncertainty. All nonmonotone velocity masks are retained; timeouts are not escapes. '
        'Source, parameters, pilot sizing, timestep refinements, tail tests and retained-node audits are saved.', '',
        'The local-steady-state assumption remains a limitation in weak optical wings. Frozen-point event '
        'comparisons do not establish internal-population equilibration along moving trajectories. '
        'These intervals exclude model, apparatus and vapor-density uncertainty; no temperature is inferred.', '',
        '![Loading versus gradient](../../../figures/mot_multilevel_population_rate_v1/'+CAMPAIGN+'/loading_rate_vs_axial_gradient.png)', '']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def run(args,root):
    rc.write_json(root/'process.json',dict(pid=os.getpid(),command=sys.argv,workers=args.workers,started_at_utc=utc()))
    validate_local(root)
    if args.stage=='validate':return
    certificate=read(root/'validation/test_status.json')
    assert certificate['passed'] and certificate['source_sha256']==CORE_SHA
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        file=root/'validation/fresh_workers.json'
        if not file.exists():
            pos,dirs=ce.geometry(PILOT_SEED,PILOT_N,POINTS,15.)
            jobs=[(p,d,j,pos[d,j],dirs[d],v,prepare_gradient(p.config()))
                for p in points() for d,j in ((0,0),(17,13)) for v in (.25,10.,40.,120.)]
            checks=list(pool.map(fresh_check,jobs))
            rc.write_json(file,dict(passed=all(c['passed'] for c in checks),checks=checks))
        assert read(file)['passed'], 'Fresh-worker numerical gate failed'
        planfile=root/'sampling_plan.json'
        if not planfile.exists():
            sizing=[]
            for p in points():
                row=run_point(p,'pilot',PILOT_N,root,pool)
                with np.load(root/'pilot'/p.key/'disc_statistics.npz') as saved:
                    samples=saved['disc_loading_atoms_per_s']
                mean=float(samples.mean()); variance=float(samples.var(ddof=1))
                if mean<=0:raise RuntimeError('Zero pilot loading: independent pilot extension required')
                n=100
                while 1.5*t.ppf(.975,n-1)**2*variance/(TARGET*mean)**2>n:
                    n+=25
                    if n>2000:raise RuntimeError('Pilot implies >2000 discs; review before production')
                sizing.append(dict(key=p.key,gradient_g_per_cm=p.coordinate,required_discs=n,
                    pilot_mean=mean,pilot_disc_variance=variance))
                rc.write_json(root/'pilot_progress.json',dict(completed=len(sizing),sizing=sizing))
            fixedfile=root/'validation/full_fixed_comparison.json'
            if not fixedfile.exists():
                jobs=[(p,root/'pilot'/p.key/'rays'/f'd{d:03d}_p{j:03d}.npz',prepare_gradient(p.config()),
                    root/'validation/fixed_comparisons'/p.key/f'd{d:03d}_p{j:03d}.npz')
                    for p in points() for d,j in ((0,0),(17,13))]
                checks=list(pool.map(fixed_comparison,jobs))
                rc.write_json(fixedfile,dict(passed=all(not c['differences'] for c in checks),checks=checks))
            assert read(fixedfile)['passed'], 'Full fixed-RK4 comparison failed'
            common_n=max(s['required_discs'] for s in sizing)
            for s,p in zip(sizing,points()):
                timing=read(root/'pilot'/p.key/'timing.json')
                hours=(timing['capture_wall_s']+timing['retention_wall_s'])*common_n/PILOT_N/3600
                s.update(disc_count=common_n,points_per_disc=POINTS,rays=common_n*POINTS,
                    estimated_hours=[1.25*hours,1.75*hours])
            rc.write_json(planfile,dict(frozen_at_utc=utc(),allocations=sizing,independent_of_pilot=True,
                common_geometry=True,relative_95_half_width_target=TARGET,
                estimated_total_hours=[sum(s['estimated_hours'][i] for s in sizing) for i in (0,1)]))
            print('FROZEN PLAN '+json.dumps(read(planfile)),flush=True)
        plan=read(planfile)
        assert read(root/'validation/full_fixed_comparison.json')['passed']
        if args.stage=='pilot':return
        rows=[]
        for p,a in zip(points(),plan['allocations']):
            assert p.key==a['key']
            row=run_point(p,'production',a['disc_count'],root,pool)
            density=ce.LOADING_RATE_PREFACTOR/(4/(np.sqrt(np.pi)*ce.THERMAL_SCALE_M2_PER_S2**1.5))
            precision=row['loading_relative_half_width'] is not None and row['loading_relative_half_width']<=TARGET
            row.update(gradient_g_per_cm=p.coordinate,loading_precision_pass=precision,
                coil_current_a=default_anti_helmholtz_config(target_gradient_g_per_cm=p.coordinate).current_a,
                loading_coefficient_m3_per_s=row['loading_rate_atoms_per_s']/density,
                loading_coefficient_95_half_width_m3_per_s=row['loading_95_half_width_atoms_per_s']/density,
                publication_status='numerically qualified' if precision else 'precision target not met')
            rc.write_json(root/'production'/p.key/'summary.json',row)
            rows.append(row);rc.write_csv(root/'production_summary.csv',rows)
            print('COMPLETE POINT '+json.dumps(row),flush=True)
        finalize(root,plan,rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('validate','pilot','all'))
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    rc.CAMPAIGN=CAMPAIGN
    root=rc.paths()['statistics'];root.mkdir(parents=True,exist_ok=True)
    with campaign_lock(root):
        try:
            root=initialize()
            run(args,root)
        except Exception as error:
            rc.write_json(root/'failure.json',dict(at_utc=utc(),error=repr(error),traceback=traceback.format_exc()))
            raise
