"""Historical scan coordinates and corrected Section-12 force campaign."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import sleep

import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .configuration import default_multilevel_mot_config, multilevel_mot_paths
from .accelerated import prepare, observable

CAMPAIGN = 'relationships_20260919'
RADII_MM = (3.,5.,8.,12.,15.,20.,25.,30.)
RAW_SATURATION = (.25,.5,.75,1.,2.,3.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,60.,70.,80.,90.,100.,110.,120.,125.)
EFFECTIVE_SATURATION = tuple(.25*i for i in range(1,21))
LOADING_DETUNING = tuple(-.25*i for i in range(2,25))
FORCE_DETUNING = tuple(round(-.5-.05*i,2) for i in range(111))
REFERENCE_GAMMA = 2*np.pi*6.07e6
REFERENCE_ISAT = 16.69
BEAM_RADIUS = .00635


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            sleep(.1)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paths():
    return {k:v/CAMPAIGN for k,v in multilevel_mot_paths().items() if k != 'root'}


def source_digest():
    root=multilevel_mot_paths()['root']
    files=list((root/'src/pmot/mot_multilevel').glob('*.py'))
    files += [root/'src/pmot'/name for name in (
        'configuration.py','magnetic_fields.py','fields.py','beams.py',
        'launch_geometry.py','loading.py')]
    digest=hashlib.sha256()
    for p in sorted(files):
        digest.update(str(p.relative_to(root)).replace('\\','/').encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadingPoint:
    study: str
    index: int
    coordinate: float
    radius_mm: float
    cooling_power_w: float
    detuning_rad_s: float

    @property
    def key(self):
        return f'{self.study}/{self.index:03d}'

    def config(self):
        return replace(default_multilevel_mot_config(),
            cooling_power_w_per_beam=self.cooling_power_w,
            cooling_detuning_rad_per_s=self.detuning_rad_s)


def loading_points():
    det=-2*np.pi*15e6
    area_factor=REFERENCE_ISAT*np.pi*BEAM_RADIUS**2/2
    return ([LoadingPoint('01_radius',i,r,r,.027,det) for i,r in enumerate(RADII_MM)]
        +[LoadingPoint('02_raw_saturation',i,s,15.,s*area_factor,det) for i,s in enumerate(RAW_SATURATION)]
        +[LoadingPoint('03_effective_saturation',i,s,15.,s*(1+4*(det/REFERENCE_GAMMA)**2)*area_factor,det) for i,s in enumerate(EFFECTIVE_SATURATION)]
        +[LoadingPoint('04_detuning',i,d,15.,.027,d*REFERENCE_GAMMA) for i,d in enumerate(LOADING_DETUNING)])


def parabolic_minimum(x,y):
    i=int(np.argmin(y))
    if i in (0,len(x)-1):
        return float(x[i]),False
    h=x[i+1]-x[i]
    curvature=y[i-1]-2*y[i]+y[i+1]
    shift=0.5*h*(y[i-1]-y[i+1])/curvature if curvature > 0 else 0.
    return float(x[i]+np.clip(shift,-h,h)),True


def force_point(index, detuning):
    config=replace(default_multilevel_mot_config(),include_gravity=False,
        cooling_detuning_rad_per_s=detuning*REFERENCE_GAMMA)
    data=prepare(config)
    zero=np.zeros(3)
    axis=np.array([0.,0.,1.])
    row={'index':index,'detuning_over_gamma':detuning,
         'detuning_mhz':detuning*6.07,'cooling_power_w_per_beam':.027}
    curves={}
    for a,label in enumerate('xyz'):
        unit=np.eye(3)[a]
        def slope(h):
            return float((observable(unit*h,zero,axis,data)[0][a]-
                          observable(-unit*h,zero,axis,data)[0][a])/(2*h))
        h=.0001
        coarse=slope(h)
        for _ in range(10):
            h/=2
            fine=slope(h)
            err=abs(coarse-fine)
            if err <= .02*max(abs(fine),1e-28):
                break
            coarse=fine
        else:
            raise RuntimeError(f'restoring derivative not converged: {detuning}, {label}')
        row[f'slope_{label}_n_per_m']=fine
        row[f'slope_{label}_error_n_per_m']=err
        row[f'slope_{label}_step_m']=h
        spacing=.125
        extent=50.
        for _ in range(8):
            velocities=np.arange(0,extent+spacing/2,spacing)
            forces=np.array([observable(zero,unit*v,axis,data)[0][a] for v in velocities])
            vc,ic=parabolic_minimum(velocities[::2],forces[::2])
            vf,iff=parabolic_minimum(velocities,forces)
            if ic and iff and abs(vf-vc) <= .05:
                break
            if not iff:
                extent*=2
            else:
                spacing/=2
        else:
            raise RuntimeError(f'damping extremum not converged: {detuning}, {label}')
        row[f'turnaround_{label}_m_per_s']=vf
        row[f'turnaround_{label}_error_m_per_s']=abs(vf-vc)
        row[f'turnaround_{label}_step_m_per_s']=spacing
        curves[f'velocity_{label}_m_per_s']=velocities
        curves[f'force_{label}_n']=forces
    return row,curves


def run_force_sweep():
    output=paths()['statistics']/'05_force'
    figures=paths()['figures']/'05_force'
    output.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    rows=[]
    digest=source_digest()
    for i,d in enumerate(FORCE_DETUNING):
        file=output/f'point_{i:03d}.json'
        if file.exists():
            row=json.loads(file.read_text())
        else:
            row={}
        if row.get('source_sha256')!=digest:
            row,curves=force_point(i,d)
            row['source_sha256']=digest
            np.savez_compressed(output/f'curves_{i:03d}.npz',**curves)
            write_json(file,row)
        rows.append(row)
        if i % 10 == 0:
            print(f'force {i+1}/111',flush=True)
    write_csv(output/'force_vs_detuning.csv',rows)
    for kind,stem,ylabel,scale in (
        ('slope','restoring_force_slope_vs_detuning','Radiation-force slope dF/dx [10⁻¹⁹ N/m]',1e19),
        ('turnaround','damping_turnaround_vs_detuning','Positive damping turnaround speed [m/s]',1.)):
        fig,ax=plt.subplots(figsize=(9,5.7),layout='constrained')
        for a,marker in zip('xyz',('o','s','^')):
            unit='n_per_m' if kind=='slope' else 'm_per_s'
            ax.errorbar([r['detuning_over_gamma'] for r in rows],
                [r[f'{kind}_{a}_{unit}']*scale for r in rows],
                yerr=[r[f'{kind}_{a}_error_{unit}']*scale for r in rows],
                label=a,fmt=marker+'-',ms=3.5 if a=='y' else 2.5,lw=1,capsize=1.5,
                markerfacecolor='none' if a=='y' else None)
        ax.set(xlabel='Cooling detuning Δ/Γ (Γ/2π = 6.07 MHz)',ylabel=ylabel,
               title='Corrected 24-state population-rate MOT · 27 mW/beam')
        ax.text(.02,.97,'Numerical refinement error bars · gravity excluded',
                transform=ax.transAxes,va='top',fontsize=9)
        ax.grid(alpha=.25);ax.legend(loc='best')
        for extension in ('png','pdf'):
            fig.savefig(figures/f'{stem}.{extension}',dpi=180)
        plt.close(fig)
    write_json(output/'summary.json',{'points':111,'axes':'xyz','converged':True,
        'error_bars':'absolute difference between successive numerical resolutions; not statistical CIs',
        'gravity_in_force':False,'model':'pmot.mot_multilevel Section-12 net stimulated force'})
    return rows


def cluster_interval(values):
    values=np.asarray(values)
    n=len(values)
    if n<2:
        raise ValueError('at least two independent direction discs required')
    mean=np.mean(values,axis=0)
    half=student_t.ppf(.975,n-1)*np.std(values,axis=0,ddof=1)/np.sqrt(n)
    return mean,half
