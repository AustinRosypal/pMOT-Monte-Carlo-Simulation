"""Exact inputs and saved launch geometry for the matched two-level benchmark.

The physical population-rate solver remains owned by ``mot_multilevel``.
Only benchmark data and configuration metadata are read from ``mot_simple``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..comparison_paths import COMPARISON_FIGURES as COMPARISON_FIGURE_ROOT
from ..comparison_paths import COMPARISON_NOTEBOOKS, COMPARISON_NOTEBOOK_NAMES

from ..configuration import default_mot_apparatus_config
from ..magnetic_fields import default_anti_helmholtz_config
from .accelerated import prepare
from .configuration import default_multilevel_mot_config
from .simulation import build_multilevel_mot_beams
from .rate_equations import build_rate_equation_model


ROOT=Path(__file__).resolve().parents[3]
NAME="matched_two_level_exact_D2_20261104"
BENCHMARK=ROOT/"outputs/statistics/mot_simple/loading_scans_exact_D2_20261103"
STATISTICS=ROOT/"outputs/statistics/mot_multilevel_population_rate_v1"/NAME
FIGURES=ROOT/"outputs/figures/mot_multilevel_population_rate_v1"/NAME
COMPARISON_FIGURES=COMPARISON_FIGURE_ROOT/NAME
CACHE=Path(os.environ.get("LOCALAPPDATA",tempfile.gettempdir()))/"pmot-campaigns"/NAME
NOTEBOOK=ROOT/"notebooks/mot_multilevel"/f"{NAME}.ipynb"
COMPARISON_NOTEBOOK=COMPARISON_NOTEBOOKS/COMPARISON_NOTEBOOK_NAMES["loading_two"]
WAVELENGTH_M=780.241209686e-9
REPUMP_WAVELENGTH_M=780.232684e-9
VELOCITIES=np.arange(0.,80.0001,.25)
BASELINE=(10.,-15.,12.7,20.,.1,12.7,0.)
REPUMP_POWERS_MW=(.1,.3,.5,.7,.9,1.1,1.3,1.5,1.7,2.,4.,6.,8.,10.,12.,14.,16.,18.,20.)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    replace_file(temp,path)


def replace_file(source,target):
    from time import sleep
    for attempt in range(100):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt==99:raise
            sleep(.1)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MatchedPoint:
    index:int
    cooling_power_mw:float=10.
    cooling_detuning_mhz:float=-15.
    cooling_diameter_mm:float=12.7
    gradient_g_per_cm:float=20.
    repump_power_mw:float=.1
    repump_diameter_mm:float=12.7
    repump_detuning_mhz:float=0.
    benchmark_index:int|None=None

    @property
    def key(self):
        return f"configuration_{self.index:03d}"

    @property
    def physical_values(self):
        return tuple(asdict(self)[key] for key in (
            "cooling_power_mw","cooling_detuning_mhz","cooling_diameter_mm",
            "gradient_g_per_cm","repump_power_mw","repump_diameter_mm","repump_detuning_mhz"))


def definitions(*,repump_diameters=None,repump_powers=None):
    original=read_json(BENCHMARK/"campaign_manifest.json")
    points=[]
    for cfg in original["configurations"]:
        points.append(MatchedPoint(cfg["index"],cfg["power_mw_per_beam"],cfg["detuning_mhz"],
                     cfg["diameter_mm"],cfg["gradient_g_per_cm"],benchmark_index=cfg["index"]))
    scans={"cooling_"+family:list(ids) for family,ids in original["scan_configuration_indices"].items()}
    repump_diameters=tuple(repump_diameters or original["scan_grids"]["diameter"])
    powers=tuple(repump_powers or REPUMP_POWERS_MW)
    specifications=(
        ("cooling_diameter_matched_repump",original["scan_grids"]["diameter"],2),
        ("repump_power",powers,4),
        ("repump_diameter",repump_diameters,5),
        ("repump_detuning",tuple(float(x) for x in range(-10,11)),6),
    )
    for family,values,dimension in specifications:
        ids=[]
        for value in values:
            physical=list(BASELINE);physical[dimension]=float(value)
            if family=="cooling_diameter_matched_repump":physical[5]=float(value)
            found=next((p.index for p in points if p.physical_values==tuple(physical)),None)
            if found is None:
                found=len(points)
                points.append(MatchedPoint(found,*physical))
            ids.append(found)
        scans[family]=ids
    return tuple(points),scans


def configuration(point):
    cfg=replace(default_multilevel_mot_config(),
                cooling_power_w_per_beam=point.cooling_power_mw*.001,
                cooling_detuning_rad_per_s=2*np.pi*point.cooling_detuning_mhz*1e6,
                cooling_wavelength_m=WAVELENGTH_M,
                repump_power_w_per_beam=point.repump_power_mw*.001,
                repump_detuning_rad_per_s=2*np.pi*point.repump_detuning_mhz*1e6,
                repump_wavelength_m=REPUMP_WAVELENGTH_M,include_gravity=True)
    app=default_mot_apparatus_config()
    app=replace(app,cooling=replace(app.cooling,beam_diameter_m=point.cooling_diameter_mm*.001),
                repump=replace(app.repump,beam_diameter_m=point.repump_diameter_mm*.001))
    beams=build_multilevel_mot_beams(app,cfg)
    # Apply both radii explicitly so this campaign does not depend on historical
    # builders that derived the repump envelope from the cooling diameter.
    beams=[replace(b,beam_radius_m=(point.cooling_diameter_mm if b.family=="cooling" else point.repump_diameter_mm)*.0005)
           for b in beams]
    coil=default_anti_helmholtz_config(target_gradient_g_per_cm=point.gradient_g_per_cm)
    assert len(beams)==12
    assert all(b.wavelength_m==WAVELENGTH_M for b in beams if b.family=="cooling")
    return cfg,app,coil,beams


def prepared(point):
    cfg,app,coil,beams=configuration(point)
    data=list(prepare(cfg,coil));data[1]=data[1].copy()
    for j,beam in enumerate(beams):
        data[1][j,0]=beam.power_w
        data[1][j,1]=beam.beam_radius_m**2
        data[1][j,2]=(np.pi*beam.beam_radius_m**2/beam.wavelength_m)**2
        data[1][j,3]=2*np.pi/beam.wavelength_m
    return tuple(data)


def load_master_geometry():
    path=BENCHMARK/"launch_geometry_master.csv"
    with path.open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
    discs=max(int(row["disc_index"]) for row in rows)+1
    npoints=max(int(row["point_index"]) for row in rows)+1
    positions=np.empty((discs,npoints,3));directions=np.empty_like(positions)
    for row in rows:
        d,p=int(row["disc_index"]),int(row["point_index"])
        positions[d,p]=[float(row[key]) for key in ("x0_m","y0_m","z0_m")]
        directions[d,p]=[float(row[key]) for key in ("vx_hat","vy_hat","vz_hat")]
    assert positions.shape==(901,50,3)
    np.testing.assert_allclose(np.linalg.norm(directions,axis=2),1.,atol=2e-15)
    assert np.all(directions==directions[:,0:1,:])
    assert np.all(np.linalg.norm(positions,axis=2)<=np.sqrt(2)*.015+1e-14)
    return positions,directions,sha256(path)


def sample_count(point):
    if point.benchmark_index is not None:
        saved=read_json(BENCHMARK/"production_plan.json")["searches"][str(point.benchmark_index)]
        return saved["disc_count"],saved["points_per_disc"]
    # A coupled cooling/repump diameter point uses exactly the same launch
    # sample as its fixed-repump counterpart and the original two-level point.
    if point.cooling_diameter_mm!=12.7:
        original=read_json(BENCHMARK/"campaign_manifest.json")["configurations"]
        index=next(p["index"] for p in original if p["diameter_mm"]==point.cooling_diameter_mm)
        saved=read_json(BENCHMARK/"production_plan.json")["searches"][str(index)]
        return saved["disc_count"],saved["points_per_disc"]
    return 901,50


def source_hashes():
    files=[path for path in Path(__file__).parent.glob("*.py")]
    files += [ROOT/"src/pmot"/name for name in ("configuration.py","cartesian_light.py","fields.py","beams.py",
                "magnetic_fields.py","state.py","loading.py","capture_statistics.py","launch_geometry.py")]
    return {str(path.relative_to(ROOT)).replace("\\","/"):sha256(path) for path in sorted(files)}


def model_notes():
    model=build_rate_equation_model()
    return dict(model="physical 24-state Section-12 local steady-state population-rate MOT",
                stimulated_coefficient="Gamma_e*|Omega_ge|^2/(Gamma_e^2+4*Delta_ge^2)",
                force="sum_b hbar*k_b*sum_ge W_ge^(b)*(p_g-p_e)",
                cooling_reference="F=2 -> F'=3",repump_reference="F=1 -> F'=2",
                cooling_wavelength_m=WAVELENGTH_M,repump_wavelength_m=REPUMP_WAVELENGTH_M,
                cooling_reference_frequency_hz=model.structure.cooling_reference_angular_frequency_rad_per_s/(2*np.pi),
                repump_reference_frequency_hz=model.structure.repump_reference_angular_frequency_rad_per_s/(2*np.pi),
                state_decay_linewidth_mhz_range=[float(model.excited_decay_rates_per_s.min()/(2*np.pi*1e6)),float(model.excited_decay_rates_per_s.max()/(2*np.pi*1e6))],
                state_decay_linewidths_mhz=(model.excited_decay_rates_per_s/(2*np.pi*1e6)).tolist(),
                benchmark_saturation_axis_linewidth_hz=6.065e6,
                benchmark_saturation_intensity_w_per_m2=16.69,
                polarization="propagation-frame (x,y,z)=(sigma+,sigma+,sigma-) in both directions for cooling and repump; physical realization of the two-level effective restoring signs",
                atomic_model_differences="Resolved hyperfine/Zeeman levels, dipoles, state populations, local polarization decomposition, ARC-derived pairwise decay and Landé factors; the narrow ARC decay-rate differences are retained as atomic physics, with benchmark axes unchanged",
                approximation="Instantaneous local steady-state populations; incoherent beams; no recoil diffusion or temperature predictions. Weak-wing population-relaxation limitations must be reported.")
