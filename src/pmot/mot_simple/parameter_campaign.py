"""Independent, resumable November 2026 two-level loading parameter scans.

Run with ``uv run python -m pmot.mot_simple.parameter_campaign``.
Pilot data select a fixed production sample size and are excluded from its
Student-t intervals. All five figures use four independent parameter sweeps.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import gzip
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone, timedelta
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from ..loading import VaporLoadingParameters, loading_integrand
from ..magnetic_fields import default_anti_helmholtz_config
from . import refined_relationship_campaign as engine
from .configuration import default_simple_mot_apparatus, default_simple_mot_config
from .campaign_io import replace_with_retry
from .power_loading_study import StudyPaths, geometry_csv_text, geometry_rows
from .simulation import build_simple_mot_beams
from .timeout_audit import capture_predicate_with_overrides, velocity_overrides_from_payload


ROOT = Path(__file__).resolve().parents[3]
NAME = "loading_scans_exact_D2_20261103"
STATISTICS = ROOT / "outputs" / "statistics" / "mot_simple" / NAME
FIGURES = ROOT / "outputs" / "figures" / "mot_simple" / NAME
NOTEBOOK = ROOT / "notebooks" / "mot_simple" / f"{NAME}.ipynb"
HTML = FIGURES / "loading_parameter_scans.html"
CACHE = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "pmot-campaigns" / NAME
WAVELENGTH_M = 780.241209686e-9
BASELINE = (10., -15., 12.7, 20.)  # mW per beam, MHz, mm, G/cm
GRIDS = {
    "power": tuple(float(x) for x in range(2,41,2)),
    "detuning": tuple(-5.-2.5*x for x in range(15)),
    "diameter": tuple(sorted({5.+2.5*x for x in range(11)} | {12.7})),
    "gradient": tuple(float(x) for x in range(2,41,2)),
}
FIELDS = {"power":0, "detuning":1, "diameter":2, "gradient":3}
CONFIGURATIONS = [BASELINE]
SCAN_INDICES = {}
for _family, _values in GRIDS.items():
    SCAN_INDICES[_family] = []
    for _value in _values:
        _config = list(BASELINE)
        _config[FIELDS[_family]] = _value
        _config = tuple(_config)
        if _config not in CONFIGURATIONS:
            CONFIGURATIONS.append(_config)
        SCAN_INDICES[_family].append(CONFIGURATIONS.index(_config))

VIEWS = (
    ("saturation", "power", "s0", "Single-beam center saturation s₀", "Loading rate vs raw saturation", "#6096ba", "#b9d7ea"),
    ("effective_saturation", "power", "seff", "Effective single-beam center saturation s_eff", "Loading rate vs effective saturation", "#588157", "#c8ddbd"),
    ("detuning", "detuning", "detuning_mhz", "Cooling detuning [MHz]", "Loading rate vs detuning", "#9575a6", "#d9c6e3"),
    ("diameter", "diameter", "diameter_mm", "Gaussian 1/e² intensity diameter [mm]", "Loading rate vs beam diameter", "#bd8057", "#eed0b6"),
    ("gradient", "gradient", "gradient_g_per_cm", "Axial gradient magnitude [G/cm]", "Loading rate vs magnetic gradient", "#ba7286", "#ebc9d4"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def config_record(index):
    power, detuning, diameter, gradient = CONFIGURATIONS[index]
    s0 = engine.on_resonance_saturation_parameter(power*1e-3, beam_diameter_m=diameter*1e-3)
    seff = engine.effective_saturation_from_s0(s0, detuning*1e6/default_simple_mot_config().linewidth_hz)
    return dict(index=index, power_mw_per_beam=power, detuning_mhz=detuning,
                diameter_mm=diameter, gradient_g_per_cm=gradient, s0=s0, seff=seff)


def point_for(index):
    cfg = config_record(index)
    return engine.RelationshipPoint(
        "configurations", index, "configuration", float(index),
        cfg["power_mw_per_beam"]*1e-3,
        cfg["detuning_mhz"]*1e6/default_simple_mot_config().linewidth_hz,
        cfg["detuning_mhz"]*1e6, cfg["s0"], cfg["seff"],
    )


def build_configuration(point):
    cfg = config_record(point.point_index)
    apparatus = default_simple_mot_apparatus()
    apparatus = replace(apparatus, cooling=replace(
        apparatus.cooling, wavelength_m=WAVELENGTH_M,
        power_w_per_beam=cfg["power_mw_per_beam"]*1e-3,
        detuning_hz=cfg["detuning_mhz"]*1e6,
        beam_diameter_m=cfg["diameter_mm"]*1e-3,
    ))
    simple = replace(default_simple_mot_config(), cooling_detuning_hz=cfg["detuning_mhz"]*1e6)
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=cfg["gradient_g_per_cm"])
    return apparatus, simple, coil, build_simple_mot_beams(apparatus, simple)


def paths_for(index, stage="production", *, working=False):
    statistics = CACHE/"statistics" if working else STATISTICS
    figures = CACHE/"figures" if working else FIGURES
    return StudyPaths(statistics/stage/point_for(index).slug, figures/stage/point_for(index).slug)


def publish_point(index, stage, source):
    """Publish a finished point once; keep large live audit writes off sync drives."""
    destination=paths_for(index,stage)
    destination.statistics.mkdir(parents=True,exist_ok=True)
    destination.figures.mkdir(parents=True,exist_ok=True)
    checksums={}
    for path in source.statistics.iterdir():
        if path.name.startswith('.') or path.name in ("run_metadata.json","capture_velocity_partial_samples.csv"):
            continue
        output=destination.statistics/path.name
        if path.name=="capture_endpoint_audit.csv":
            output=output.with_suffix(".csv.gz")
            temporary=output.with_suffix(output.suffix+".tmp")
            with path.open("rb") as src, gzip.open(temporary,"wb",compresslevel=1) as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            replace_with_retry(temporary,output)
        else:
            temporary=output.with_suffix(output.suffix+".tmp")
            shutil.copyfile(path,temporary)
            replace_with_retry(temporary,output)
        checksums[output.name]=hashlib.sha256(output.read_bytes()).hexdigest()
    for path in source.figures.iterdir():
        if path.is_file(): shutil.copyfile(path,destination.figures/path.name)
    write_json(destination.statistics/"publication.json",dict(
        published_utc=now(),source_directory=str(source.statistics),
        endpoint_audit="capture_endpoint_audit.csv.gz (lossless gzip; every evaluated node retained)",
        files_sha256=checksums,
    ))
    # Publish completion last so readers never accept a half-copied point.
    write_json(destination.metadata_json,read_json(source.metadata_json))


def search_for(discs, points, seed):
    return replace(engine.default_search_config(seed=seed), disc_count=discs,
                   points_per_disc=points, trajectory_backend="numba",
                   analysis_velocity_max_m_per_s=80.0, save_every=5000)


def run_point(index, search, workers, stage):
    # A direct 321-speed audit stores complete trajectory evidence in JSON
    # CSV fields, which can exceed Python's small default reader limit.
    csv.field_size_limit(max(csv.field_size_limit(),128*1024*1024))
    point, paths = point_for(index), paths_for(index, stage, working=True)
    discs, points = engine.generate_common_geometry(search)
    geometry = geometry_csv_text(geometry_rows(discs, points))
    digest = hashlib.sha256(geometry.encode()).hexdigest()
    previous_builder, previous_name = engine.build_point_configuration, engine.CAMPAIGN_NAME
    previous_progress = engine.PROGRESS_EVERY
    previous_checkpoint = engine.CHECKPOINT_EVERY
    engine.build_point_configuration, engine.CAMPAIGN_NAME = build_configuration, NAME
    engine.PROGRESS_EVERY = 500
    engine.CHECKPOINT_EVERY = search.save_every
    try:
        published=paths_for(index,stage)
        if published.metadata_json.exists() and (published.statistics/"capture_counts_by_disc.json").exists():
            prior=read_json(published.metadata_json)
            expected=engine._signature(engine._point_signature_payload(point,search,digest))
            if prior.get("status")=="completed" and prior.get("run_signature_sha256")==expected:
                return read_json(published.capture_summary_json)
            compatibility_path=STATISTICS/"terminal_radius_bookkeeping_migration.json"
            if prior.get("status")=="completed" and compatibility_path.exists():
                compatibility=read_json(compatibility_path)
                accepted=compatibility.get("completed_point_signatures",{}).get(str(index),{})
                if (accepted.get("original")==prior.get("run_signature_sha256")
                    and accepted.get("current")==expected
                    and compatibility.get("current_physics_source_sha256")==engine._source_hashes()):
                    return read_json(published.capture_summary_json)
        complete = paths.metadata_json.exists() and read_json(paths.metadata_json).get("status") == "completed"
        summary = engine.run_relationship_point(
            point, search=search, discs=discs, points=points, geometry_text=geometry,
            geometry_hash=digest, paths=paths, worker_count=workers, resume=True,
            analyze_only=complete,
        )
    finally:
        engine.build_point_configuration, engine.CAMPAIGN_NAME = previous_builder, previous_name
        engine.PROGRESS_EVERY = previous_progress
        engine.CHECKPOINT_EVERY = previous_checkpoint
    export_disc_spectra(index, search, paths)
    publish_point(index,stage,paths)
    return summary


def export_disc_spectra(index, search, paths):
    """Keep per-disc capture counts, including explicit unknown zero nodes."""
    samples = engine.load_capture_velocity_samples(paths.final_samples_csv)
    overrides = velocity_overrides_from_payload(read_json(engine._overrides_path(paths)))
    velocity = engine._velocity_grid(samples, search)
    if max(s.untrapped_velocity_upper_m_per_s for s in samples) >= velocity[-1]-5.0:
        raise ValueError("capture boundary approaches spectrum limit; extend the speed domain")
    predicate = capture_predicate_with_overrides(samples, velocity, overrides)
    counts = []
    for disc in range(search.disc_count):
        rays = [x for x in samples if x.disc_index == disc]
        values = []
        for speed in velocity:
            decisions = [predicate(x, float(speed)) for x in rays]
            values.append(None if any(x is None for x in decisions) else int(sum(decisions)))
        counts.append(values)
    if any(any(value not in (0, None) for value in row[-4:]) for row in counts):
        raise ValueError("capture spectrum has not reached an empty high-speed tail")
    write_json(paths.statistics/"capture_counts_by_disc.json", dict(
        schema="pmot.capture-counts-by-disc.v1", configuration=config_record(index),
        geometry_sha256=read_json(paths.metadata_json)["geometry_sha256"],
        velocity_m_per_s=velocity.tolist(), captured_counts=counts,
        points_per_disc=search.points_per_disc, disc_area_m2=pi*search.disc_radius_m**2,
        unknown_policy="null is allowed only at exact zero speed; g(0)=0, no sigma(0) imputation",
    ))


def reweight_counts(payload, temperature_k=296.483, pressure_torr=1e-10):
    """Recompute each disc integral before constructing the Student-t interval."""
    vapor = VaporLoadingParameters(temperature_k=temperature_k, rb87_partial_pressure_torr=pressure_torr)
    velocity = np.asarray(payload["velocity_m_per_s"])
    counts = np.asarray([[np.nan if c is None else c for c in row] for row in payload["captured_counts"]], float)
    if np.any(~np.isfinite(counts[:, velocity > 0])):
        raise ValueError("unresolved positive-flux capture node")
    if np.any(counts[np.isfinite(counts)] < 0) or np.any(counts[np.isfinite(counts)] > payload["points_per_disc"]):
        raise ValueError("invalid capture count")
    sigma = counts * payload["disc_area_m2"] / payload["points_per_disc"]
    integrands = np.asarray([loading_integrand(velocity, row, vapor) for row in sigma])
    if velocity[0] == 0:
        integrands[:,0] = 0.0
    else:
        velocity = np.r_[0., velocity]
        integrands = np.column_stack((np.zeros(len(sigma)), integrands))
    values = vapor.loading_rate_prefactor*np.trapezoid(integrands, velocity, axis=1)
    mean = float(np.mean(values))
    critical = float(student_t.ppf(.975, len(values)-1))
    half = float(critical*np.std(values, ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean, lower=mean-half, upper=mean+half,
                relative_half_width=half/mean if mean > 0 else None,
                disc_count=len(values), student_t_critical_95=critical,
                disc_loading_rates_atoms_per_s=values.tolist())


def manifest():
    return dict(schema="pmot.independent-parameter-campaign.v1", campaign=NAME,
                wavelength_m=WAVELENGTH_M, baseline=dict(zip(
                    ("power_mw_per_traveling_beam", "detuning_mhz", "diameter_mm", "axial_gradient_g_per_cm"), BASELINE)),
                vapor=asdict(VaporLoadingParameters()), temperature_reweighting="fixed Rb87 partial pressure",
                scan_grids=GRIDS, scan_configuration_indices=SCAN_INDICES,
                configurations=[config_record(i) for i in range(len(CONFIGURATIONS))],
                geometry="full sphere, uniform area, parallel launch velocities; disc radius and center distance both 15 mm",
                model="deterministic effective two-level mean force, gravity on, recoil diffusion off; no repump",
                saturation="single-beam Gaussian-center s0=I0/Isat; seff=s0/(1+(2*detuning_hz/linewidth_hz)^2)",
                diameter_rule="10 mW per traveling beam held fixed",
                storage="live checkpoints in local application storage; completed immutable points published to the repository; audit CSVs losslessly gzip-compressed",
                statistics="two-sided pointwise 95% Student-t across direction discs; pilot excluded",
                precision_target_relative_half_width=.10,
                source_sha256={**engine._source_hashes(),
                    "pmot/mot_simple/parameter_campaign.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def fixed_text(family):
    parts = []
    if family != "power": parts.append("P = 10 mW per traveling beam")
    if family != "detuning": parts.append("Δ = −15 MHz")
    if family != "diameter": parts.append("D = 12.7 mm (1/e²)")
    if family != "gradient": parts.append("Gz = 20 G/cm")
    return "; ".join(parts)


def completed_payloads():
    result = {}
    for i in range(len(CONFIGURATIONS)):
        paths = paths_for(i)
        meta = paths.metadata_json
        data = paths.statistics/"capture_counts_by_disc.json"
        if meta.exists() and data.exists() and read_json(meta).get("status") == "completed":
            result[i] = read_json(data)
    return result


def plot_figures(payloads=None, *, temperature_k=296.483, pressure_torr=1e-10, save=True):
    payloads = completed_payloads() if payloads is None else payloads
    figures = []
    for key, family, xfield, xlabel, title, color, pastel in VIEWS:
        rows = []
        for i in SCAN_INDICES[family]:
            if i in payloads:
                rows.append((config_record(i)[xfield], reweight_counts(payloads[i], temperature_k, pressure_torr)))
        rows.sort(key=lambda row:row[0])
        fig, ax = plt.subplots(figsize=(10.5,7.2))
        fig.patch.set_facecolor("#fcfbf8")
        ax.set_facecolor("#fcfbf8")
        if rows:
            x=np.asarray([v[0] for v in rows])
            y=np.asarray([v[1]["mean"] for v in rows])/1e6
            lo=np.asarray([v[1]["lower"] for v in rows])/1e6
            hi=np.asarray([v[1]["upper"] for v in rows])/1e6
            ax.fill_between(x,lo,hi,color=pastel,alpha=.60,label="Pointwise 95% Student-t interval")
            ax.plot(x,y,"o-",color=color,linewidth=2,markersize=4,label="Direction-disc mean")
        ax.set_title(title, fontsize=16, pad=16)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loading rate [10⁶ atoms/s]")
        ax.grid(alpha=.18)
        ax.set_ylim(bottom=0)
        if rows: ax.legend(frameon=False,fontsize=9)
        if family == "power":
            factor = config_record(SCAN_INDICES[family][0])[xfield] / GRIDS[family][0]
            top = ax.secondary_xaxis("top",functions=(lambda x,scale=factor:x/scale,lambda p,scale=factor:p*scale))
            top.set_xlabel("Cooling power per traveling beam [mW]")
        ns = [len(payloads[i]["captured_counts"]) for i in SCAN_INDICES[family] if i in payloads]
        ntext = str(min(ns)) if ns and min(ns)==max(ns) else f"{min(ns)}–{max(ns)}" if ns else "0"
        points = next(iter(payloads.values()),{}).get("points_per_disc",0)
        caption = (fixed_text(family)+f"\nλ = 780.241209686 nm; vapor T = {temperature_k:g} K; Rb-87 pressure = {pressure_torr:g} Torr (fixed)"
                   +f"\n{ntext} full-sphere discs × {points} uniform-area points/disc; disc radius = 15 mm; parallel launches"
                   +"\nDeterministic two-level mean force; gravity on; recoil diffusion off. Cross section averaged over directions.")
        if len(rows) != len(SCAN_INDICES[family]):
            caption += f"\nIN PROGRESS: {len(rows)}/{len(SCAN_INDICES[family])} scan points complete"
        fig.text(.5,.025,caption,ha="center",va="bottom",fontsize=8.3,color="#475569")
        fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.78 if family=="power" else .89)
        if save:
            FIGURES.mkdir(parents=True,exist_ok=True)
            fig.savefig(FIGURES/f"loading_vs_{key}.png",dpi=170)
            fig.savefig(FIGURES/f"loading_vs_{key}.svg")
        figures.append(fig)
    return figures


def write_notebook():
    cells = []
    def markdown(text):
        cells.append(dict(cell_type="markdown",metadata={},source=text.splitlines(True)))
    def code(text, outputs=None):
        cells.append(dict(cell_type="code",metadata={},execution_count=None,
                          source=text.splitlines(True),outputs=outputs or []))
    markdown("# Two-level MOT loading parameter scans\n\nFour independent sweeps; raw/effective saturation reuse the same power data. "
             "All fixed settings appear on each plot. Intervals are pointwise 95% Student-t across independent direction discs. "
             "The pilot is excluded. Run the cells again to refresh an ongoing campaign.\n\n"
             "Capture spectra and per-disc counts are retained so temperature can be changed without new trajectories. "
             "Rb-87 partial pressure stays fixed; number density is recomputed as p/(k_B T). "
             "The spectra describe the specified finite 15 mm launch aperture and launch plane.")
    code("from pathlib import Path\nimport sys\n"
         "root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src' / 'pmot').is_dir())\n"
         "sys.path.insert(0, str(root / 'src'))\n"
         "from pmot.mot_simple import parameter_campaign as campaign\n"
         "import matplotlib.pyplot as plt\n"
         "temperature_k = 296.483\nrb87_partial_pressure_torr = 1e-10\n"
         "payloads = campaign.completed_payloads()\n"
         "print(f'{len(payloads)}/{len(campaign.CONFIGURATIONS)} unique configurations complete')")
    code("figures = campaign.plot_figures(payloads, temperature_k=temperature_k,\n"
         "    pressure_torr=rb87_partial_pressure_torr, save=False)\nplt.close('all')")
    for index,view in enumerate(VIEWS):
        path=FIGURES/f"loading_vs_{view[0]}.png"
        outputs=[]
        if path.exists():
            outputs=[dict(output_type="display_data",metadata={},data={
                "image/png":base64.b64encode(path.read_bytes()).decode(),
                "text/plain":[view[4]]})]
        code(f"display(figures[{index}])",outputs)
    code("# Reload individual disc rates and uncertainties at the temperature above.\n"
         "import pandas as pd\n"
         "results = pd.DataFrame([{**campaign.config_record(i),\n"
         "    **{k:v for k,v in campaign.reweight_counts(p, temperature_k, rb87_partial_pressure_torr).items()\n"
         "       if k != 'disc_loading_rates_atoms_per_s'}} for i,p in payloads.items()])\nresults")
    write_json(NOTEBOOK,dict(nbformat=4,nbformat_minor=5,cells=[{**c,"id":f"cell-{i:03d}"} for i,c in enumerate(cells)],
                            metadata=dict(kernelspec=dict(display_name="Python 3 (project .venv)",language="python",name="python3"))))


def refresh_report(status="running"):
    payloads = completed_payloads()
    for fig in plot_figures(payloads): plt.close(fig)
    rows = [{**config_record(i), **{k:v for k,v in reweight_counts(p).items() if k!="disc_loading_rates_atoms_per_s"}}
            for i,p in sorted(payloads.items())]
    if rows:
        with (STATISTICS/"aggregate.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(STATISTICS/"status.json",dict(status=status,updated_utc=now(),completed=len(payloads),
                expected=len(CONFIGURATIONS),points_under_10_percent=sum(
                    r["relative_half_width"] is not None and r["relative_half_width"]<.10 for r in rows)))
    write_notebook()
    from .parameter_campaign_html import write_html
    write_html(payloads, status)


def run_pilot(workers):
    indices=[0,SCAN_INDICES["power"][0],SCAN_INDICES["diameter"][0],
             SCAN_INDICES["gradient"][0],SCAN_INDICES["detuning"][-1],
             SCAN_INDICES["power"][-1],SCAN_INDICES["diameter"][-1],
             SCAN_INDICES["gradient"][-1],SCAN_INDICES["detuning"][0]]
    results=read_json(STATISTICS/"pilot_results.json") if (STATISTICS/"pilot_results.json").exists() else []
    search=search_for(12,50,2026110301)
    for i in indices:
        if any(r["index"]==i for r in results):
            continue
        started=time.monotonic()
        summary=run_point(i,search,workers,"pilot")
        data=read_json(paths_for(i,"pilot").statistics/"capture_counts_by_disc.json")
        value=reweight_counts(data)
        elapsed=time.monotonic()-started
        results.append(dict(index=i,elapsed_wall_time_s=elapsed,relative_half_width=value["relative_half_width"],
                            mean_atoms_per_s=value["mean"],velocity_override_count=summary["velocity_resolved_override_count"]))
        write_json(STATISTICS/"pilot_results.json",results)
    return results


def select_plan(pilot, workers, budget_hours):
    # Freeze production size from independent pilot; no sequential stopping on
    # production interval widths. Retain a 30% duration/audit reserve.
    family_targets={}
    family_costs={}
    for family,indices in SCAN_INDICES.items():
        relevant=[r for r in pilot if r.get("index",0)==0 or r.get("index") in indices]
        target=max([1250]+[int(np.ceil(600*(r["relative_half_width"]/.08)**2))
                           for r in relevant if r["relative_half_width"] is not None])
        family_targets[family]=target
        # Include report/export and large-checkpoint overhead beyond small pilot.
        family_costs[family]=max(r["elapsed_wall_time_s"]/600 for r in relevant)*1.20
    desired={}
    costs={}
    for i in range(len(CONFIGURATIONS)):
        families=[family for family,ids in SCAN_INDICES.items() if i in ids]
        desired[i]=max(family_targets[f] for f in families)
        costs[i]=max(family_costs[f] for f in families)
    elapsed=sum(r["elapsed_wall_time_s"] for r in pilot)
    available=max(0., budget_hours*3600-elapsed)*.70
    desired_cost=sum(desired[i]*costs[i] for i in desired)
    factor=min(1.,available/max(desired_cost,1e-9))
    searches={str(i):asdict(search_for(max(25,int(np.ceil(desired[i]*factor/50))),50,2026110302)) for i in desired}
    master=max(searches.values(),key=lambda s:s["disc_count"])
    projected=sum(s["disc_count"]*50*costs[int(i)] for i,s in searches.items())
    return dict(created_utc=now(),workers=workers,budget_hours=budget_hours,
                deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=budget_hours*3600-elapsed)).isoformat(),
                pilot_excluded_from_production=True,search=master,searches=searches,
                family_target_rays=family_targets,budget_scale_factor=factor,
                fixed_rays_per_configuration={i:s["disc_count"]*50 for i,s in searches.items()},
                projected_production_hours=projected/3600,
                geometry_policy="each configuration uses a prefix of one saved master geometry; 50 points per disc",
                precision="aim <10% relative half-width; achieved values reported, not guaranteed",
                deadline_policy="checkpoint at 24h; never replace unresolved classifications with escape")


def audit_campaign():
    payloads=completed_payloads()
    if len(payloads)!=len(CONFIGURATIONS):
        raise ValueError("campaign has incomplete configurations")
    master_text=(STATISTICS/"launch_geometry_master.csv").read_text(encoding="utf-8")
    geometry=set()
    checks=[]
    for i,p in sorted(payloads.items()):
        paths=paths_for(i)
        meta=read_json(paths.metadata_json)
        summary=read_json(paths.capture_summary_json)
        geometry.add(p["geometry_sha256"])
        if not master_text.startswith(paths.geometry_csv.read_text(encoding="utf-8")):
            raise ValueError("configuration launch geometry is not a prefix of the common master")
        if summary["unresolved_timeout_count"]!=0:
            raise ValueError("unresolved capture classifications")
        publication=read_json(paths.statistics/"publication.json")
        for filename,digest in publication["files_sha256"].items():
            if hashlib.sha256((paths.statistics/filename).read_bytes()).hexdigest()!=digest:
                raise ValueError(f"published capture evidence checksum mismatch: {i}/{filename}")
        if meta["signature_payload"]["apparatus_config"]["cooling"]["wavelength_m"] != WAVELENGTH_M:
            raise ValueError("wrong cooling wavelength")
        loading=read_json(paths.loading_json)
        reweighted=reweight_counts(p)
        for actual,field in ((reweighted["mean"],"loading_rate_mean_atoms_per_s"),
                             (max(0., reweighted["lower"]),"loading_rate_t95_lower_atoms_per_s"),
                             (reweighted["upper"],"loading_rate_t95_upper_atoms_per_s")):
            if not np.isclose(actual,loading[field],rtol=2e-12,atol=1e-6):
                raise ValueError(f"per-disc reweighting disagrees with engine: {i}/{field}")
        checks.append(dict(index=i,relative_half_width=reweighted["relative_half_width"],status="passed"))
    result=dict(status="passed",audited_utc=now(),configuration_count=len(checks),
                geometry_sha256=sorted(geometry),checks=checks,
                precision_target_met_for_all=all(x["relative_half_width"] is not None and x["relative_half_width"]<.10 for x in checks))
    write_json(STATISTICS/"final_audit.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",nargs="?",default="all",choices=("pilot","all","production","report","audit"))
    parser.add_argument("--workers",type=int,default=min(20,os.cpu_count() or 1))
    parser.add_argument("--budget-hours",type=float,default=24.)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=24: parser.error("workers must be 1..24")
    STATISTICS.mkdir(parents=True,exist_ok=True)
    current=manifest()
    manifest_path=STATISTICS/"campaign_manifest.json"
    if manifest_path.exists() and read_json(manifest_path)!=json.loads(json.dumps(current)):
        raise ValueError("campaign manifest changed; use a distinct output root")
    write_json(manifest_path,current)
    if args.stage=="report": refresh_report();return 0
    if args.stage=="audit": print(json.dumps(audit_campaign(),indent=2));return 0
    try:
        plan_path=STATISTICS/"production_plan.json"
        if args.stage=="pilot" or (args.stage=="all" and not plan_path.exists()):
            pilot=run_pilot(args.workers)
            write_json(plan_path,select_plan(pilot,args.workers,args.budget_hours))
        if args.stage=="pilot": return 0
        if not plan_path.exists(): raise ValueError("run pilot before production")
        plan=read_json(plan_path)
        master_search=replace(engine.default_search_config(),**plan["search"])
        d,p=engine.generate_common_geometry(master_search)
        master_geometry=geometry_csv_text(geometry_rows(d,p))
        master_path=STATISTICS/"launch_geometry_master.csv"
        if master_path.exists() and master_path.read_text(encoding="utf-8")!=master_geometry:
            raise ValueError("master launch geometry changed")
        if not master_path.exists():
            master_path.write_text(master_geometry,encoding="utf-8",newline="")
        refresh_report()
        # Baseline first, then round-robin families for useful partial previews.
        order=[0]
        for j in range(max(map(len,SCAN_INDICES.values()))):
            for ids in SCAN_INDICES.values():
                if j<len(ids) and ids[j] not in order: order.append(ids[j])
        for i in order:
            if (not plan.get("continue_until_complete",False)
                and datetime.now(timezone.utc)>=datetime.fromisoformat(plan["deadline_utc"])):
                refresh_report("budget_reached_partial")
                return 2
            search=replace(engine.default_search_config(),**plan["searches"][str(i)])
            run_point(i,search,args.workers,"production")
            refresh_report()
        audit_campaign()
        refresh_report("completed")
    except BaseException as exc:
        write_json(STATISTICS/"failure.json",dict(time_utc=now(),error=f"{type(exc).__name__}: {exc}"))
        refresh_report("failed")
        raise
    return 0


if __name__=="__main__":
    raise SystemExit(main())
