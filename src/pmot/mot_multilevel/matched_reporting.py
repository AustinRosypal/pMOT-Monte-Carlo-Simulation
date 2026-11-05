"""Notebook, browser, and matched two-model views of validated loading data."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from ..configuration import RB87_MASS_KG

from . import matched_benchmark as b
from .matched_campaign import completed,paths,points_and_scans,reweight_counts,write_csv


VIEWS=(
 ("cooling_saturation","cooling_power","cooling_s0","Nominal single-beam center saturation s₀","Loading vs cooling saturation"),
 ("cooling_effective_saturation","cooling_power","cooling_seff","Nominal effective center saturation s_eff","Loading vs effective cooling saturation"),
 ("cooling_detuning","cooling_detuning","cooling_detuning_mhz","Cooling detuning [MHz]","Loading vs cooling detuning"),
 ("cooling_diameter_fixed_repump","cooling_diameter","cooling_diameter_mm","Cooling 1/e² intensity diameter [mm]","Cooling diameter · repump diameter fixed"),
 ("cooling_gradient","cooling_gradient","gradient_g_per_cm","Axial gradient magnitude [G/cm]","Loading vs magnetic gradient"),
 ("cooling_diameter_matched_repump","cooling_diameter_matched_repump","cooling_diameter_mm","Cooling / repump 1/e² intensity diameter [mm]","Cooling diameter · repump diameter matched"),
 ("repump_power","repump_power","repump_power_mw","Repump power per traveling beam [mW]","Loading vs repump power"),
 ("repump_diameter","repump_diameter","repump_diameter_mm","Repump 1/e² intensity diameter [mm]","Loading vs repump diameter"),
 ("repump_detuning","repump_detuning","repump_detuning_mhz","Repump detuning from F=1 → F′=2 [MHz]","Loading vs repump detuning"),
)
COLORS={"two_level":("#147d87","#b9dde0","Effective two-level MOT"),
        "multilevel":("#8855a0","#d9c4e6","Physical 24-state MOT")}


def point_records(points):
    records={}
    for p in points:
        r=asdict(p);s0=2*p.cooling_power_mw*.001/(np.pi*(p.cooling_diameter_mm*.0005)**2*16.69)
        r.update(cooling_s0=s0,cooling_seff=s0/(1+(2*p.cooling_detuning_mhz/6.065)**2))
        records[p.index]=r
    return records


def fixed_text(family):
    lines=[]
    if family!="cooling_power":lines.append("Pcool = 10 mW/beam")
    if family!="cooling_detuning":lines.append("Δcool = −15 MHz")
    if family not in ("cooling_diameter","cooling_diameter_matched_repump"):lines.append("Dcool = 12.7 mm")
    if family!="cooling_gradient":lines.append("Gz = 20 G/cm")
    if family!="repump_power":lines.append("Prep = 0.1 mW/beam")
    if family!="repump_detuning":lines.append("Δrep = 0 MHz (F=1 → F′=2)")
    if family=="cooling_diameter_matched_repump":lines.append("Drep = Dcool")
    elif family!="repump_diameter":lines.append("Drep = 12.7 mm")
    return "; ".join(lines)


def baseline_payloads():
    result={}
    original=b.read_json(b.BENCHMARK/"campaign_manifest.json")
    for cfg in original["configurations"]:
        index=cfg["index"]
        folder=b.BENCHMARK/"production"/f"{index:03d}_configuration_{index}"
        result[index]=b.read_json(folder/"capture_counts_by_disc.json")
    return result


def report_bundle(comparison=False):
    points,scans=points_and_scans();records=point_records(points)
    datasets={"multilevel":completed()}
    completed_total=len(datasets["multilevel"])
    if comparison and completed_total:
        approval=b.read_json(b.STATISTICS/"comparison_manifest.json")
        if approval["status"]!="approved_and_matched" or approval["common_configuration_count"]!=64:
            raise ValueError("comparison input review/matching gate has not passed")
    if comparison:datasets["two_level"]=baseline_payloads()
    views=[]
    for key,family,field,label,title in VIEWS:
        if comparison and family.startswith("repump_"):continue
        series=[dict(model="multilevel",ids=scans[family])]
        if comparison:
            original_family="cooling_diameter" if family=="cooling_diameter_matched_repump" else family
            series.insert(0,dict(model="two_level",ids=scans[original_family]))
        views.append(dict(id=key,family=family,field=field,label=label,title=title,fixed=fixed_text(family),series=series))
    for model,dataset in datasets.items():
        needed={index for view in views for series in view["series"] if series["model"]==model for index in series["ids"]}
        datasets[model]={index:p for index,p in dataset.items() if index in needed}
    return dict(points=records,views=views,datasets=datasets,comparison=comparison,completed_total=completed_total)


def plot_views(comparison=False,*,temperature_k=296.483,pressure_torr=1e-10,save=True):
    bundle=report_bundle(comparison);figures=[]
    output=b.COMPARISON_FIGURES if comparison else b.FIGURES
    for view in bundle["views"]:
        fig,ax=plt.subplots(figsize=(11.5,8.1));fig.patch.set_facecolor("#fcfbf8");ax.set_facecolor("#fcfbf8")
        counts=[];under=0;finished=0
        for series in view["series"]:
            model=series["model"];line,pastel,title=COLORS[model];rows=[]
            for index in series["ids"]:
                if index in bundle["datasets"][model]:
                    payload=bundle["datasets"][model][index];result=reweight_counts(payload,temperature_k,pressure_torr)
                    rows.append((bundle["points"][index][view["field"]],result))
                    counts.append(result["disc_count"])
                    if model=="multilevel":
                        finished+=1;under+=result["relative_half_width"] is not None and result["relative_half_width"]<.10
            rows.sort(key=lambda row:row[0])
            if not rows:continue
            x=np.asarray([r[0] for r in rows]);mean=np.asarray([r[1]["mean"] for r in rows])/1e6
            lo=np.asarray([r[1]["lower"] for r in rows])/1e6;hi=np.asarray([r[1]["upper"] for r in rows])/1e6
            ax.fill_between(x,lo,hi,color=pastel,alpha=.55,linewidth=0)
            ax.plot(x,mean,"o-",label=title,color=line,markersize=4.2,linewidth=2)
        ax.set_title(view["title"],fontsize=16,pad=15)
        ax.set_xlabel(view["label"]);ax.set_ylabel("Loading rate [10⁶ atoms/s]")
        ax.set_ylim(bottom=0);ax.grid(alpha=.18)
        if counts:ax.legend(frameon=False,fontsize=10)
        if view["family"]=="cooling_power":
            first=bundle["points"][view["series"][-1]["ids"][0]]
            scale=first[view["field"]]/first["cooling_power_mw"]
            secondary=ax.secondary_xaxis("top",functions=(lambda x,s=scale:x/s,lambda p,s=scale:p*s))
            secondary.set_xlabel("Cooling power per traveling beam [mW]")
        title_fixed=view["fixed"].split("; ")
        split=min(4,len(title_fixed))
        ntext=str(min(counts)) if counts and min(counts)==max(counts) else f"{min(counts)}–{max(counts)}" if counts else "pending"
        caption=("; ".join(title_fixed[:split])+"\n"+"; ".join(title_fixed[split:])
                 +f"\nλcool = 780.241209686 nm; vapor T = {temperature_k:g} K; Rb-87 pressure = {pressure_torr:g} Torr (fixed)"
                 +f"\n{ntext} full-sphere direction discs/point; 50 area-uniform points/disc; disc radius and launch-plane distance 15 mm."
                 +f"\nPointwise 95% Student-t across direction discs; gravity on; recoil diffusion off. {under}/{finished} multilevel points below 10%.")
        if comparison:caption+="\nRepump and state-resolved populations are multilevel-only; matched saved launch coordinates and benchmark controls."
        if finished!=len(view["series"][-1]["ids"]):caption+=f"\nIN PROGRESS: {finished}/{len(view['series'][-1]['ids'])} multilevel scan points complete."
        fig.text(.5,.025,caption,ha="center",va="bottom",fontsize=8.2,color="#475569")
        fig.subplots_adjust(left=.12,right=.98,bottom=.27 if comparison else .25,top=.79 if view["family"]=="cooling_power" else .89)
        if save:
            output.mkdir(parents=True,exist_ok=True);fig.savefig(output/f"loading_vs_{view['id']}.png",dpi=170)
            fig.savefig(output/f"loading_vs_{view['id']}.svg")
        figures.append(fig)
    return figures


def write_notebook(comparison=False):
    path=b.COMPARISON_NOTEBOOK if comparison else b.NOTEBOOK
    directory=b.COMPARISON_FIGURES if comparison else b.FIGURES
    bundle=report_bundle(comparison);cells=[]
    def md(text):cells.append(dict(cell_type="markdown",metadata={},source=text.splitlines(True)))
    def code(text,outputs=None):cells.append(dict(cell_type="code",metadata={},execution_count=None,source=text.splitlines(True),outputs=outputs or []))
    md(("# Two-level and physical multilevel MOT comparison" if comparison else "# Physical multilevel MOT loading scans")+
       "\n\nThe physical model uses the replacement 24-state Section-12 population-rate equations. "
       "Capture evidence and per-disc cross sections are saved; all powers are per traveling component and diameters are Gaussian 1/e². "
       "The cooling recoil/Doppler wavelength is fixed at 780.241209686 nm. The repump zero is the F=1 → F′=2 resonance. "
       "Intervals are pointwise 95% Student-t across independent direction discs.\n\n"
       "Edit the vapor temperature below and rerun to reweight each disc's loading integral and confidence interval at fixed pressure. "
       "This does not rerun trajectories. The two nominal cooling saturation views reuse one power dataset. "
       "These finite-aperture results use the exact saved benchmark launch points; local steady-state/incoherent-light assumptions remain model limits.")
    code("from pathlib import Path\nimport sys\n"
         "root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'src'/'pmot').is_dir())\n"
         "sys.path.insert(0,str(root/'src'))\n"
         "from pmot.mot_multilevel import matched_reporting as reports\n"
         "import matplotlib.pyplot as plt\n"
         "temperature_k=296.483\nrb87_partial_pressure_torr=1e-10\n"
         f"figures=reports.plot_views(comparison={comparison},temperature_k=temperature_k,pressure_torr=rb87_partial_pressure_torr,save=False)\n"
         "plt.close('all')")
    for j,view in enumerate(bundle["views"]):
        image_path=directory/f"loading_vs_{view['id']}.png";outputs=[]
        if image_path.exists():outputs=[dict(output_type="display_data",metadata={},data={"image/png":base64.b64encode(image_path.read_bytes()).decode(),"text/plain":[view["title"]]})]
        code(f"display(figures[{j}])",outputs)
    code("# Inspect and export numeric rates at the temperature above.\n"
         "from pmot.mot_multilevel.matched_campaign import completed,reweight_counts\n"
         "import pandas as pd\n"
         "results=pd.DataFrame([{**p['configuration'],**{k:v for k,v in reweight_counts(p,temperature_k,rb87_partial_pressure_torr).items() if k!='disc_loading_rates_atoms_per_s'}} for p in completed().values()])\nresults")
    b.write_json(path,dict(nbformat=4,nbformat_minor=5,cells=[{**cell,"id":f"cell-{j:03d}"} for j,cell in enumerate(cells)],
                  metadata=dict(kernelspec=dict(display_name="Python 3 (project .venv)",language="python",name="python3"))))


def write_html(comparison=False,status="running"):
    from .matched_report_html import TEMPLATE
    bundle=report_bundle(comparison)
    for dataset in bundle["datasets"].values():
        for payload in dataset.values():
            rows=payload.pop("captured_counts")
            payload["critical"]=float(t.ppf(.975,len(rows)-1))
            payload["disc_count"]=len(rows)
            packed=bytes(255 if value is None else value for row in rows for value in row)
            payload["counts_base64"]=base64.b64encode(packed).decode()
    data=dict(**bundle,status=status,updated_utc=b.now(),expected=len(points_and_scans()[0]),colors=COLORS,
              mass_kg=RB87_MASS_KG)
    directory=b.COMPARISON_FIGURES if comparison else b.FIGURES;directory.mkdir(parents=True,exist_ok=True)
    target=directory/("two_level_multilevel_comparison.html" if comparison else "multilevel_loading_scans.html")
    temporary=target.with_suffix(".html.tmp")
    temporary.write_text(TEMPLATE.replace("__DATA__",json.dumps(data,separators=(",",":"),allow_nan=False).replace("</","<\\/")),encoding="utf-8")
    b.replace_file(temporary,target)


def refresh_report(status="running"):
    payloads=completed();points,scans=points_and_scans();records=point_records(points)
    values=[]
    for i,p in sorted(payloads.items()):values.append(dict(**records[i],**{k:v for k,v in reweight_counts(p).items() if k!="disc_loading_rates_atoms_per_s"}))
    if values:write_csv(b.STATISTICS/"aggregate.csv",values)
    b.write_json(b.STATISTICS/"status.json",dict(status=status,updated_utc=b.now(),completed=len(payloads),expected=len(points),
                 points_under_10_percent=sum(v["relative_half_width"] is not None and v["relative_half_width"]<.1 for v in values)))
    for comparison in (False,True):
        for fig in plot_views(comparison):plt.close(fig)
        write_notebook(comparison);write_html(comparison,status)


def audit_campaign():
    points,scans=points_and_scans();payloads=completed()
    if len(payloads)!=len(points):raise ValueError("campaign has incomplete physical settings")
    plan=b.read_json(b.STATISTICS/"campaign_manifest.json")
    checks=[]
    for point in points:
        _,folder,_=paths(point)
        publication=b.read_json(folder/"publication.json")
        for filename,digest in publication["files_sha256"].items():
            if b.sha256(folder/filename)!=digest:raise ValueError(f"published evidence checksum mismatch: {point.index}/{filename}")
        summary=b.read_json(folder/"summary.json")
        if summary["unresolved_positive_speed_count"]:raise ValueError("unresolved positive-flux capture nodes")
        if b.read_json(folder/"retention_validation.json")["status"]!="passed":
            raise ValueError("longer-trajectory validation did not pass")
        p=payloads[point.index];result=reweight_counts(p)
        for field in ("mean","lower","upper"):
            if not np.isclose(result[field],summary["loading"][field],rtol=2e-12,atol=1e-6):raise ValueError("loading reduction mismatch")
        expected=b.sample_count(point)
        assert (len(p["captured_counts"]),p["points_per_disc"])==expected
        assert p["master_geometry_sha256"]==plan["master_geometry_sha256"]
        checks.append(dict(index=point.index,status="passed",relative_half_width=result["relative_half_width"]))
    result=dict(status="passed",audited_utc=b.now(),configuration_count=len(points),checks=checks,
                all_under_10_percent=all(row["relative_half_width"] is not None and row["relative_half_width"]<.1 for row in checks))
    b.write_json(b.STATISTICS/"final_audit.json",result)
    return result
