"""Resumable matched physical multilevel MOT capture/loading campaign."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Isolate compiled callers from unrelated campaigns and historical caches.
_local_cache=Path(os.environ.get("LOCALAPPDATA",tempfile.gettempdir()))/"pmot-campaigns"/"matched_two_level_exact_D2_20261104"
os.environ.setdefault("NUMBA_CACHE_DIR",str(_local_cache/"compiled_cache_v1"))
os.environ.setdefault("MPLBACKEND","Agg")

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import platform
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t

from ..loading import VaporLoadingParameters
from . import matched_benchmark as b
from .matched_capture import AUDIT_COLUMNS,RayEvaluator,solve_ray


_DATA=None
_POINT=None
_SIGNATURE=None
_RUN_LOCK=None


def library_versions():
    return dict(python=platform.python_version(),packages={name:version(name)
                for name in ("numpy","scipy","numba","arc-alkali-rydberg-calculator")})


def acquire_run_lock():
    """Keep an OS-held lock so a monitor cannot duplicate an active pool."""
    global _RUN_LOCK
    b.CACHE.mkdir(parents=True,exist_ok=True)
    handle=(b.CACHE/"runner.lock").open("a+b")
    handle.seek(0,2)
    if handle.tell()==0:handle.write(b"0");handle.flush()
    handle.seek(0)
    try:
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError("a matched campaign process already holds the execution lock")
    _RUN_LOCK=handle
    b.write_json(b.STATISTICS/"run_owner.json",dict(pid=os.getpid(),started_utc=b.now(),command="physical matched multilevel campaign"))


def worker_init(data,point,signature):
    global _DATA,_POINT,_SIGNATURE
    _DATA=data;_POINT=point;_SIGNATURE=signature


def atomic_npz(path,**payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:np.savez_compressed(stream,**payload)
    b.replace_file(temp,path)


def ray_worker(task):
    disc,point,position,direction,path,representative=task
    path=Path(path);records=[];previous_elapsed=0.
    if path.exists():
        with np.load(path) as z:
            if str(z["signature"])!=_SIGNATURE:raise ValueError("ray resume signature mismatch")
            np.testing.assert_array_equal(z["position_m"],position)
            np.testing.assert_array_equal(z["direction"],direction)
            if str(z["status"])=="completed":
                return dict(disc=disc,point=point,status="completed",elapsed_wall_s=float(z["elapsed_wall_s"]))
            records=z["evaluations"].copy();previous_elapsed=float(z["elapsed_wall_s"])
    started=perf_counter()
    def save(evaluator,status="running",result=None,error=""):
        payload=dict(signature=np.array(_SIGNATURE),position_m=position,direction=direction,
                     disc=np.array(disc),point=np.array(point),evaluations=np.asarray(evaluator.records).reshape((-1,len(AUDIT_COLUMNS))),
                     status=np.array(status),error=np.array(error),elapsed_wall_s=np.array(previous_elapsed+perf_counter()-started),
                     representative=np.array(representative))
        if result is not None:
            payload.update(mask=result["mask"],mode=np.array(result["mode"]),
                boundary_json=np.array(json.dumps(result["boundary"])),
                complete_searches_json=np.array(json.dumps(result["complete_searches"])),
                nonmonotone=np.array(result["nonmonotone"]))
        atomic_npz(path,**payload)
    evaluator=RayEvaluator(position,direction,_DATA,records=records,checkpoint=save)
    try:
        result=solve_ray(evaluator,representative_grid=representative)
        save(evaluator,"completed",result)
        return dict(disc=disc,point=point,status="completed",elapsed_wall_s=previous_elapsed+perf_counter()-started,
                    mode=result["mode"],nonmonotone=result["nonmonotone"])
    except Exception as exc:
        save(evaluator,"failed",error=f"{type(exc).__name__}: {exc}")
        raise


def paths(point,mode="production"):
    return b.CACHE/mode/point.key,b.STATISTICS/mode/point.key,b.FIGURES/mode/point.key


def selected_settings():
    path=b.STATISTICS/"requested_settings.json"
    return b.read_json(path) if path.exists() else dict(
        repump_diameters_mm=None,repump_powers_mw=list(b.REPUMP_POWERS_MW),continue_until_complete=True,
        source="user-confirmed clarification on November 4, 2026")


def points_and_scans():
    settings=selected_settings()
    return b.definitions(repump_diameters=settings.get("repump_diameters_mm"),repump_powers=settings.get("repump_powers_mw"))


def point_signature(point,mode,disc_count,points_per_disc):
    cfg,app,coil,beams=b.configuration(point)
    payload=dict(schema=1,point=asdict(point),mode=mode,disc_count=disc_count,points_per_disc=points_per_disc,
                 master_point_indices=[0,25] if mode=="probe" else list(range(points_per_disc)),
                 model=b.model_notes(),config=asdict(cfg),coil=asdict(coil),beams=[asdict(x) for x in beams],
                 master_geometry_sha256=b.sha256(b.BENCHMARK/"launch_geometry_master.csv"),
                 trajectory_protocol="fixed RK4 5 us base; dual-step 200/250/400 ms boundaries; unresolved grid nodes only to 1 and 2 s; original 2mm/5ms-or-two-entry criterion",
                 velocity_m_per_s=b.VELOCITIES.tolist(),source_sha256=b.source_hashes(),libraries=library_versions())
    signature=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return signature,payload


def publish_point(point,folder):
    _,destination,_=paths(point)
    destination.mkdir(parents=True,exist_ok=True)
    hashes={}
    # Preserve every ray's evaluated nodes and authoritative masks in one
    # immutable archive, rather than syncing thousands of live small files.
    archive=destination/"capture_evidence.tar.gz";temp=archive.with_suffix(".gz.tmp")
    with tarfile.open(temp,"w:gz",compresslevel=1) as tar:
        tar.add(folder/"rays",arcname="rays",recursive=True)
    b.replace_file(temp,archive);hashes[archive.name]=b.sha256(archive)
    for source in folder.iterdir():
        if not source.is_file() or source.name.startswith('.') or source.name=="run_metadata.json":continue
        target=destination/source.name;temp=target.with_suffix(target.suffix+".tmp")
        shutil.copyfile(source,temp);b.replace_file(temp,target);hashes[target.name]=b.sha256(target)
    b.write_json(destination/"publication.json",dict(status="completed",published_utc=b.now(),files_sha256=hashes,
                  evidence="All attempted nodes and authoritative masks are losslessly retained in capture_evidence.tar.gz"))
    b.write_json(destination/"run_metadata.json",b.read_json(folder/"run_metadata.json"))


def reweight_counts(payload,temperature_k=296.483,pressure_torr=1e-10):
    vapor=VaporLoadingParameters(temperature_k=temperature_k,rb87_partial_pressure_torr=pressure_torr)
    v=np.asarray(payload["velocity_m_per_s"],float)
    counts=np.asarray([[np.nan if x is None else x for x in row] for row in payload["captured_counts"]],float)
    if np.any(~np.isfinite(counts[:,v>0])):raise ValueError("unresolved positive-flux capture state")
    sigma=counts*payload["disc_area_m2"]/payload["points_per_disc"]
    g=sigma*v[None,:]**3*np.exp(-v[None,:]**2/vapor.thermal_scale_m2_per_s2)
    if v[0]==0:g[:,0]=0.
    else:
        v=np.r_[0.,v];g=np.column_stack((np.zeros(len(g)),g))
    values=vapor.loading_rate_prefactor*np.trapezoid(g,v,axis=1)
    critical=float(student_t.ppf(.975,len(values)-1));mean=float(values.mean())
    half=float(critical*values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean,lower=mean-half,upper=mean+half,relative_half_width=half/mean if mean>0 else None,
                disc_count=len(values),student_t_critical_95=critical,disc_loading_rates_atoms_per_s=values.tolist())


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    b.replace_file(temporary,path)


def summarize(point,folder,discs,npoints,signature):
    counts=np.zeros((discs,len(b.VELOCITIES)),int);unknown=np.zeros(discs,bool)
    samples=[];direct=0;nonmonotone=0;evaluations=0
    for d in range(discs):
        for p in range(npoints):
            with np.load(folder/"rays"/f"d{d:04d}_p{p:02d}.npz") as z:
                assert str(z["status"])=="completed" and str(z["signature"])==signature
                mask=z["mask"];assert np.all(mask[1:]>=0)
                counts[d]+=np.maximum(mask,0);unknown[d]|=mask[0]<0
                mode=str(z["mode"]);direct+=mode.startswith("direct");nonmonotone+=bool(z["nonmonotone"])
                boundary=json.loads(str(z["boundary_json"]));evaluations+=len(z["evaluations"])
                samples.append(dict(disc_index=d,point_index=p,mode=mode,lower_boundary_m_per_s=boundary["lower"],
                                    upper_boundary_m_per_s=boundary["upper"],nonmonotone=bool(z["nonmonotone"]),
                                    zero_speed_indeterminate=bool(mask[0]<0)))
    captured=counts.astype(object).tolist()
    for d in np.flatnonzero(unknown):captured[d][0]=None
    payload=dict(schema="pmot.capture-counts-by-disc.v1",configuration=asdict(point),
                 velocity_m_per_s=b.VELOCITIES.tolist(),captured_counts=captured,
                 points_per_disc=npoints,disc_area_m2=float(np.pi*.015**2),master_geometry_sha256=b.sha256(b.BENCHMARK/"launch_geometry_master.csv"))
    b.write_json(folder/"capture_counts_by_disc.json",payload)
    result=reweight_counts(payload)
    write_csv(folder/"loading_rate_by_disc.csv",[dict(disc_index=d,loading_rate_atoms_per_s=value) for d,value in enumerate(result["disc_loading_rates_atoms_per_s"])])
    spectrum=[];area=float(np.pi*.015**2)
    for j,v in enumerate(b.VELOCITIES):
        if j==0 and np.any(unknown):continue
        disc_sigma=area*counts[:,j]/npoints;mean=float(disc_sigma.mean())
        half=float(result["student_t_critical_95"]*disc_sigma.std(ddof=1)/np.sqrt(discs))
        spectrum.append(dict(velocity_m_per_s=float(v),capture_cross_section_m2=mean,
                             lower_95_m2=max(0.,mean-half),upper_95_m2=min(area,mean+half)))
    write_csv(folder/"capture_cross_section.csv",spectrum);write_csv(folder/"capture_samples.csv",samples)
    summary=dict(status="completed",configuration=asdict(point),signature=signature,disc_count=discs,
                 points_per_disc=npoints,sample_count=discs*npoints,evaluation_count=evaluations,
                 direct_mask_count=direct,nonmonotone_ray_count=nonmonotone,indeterminate_zero_flux_discs=int(unknown.sum()),
                 unresolved_positive_speed_count=0,loading={k:v for k,v in result.items() if k!="disc_loading_rates_atoms_per_s"})
    b.write_json(folder/"summary.json",summary)
    return summary


def run_point(point,workers=20,*,probe=False):
    mode="probe" if probe else "production"
    folder,published,_=paths(point,mode);folder.mkdir(parents=True,exist_ok=True)
    discs,npoints=(12,2) if probe else b.sample_count(point)
    signature,payload=point_signature(point,mode,discs,npoints)
    frozen_manifest=b.STATISTICS/"campaign_manifest.json"
    if not probe and frozen_manifest.exists():
        frozen=b.read_json(frozen_manifest)
        if payload["source_sha256"]!=frozen["source_sha256"] or payload["libraries"]!=frozen["libraries"]:
            raise RuntimeError("frozen campaign source changed; preserve checkpoints and record a validated revision")
    if not probe and (published/"run_metadata.json").exists():
        prior=b.read_json(published/"run_metadata.json")
        if prior["signature"]!=signature:raise ValueError("published configuration/source signature mismatch")
        if prior["status"]=="completed":return b.read_json(published/"summary.json")
    meta_path=folder/"run_metadata.json"
    prior=b.read_json(meta_path) if meta_path.exists() else None
    if prior is not None and prior["signature"]!=signature:raise ValueError("point resume signature mismatch")
    metadata=dict(status="running",signature=signature,inputs=payload,started_utc=prior["started_utc"] if prior else b.now(),
                  updated_utc=b.now(),expected_samples=discs*npoints,completed_samples=0)
    positions,directions,_=b.load_master_geometry();data=b.prepared(point)
    tasklist=[];complete=0
    for d in range(discs):
        for p in range(npoints):
            original_p=p*25 if probe else p
            path=folder/"rays"/f"d{d:04d}_p{p:02d}.npz"
            if path.exists():
                with np.load(path) as z:
                    if str(z["signature"])!=signature:raise ValueError("checkpoint ray signature mismatch")
                    if str(z["status"])=="completed":complete+=1;continue
            tasklist.append((d,p,positions[d,original_p],directions[d,original_p],str(path),bool(p==0 and d%40==0)))
    metadata["completed_samples"]=complete;b.write_json(meta_path,metadata)
    print(f"[{b.now()}] {mode} {point.key}: {complete}/{discs*npoints} complete; {workers} workers",flush=True)
    started=perf_counter()
    pool=None
    try:
        pool=ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(data,point,signature))
        futures=[pool.submit(ray_worker,task) for task in tasklist]
        for future in as_completed(futures):
            future.result();complete+=1
            if complete%25==0 or complete==discs*npoints:
                elapsed=perf_counter()-started
                metadata.update(completed_samples=complete,updated_utc=b.now(),segment_wall_s=elapsed)
                b.write_json(meta_path,metadata)
                print(f"[{b.now()}] {mode} {point.key}: {complete}/{discs*npoints}; segment {elapsed/60:.2f} min",flush=True)
    except BaseException as exc:
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=True);pool=None
        metadata.update(status="failed",error=f"{type(exc).__name__}: {exc}",updated_utc=b.now())
        b.write_json(meta_path,metadata);raise
    finally:
        if pool is not None:pool.shutdown(wait=True)
    if probe:
        metadata.update(status="completed",completed_samples=complete,segment_wall_s=perf_counter()-started)
        b.write_json(meta_path,metadata)
        b.write_json(b.STATISTICS/"validation/timing_probe.json",dict(status="passed",configuration=asdict(point),
                     rays=complete,wall_s=perf_counter()-started,notes="Exact saved launch positions; timing/validation only, excluded from production statistics."))
        return metadata
    from .matched_validation import validate_point_retention
    validate_point_retention(point,folder)
    if point_signature(point,mode,discs,npoints)[0]!=signature:
        raise RuntimeError("configuration/source changed during execution; evidence retained without publication")
    summary=summarize(point,folder,discs,npoints,signature)
    metadata.update(status="completed",completed_samples=complete,updated_utc=b.now(),segment_wall_s=perf_counter()-started)
    b.write_json(meta_path,metadata);publish_point(point,folder)
    return summary


def completed():
    result={}
    points,_=points_and_scans()
    for point in points:
        _,folder,_=paths(point)
        if (folder/"run_metadata.json").exists() and (folder/"summary.json").exists():
            if b.read_json(folder/"run_metadata.json")["status"]=="completed":
                result[point.index]=b.read_json(folder/"capture_counts_by_disc.json")
    return result


def comparison_manifest(points,master_digest):
    """Verify the reviewed common inputs against immutable benchmark files."""
    original=b.read_json(b.BENCHMARK/"campaign_manifest.json")
    assert original["vapor"]==asdict(VaporLoadingParameters())
    master=(b.BENCHMARK/"launch_geometry_master.csv").read_text(encoding="utf-8")
    rows=[]
    for point in points[:64]:
        source=b.BENCHMARK/"production"/f"{point.index:03d}_configuration_{point.index}"
        prior=b.read_json(source/"run_metadata.json")
        payload=prior["signature_payload"];cooling=payload["apparatus_config"]["cooling"]
        cfg,app,coil,beams=b.configuration(point)
        assert cooling["wavelength_m"]==cfg.cooling_wavelength_m==b.WAVELENGTH_M
        assert cooling["power_w_per_beam"]==cfg.cooling_power_w_per_beam
        assert cooling["detuning_hz"]==point.cooling_detuning_mhz*1e6
        assert cooling["beam_diameter_m"]==point.cooling_diameter_mm*.001
        assert payload["coil_config"]==asdict(coil)
        assert payload["simple_mot_config"]["include_gravity"]==cfg.include_gravity
        assert b.sha256(source/"launch_geometry.csv")==prior["geometry_sha256"]
        publication=b.read_json(source/"publication.json")
        counts_digest=b.sha256(source/"capture_counts_by_disc.json")
        assert counts_digest==publication["files_sha256"]["capture_counts_by_disc.json"]
        assert master.startswith((source/"launch_geometry.csv").read_text(encoding="utf-8"))
        old_search=payload["search_config"]
        assert b.sample_count(point)==(old_search["disc_count"],old_search["points_per_disc"])
        for name,value in (("disc_radius_m",.015),("radial_distance_m",.015),("time_step_s",5e-6),
                           ("trap_core_radius_m",.002),("escape_radius_m",.03),("required_core_entries",2),
                           ("bounded_core_residence_s",.005),("analysis_velocity_step_m_per_s",.25),
                           ("analysis_velocity_max_m_per_s",80.)):
            assert old_search[name]==value,(point.index,name)
        rows.append(dict(index=point.index,controls=asdict(point),geometry_sha256=prior["geometry_sha256"],
                         benchmark_capture_counts_sha256=counts_digest,
                         disc_count=old_search["disc_count"],points_per_disc=50,status="matched"))
    record=dict(status="approved_and_matched",reviewed_utc=b.now(),
                authorization="User explicitly requested identical benchmark controls and exact saved samples, plus the specified multilevel-only repump studies; confirmed the grids and extended runtime on November 4, 2026.",
                common_configuration_count=64,master_geometry_sha256=master_digest,common_controls=rows,
                intended_atomic_model_differences=b.model_notes(),
                comparison_views="Original five views plus a second cooling-diameter comparison with matched repump diameter; repump-only scans have no two-level analogue.")
    b.write_json(b.STATISTICS/"comparison_manifest.json",record)
    return record


def initialize():
    points,scans=points_and_scans();positions,directions,digest=b.load_master_geometry()
    assert b.read_json(b.BENCHMARK/"final_audit.json")["status"]=="passed"
    manifest=dict(schema=1,campaign=b.NAME,created_from_two_level=str(b.BENCHMARK),
                  benchmark_manifest_sha256=b.sha256(b.BENCHMARK/"campaign_manifest.json"),
                  benchmark_final_audit_sha256=b.sha256(b.BENCHMARK/"final_audit.json"),
                  model=b.model_notes(),settings=selected_settings(),points=[asdict(p) for p in points],scans=scans,
                  per_point_sampling={str(p.index):dict(zip(("discs","points_per_disc"),b.sample_count(p))) for p in points},
                  master_geometry_sha256=digest,velocities_m_per_s=b.VELOCITIES.tolist(),
                  vapor=asdict(VaporLoadingParameters()),source_sha256=b.source_hashes(),libraries=library_versions())
    path=b.STATISTICS/"campaign_manifest.json"
    if path.exists() and b.read_json(path)!=json.loads(json.dumps(manifest)):
        raise ValueError("campaign manifest changed; preserve existing results and record a reviewed revision")
    b.write_json(path,manifest)
    comparison_manifest(points,digest)
    snapshot=b.STATISTICS/"source_snapshot"
    for relative,digest in manifest["source_sha256"].items():
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(b.ROOT/relative,target)
        if b.sha256(target)!=digest:raise ValueError("saved source snapshot disagrees with frozen manifest")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("probe","validate","production","report","audit","all"),nargs="?",default="all")
    parser.add_argument("--workers",type=int,default=20)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=24:parser.error("workers must be 1..24")
    b.STATISTICS.mkdir(parents=True,exist_ok=True)
    points,scans=points_and_scans()
    if args.stage in ("probe","production","all"):acquire_run_lock()
    if args.stage=="probe":run_point(points[0],args.workers,probe=True);return 0
    if args.stage in ("validate","all"):
        from .matched_validation import run_preflight
        run_preflight(points)
        if args.stage=="validate":return 0
    initialize()
    from .matched_reporting import refresh_report,audit_campaign
    if args.stage=="audit":audit_campaign();return 0
    if args.stage=="report":refresh_report();return 0
    validation=b.read_json(b.STATISTICS/"validation/preflight.json")
    if validation["status"]!="passed" or validation["checked_configurations"]!=len(points):
        raise ValueError("complete physical-kernel validation is required before production")
    refresh_report()
    try:
        order=[0]
        # Complete shared benchmark sweeps before the additional repump studies.
        for ids in scans.values():
            for index in ids:
                if index not in order:order.append(index)
        for i in order:
            run_point(points[i],args.workers);refresh_report()
        audit_campaign();refresh_report("completed")
    except BaseException as exc:
        b.write_json(b.STATISTICS/"failure.json",dict(time_utc=b.now(),error=f"{type(exc).__name__}: {exc}"))
        refresh_report("failed");raise
    return 0


if __name__=="__main__":raise SystemExit(main())
