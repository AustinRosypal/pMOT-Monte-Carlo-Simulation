"""Reproduce and repair the timeout endpoint minimum-radius bookkeeping only.

Original published data and source snapshots are retained. This migration
does not change trajectories, capture decisions, or loading estimates.
"""
import ast
import csv
import gzip
import hashlib
import json
import shutil
from dataclasses import asdict, replace

import numpy as np

from pmot.mot_simple import parameter_campaign as c
from pmot.mot_simple import refined_relationship_campaign as engine
from pmot.mot_simple.accelerated import classify_trajectory
from pmot.mot_simple.timeout_audit import point_from_sample, velocity_overrides_from_payload


def function_ast(path,name):
    tree=ast.parse(path.read_text(encoding="utf-8"))
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))


def main():
    csv.field_size_limit(128*1024*1024)
    record_path=c.STATISTICS/"terminal_radius_bookkeeping_migration.json"
    if record_path.exists():
        raise RuntimeError("migration already recorded; do not apply it twice")
    old_manifest=c.read_json(c.STATISTICS/"campaign_manifest.json")
    current_manifest=c.manifest()
    changed={k for k in current_manifest["source_sha256"]
             if current_manifest["source_sha256"][k]!=old_manifest["source_sha256"][k]}
    assert changed=={"pmot/mot_simple/accelerated.py","pmot/mot_simple/parameter_campaign.py"},changed
    snapshot=c.STATISTICS/"source_snapshot"/"src"/"pmot"/"mot_simple"
    original_accelerator=snapshot/"accelerated.py"
    current_accelerator=c.ROOT/"src"/"pmot"/"mot_simple"/"accelerated.py"
    assert hashlib.sha256(original_accelerator.read_bytes()).hexdigest()==old_manifest["source_sha256"]["pmot/mot_simple/accelerated.py"]
    for name in ("elliptic_ke","magnetic_field","acceleration","rk4","pack_parameters"):
        assert function_ast(original_accelerator,name)==function_ast(current_accelerator,name),name
    old_tree=ast.parse(original_accelerator.read_text(encoding="utf-8"))
    new_tree=ast.parse(current_accelerator.read_text(encoding="utf-8"))
    old_classifier=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name=="classify_numeric")
    new_classifier=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name=="classify_numeric")
    # The only classifier change is an extra endpoint-minimum update before
    # the existing return; the full integration/event loop is identical.
    del new_classifier.body[-2]
    assert ast.dump(new_classifier)==ast.dump(old_classifier)

    index=45
    paths=c.paths_for(index,working=True)
    metadata=c.read_json(paths.metadata_json)
    search=replace(engine.default_search_config(),**metadata["signature_payload"]["search_config"])
    samples=engine.load_capture_velocity_samples(paths.partial_samples_csv)
    sample_map={(s.disc_index,s.point_index):s for s in samples}
    rows=engine._load_audit_rows(engine._audit_path(paths))
    assert set(rows)==set(sample_map)
    _,simple,coil,beams=c.build_configuration(c.point_for(index))
    repairs=[]
    for key,row in rows.items():
        evidence=json.loads(row["adaptive_audit_evidence_json"] or "[]")
        touched=False
        for j,node in enumerate(evidence):
            for kind,step in (("coarse_result","coarse_time_step_s"),("fine_result","fine_time_step_s")):
                result=node[kind]
                if result["minimum_radius_m"]<=result["final_radius_m"]+1e-12:
                    continue
                assert result["termination_reason"]=="timeout"
                assert node["velocity_m_per_s"]==0.0
                replay=asdict(classify_trajectory(beams,point_from_sample(sample_map[key]),
                    node["velocity_m_per_s"],coil,simple,
                    replace(search,time_step_s=node[step],max_simulation_time_s=node["duration_s"])))
                for field in ("trapped","termination_reason","entered_trap_core","core_entry_count"):
                    assert result[field]==replay[field],(key,field)
                for field in ("elapsed_time_s","final_radius_m","final_position_m","final_velocity_m_per_s"):
                    np.testing.assert_allclose(result[field],replay[field],rtol=1e-10,atol=2e-12)
                corrected=min(result["minimum_radius_m"],result["final_radius_m"])
                assert abs(corrected-replay["minimum_radius_m"])<2e-12
                repairs.append(dict(key=list(key),node=j,kind=kind,original=result.copy(),
                                    corrected_minimum_radius_m=corrected,replayed=replay))
                result["minimum_radius_m"]=corrected
                touched=True
        if touched:
            row["adaptive_audit_evidence_json"]=json.dumps(evidence,separators=(",",":"))
    assert repairs,"expected inconsistent timeout diagnostics were not found"
    overrides=velocity_overrides_from_payload(c.read_json(engine._overrides_path(paths)))
    override_map={x.key:x for x in overrides}
    engine._validate_completed_audit_ledger(sample_map,rows,override_map,search=search)

    backup=c.STATISTICS/"provenance"/"terminal_radius_before_repair"
    backup.mkdir(parents=True,exist_ok=True)
    with engine._audit_path(paths).open("rb") as src,gzip.open(backup/"original_partial_audit.csv.gz","wb",compresslevel=1) as dst:
        shutil.copyfileobj(src,dst,1024*1024)
    c.write_json(backup/"original_partial_metadata.json",metadata)
    c.write_json(backup/"original_campaign_manifest.json",old_manifest)
    for filename in ("accelerated.py","parameter_campaign.py"):
        shutil.copyfile(snapshot/filename,backup/filename)

    plan=c.read_json(c.STATISTICS/"production_plan.json")
    completed={}
    prior_builder=engine.build_point_configuration
    engine.build_point_configuration=c.build_configuration
    try:
        for i in c.completed_payloads():
            published=c.paths_for(i)
            prior=c.read_json(published.metadata_json)
            point_search=replace(engine.default_search_config(),**plan["searches"][str(i)])
            payload=engine._point_signature_payload(c.point_for(i),point_search,prior["geometry_sha256"])
            completed[str(i)]=dict(original=prior["run_signature_sha256"],current=engine._signature(payload))
        updated_payload=engine._point_signature_payload(c.point_for(index),search,metadata["geometry_sha256"])
    finally:
        engine.build_point_configuration=prior_builder
    original_signature=metadata["run_signature_sha256"]
    metadata["signature_payload"]=updated_payload
    metadata["run_signature_sha256"]=engine._signature(updated_payload)
    metadata["status"]="interrupted"
    metadata["terminal_radius_bookkeeping_migration"]=str(record_path)
    engine._atomic_write_csv(engine._audit_path(paths),[rows[k] for k in sorted(rows)],engine.ENDPOINT_AUDIT_FIELDNAMES)
    c.write_json(paths.metadata_json,metadata)
    record=dict(status="verified_and_applied",updated_utc=c.now(),
                unchanged_force_field_and_rk4_ast=True,unchanged_integration_and_event_loop_ast=True,
                current_physics_source_sha256=engine._source_hashes(),
                completed_point_signatures=completed,
                original_partial_signature=original_signature,current_partial_signature=metadata["run_signature_sha256"],
                preserved_checkpoint_ray_count=len(samples),repairs=repairs,
                original_evidence_directory=str(backup),
                interpretation="Only timeout minimum-radius diagnostics changed. All captured/escaped decisions and loading estimates remain unchanged; original published results retain their original signatures.")
    c.write_json(record_path,record)
    c.write_json(c.STATISTICS/"campaign_manifest.json",current_manifest)
    for filename in ("accelerated.py","parameter_campaign.py"):
        shutil.copyfile(c.ROOT/"src"/"pmot"/"mot_simple"/filename,snapshot/filename)
    print(json.dumps(dict(repaired_diagnostics=len(repairs),preserved_rays=len(samples),completed_results_retained=len(completed))))


if __name__=="__main__":
    main()
