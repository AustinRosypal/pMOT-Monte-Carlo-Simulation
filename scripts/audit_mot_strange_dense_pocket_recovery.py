"""Independently replay each recovered pocket witness and its guarded search."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json
from time_mot_strange_loading_rays import configure_worker


def compare(actual, expected):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Pocket replay fields differ")
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, str):
        if actual != expected:
            raise ValueError("Pocket replay status differs")
    else:
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)


def missing_checkpoint_coverage(base, verified):
    """A lost trace must not silently remove a recovered pocket from the audit."""
    covered = {(str(Path(item["checkpoint"]).resolve()), item["node"]) for item in verified}
    missing = []
    for path in sorted(base.glob("configuration_*/disc_*_point_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            pockets = json.loads(z["pocket_json"].item())
        for node, pocket in pockets.items():
            if ("equilibrium_search_fallback" in pocket
                    and (str(path.resolve()), int(node)) not in covered):
                missing.append(dict(checkpoint=str(path), node=int(node)))
    return missing


def audit():
    configure_worker()
    from mot_strange_pocket_recovery import check_pocket
    from pmot.mot_strange.loading_scan_retention import observe

    revision = ROOT / "revisions" / "dense_pocket_recovery_20261108"
    preflight = json.loads((revision / "pre_resume_validation.json").read_text())
    source = Path("scripts/mot_strange_pocket_recovery.py")
    expected_source = preflight["source_sha256"][str(source)]
    if sha256(source.read_bytes()).hexdigest() != expected_source:
        raise ValueError("Guarded recovery source changed")
    manifest = json.loads((ROOT / "pilot_manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        if sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {path}")
    base = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "pilot"
    auditor_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    report_path = ROOT / "dense_pocket_recovery_audit.json"
    cached = {}
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if (previous.get("auditor_sha256") == auditor_sha
                and previous.get("fallback_source_sha256") == expected_source):
            cached = {(item["trace_sha256"], item["checkpoint_sha256"]): item
                      for item in previous["witness_and_pocket_replays"]}
    verified, pending = [], []
    for trace in sorted(base.glob("configuration_*/pocket_search_recovery/*.json")):
        content = trace.read_bytes()
        record = json.loads(content)
        configuration, disc, point = (record[key] for key in ("configuration", "disc", "point"))
        ray = base / f"configuration_{configuration:03d}" / f"disc_{disc:04d}_point_{point:02d}.npz"
        if not ray.exists():
            pending.append(str(trace))
            continue
        with np.load(ROOT / "pilot" / ray.parent.name / "runtime_inputs.npz", allow_pickle=False) as z:
            data = tuple(z[f"arr_{i}"] for i in range(12))
        for array in data:
            array.setflags(write=False)
        if (digest(data) != record["prepared_sha256"]
                or record["fallback_source_sha256"] != expected_source
                or record["cell_half_length_m"] != data[11][15]):
            raise ValueError("Recovery trace input identity differs")
        expected = np.asarray(record["observation"])
        rejected = np.asarray(record["pocket"]["equilibrium_search_fallback"]["rejected_probe_position_m"])
        if (np.linalg.norm(rejected - expected[2:5]) <= .001
                or np.max(np.abs(rejected)) <= data[11][15]):
            raise ValueError("Recovery did not reject an out-of-cell probe")
        raw_sha = sha256(ray.read_bytes()).hexdigest()
        cache_key = sha256(content).hexdigest(), raw_sha
        if cache_key in cached:
            verified.append(cached[cache_key])
            continue
        with np.load(ray, allow_pickle=False) as z:
            if z["prepared_sha256"].item() != record["prepared_sha256"]:
                raise ValueError("Recovered checkpoint input identity differs")
            matching = np.flatnonzero(np.all(z["attempts"] == expected, axis=1))
            if not len(matching):
                raise ValueError("Recovery witness is not a saved attempt")
            node, method, resolution = z["attempt_keys"][matching[-1]]
            node, resolution = int(node), float(resolution)
            if json.loads(z["pocket_json"].item())[str(node)] != record["pocket"]:
                raise ValueError("Recovered checkpoint pocket differs from trace")
            observation = observe(z["position"], z["speeds"][node] * z["direction"], data,
                                  2., .1, resolution if method == "adaptive" else 1.,
                                  resolution if method == "fixed" else 0.)
        packed = np.r_[observation[0], observation[1], observation[2], observation[3],
                       observation[4], observation[5], observation[6:]]
        np.testing.assert_allclose(packed, expected, rtol=0., atol=1e-12)
        compare(check_pocket(observation, data), record["pocket"])
        verified.append(dict(trace=str(trace), trace_sha256=sha256(content).hexdigest(),
                             checkpoint=str(ray), checkpoint_sha256=raw_sha,
                             node=node, method=str(method), resolution=resolution))
        write_json(report_path, dict(status="running", checked_utc=now(),
                                    witness_and_pocket_replays=verified, pending_traces=pending,
                                    fallback_source_sha256=expected_source, auditor_sha256=auditor_sha,
                                    scope="Verified rows are resumable; complete checkpoint coverage is checked at the end."))
    required = [base / "configuration_062" / name for name in preflight["missing_rays"]]
    missing = [str(path) for path in required if not path.exists()]
    missing_coverage = missing_checkpoint_coverage(base, verified)
    report = dict(status="pending" if pending or missing or missing_coverage else "passed", checked_utc=now(),
                  witness_and_pocket_replays=verified, pending_traces=pending,
                  pending_recovered_rays=missing, fallback_source_sha256=expected_source,
                  missing_checkpoint_recovery_coverage=missing_coverage,
                  auditor_sha256=auditor_sha,
                  frozen_sources_unchanged=True, production_intervals_unchanged=True)
    write_json(ROOT / "dense_pocket_recovery_audit.json", report)
    print(f"Recovery witnesses replayed: {len(verified)}; status: {report['status']}", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit()
