"""Independently verify excluded dense masks without publishing pilot rates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np

from mot_strange_review_overlays import load_resolution
from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json

ADAPTIVE = ((1., 2.), (4., 8.))
FIXED = ((5e-6, 2.5e-6), (1.25e-6, .625e-6), (.3125e-6, .15625e-6))
SPEEDS = np.r_[np.arange(0., 80.25, .25), np.arange(85., 121., 5.)]
BASE = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "pilot"
_AUDITOR_FINGERPRINT = sha256(Path(__file__).read_bytes() + Path("scripts/mot_strange_review_overlays.py").read_bytes()).hexdigest()


def auditor_fingerprint():
    return _AUDITOR_FINGERPRINT


def pair_codes(a, b):
    """Vector form of the declared category, horizon and 10 µm spatial gate."""
    with np.errstate(invalid="ignore"):
        same = (a[:, 0] >= 0) & (a[:, 0] == b[:, 0])
        spatial = ((a[:, 1] == b[:, 1])
                   & (np.linalg.norm(a[:, 2:5] - b[:, 2:5], axis=1) <= 1e-5)
                   & (np.linalg.norm(a[:, 9:12] - b[:, 9:12], axis=1) <= 1e-5)
                   & (np.max(np.abs(a[:, 12:15] - b[:, 12:15]), axis=1) <= 1e-5))
    return np.where(same & ((a[:, 0] == 0) | spatial), a[:, 0], -1).astype(int)


def edge_neighbours(codes):
    labels = np.where(np.isin(codes, (1, 2)), codes, 0)
    return {j for edge in np.flatnonzero(labels[:-1] != labels[1:])
            for j in range(max(0, edge - 1), min(len(codes), edge + 3))}


def raw_mask_proof(codes, keys, records, failures, half_length):
    n = len(codes)
    controls = [(method, step) for method, pairs in (("adaptive", ADAPTIVE), ("fixed", FIXED))
                for pair in pairs for step in pair]
    values = {key: np.full((n, 15), np.nan) for key in controls}
    tried = {key: np.zeros(n, dtype=bool) for key in controls}
    if records.shape != (len(keys), 15) or not np.isin(records[:, 0], (-1, 0, 1, 2, 3)).all():
        raise ValueError("Invalid dense observation rows")
    if not np.isfinite(records[:, 1:12]).all():
        raise ValueError("Nonfinite dense trajectory state")
    retained = records[:, 0] > 0
    if (not np.isfinite(records[retained, 12:]).all()
            or not np.all(records[retained, 1] == 2.)):
        raise ValueError("Retained evidence did not reach the declared horizon")
    classifications = np.where((records[:, 8] >= 0) & (records[:, 13] <= .002), 1,
                               np.where((records[:, 12] > .002) & (records[:, 14] <= .001), 2, 3))
    if not np.array_equal(classifications[retained], records[retained, 0]):
        raise ValueError("Category does not match saved residence evidence")
    wall = records[:, 0] == 0
    if not np.all(np.max(np.abs(records[wall, 2:5]), axis=1) > half_length):
        raise ValueError("Wall classification lacks an actual cell exit")
    for key, row in [*zip(keys, records, strict=True), *((key, None) for key, _ in failures)]:
        index, method, step = int(key[0]), str(key[1]), float(key[2])
        control = method, step
        if control not in values or not 0 <= index < n or tried[control][index]:
            raise ValueError("Duplicate or undeclared observation control")
        tried[control][index] = True
        if row is not None:
            values[control][index] = row
    if not (tried[("adaptive", 1.)] & tried[("adaptive", 2.)]).all():
        raise ValueError("Missing initial paired observations")
    first = pair_codes(values[("adaptive", 1.)], values[("adaptive", 2.)])
    second = pair_codes(values[("adaptive", 4.)], values[("adaptive", 8.)])
    adaptive = np.where(first >= 0, first, second)
    fixed = np.full(n, -1)
    for a, b in FIXED:
        candidate = pair_codes(values[("fixed", a)], values[("fixed", b)])
        fixed = np.where(fixed >= 0, fixed, candidate)
    fixed_tried = np.logical_or.reduce([tried[("fixed", step)] for pair in FIXED for step in pair])
    mandatory = set(np.flatnonzero((first < 0) | tried[("adaptive", 4.)])) | {0} | edge_neighbours(codes)
    if not mandatory.issubset(set(np.flatnonzero(fixed_tried))):
        raise ValueError("Original adaptive disagreements or edges lack fixed proof")
    expected = np.where(fixed_tried, fixed, adaptive)
    if not np.array_equal(expected, codes):
        raise ValueError("Stored mask disagrees with independent paired reduction")
    return set(np.flatnonzero(fixed_tried))


def extra_fixed_proof(path, signature, codes):
    """Accept only original-neighbour categories with declared fixed evidence."""
    verified, hashes = set(), {}
    raw_sha = sha256(path.read_bytes()).hexdigest()
    for source in (path.parent / "fixed_neighbour_review").glob(f"{path.stem}_*.json"):
        content = source.read_bytes()
        report = json.loads(content)
        if (report.get("original_checkpoint_sha256") != raw_sha
                or report.get("prepared_sha256") != signature or report.get("ray") != path.name
                or report.get("horizon_s") != 2. or report.get("window_s") != .1
                or report.get("observer_source_sha256") != sha256(Path("src/pmot/mot_strange/loading_scan_retention.py").read_bytes()).hexdigest()):
            raise ValueError("Neighbour review inputs differ")
        for node in report["nodes"]:
            index = node["node"]
            if node["speed_m_per_s"] != SPEEDS[index] or codes[index] < 0:
                raise ValueError("Neighbour review is not for an original definitive node")
            pair = tuple(node["fixed_pair_s"])
            if pair not in FIXED:
                raise ValueError("Undeclared neighbour fixed controls")
            from mot_strange_review_overlays import record
            rows = {a["resolution"]: record(a) for a in node["attempts"]
                    if a["method"] == "fixed" and "code" in a}
            if not all(step in rows for step in pair):
                continue
            packed = [np.r_[r[0], r[1], r[2], r[3], r[4], r[5], r[6:]] for r in (rows[s] for s in pair)]
            category = int(pair_codes(np.array([packed[0]]), np.array([packed[1]]))[0])
            if category >= 0 and category == codes[index] == node.get("fixed_agreement"):
                verified.add(index)
                hashes[str(index)] = sha256(content).hexdigest()
    return verified, hashes


def check_ray(payload):
    index, disc, point, signature, positions, directions, half_length = payload
    path = BASE / f"configuration_{index:03d}" / f"disc_{disc:04d}_point_{point:02d}.npz"
    content = path.read_bytes()
    with np.load(BytesIO(content), allow_pickle=False) as z:
        if (tuple(int(z[k]) for k in ("configuration", "disc", "point")) != (index, disc, point)
                or z["prepared_sha256"].item() != signature):
            raise ValueError(f"Dense input identity differs in {path}")
        np.testing.assert_array_equal(z["position"], positions[disc, point])
        np.testing.assert_array_equal(z["direction"], directions[disc])
        np.testing.assert_array_equal(z["speeds"], SPEEDS)
        codes = z["codes"]
        fixed = raw_mask_proof(codes, z["attempt_keys"], z["attempts"], json.loads(z["failure_json"].item()), half_length)
        overlay = load_resolution(path, np.arange(len(codes))[1:], codes[1:], signature, SPEEDS)
        effective = codes.copy()
        verified = z["offcore_verified"].copy()
        pockets = json.loads(z["pocket_json"].item())
        if set(pockets) != {str(k) for k in np.flatnonzero(codes == 2)}:
            raise ValueError("Original pocket witnesses differ from candidate mask")
        for node, (category, pocket, _) in overlay.items():
            effective[node] = category
            if pocket is not None:
                pockets[str(node)] = pocket
                verified[node] = pocket["status"] == "stable_pocket_and_return_verified"
        if set(pockets) != {str(k) for k in np.flatnonzero(effective == 2)}:
            raise ValueError("Effective pocket witnesses differ")
        if set(map(str, np.flatnonzero(verified))) != {k for k, v in pockets.items() if v["status"] == "stable_pocket_and_return_verified"}:
            raise ValueError("Verified off-core labels differ from pocket proof")
        if np.any((SPEEDS >= 79.75) & ((effective == 1) | verified)) or bool(z["near_tail"]):
            raise ValueError("Dense capture requires velocity-tail extension")
        if int(z["unresolved"]) != np.count_nonzero(codes[1:] < 0):
            raise ValueError("Dense unresolved count differs")
        extra, extra_hashes = extra_fixed_proof(path, signature, codes)
        missing = sorted(edge_neighbours(effective) - fixed - extra)
        return dict(ray=path.name, sha256=sha256(content).hexdigest(),
                    summary_row=[disc, point, float(z["elapsed_s"]), int(z["unresolved"]),
                                 bool(z["near_tail"]), int(z["offcore_verified"].sum())],
                    missing_fixed_neighbours=missing, missing_speeds_m_per_s=SPEEDS[missing].tolist(),
                    numerical_review_sha256={str(k): v[2] for k, v in overlay.items()},
                    neighbour_review_sha256=extra_hashes,
                    unknown_zero=bool(effective[0] < 0), nodes=len(codes))


def audit(index):
    manifest = json.loads((ROOT / "pilot_manifest.json").read_text())
    published = ROOT / "pilot" / f"configuration_{index:03d}"
    summary = json.loads((published / "checkpoint_summary.json").read_text())
    n, m = manifest["discs"], manifest["points_per_disc"]
    if len(summary["rows"]) != n * m:
        raise ValueError("Dense setting is incomplete")
    with np.load(published / "runtime_inputs.npz", allow_pickle=False) as z:
        data = tuple(z[f"arr_{i}"] for i in range(12))
    signature = digest(data)
    if summary["data_sha256"] != signature or len(list((BASE / published.name).glob("disc_*_point_*.npz"))) != n * m:
        raise ValueError("Dense summary fingerprint or checkpoint count differs")
    replays = [json.loads(p.read_text()) for p in published.glob("independent_spot_replay_*.json")]
    if not any(r.get("status") == "passed" and r.get("input_sha256") == signature
               and r.get("requested_category") is None and len(r.get("rays", [])) >= 2 for r in replays):
        raise ValueError("Dense independent two-ray replay is pending")
    production = BASE.parent / "sampled_production" / published.name / "runtime_inputs.npz"
    with np.load(production, allow_pickle=False) as z:
        if signature != digest(tuple(z[f"arr_{i}"] for i in range(12))):
            raise ValueError("Dense and independently audited production physical inputs differ")
    with np.load(ROOT / "pilot_launch_geometry.npz", allow_pickle=False) as z:
        positions, directions = z["positions"], z["directions"]
    tasks = [(index, d, p, signature, positions, directions, data[11][15]) for d, p in np.ndindex(n, m)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rays = list(pool.map(check_ray, tasks))
    if [r["summary_row"] for r in rays] != summary["rows"]:
        raise ValueError("Dense summary rows differ from immutable checkpoint metadata")
    missing = [r for r in rays if r["missing_fixed_neighbours"]]
    stream = sha256()
    for r in rays:
        stream.update(bytes.fromhex(r["sha256"]))
    report = dict(status="requires_fixed_neighbour_review" if missing else "passed", configuration=index,
                  checked_utc=now(), rays=n * m, nodes=sum(r["nodes"] for r in rays),
                  prepared_sha256=signature, checkpoint_stream_sha256=stream.hexdigest(),
                  auditor_sha256=auditor_fingerprint(),
                  unknown_zero_rays=sum(r["unknown_zero"] for r in rays),
                  numerical_review_sha256={r["ray"]: r["numerical_review_sha256"] for r in rays if r["numerical_review_sha256"]},
                  neighbour_review_sha256={r["ray"]: r["neighbour_review_sha256"] for r in rays if r["neighbour_review_sha256"]},
                  missing_fixed_neighbours=missing, production_intervals_unchanged=True,
                  role="Excluded dense diagnostic evidence only; no pilot loading rates published")
    write_json(published / "dense_evidence_audit.json", report)
    print(f"Dense {index:03d}: {report['status']}", flush=True)
    return report


def finalize_dense():
    """Recheck retained bytes and review dependencies after all settings pass."""
    from audit_mot_strange_sampled_production import verify_retained_evidence
    manifest = json.loads((ROOT / "pilot_manifest.json").read_text())
    hashes, zero_count, nodes = {}, 0, 0
    for index in range(len(manifest["points"])):
        name = f"configuration_{index:03d}"
        path = ROOT / "pilot" / name / "dense_evidence_audit.json"
        result = json.loads(path.read_text())
        if (result["status"] != "passed" or result["configuration"] != index
                or result["auditor_sha256"] != auditor_fingerprint()):
            raise ValueError(f"Dense setting {index} is not completely verified")
        folder = BASE / name
        spec = dict(discs=manifest["discs"], points_per_disc=manifest["points_per_disc"])
        verify_retained_evidence(folder, spec, result)
        with np.load(ROOT / "pilot" / name / "runtime_inputs.npz", allow_pickle=False) as z:
            if digest(tuple(z[f"arr_{i}"] for i in range(12))) != result["prepared_sha256"]:
                raise ValueError("Retained dense runtime inputs changed")
        for ray, references in result["neighbour_review_sha256"].items():
            available = {sha256(p.read_bytes()).hexdigest()
                         for p in (folder / "fixed_neighbour_review").glob(f"{Path(ray).stem}_*.json")}
            if not set(references.values()).issubset(available):
                raise ValueError("Retained neighbour review changed")
        hashes[str(index)] = sha256(path.read_bytes()).hexdigest()
        zero_count += result["unknown_zero_rays"]
        nodes += result["nodes"]
    report = dict(status="passed", checked_utc=now(), settings=len(hashes),
                  launch_lines=len(hashes) * manifest["discs"] * manifest["points_per_disc"],
                  velocity_nodes=nodes, unresolved_positive_nodes=0, unknown_literal_zero_nodes=zero_count,
                  raw_checkpoint_bytes_and_review_dependencies_rechecked=True,
                  independent_geometry_regeneration=True, setting_audit_sha256=hashes,
                  auditor_sha256=auditor_fingerprint(), production_intervals_unchanged=True,
                  role="Excluded dense diagnostic only; no pooling or rate promotion")
    write_json(ROOT / "final_dense_audit.json", report)
    print("All dense diagnostic evidence passed final retained-file verification.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int, nargs="?")
    parser.add_argument("--available", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if sum((args.configuration is not None, args.available, args.finalize)) != 1:
        parser.error("Choose one setting, --available, or --finalize")
    from audit_mot_strange_sampled_production import load_provenance
    load_provenance()
    from pmot.mot_strange.loading_scan_inputs import current_symmetric_surrogate, sample_launch_geometry
    manifest = json.loads((ROOT / "pilot_manifest.json").read_text())
    assert manifest["seed"] == 2026110601 and manifest["horizon_s"] == 2. and manifest["final_window_s"] == .1
    expected = sample_launch_geometry(manifest["discs"], manifest["points_per_disc"], manifest["seed"], current_symmetric_surrogate())
    with np.load(ROOT / "pilot_launch_geometry.npz", allow_pickle=False) as z:
        for key, values in zip(("positions", "directions", "areas"), expected, strict=True):
            np.testing.assert_array_equal(z[key], values)
    if args.configuration is not None:
        audit(args.configuration)
    elif args.available:
        fingerprint = auditor_fingerprint()
        for i in range(69):
            folder = ROOT / "pilot" / f"configuration_{i:03d}"
            if not (folder / "checkpoint_summary.json").is_file():
                continue
            cached = folder / "dense_evidence_audit.json"
            if cached.is_file():
                saved = json.loads(cached.read_text())
                if saved["status"] == "passed" and saved["auditor_sha256"] == fingerprint:
                    continue
            try:
                audit(i)
            except Exception as error:
                write_json(ROOT / "dense_evidence_failure.json", {
                    "configuration": i, "checked_utc": now(), "error": repr(error),
                    "later_settings": "not run by this audit invocation", "production_intervals_unchanged": True,
                })
                raise
    elif args.finalize:
        finalize_dense()
