"""Verify production checkpoints and publish independently reduced loading rates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

from time_mot_strange_loading_rays import configure_worker

if __name__ == "__main__":
    configure_worker()

import numpy as np

from mot_strange_sampled_design import FINE_GRID, contribution, draw_nodes, strata_for_grid
from mot_strange_review_overlays import effective_classification, load_resolution, verify_reused_attempts
from plan_mot_strange_loading_production import read_json
from pmot.loading import VaporLoadingParameters
from pmot.mot_strange.loading_scan_retention import CORE, OFF_CORE, OTHER, WALL
from pmot.mot_strange.loading_scan_statistics import loading_interval
from resume_mot_strange_loading_pilot import write_with_retry
from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json


AUDIT_SEED = 2026110610
CACHE = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_production"
OUTPUT = ROOT / "production"


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def load_provenance():
    plan_path = ROOT / "sampled_production_plan.json"
    plan = read_json(plan_path)
    manifest_path = ROOT / "pilot_manifest.json"
    manifest = read_json(manifest_path)
    protocol = read_json(CACHE / "runtime_protocol.json")
    if (plan["status"] != "frozen" or protocol["production_plan_sha256"] != file_sha(plan_path)
            or plan["dense_manifest_sha256"] != file_sha(manifest_path)):
        raise ValueError("Production plan or source manifest changed")
    for name, expected in {**manifest["source_sha256"], **protocol["source_sha256"]}.items():
        if file_sha(Path(name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    with np.load(CACHE / "launch_geometry.npz", allow_pickle=False) as z:
        geometry = tuple(z[key] for key in ("positions", "directions", "areas"))
    vapor = VaporLoadingParameters(**protocol["vapor"])
    return plan, manifest, protocol, geometry, strata_for_grid(FINE_GRID, vapor)


def check_sample(ray, index, disc, point, data_sha, geometry, strata, seed, overlay=None):
    draws = draw_nodes(strata, seed, index, disc, point)
    raw_unknown = np.unique(ray["node_indices"][ray["codes"] < 0])
    codes, verified, pockets = effective_classification(ray, overlay or {})
    if (tuple(int(ray[key]) for key in ("configuration", "disc", "point")) != (index, disc, point)
            or ray["prepared_sha256"].item() != data_sha
            or not np.array_equal(ray["position"], geometry[0][disc, point])
            or not np.array_equal(ray["direction"], geometry[1][disc])
            or not np.array_equal(ray["node_indices"], draws)
            or codes.shape != draws.shape or verified.shape != draws.shape
            or not np.isin(codes, (WALL, CORE, OFF_CORE, OTHER)).all()
            or int(ray["unresolved"]) != len(raw_unknown)
            or bool(ray["tail_hit"])):
        raise ValueError(f"Unresolved, tail, or incompatible evidence at {index}/{disc}/{point}")
    if np.any((FINE_GRID[draws] >= 79.75) & ((codes == CORE) | verified)):
        raise ValueError(f"Capture approaches the velocity cutoff at {index}/{disc}/{point}")
    if (set(pockets) != {str(x) for x in draws[codes == OFF_CORE]}
            or {str(x) for x in draws[verified]} !=
            {key for key, entry in pockets.items() if entry["status"] == "stable_pocket_and_return_verified"}):
        raise ValueError(f"Off-core witnesses do not match at {index}/{disc}/{point}")
    return {
        "core": codes == CORE, "offcore": verified, "wall": codes == WALL,
        "localized_only": (codes == OFF_CORE) & ~verified, "other_retained": codes == OTHER,
    }


def replay_ray(path, data, protocol, dense, overlay=None):
    """Recompute selected observations with the preserved solver and finer mask."""
    from pmot.mot_strange.loading_scan_retention import agreed_category, observe
    from run_mot_strange_sampled_production import classify, evidence_arrays

    with np.load(path, allow_pickle=False) as ray:
        r, direction, nodes = ray["position"], ray["direction"], ray["node_indices"]
        codes, verified, saved_pockets = effective_classification(ray, overlay or {})
        trace = list(zip(ray["attempt_keys"], ray["attempts"], strict=True))
    marked = {int(nodes.flat[0]), int(nodes.flat[-1])}
    for category in (CORE, OFF_CORE):
        if np.any(codes == category):
            candidates = nodes[codes == category]
            marked.add(int(candidates[len(candidates) // 2]))
    checked = 0
    for key, expected in trace:
        node, method, resolution = int(key[0]), key[1], float(key[2])
        if node not in marked or expected[0] < 0:
            continue
        actual = observe(r, FINE_GRID[node] * direction, data, protocol["horizon_s"],
                         protocol["window_s"], resolution if method == "adaptive" else 1.,
                         resolution if method == "fixed" else 0.)
        row = np.r_[actual[0], actual[1], actual[2], actual[3], actual[4], actual[5], actual[6:]]
        np.testing.assert_allclose(row, expected, rtol=0., atol=1e-12)
        checked += 1
    if not checked:
        raise ValueError(f"No sampled attempts were independently replayed in {path.name}")
    result = {"ray": path.name, "independently_replayed_attempts": checked}
    if dense:
        speeds = np.r_[FINE_GRID, np.arange(85., 121., 5.)]
        observed, confirmed, attempts, pockets = classify(
            r, direction, speeds, data, protocol["horizon_s"], protocol["window_s"],
        )
        # Preserve the dense test even if its acceptance gates fail.
        output = path.parent / "audit"
        output.mkdir(exist_ok=True)
        from run_mot_strange_sampled_production import save_checkpoint
        save_checkpoint(output / path.name, {
            "velocities": speeds, "codes": observed, "offcore_verified": confirmed,
            "pocket_json": json.dumps(pockets), **evidence_arrays(attempts),
        })
        for node, (category, pocket, _) in (overlay or {}).items():
            verified_pocket = bool(pocket and pocket["status"] == "stable_pocket_and_return_verified")
            if observed[node] >= 0 and (observed[node] != category or confirmed[node] != verified_pocket):
                raise ValueError(f"Dense classification disagrees with the independent numerical review in {path.name}")
            observed[node] = category
            confirmed[node] = verified_pocket
        if np.any(observed[1:] < 0) or np.any((speeds >= 79.75) & ((observed == CORE) | confirmed)):
            raise ValueError(f"Dense validation has an unresolved or tail capture node in {path.name}")
        np.testing.assert_array_equal(observed[nodes], codes)
        np.testing.assert_array_equal(confirmed[nodes], verified)
        result["dense_node_count"] = len(speeds)
    if np.any(codes == CORE):
        positive = nodes[codes == CORE]
        speed = float(FINE_GRID[positive[len(positive) // 2]])
        fixed = [observe(r, speed * direction, data, 2.25, protocol["window_s"], 1., step)
                 for step in (5e-6, 2.5e-6)]
        if agreed_category(*fixed) != CORE:
            raise ValueError(f"Longer independent fixed core replay did not agree in {path.name}")
        result["extended_core"] = {
            "speed_m_per_s": speed, "horizon_s": 2.25, "fixed_steps_s": [5e-6, 2.5e-6],
            "maximum_final_window_radius_m": [float(obs[7]) for obs in fixed],
        }
    if np.any(verified):
        from mot_strange_pocket_recovery import check_pocket
        from pmot.mot_strange.loading_scan_restoring import NEIGHBOURHOOD_M

        selected = nodes[verified]
        node = int(selected[len(selected) // 2])
        speed = float(FINE_GRID[node])
        fixed = [observe(r, speed * direction, data, 2.25, protocol["window_s"], 1., step)
                 for step in (5e-6, 2.5e-6)]
        if agreed_category(*fixed) != OFF_CORE:
            raise ValueError(f"Longer independent fixed off-core replay did not agree in {path.name}")
        pocket = check_pocket(fixed[-1], data)
        if (pocket["status"] != "stable_pocket_and_return_verified"
                or np.linalg.norm(np.asarray(pocket["center_m"]) -
                                  saved_pockets[str(node)]["center_m"]) > NEIGHBOURHOOD_M):
            raise ValueError(f"Longer replay did not verify the same restoring pocket in {path.name}")
        result["extended_offcore"] = {
            "speed_m_per_s": speed, "horizon_s": 2.25, "fixed_steps_s": [5e-6, 2.5e-6],
            "pocket": pocket,
        }
    return result


def audit(index, reviewed=False):
    plan, _, protocol, geometry, strata = load_provenance()
    spec = plan["settings"][str(index)]
    folder = CACHE / f"configuration_{index:03d}"
    summary = read_json(folder / "summary.json")
    accepted = ("checkpointed_pending_audit", "requires_review") if reviewed else ("checkpointed_pending_audit",)
    if summary["status"] not in accepted or summary["rays"] != spec["rays"]:
        raise ValueError(f"Configuration {index} is incomplete or requires review")
    with np.load(folder / "runtime_inputs.npz", allow_pickle=False) as z:
        data = tuple(z[f"arr_{i}"] for i in range(len(z.files)))
    for array in data:
        array.setflags(write=False)
    signature = digest(data)
    if signature != spec["prepared_sha256"]:
        raise ValueError(f"Prepared input hash changed for {index}")
    masses = np.array([weight.sum() for _, weight in strata])
    n, m = spec["discs"], spec["points_per_disc"]
    categories = ("core", "offcore", "wall", "localized_only", "other_retained")
    disc_rates = {name: np.zeros(n) for name in categories}
    counts = {name: 0 for name in categories}
    evidence = sha256()
    first = {}
    overlays = {}
    for disc, point in np.ndindex(n, m):
        path = folder / f"disc_{disc:04d}_point_{point:02d}.npz"
        evidence.update(bytes.fromhex(file_sha(path)))
        with np.load(path, allow_pickle=False) as ray:
            overlay = (load_resolution(path, ray["node_indices"], ray["codes"], signature)
                       if reviewed and int(ray["unresolved"]) else {})
            if overlay:
                overlays[path] = overlay
            masks = check_sample(ray, index, disc, point, signature, geometry, strata,
                                 protocol["speed_seed"], overlay)
        for name, mask in masks.items():
            disc_rates[name][disc] += contribution(mask, masses, geometry[2][disc]) / m
            observed = int(np.count_nonzero(mask))
            counts[name] += observed
            if observed:
                first.setdefault(name, path)
    rng = np.random.default_rng(np.random.SeedSequence([AUDIT_SEED, index]))
    d, p = int(rng.integers(n)), int(rng.integers(m))
    random_ray = folder / f"disc_{d:04d}_point_{p:02d}.npz"
    representatives = list(dict.fromkeys([random_ray, *[first[key] for key in
                                ("core", "offcore", "localized_only") if key in first]]))
    replays = [replay_ray(path, data, protocol, dense=path == random_ray or path == first.get("core"),
                          overlay=overlays.get(path))
               for path in representatives]
    folder_out = OUTPUT / f"configuration_{index:03d}"
    folder_out.mkdir(parents=True, exist_ok=True)
    np.savez(folder_out / "direction_disc_rates.npz", **disc_rates)
    record = {
        "status": "passed", "configuration": index, "discs": n, "points_per_disc": m,
        "prepared_sha256": signature, "production_plan_sha256": protocol["production_plan_sha256"],
        "checkpoint_stream_sha256": evidence.hexdigest(), "counts": counts,
        "numerical_review_sha256": {path.name: {str(node): resolution[2]
                                     for node, resolution in resolved.items()}
                                     for path, resolved in overlays.items()},
        "replays": replays, "disc_rates_sha256": file_sha(folder_out / "direction_disc_rates.npz"),
        **{name: loading_interval(rates) for name, rates in disc_rates.items()},
        "checked_utc": now(),
    }
    write_with_retry(folder_out / "audit.json", record)
    print(f"Audited production setting {index}: {n * m} launches, {len(replays)} independent replays.")


def verify_retained_evidence(folder, spec, result):
    """Recheck every retained checkpoint and referenced review after point audits."""
    files = list(folder.glob("disc_*_point_*.npz"))
    if len(files) != spec["discs"] * spec["points_per_disc"]:
        raise ValueError(f"Retained checkpoint count differs in {folder.name}")
    evidence = sha256()
    paths = (folder / f"disc_{disc:04d}_point_{point:02d}.npz"
             for disc, point in np.ndindex(spec["discs"], spec["points_per_disc"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        # map returns checksums in the original direction-disc/point order.
        for checksum in pool.map(file_sha, paths):
            evidence.update(bytes.fromhex(checksum))
    if evidence.hexdigest() != result["checkpoint_stream_sha256"]:
        raise ValueError(f"Retained checkpoint bytes changed in {folder.name}")
    for ray, nodes in result["numerical_review_sha256"].items():
        stem = Path(ray).stem
        available = {file_sha(path): path for path in (folder / "numerical_review").glob(f"{stem}_*.json")}
        if not set(nodes.values()).issubset(available):
            raise ValueError(f"Referenced numerical review changed for {folder.name}/{ray}")
        for checksum in set(nodes.values()):
            path = available[checksum]
            verify_reused_attempts(path, read_json(path))


def finalize():
    from scipy.stats import t
    from pmot.mot_strange.loading_scan_inputs import current_symmetric_surrogate, sample_launch_geometry

    plan, manifest, protocol, geometry, _ = load_provenance()
    independently_generated = sample_launch_geometry(
        geometry[0].shape[0], geometry[0].shape[1], plan["production_launch_seed"],
        current_symmetric_surrogate(),
    )
    for expected, actual in zip(geometry, independently_generated, strict=True):
        np.testing.assert_array_equal(expected, actual)
    results, checksums = {}, {}
    for index in range(len(manifest["points"])):
        folder = OUTPUT / f"configuration_{index:03d}"
        path = folder / "audit.json"
        result = read_json(path)
        if (result["status"] != "passed" or result["configuration"] != index
                or result["production_plan_sha256"] != protocol["production_plan_sha256"]
                or file_sha(folder / "direction_disc_rates.npz") != result["disc_rates_sha256"]):
            raise ValueError(f"Invalid final audit input for configuration {index}")
        spec = plan["settings"][str(index)]
        if (result["discs"], result["points_per_disc"]) != (spec["discs"], spec["points_per_disc"]):
            raise ValueError(f"Frozen counts changed for configuration {index}")
        verify_retained_evidence(CACHE / f"configuration_{index:03d}", spec, result)
        with np.load(folder / "direction_disc_rates.npz", allow_pickle=False) as archive:
            for name in ("core", "offcore", "wall", "localized_only", "other_retained"):
                rates = archive[name]
                if rates.shape != (spec["discs"],) or not np.all(np.isfinite(rates) & (rates >= 0)):
                    raise ValueError(f"Invalid direction-disc rates for {index}/{name}")
                mean = float(np.sum(rates) / len(rates))
                np.testing.assert_allclose(result[name]["mean_atoms_per_s"], mean, rtol=1e-14)
                if mean:
                    scatter = np.sum((rates - mean)**2)
                    half = t.ppf(.975, len(rates) - 1) * np.sqrt(scatter / (len(rates) * (len(rates) - 1)))
                    np.testing.assert_allclose(
                        [result[name]["lower_atoms_per_s"], result[name]["upper_atoms_per_s"],
                         result[name]["relative_half_width"]],
                        [mean - half, mean + half, half / mean], rtol=1e-12,
                    )
                elif any(result[name][key] is not None for key in (
                    "lower_atoms_per_s", "upper_atoms_per_s", "relative_half_width",
                )):
                    raise ValueError(f"Zero-event precision was assigned at {index}/{name}")
        results[str(index)] = result
        checksums[str(index)] = file_sha(path)
    destination = ROOT / "final_loading_results.json"
    write_with_retry(destination, results)
    write_with_retry(ROOT / "final_audit.json", {
        "status": "passed", "checked_utc": now(), "configuration_audit_sha256": checksums,
        "production_plan_sha256": protocol["production_plan_sha256"],
        "result_sha256": file_sha(destination),
        "retained_checkpoint_streams_and_review_hashes_rechecked": True,
        "production_launches": sum(spec["rays"] for spec in plan["settings"].values()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int, nargs="?")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--reviewed", action="store_true")
    args = parser.parse_args()
    if (args.configuration is None) == (not args.finalize):
        parser.error("Specify a configuration or --finalize")
    if args.finalize:
        finalize()
    else:
        audit(args.configuration, reviewed=args.reviewed)
