"""Independently reduce and replay the added strange-MOT gradient settings."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path

os.environ["NUMBA_CACHE_DIR"] = str(
    Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / "strange-gradient-extension-audit" / f"process_{os.getpid()}")

import extend_mot_strange_gradient_loading as campaign
import numpy as np


def audit_setting(job):
    phase, index, dense = job
    from time_mot_strange_loading_rays import configure_worker

    configure_worker()
    from audit_mot_strange_sampled_production import check_sample, replay_ray
    from mot_strange_review_overlays import load_resolution
    from mot_strange_sampled_design import FINE_GRID, contribution, strata_for_grid
    from pmot.loading import VaporLoadingParameters
    from pmot.mot_strange.loading_scan_statistics import loading_interval

    document = campaign.read(campaign.OUTPUT / "manifest.json")
    cache = campaign.CACHE / phase
    folder = cache / f"configuration_{index:03d}"
    with np.load(cache / "launch_geometry.npz") as z:
        geometry = tuple(z[k] for k in ("positions", "directions", "areas"))
    with np.load(campaign.CACHE / folder.name / "runtime_inputs.npz") as z:
        data = tuple(z[f"arr_{i}"] for i in range(len(z.files)))
    for array in data:
        array.setflags(write=False)
    signature = campaign.digest(data)
    protocol = {**document, "speed_seed": document[f"{phase}_speed_seed"]}
    strata = strata_for_grid(FINE_GRID, VaporLoadingParameters(**document["vapor"]))
    masses = np.array([w.sum() for _, w in strata])
    n, m = geometry[0].shape[:2]
    categories = ("core", "offcore", "wall", "localized_only", "other_retained")
    rates = {key: np.zeros(n) for key in categories}
    counts = {key: 0 for key in categories}
    first = {}
    overlays = {}
    evidence = sha256()
    for disc, point in np.ndindex(n, m):
        path = folder / f"disc_{disc:04d}_point_{point:02d}.npz"
        evidence.update(bytes.fromhex(campaign.checksum(path)))
        with np.load(path, allow_pickle=False) as ray:
            overlay = load_resolution(path, ray["node_indices"], ray["codes"], signature) if int(ray["unresolved"]) else {}
            if overlay:
                overlays[path] = overlay
            masks = check_sample(ray, index, disc, point, signature, geometry, strata, protocol["speed_seed"], overlay)
        for name, mask in masks.items():
            rates[name][disc] += contribution(mask, masses, geometry[2][disc]) / m
            observed = int(np.count_nonzero(mask))
            counts[name] += observed
            if observed:
                first.setdefault(name, path)
    rng = np.random.default_rng(np.random.SeedSequence([2026111103, index, phase == "production"]))
    random = folder / f"disc_{rng.integers(n):04d}_point_{rng.integers(m):02d}.npz"
    paths = list(dict.fromkeys([random, *[first[k] for k in ("core", "offcore", "localized_only") if k in first],
                               *overlays]))
    replays = [replay_ray(path, data, protocol,
                          dense=dense and (path == random or path == first.get("core")),
                          overlay=overlays.get(path))
               for path in paths]
    output = campaign.OUTPUT / phase / folder.name
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "direction_disc_rates.npz", **rates)
    result = {
        "status": "passed", "configuration": index, "discs": n, "points_per_disc": m,
        "prepared_sha256": signature, "counts": counts,
        "checkpoint_stream_sha256": evidence.hexdigest(),
        "numerical_review_sha256": {
            path.name: {str(node): evidence[2] for node, evidence in overlay.items()}
            for path, overlay in overlays.items()},
        "disc_rates_sha256": campaign.checksum(output / "direction_disc_rates.npz"),
        "manifest_sha256": campaign.checksum(campaign.OUTPUT / "manifest.json"),
        "audit_source_sha256": campaign.checksum(Path(__file__)),
        "replays": replays, **{k: loading_interval(v) for k, v in rates.items()},
    }
    campaign.write_json(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "production"))
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=16)
    args = parser.parse_args()
    document = campaign.read(campaign.OUTPUT / "manifest.json")
    for name, expected in document["source_sha256"].items():
        if campaign.checksum(Path(name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if args.phase == "production":
        from validate_mot_strange_gradient_extension import main as validate_physics

        validate_physics()
    completion = campaign.read(campaign.OUTPUT / f"{args.phase}_completion.json")
    indices = sorted(map(int, completion))
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(audit_setting, (args.phase, i, args.phase == "production")): i for i in indices}
        for future in as_completed(jobs):
            result = future.result()
            results[str(jobs[future])] = result
            campaign.write_json(campaign.OUTPUT / f"{args.phase}_audit_progress.json",
                                {"finished": len(results), "expected": len(indices)})
            print(f"Audited {args.phase} configuration {jobs[future]}", flush=True)
    if args.phase == "production":
        if len(results) != 20:
            raise ValueError("All twenty extension settings are required")
        if campaign.checksum(campaign.ORIGINAL / "final_loading_results.json") != document["original_results_sha256"]:
            raise ValueError("Original results changed during the extension")
    campaign.write_json(campaign.OUTPUT / f"{args.phase}_results.json", results)
    print(f"All {len(results)} {args.phase} settings audited", flush=True)


if __name__ == "__main__":
    main()
