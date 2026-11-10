"""Compare sampled-node and dense trajectories on separate validation launches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path

from time_mot_strange_loading_rays import configure_worker


VALIDATION_SEED = 2026110605
SPEED_SEED = 2026110606


def validate_point(index, prepared, positions, directions, folder, manifest):
    configure_worker()
    import numpy as np

    from benchmark_mot_strange_stratified_rays import compare_nodes
    from diagnose_mot_strange_velocity_sampling import velocity_strata
    from pmot.loading import VaporLoadingParameters
    from pmot.mot_strange.loading_scan_retention import observe
    from run_mot_strange_loading_pilot import digest, initialize_worker, now, run_ray, write_json

    folder.mkdir(parents=True, exist_ok=True)
    signature = digest(prepared)
    initialize_worker(index, prepared, signature, positions, directions, folder)
    np.savez(folder / "runtime_inputs.npz", *prepared)
    strata = velocity_strata(VaporLoadingParameters(**manifest["vapor"]))
    rows = []
    rng = np.random.default_rng(np.random.SeedSequence([SPEED_SEED, index]))
    observe(positions[0, 0], directions[0] * 80., prepared, manifest["horizon_s"],
            manifest["final_window_s"], 1.)
    for disc, point in np.ndindex(positions.shape[:2]):
        metadata = run_ray((index, disc, point))
        path = folder / f"disc_{disc:04d}_point_{point:02d}.npz"
        if metadata[3] or metadata[4]:
            rows.append({"ray": path.name, "status": "dense_requires_review",
                         "unresolved_positive_nodes": metadata[3], "near_tail": metadata[4]})
        else:
            rows.append(compare_nodes(path, prepared, signature, strata, rng, manifest))
        write_json(folder / "progress.json", {"status": "running", "rows": rows, "updated_utc": now()})
    return {
        "configuration": index, "input_sha256": signature,
        "status": ("requires_review" if any(row.get("status") or row.get("differences") for row in rows)
                   else "diagnostic_agreement"),
        "rows": rows,
    }


def main():
    import numpy as np

    from plan_mot_strange_loading_production import read_json
    from pmot.mot_strange.loading_scan_inputs import (
        current_symmetric_surrogate, field_config, prepare, sample_launch_geometry, scan_points,
    )
    from resume_mot_strange_loading_pilot import write_with_retry
    from run_mot_strange_loading_pilot import NAME, ROOT, now, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = read_json(ROOT / "pilot_manifest.json")
    if manifest["seed"] == VALIDATION_SEED:
        raise ValueError("Validation launches must be independent of the pilot")
    geometry = current_symmetric_surrogate()
    points, scans = scan_points()
    indices = list(dict.fromkeys([
        0, points.index(scans["trap_power_isotropic"][0]),
        points.index(scans["trap_power_axial"][0]),
        points.index(scans["trap_power_axial"][-1]),
        points.index(scans["trap_power_isotropic"][-1]),
        points.index(scans["cooling_detuning"][-1]),
        points.index(scans["cooling_diameter"][0]),
    ]))
    positions, directions, areas = sample_launch_geometry(2, 1, VALIDATION_SEED, geometry)
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_node_holdout"
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / "launch_geometry.npz", positions=positions, directions=directions, areas=areas)
    payloads = [prepare(points[i], field_config(points[i], geometry, isotropic_axial_helicity=-1))
                for i in indices]
    report = {"status": "running", "launch_seed": VALIDATION_SEED, "speed_seed": SPEED_SEED,
              "configurations": indices, "points": []}
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        futures = [
            pool.submit(validate_point, index, data, positions, directions,
                        cache / f"configuration_{index:03d}", manifest)
            for index, data in zip(indices, payloads, strict=True)
        ]
        for future in as_completed(futures):
            result = future.result()
            report["points"].append(result)
            write_json(cache / "validation_progress.json", {**report, "updated_utc": now()})
            print(result["configuration"], result["status"], flush=True)
    report["status"] = ("requires_review" if any(r["status"] != "diagnostic_agreement"
                                               for r in report["points"]) else "diagnostic_agreement")
    write_json(cache / "validation_progress.json", {**report, "updated_utc": now()})
    write_with_retry(ROOT / "pilot_diagnostics" / "sampled_node_holdout.json", {**report, "updated_utc": now()})


if __name__ == "__main__":
    main()
