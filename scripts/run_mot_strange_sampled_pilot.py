"""Run the independent planning pilot for sampled velocity quadrature."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
import msvcrt
import os
from pathlib import Path
from time import perf_counter

from time_mot_strange_loading_rays import configure_worker


LAUNCH_SEED = 2026110607
SPEED_SEED = 2026110608
DISCS, POINTS_PER_DISC = 8, 8
STATE = None


def initialize(index, arrays, fingerprint, positions, directions, folder, vapor, horizon, window):
    configure_worker()
    from diagnose_mot_strange_velocity_sampling import velocity_strata
    from run_mot_strange_loading_pilot import digest

    for array in arrays:
        array.setflags(write=False)
    if digest(arrays) != fingerprint:
        raise ValueError("Sampled pilot runtime array fingerprint mismatch")
    global STATE
    STATE = (index, arrays, fingerprint, positions, directions, folder,
             velocity_strata(vapor), horizon, window)


def run_ray(task):
    import numpy as np

    from plan_mot_strange_loading_production import GRID
    from pmot.mot_strange.loading_scan_restoring import check_off_core_pocket
    from pmot.mot_strange.loading_scan_retention import OFF_CORE, observe_grid

    index, disc, point = task
    bound, arrays, fingerprint, positions, directions, folder, strata, horizon, window = STATE
    if index != bound:
        raise ValueError("Worker received a different configuration")
    r, direction = positions[disc, point], directions[disc]
    rng = np.random.default_rng(np.random.SeedSequence([SPEED_SEED, index, disc, point]))
    draws = np.array([rng.choice(nodes, size=2, p=weights / weights.sum()) for nodes, weights in strata])
    started = perf_counter()
    labels, evidence, keys, failures, pockets = {}, [], [], [], {}
    for node in sorted(np.unique(draws)):
        codes, attempts = observe_grid(r, direction, GRID[[node]], arrays, horizon=horizon, window=window)
        code = int(codes[0])
        verified = False
        for (_, method, resolution), record in attempts.items():
            key = int(node), method, resolution
            if isinstance(record, str):
                failures.append((key, record))
            else:
                keys.append(key)
                evidence.append(np.r_[record[0], record[1], record[2], record[3],
                                      record[4], record[5], record[6:]])
        if code == OFF_CORE:
            witness = next(record for record in reversed(attempts.values())
                           if not isinstance(record, str) and record[0] == OFF_CORE)
            pocket = check_off_core_pocket(witness, arrays)
            pockets[int(node)] = pocket
            verified = pocket["status"] == "stable_pocket_and_return_verified"
        labels[int(node)] = code, verified
    codes = np.array([[labels[int(node)][0] for node in row] for row in draws], dtype=np.int8)
    verified = np.array([[labels[int(node)][1] for node in row] for row in draws], dtype=bool)
    elapsed = perf_counter() - started
    path = folder / f"disc_{disc:04d}_point_{point:02d}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, configuration=index, disc=disc, point=point, position=r,
            direction=direction, prepared_sha256=fingerprint, elapsed_s=elapsed,
            node_indices=draws, codes=codes, offcore_verified=verified, attempts=evidence,
            attempt_keys=keys, failure_json=json.dumps(failures), pocket_json=json.dumps(pockets),
        )
    temporary.replace(path)
    return disc, point, elapsed, int(np.count_nonzero(codes < 0))


def summary_row(path, signature, index, disc, point, position, direction):
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        if (z["prepared_sha256"].item() != signature
                or tuple(int(z[key]) for key in ("configuration", "disc", "point")) != (index, disc, point)
                or not np.array_equal(z["position"], position)
                or not np.array_equal(z["direction"], direction)):
            raise ValueError(f"Checkpoint inputs differ in {path}")
        return disc, point, float(z["elapsed_s"]), int(np.count_nonzero(z["codes"] < 0))


def run_pilot(workers, cache):
    import numpy as np

    from plan_mot_strange_loading_production import read_json
    from pmot.loading import VaporLoadingParameters
    from pmot.mot_strange.loading_scan_inputs import (
        current_symmetric_surrogate, field_config, prepare, sample_launch_geometry, scan_points,
    )
    from run_mot_strange_loading_pilot import ROOT, digest, now, write_json
    from diagnose_mot_strange_velocity_sampling import STRATUM_EDGES_M_PER_S

    dense_path = ROOT / "pilot_manifest.json"
    dense = read_json(dense_path)
    vapor = VaporLoadingParameters(**dense["vapor"])
    geometry = current_symmetric_surrogate()
    positions, directions, areas = sample_launch_geometry(DISCS, POINTS_PER_DISC, LAUNCH_SEED, geometry)
    settings, scans = scan_points()
    priority = [0, settings.index(scans["trap_power_isotropic"][0]),
                settings.index(scans["trap_power_axial"][0]),
                settings.index(scans["trap_power_isotropic"][-1]),
                settings.index(scans["trap_power_axial"][-1]),
                settings.index(scans["cooling_detuning"][-1]),
                settings.index(scans["cooling_diameter"][0])]
    plan = {
        "stage": "sampled_pilot_diagnostic", "dense_manifest_sha256": sha256(dense_path.read_bytes()).hexdigest(),
        "launch_seed": LAUNCH_SEED, "speed_seed": SPEED_SEED, "discs": DISCS,
        "points_per_disc": POINTS_PER_DISC, "draws_per_stratum": 2,
        "velocity_stratum_edges_m_per_s": STRATUM_EDGES_M_PER_S.tolist(),
        "settings": [asdict(point) for point in settings],
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    frozen = cache / "manifest.json"
    if frozen.exists() and read_json(frozen) != plan:
        raise ValueError("Sampled pilot manifest differs from the frozen plan")
    write_json(frozen, plan)
    np.savez(cache / "launch_geometry.npz", positions=positions, directions=directions, areas=areas)
    completed = 0
    for index in dict.fromkeys([*priority, *range(len(settings))]):
        point = settings[index]
        folder = cache / f"configuration_{index:03d}"
        folder.mkdir(exist_ok=True)
        arrays = prepare(point, field_config(point, geometry, isotropic_axial_helicity=-1))
        signature = digest(arrays)
        np.savez(folder / "runtime_inputs.npz", *arrays)
        tasks, rows = [], []
        for disc, ray in np.ndindex(positions.shape[:2]):
            path = folder / f"disc_{disc:04d}_point_{ray:02d}.npz"
            if path.exists():
                rows.append(summary_row(path, signature, index, disc, ray, positions[disc, ray],
                                        directions[disc]))
            else:
                tasks.append((index, disc, ray))

        def progress():
            write_json(cache / "progress.json", {
                "status": "running", "updated_utc": now(), "configuration": index,
                "point": asdict(point), "active_point_finished": len(rows),
                "total_finished": completed + len(rows), "total_expected": len(settings) * DISCS * POINTS_PER_DISC,
                "active_worker_seconds": sum(row[2] for row in rows),
                "unresolved_draws": sum(row[3] for row in rows),
            })

        progress()
        if tasks:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=initialize,
                initargs=(index, arrays, signature, positions, directions, folder, vapor,
                          dense["horizon_s"], dense["final_window_s"]),
            ) as pool:
                for future in as_completed([pool.submit(run_ray, task) for task in tasks]):
                    rows.append(future.result())
                    progress()
        write_json(folder / "summary.json", {
            "status": "requires_review" if any(row[3] for row in rows) else "checkpointed",
            "configuration": index, "rows": sorted(rows), "prepared_sha256": signature,
        })
        print(f"Sampled pilot {index:03d}: {len(rows)} rays, "
              f"{sum(row[2] for row in rows) / len(rows):.3f} worker-seconds per ray.", flush=True)
        completed += len(rows)
    write_json(cache / "progress.json", {
        "status": "checkpointed_pending_validation", "updated_utc": now(),
        "total_finished": completed, "total_expected": len(settings) * DISCS * POINTS_PER_DISC,
    })


if __name__ == "__main__":
    from run_mot_strange_loading_pilot import NAME, now, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_pilot"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run_pilot(args.workers, cache)
        except Exception as error:
            write_json(cache / "failure.json", {"updated_utc": now(), "error": repr(error)})
            raise
