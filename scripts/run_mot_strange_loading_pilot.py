"""Run the independent two-second strange-MOT loading pilot."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import msvcrt
import os
from pathlib import Path
from time import perf_counter

from time_mot_strange_loading_rays import configure_worker


NAME = "symmetric_exact_D2_loading_20261105"
ROOT = Path("outputs/statistics/mot_strange") / NAME
DISC_COUNT, POINT_COUNT, PILOT_SEED = 32, 16, 2026110601
HORIZON_S, WINDOW_S = 2., .100
STATE = None


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(data):
    checksum = sha256()
    for array in data:
        checksum.update(str((array.dtype.str, array.shape)).encode())
        checksum.update(array.tobytes())
    return checksum.hexdigest()


def initialize_worker(configuration, arrays, expected_digest, positions, directions, folder):
    configure_worker()
    global STATE
    for array in arrays:
        array.setflags(write=False)
    if digest(arrays) != expected_digest:
        raise RuntimeError("prepared array fingerprint mismatch")
    STATE = (configuration, arrays, expected_digest, positions, directions, Path(folder))


def run_ray(task):
    import numpy as np
    from pmot.mot_strange.loading_scan_restoring import check_off_core_pocket
    from pmot.mot_strange.loading_scan_retention import CORE, OFF_CORE, observe_grid

    configuration, disc, point = task
    bound, data, signature, positions, directions, folder = STATE
    if bound != configuration:
        raise RuntimeError("worker received a different configuration")
    position, direction = positions[disc, point], directions[disc]
    speeds = np.r_[np.arange(0., 80.25, .25), np.arange(85., 121., 5.)]
    started = perf_counter()
    codes, attempts = observe_grid(position, direction, speeds, data,
                                   horizon=HORIZON_S, window=WINDOW_S)
    observations, keys, failed, pockets = [], [], [], {}
    offcore = np.zeros(len(speeds), dtype=bool)
    for (index, method, resolution), record in attempts.items():
        key = (index, method, resolution)
        if isinstance(record, str):
            failed.append((key, record))
            continue
        keys.append(key)
        observations.append(np.r_[record[0], record[1], record[2], record[3], record[4],
                                  record[5], record[6:]])
    for index in np.flatnonzero(codes == OFF_CORE):
        witness = next(record for (node, _, _), record in reversed(attempts.items())
                       if node == index and not isinstance(record, str) and record[0] == OFF_CORE)
        pocket = check_off_core_pocket(witness, data)
        pockets[int(index)] = pocket
        offcore[index] = pocket["status"] == "stable_pocket_and_return_verified"
    unresolved = int(np.count_nonzero(codes[1:] < 0))
    near_tail = bool(np.any((speeds >= 79.75) & ((codes == CORE) | offcore)))
    elapsed = perf_counter() - started
    output = folder / f"disc_{disc:04d}_point_{point:02d}.npz"
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, configuration=configuration, disc=disc, point=point, position=position,
            direction=direction, speeds=speeds, codes=codes, offcore_verified=offcore,
            prepared_sha256=signature, attempts=observations, attempt_keys=keys,
            failure_json=json.dumps(failed), pocket_json=json.dumps(pockets),
            elapsed_s=elapsed, unresolved=unresolved, near_tail=near_tail,
        )
    temporary.replace(output)
    return disc, point, elapsed, unresolved, near_tail, int(np.count_nonzero(offcore))


def run_pilot(workers):
    import numpy as np
    from pmot.loading import VaporLoadingParameters
    from pmot.mot_strange.loading_scan_inputs import (
        cooling_saturation, current_symmetric_surrogate, field_axes, field_config,
        prepare, sample_launch_geometry, scan_points,
    )

    configs, scans = scan_points()
    geometry = current_symmetric_surrogate()
    positions, directions, areas = sample_launch_geometry(DISC_COUNT, POINT_COUNT, PILOT_SEED, geometry)
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "pilot"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": NAME, "seed": PILOT_SEED, "discs": DISC_COUNT, "points_per_disc": POINT_COUNT,
        "horizon_s": HORIZON_S, "final_window_s": WINDOW_S, "radius_m": .015,
        "plane_distance_m": .010, "vapor": asdict(VaporLoadingParameters()),
        "geometry": geometry.metadata(),
        "scans": {name: [configs.index(point) for point in points] for name, points in scans.items()},
        "points": [
            {**asdict(point), **field_axes(field_config(point, geometry, isotropic_axial_helicity=-1)),
             **cooling_saturation(point)}
            for point in configs
        ],
        "source_sha256": {
            path.as_posix(): sha256(path.read_bytes()).hexdigest() for path in (
                Path(__file__).resolve().relative_to(Path.cwd()),
                *sorted(Path("src/pmot/mot_strange").glob("*.py")),
                Path("src/pmot/mot_multilevel/accelerated.py"),
                Path("src/pmot/mot_multilevel/adaptive.py"),
            )
        },
    }
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    manifest_path = ROOT / "pilot_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("saved pilot manifest differs from current inputs")
    write_json(manifest_path, manifest)
    np.savez(ROOT / "pilot_launch_geometry.npz", positions=positions, directions=directions, areas=areas)
    # Time widely separated regimes first, before the remainder of the ordered scan.
    priority = [0, 1, configs.index(scans["trap_power_axial"][0]),
                configs.index(scans["trap_power_axial"][-1]), configs.index(scans["trap_power_isotropic"][-1]),
                configs.index(scans["cooling_detuning"][-1]), configs.index(scans["cooling_diameter"][0])]
    completed = 0
    expected = len(configs) * DISC_COUNT * POINT_COUNT
    for index in dict.fromkeys([*priority, *range(len(configs))]):
        point = configs[index]
        folder = cache / f"configuration_{index:03d}"
        folder.mkdir(exist_ok=True)
        data = prepare(point, field_config(point, geometry, isotropic_axial_helicity=-1))
        signature = digest(data)
        published = ROOT / "pilot" / f"configuration_{index:03d}"
        published.mkdir(parents=True, exist_ok=True)
        np.savez(published / "runtime_inputs.npz", *data)
        tasks, rows = [], []
        for disc in range(DISC_COUNT):
            for ray in range(POINT_COUNT):
                path = folder / f"disc_{disc:04d}_point_{ray:02d}.npz"
                if not path.exists():
                    tasks.append((index, disc, ray))
                    continue
                with np.load(path) as saved:
                    if (saved["prepared_sha256"].item() != signature
                            or not np.array_equal(saved["position"], positions[disc, ray])
                            or not np.array_equal(saved["direction"], directions[disc])):
                        raise RuntimeError(f"checkpoint inputs changed in {path}")
                    rows.append((disc, ray, float(saved["elapsed_s"]), int(saved["unresolved"]),
                                 bool(saved["near_tail"]), int(saved["offcore_verified"].sum())))
        print(f"Pilot configuration {index:03d} {point}: {len(tasks)} pending rays", flush=True)

        def progress():
            write_json(ROOT / "pilot_progress.json", {
                "status": "running", "updated_utc": now(), "pid": os.getpid(),
                "configuration": index, "point": asdict(point),
                "active_point_finished": len(rows), "active_point_total": DISC_COUNT * POINT_COUNT,
                "total_finished": completed + len(rows), "total_expected": expected,
                "active_worker_seconds": sum(row[2] for row in rows),
                "unresolved_positive_nodes": sum(row[3] for row in rows),
                "rays_requiring_tail_extension": sum(row[4] for row in rows),
            })

        progress()
        if tasks:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=initialize_worker,
                initargs=(index, data, signature, positions, directions, folder),
            ) as pool:
                for future in as_completed([pool.submit(run_ray, task) for task in tasks]):
                    rows.append(future.result())
                    progress()
                    if len(rows) % 16 == 0 or len(rows) == 1:
                        print(f"Pilot {index:03d}: {len(rows)}/{DISC_COUNT * POINT_COUNT} rays", flush=True)
        write_json(published / "checkpoint_summary.json", {
            "status": "requires_numerical_review" if any(row[3] or row[4] for row in rows)
            else "ready_for_independent_replay",
            "data_sha256": signature, "rows": sorted(rows), "checkpoint_directory": str(folder),
        })
        completed += len(rows)
    write_json(ROOT / "pilot_progress.json", {
        "status": "checkpointed_pending_independent_replay", "updated_utc": now(),
        "total_finished": completed, "total_expected": expected,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "pilot.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run_pilot(args.workers)
        except Exception as error:
            write_json(ROOT / "pilot_failure.json", {
                "status": "failed", "updated_utc": now(), "error": repr(error),
            })
            raise


if __name__ == "__main__":
    main()
