"""Run the frozen fine-grid strange-MOT production plan."""

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


STATE = None


def initialize(index, data, signature, geometry, folder, protocol):
    configure_worker()
    from mot_strange_sampled_design import FINE_GRID, strata_for_grid
    from pmot.loading import VaporLoadingParameters
    from run_mot_strange_loading_pilot import digest

    for array in data:
        array.setflags(write=False)
    if digest(data) != signature:
        raise ValueError("Production worker input signature changed")
    global STATE
    STATE = (index, data, signature, geometry, folder, protocol,
             strata_for_grid(FINE_GRID, VaporLoadingParameters(**protocol["vapor"])))


def classify(position, direction, speeds, data, horizon, window):
    import numpy as np

    from mot_strange_pocket_recovery import check_pocket
    from pmot.mot_strange.loading_scan_retention import OFF_CORE, observe_grid

    codes, attempts = observe_grid(position, direction, speeds, data, horizon=horizon, window=window)
    pockets = {}
    verified = np.zeros(len(speeds), dtype=bool)
    for node in np.flatnonzero(codes == OFF_CORE):
        witness = next(record for (j, _, _), record in reversed(attempts.items())
                       if j == node and not isinstance(record, str) and record[0] == OFF_CORE)
        pockets[int(node)] = check_pocket(witness, data)
        verified[node] = pockets[int(node)]["status"] == "stable_pocket_and_return_verified"
    return codes, verified, attempts, pockets


def evidence_arrays(attempts):
    import numpy as np

    successful = [(key, row) for key, row in attempts.items() if not isinstance(row, str)]
    return {
        "attempt_keys": [key for key, _ in successful],
        "attempts": [np.r_[row[0], row[1], row[2], row[3], row[4], row[5], row[6:]]
                     for _, row in successful],
        "failure_json": json.dumps([(key, row) for key, row in attempts.items() if isinstance(row, str)]),
    }


def save_checkpoint(path, payload):
    import numpy as np

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def run_ray(task):
    import numpy as np

    from mot_strange_sampled_design import FINE_GRID, draw_nodes
    from pmot.mot_strange.loading_scan_retention import CORE

    index, disc, point = task
    bound, data, signature, geometry, folder, protocol, strata = STATE
    if index != bound:
        raise ValueError("Production worker received a different setting")
    positions, directions, _ = geometry
    r, direction = positions[disc, point], directions[disc]
    draws = draw_nodes(strata, protocol["speed_seed"], index, disc, point)
    unique, inverse = np.unique(draws, return_inverse=True)
    started = perf_counter()
    # Evaluate disconnected sampled speeds separately. Their gaps are not capture edges.
    outcomes = [
        classify(r, direction, FINE_GRID[[node]], data, protocol["horizon_s"], protocol["window_s"])
        for node in unique
    ]
    labels = np.array([codes[0] for codes, _, _, _ in outcomes], dtype=np.int8)
    verified = np.array([found[0] for _, found, _, _ in outcomes], dtype=bool)
    attempts = {(int(node), method, resolution): row
                for node, (_, _, trace, _) in zip(unique, outcomes, strict=True)
                for (_, method, resolution), row in trace.items()}
    pockets = {int(node): proof[0] for node, (_, _, _, proof) in zip(unique, outcomes, strict=True) if proof}
    inverse = inverse.reshape(draws.shape)
    tail_hit = bool(np.any((FINE_GRID[unique] >= 79.75) & ((labels == CORE) | verified)))
    unresolved = int(np.count_nonzero(labels < 0))
    payload = {
        "configuration": index, "disc": disc, "point": point, "position": r, "direction": direction,
        "prepared_sha256": signature, "node_indices": draws, "codes": labels[inverse],
        "offcore_verified": verified[inverse], "pocket_json": json.dumps(pockets),
        "elapsed_s": perf_counter() - started, "tail_hit": tail_hit, "unresolved": unresolved,
        **evidence_arrays(attempts),
    }
    save_checkpoint(folder / f"disc_{disc:04d}_point_{point:02d}.npz", payload)
    return disc, point, float(payload["elapsed_s"]), unresolved, tail_hit


def saved_row(path, index, disc, point, geometry, fingerprint):
    import numpy as np

    with np.load(path, allow_pickle=False) as ray:
        if (ray["prepared_sha256"].item() != fingerprint
                or not np.array_equal(ray["position"], geometry[0][disc, point])
                or not np.array_equal(ray["direction"], geometry[1][disc])
                or tuple(int(ray[k]) for k in ("configuration", "disc", "point")) != (index, disc, point)):
            raise ValueError(f"Production checkpoint input mismatch in {path}")
        return disc, point, float(ray["elapsed_s"]), int(ray["unresolved"]), bool(ray["tail_hit"])


def run(workers, cache):
    import numpy as np

    from mot_strange_sampled_design import FINE_GRID
    from plan_mot_strange_loading_production import read_json
    from pmot.mot_strange.loading_scan_inputs import (
        current_symmetric_surrogate, field_config, prepare, sample_launch_geometry, scan_points,
    )
    from resume_mot_strange_loading_pilot import write_with_retry
    from run_mot_strange_loading_pilot import ROOT, digest, now, write_json

    plan_path = ROOT / "sampled_production_plan.json"
    plan = read_json(plan_path)
    dense_path = ROOT / "pilot_manifest.json"
    dense = read_json(dense_path)
    if (plan["status"] != "frozen" or plan["dense_manifest_sha256"] != sha256(dense_path.read_bytes()).hexdigest()
            or not np.array_equal(plan["velocity_grid_m_per_s"], FINE_GRID)):
        raise ValueError("Production requires the compatible frozen fine-grid plan")
    for path, expected in dense["source_sha256"].items():
        if sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source differs: {path}")
    for index, expected in enumerate(plan["refinement_check_sha256"]):
        path = ROOT / "pilot_diagnostics" / f"velocity_convergence_{index:03d}.json"
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen refinement evidence differs: {path}")
    fine_replay = ROOT / "pilot_diagnostics" / "fine_node_replay.json"
    if sha256(fine_replay.read_bytes()).hexdigest() != plan["fine_replay_sha256"]:
        raise ValueError("Frozen fine-grid replay evidence differs")
    point_definitions, _ = scan_points()
    if set(plan["settings"]) != set(map(str, range(len(point_definitions)))):
        raise ValueError("Missing setting in the frozen plan")
    protocol = {
        "production_plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
        "source_sha256": {str(Path(name)): sha256(Path(name).read_bytes()).hexdigest() for name in (
            Path(__file__).resolve().relative_to(Path.cwd()), "scripts/mot_strange_sampled_design.py",
            "scripts/mot_strange_pocket_recovery.py",
        )},
        "horizon_s": dense["horizon_s"], "window_s": dense["final_window_s"],
        "vapor": dense["vapor"], "speed_seed": plan["production_speed_seed"],
    }
    frozen = cache / "runtime_protocol.json"
    if frozen.exists() and read_json(frozen) != protocol:
        raise ValueError("Existing production runtime protocol differs")
    write_json(frozen, protocol)
    geometry = current_symmetric_surrogate()
    all_launches = sample_launch_geometry(
        max(row["discs"] for row in plan["settings"].values()),
        max(row["points_per_disc"] for row in plan["settings"].values()),
        plan["production_launch_seed"], geometry,
    )
    np.savez(cache / "launch_geometry.npz", positions=all_launches[0], directions=all_launches[1],
             areas=all_launches[2])
    total = sum(row["rays"] for row in plan["settings"].values())
    finished = 0
    for index, point in enumerate(point_definitions):
        setting = plan["settings"][str(index)]
        if asdict(point) != {key: dense["points"][index][key] for key in asdict(point)}:
            raise ValueError(f"Physical setting {index} differs from the manifest")
        n, m = setting["discs"], setting["points_per_disc"]
        geometry_subset = (all_launches[0][:n, :m], all_launches[1][:n], all_launches[2][:n])
        data = prepare(point, field_config(point, geometry, isotropic_axial_helicity=-1))
        signature = digest(data)
        if signature != setting["prepared_sha256"]:
            raise ValueError(f"Production ARC or field inputs differ for setting {index}")
        folder = cache / f"configuration_{index:03d}"
        folder.mkdir(exist_ok=True)
        np.savez(folder / "runtime_inputs.npz", *data)
        rows, tasks = [], []
        for disc, ray in np.ndindex(n, m):
            path = folder / f"disc_{disc:04d}_point_{ray:02d}.npz"
            if path.exists():
                rows.append(saved_row(path, index, disc, ray, geometry_subset, signature))
            else:
                tasks.append((index, disc, ray))

        def progress():
            write_json(cache / "progress.json", {
                "status": "running", "updated_utc": now(), "configuration": index, "point": asdict(point),
                "active_point_finished": len(rows), "active_point_expected": n * m,
                "total_finished": finished + len(rows), "total_expected": total,
                "unresolved_nodes": sum(row[3] for row in rows), "tail_hits": sum(row[4] for row in rows),
            })

        progress()
        if tasks:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(
                index, data, signature, geometry_subset, folder, protocol,
            )) as pool:
                for future in as_completed([pool.submit(run_ray, task) for task in tasks]):
                    rows.append(future.result())
                    if len(rows) % 16 == 0:
                        progress()
        progress()
        review = any(row[3] or row[4] for row in rows)
        write_json(folder / "summary.json", {
            "status": "requires_review" if review else "checkpointed_pending_audit",
            "configuration": index, "prepared_sha256": signature,
            "rays": len(rows), "worker_seconds": sum(row[2] for row in rows),
        })
        print(f"Production {index:03d}: {len(rows)} rays, review={review}", flush=True)
        finished += len(rows)
    record = {"status": "checkpointed_pending_audit", "updated_utc": now(),
              "total_finished": finished, "total_expected": total}
    write_json(cache / "progress.json", record)
    write_with_retry(ROOT / "production_progress.json", record)


if __name__ == "__main__":
    from run_mot_strange_loading_pilot import NAME, now, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_production"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run(args.workers, cache)
        except Exception as error:
            write_json(cache / "failure.json", {"updated_utc": now(), "error": repr(error)})
            raise
