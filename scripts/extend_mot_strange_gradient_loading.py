"""Extend the frozen strange-MOT scans to 40 G/cm without rerunning old points."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np

from run_mot_strange_loading_pilot import ROOT as ORIGINAL, digest, write_json
from pmot.mot_strange import loading_scan_inputs as inputs


OUTPUT = ORIGINAL / "extension_40G_20261111"
CACHE = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / ORIGINAL.name / OUTPUT.name
CHICAGO = ZoneInfo("America/Chicago")
GRADIENTS = tuple(range(22, 41, 2))
DISCS, POINTS = 800, 16
PILOT_LAUNCH_SEED, PILOT_SPEED_SEED = 2026111101, 2026111102
PRODUCTION_LAUNCH_SEED, PRODUCTION_SPEED_SEED = 2026110602, 2026110609
WORKER_INPUTS = None


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def point_definitions():
    geometry = inputs.current_symmetric_surrogate()
    rows = []
    for name in ("isotropic", "axial"):
        unit = inputs.ScanPoint(name, 1.)
        calibration = inputs.field_axes(inputs.field_config(unit, geometry, isotropic_axial_helicity=-1))
        slope = abs(calibration["origin_surrogate_field_jacobian_g_per_cm"][2][2])
        for gradient in GRADIENTS:
            point = inputs.ScanPoint(name, gradient / slope)
            field = inputs.field_config(point, geometry, isotropic_axial_helicity=-1)
            rows.append({
                **asdict(point), **inputs.field_axes(field), **inputs.cooling_saturation(point),
                "target_axial_gradient_g_per_cm": gradient,
            })
    return rows


def physical_point(row):
    return inputs.ScanPoint(**{key: row[key] for key in inputs.ScanPoint.__dataclass_fields__})


def prepare():
    original = read(ORIGINAL / "pilot_manifest.json")
    for name, expected in original["source_sha256"].items():
        if checksum(Path(name)) != expected:
            raise ValueError(f"Original frozen source changed: {name}")
    sources = {**original["source_sha256"], **{
        name: checksum(Path(name)) for name in (
            "scripts/extend_mot_strange_gradient_loading.py",
            "scripts/run_mot_strange_sampled_production.py",
            "scripts/mot_strange_sampled_design.py",
            "scripts/mot_strange_pocket_recovery.py",
        )
    }}
    rows = point_definitions()
    old_max = max(abs(p["origin_surrogate_field_jacobian_g_per_cm"][2][2])
                  for p in original["points"])
    if any(row["target_axial_gradient_g_per_cm"] <= old_max for row in rows):
        raise ValueError("Extension overlaps an existing gradient")
    document = {
        "source_sha256": sources,
        "original_manifest_sha256": checksum(ORIGINAL / "pilot_manifest.json"),
        "original_results_sha256": checksum(ORIGINAL / "final_loading_results.json"),
        "points": rows, "discs": DISCS, "points_per_disc": POINTS,
        "pilot_launch_seed": PILOT_LAUNCH_SEED, "pilot_speed_seed": PILOT_SPEED_SEED,
        "production_launch_seed": PRODUCTION_LAUNCH_SEED,
        "production_speed_seed": PRODUCTION_SPEED_SEED,
        "configuration_offset": len(original["points"]),
        "horizon_s": original["horizon_s"], "window_s": original["final_window_s"],
        "vapor": original["vapor"], "geometry": original["geometry"],
        "allocation": "Fixed 800 direction discs x 16 points, matching the original production budget per setting.",
        "numerical_policy": "Original 0.125 m/s stratified quadrature, two draws per stratum, unchanged retention and pocket tests.",
    }
    OUTPUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "manifest.json"
    if path.exists() and read(path) != document:
        raise ValueError("Extension manifest or frozen source differs")
    write_json(path, document)
    prepared = []
    for i, row in enumerate(rows):
        point = physical_point(row)
        field = inputs.field_config(point, inputs.current_symmetric_surrogate(), isotropic_axial_helicity=-1)
        data = inputs.prepare(point, field)
        for array in data:
            array.setflags(write=False)
        folder = CACHE / f"configuration_{i + document['configuration_offset']:03d}"
        folder.mkdir(exist_ok=True)
        file = folder / "runtime_inputs.npz"
        if file.exists():
            with np.load(file) as saved:
                if digest(tuple(saved[f"arr_{j}"] for j in range(len(saved.files)))) != digest(data):
                    raise ValueError("Saved extension inputs changed")
        else:
            np.savez(file, *data)
        prepared.append(data)
    return document, prepared


def initialize_worker(prepared, geometry, phase, protocol, indices):
    from time_mot_strange_loading_rays import configure_worker
    from mot_strange_sampled_design import FINE_GRID, strata_for_grid
    from pmot.loading import VaporLoadingParameters

    configure_worker()
    strata = strata_for_grid(FINE_GRID, VaporLoadingParameters(**protocol["vapor"]))
    global WORKER_INPUTS
    WORKER_INPUTS = {}
    for index in indices:
        data = prepared[index - protocol["configuration_offset"]]
        for array in data:
            array.setflags(write=False)
        folder = CACHE / phase / f"configuration_{index:03d}"
        WORKER_INPUTS[index] = (index, data, digest(data), geometry, folder, protocol, strata)


def run_ray(task):
    import run_mot_strange_sampled_production as runner

    runner.STATE = WORKER_INPUTS[task[0]]
    return runner.run_ray(task)


def run_phase(phase, workers, document, prepared):
    import run_mot_strange_sampled_production as runner

    pilot = phase == "pilot"
    n, m = (8, 4) if pilot else (DISCS, POINTS)
    seed = document[f"{phase}_launch_seed"]
    geometry = inputs.sample_launch_geometry(n, m, seed, inputs.current_symmetric_surrogate())
    start_index = document["configuration_offset"]
    indices = [start_index+i for i in (0, 9, 10, 19)] if pilot else list(range(start_index, start_index+20))
    protocol = {**document, "speed_seed": document[f"{phase}_speed_seed"],
                "production_plan_sha256": checksum(OUTPUT / "manifest.json")}
    phase_cache = CACHE / phase
    phase_cache.mkdir(exist_ok=True)
    np.savez(phase_cache / "launch_geometry.npz", positions=geometry[0], directions=geometry[1], areas=geometry[2])
    completed, pending = [], []
    for index in indices:
        folder = phase_cache / f"configuration_{index:03d}"
        folder.mkdir(exist_ok=True)
        signature = digest(prepared[index-start_index])
        for disc, point in np.ndindex(n, m):
            file = folder / f"disc_{disc:04d}_point_{point:02d}.npz"
            if file.exists():
                completed.append((index, *runner.saved_row(file, index, disc, point, geometry, signature)))
            else:
                pending.append((index, disc, point))
    total, initial = len(completed)+len(pending), len(completed)
    began = perf_counter()

    def progress():
        elapsed = perf_counter()-began
        fresh = len(completed)-initial
        now = datetime.now(CHICAGO)
        eta = now + timedelta(seconds=elapsed*(total-len(completed))/fresh) if fresh >= 32 else None
        row = {
            "phase": phase, "pid": os.getpid(), "workers": workers,
            "updated_chicago": now.isoformat(), "finished": len(completed), "expected": total,
            "elapsed_process_s": elapsed, "eta_chicago": eta.isoformat() if eta else None,
            "unresolved_nodes": sum(r[4] for r in completed), "tail_hits": sum(r[5] for r in completed),
        }
        write_json(OUTPUT / f"{phase}_progress.json", row)
        print(json.dumps(row), flush=True)

    progress()
    last = perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                 initargs=(prepared, geometry, phase, protocol, indices)) as pool:
            queue = iter(pending)
            active = {}

            def schedule():
                job = next(queue, None)
                if job is not None:
                    active[pool.submit(run_ray, job)] = job[0]

            for _ in range(workers*2):
                schedule()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    completed.append((active.pop(future), *future.result()))
                    schedule()
                if perf_counter()-last > 30:
                    progress()
                    last = perf_counter()
    progress()
    by_setting = {}
    for index in indices:
        rows = [r for r in completed if r[0] == index]
        by_setting[str(index)] = {
            "rays": len(rows), "worker_seconds": sum(r[3] for r in rows),
            "unresolved_nodes": sum(r[4] for r in rows), "tail_hits": sum(r[5] for r in rows),
        }
    write_json(OUTPUT / f"{phase}_completion.json", by_setting)
    if any(row["unresolved_nodes"] or row["tail_hits"] for row in by_setting.values()):
        raise ValueError(f"{phase} has unresolved or near-tail evidence requiring review")
    return by_setting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "production", "run"))
    parser.add_argument("--workers", type=int, choices=range(1,17), default=16)
    args = parser.parse_args()
    document, prepared = prepare()
    if args.phase == "prepare":
        print(f"Prepared {len(prepared)} new settings at 22..40 G/cm")
        return
    import msvcrt

    with (CACHE / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if args.phase in ("pilot", "run"):
            run_phase("pilot", args.workers, document, prepared)
        if args.phase in ("production", "run"):
            pilot = read(OUTPUT / "pilot_completion.json")
            if any(row["unresolved_nodes"] or row["tail_hits"] for row in pilot.values()):
                raise ValueError("Pilot evidence needs review")
            run_phase("production", args.workers, document, prepared)


if __name__ == "__main__":
    main()
