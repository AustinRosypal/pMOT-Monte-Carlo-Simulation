"""Time independent full-grid rays with the provisional retention criterion."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import os
from pathlib import Path
from time import perf_counter


def prepare_probe(point, seed, disc, point_index):
    """Prepare ARC-backed data in the parent before starting Windows workers."""
    from pmot.mot_strange.loading_scan_inputs import (
        current_symmetric_surrogate, field_config, prepare, sample_launch_geometry,
    )

    geometry = current_symmetric_surrogate()
    positions, directions, _ = sample_launch_geometry(disc + 1, point_index + 1, seed, geometry)
    field = field_config(point, geometry, isotropic_axial_helicity=-1)
    return positions[disc, point_index].copy(), directions[disc].copy(), prepare(point, field)


def time_ray(job, prepared):
    import numpy as np
    from pmot.mot_strange.loading_scan_retention import OFF_CORE, CORE, observe, observe_grid

    index, config, horizon = job
    r, direction, data = prepared
    observe(r, 80 * direction, data, horizon, .1, 1.)
    speeds = np.r_[np.arange(0., 80.25, .25), np.arange(85., 121., 5.)]
    began = perf_counter()
    codes, observations = observe_grid(r, direction, speeds, data, horizon=horizon)
    elapsed = perf_counter() - began
    return {
        "index": index, "point": config, "horizon_s": horizon,
        "launch_position_m": r.tolist(), "direction": direction.tolist(),
        "speeds_m_per_s": speeds.tolist(), "codes": codes.tolist(), "elapsed_s": elapsed,
        "adaptive_evaluations": sum(method == "adaptive" for _, method, _ in observations),
        "fixed_evaluations": sum(method == "fixed" for _, method, _ in observations),
        "unresolved_positive_nodes": speeds[(codes < 0) & (speeds > 0)].tolist(),
        "failed_attempts": [
            {"speed_m_per_s": float(speeds[node]), "method": method, "resolution": resolution,
             "error": value if isinstance(value, str) else None,
             "elapsed_physical_s": None if isinstance(value, str) else value[1]}
            for (node, method, resolution), value in observations.items()
            if isinstance(value, str) or value[0] < 0
        ],
        "tail_capture_nodes": speeds[np.isin(codes, (CORE, OFF_CORE)) & (speeds >= 79.75)].tolist(),
    }


def configure_worker():
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / "strange-loading-preflight"
    os.environ["NUMBA_CACHE_DIR"] = str(cache / f"process_{os.getpid()}")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, required=True)
    args = parser.parse_args()
    if not .1 < args.horizon <= 2:
        parser.error("probe horizon must be greater than 0.1 s and at most 2 s")
    from pmot.mot_strange.loading_scan_inputs import ScanPoint
    from validate_mot_strange_loading_scans import write_json

    points = (
        ScanPoint(), ScanPoint("axial", 1), ScanPoint("axial", 40),
        ScanPoint("isotropic", 1), ScanPoint("isotropic", 40),
        ScanPoint("axial", 10, -40), ScanPoint("axial", 10, -5),
        ScanPoint("axial", 10, -15, 5),
    )
    jobs = [(i, asdict(point), args.horizon) for i, point in enumerate(points)]
    inputs = [prepare_probe(point, 2026110503, index, 0) for index, point in enumerate(points)]
    results = []
    with ProcessPoolExecutor(
        max_workers=len(jobs), max_tasks_per_child=1, initializer=configure_worker,
    ) as pool:
        pending = {pool.submit(time_ray, job, prepared): job
                   for job, prepared in zip(jobs, inputs, strict=True)}
        for future in as_completed(pending):
            try:
                row = future.result()
            except Exception as error:
                index, point, horizon = pending[future]
                row = {"index": index, "point": point, "horizon_s": horizon, "error": repr(error)}
            results.append(row)
            output = f"parent_prepared_full_grid_{args.horizon:g}s_timing.json"
            write_json(output, sorted(results, key=lambda x: x["index"]))
            print({k: v for k, v in row.items() if k not in ("codes", "speeds_m_per_s")}, flush=True)


if __name__ == "__main__":
    main()
