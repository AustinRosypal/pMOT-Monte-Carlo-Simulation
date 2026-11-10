"""Audit selected two-second retention nodes and keep partial attempt evidence."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from time import perf_counter

from time_mot_strange_loading_rays import configure_worker, prepare_probe


def probe(job, point, prepared):
    import numpy as np

    from pmot.mot_strange.loading_scan_restoring import check_off_core_pocket
    from pmot.mot_strange.loading_scan_retention import OFF_CORE, observe, observe_grid
    from probe_mot_strange_loading_retention import record

    speeds = job[-1]
    r, direction, data = prepared
    observe(r, 80 * direction, data, 2., .1, 1.)
    start = perf_counter()
    codes, attempts = observe_grid(r, direction, np.array(speeds), data, horizon=2.)
    outcomes = []
    for index, (speed, code) in enumerate(zip(speeds, codes, strict=True)):
        witnesses = [
            (method, resolution, value)
            for (node, method, resolution), value in attempts.items() if node == index
        ]
        row = {
            "speed_m_per_s": speed,
            "category": int(code),
            "attempts": [
                {
                    "method": method, "resolution": resolution,
                    "observation": record(value) if not isinstance(value, str) else None,
                    "error": value if isinstance(value, str) else None,
                }
                for method, resolution, value in witnesses
            ],
        }
        if code == OFF_CORE:
            localized = [value for _, _, value in witnesses
                         if not isinstance(value, str) and value[0] == OFF_CORE]
            row["off_core_pocket"] = check_off_core_pocket(localized[-1], data)
        outcomes.append(row)
    return {
        "job": job, "point": asdict(point), "position_m": r.tolist(),
        "direction": direction.tolist(), "elapsed_s": perf_counter() - start, "outcomes": outcomes,
    }


def audit_iterations(disc, point, speeds, prepared):
    from pmot.mot_strange.loading_scan_retention import observe
    from probe_mot_strange_loading_retention import record

    r, direction, data = prepared
    observe(r, 80 * direction, data, 1.6, .1, 1.)
    attempts = []
    for speed in speeds:
        for refinement in (1., 2.):
            start = perf_counter()
            row = {"speed_m_per_s": float(speed), "refinement": refinement}
            try:
                row["observation"] = record(observe(r, speed * direction, data, 1.6, .1, refinement))
            except (ValueError, FloatingPointError) as error:
                row["error"] = repr(error)
            row["elapsed_s"] = perf_counter() - start
            attempts.append(row)
    return {"disc": disc, "point": asdict(point), "attempts": attempts}


def main():
    from pmot.mot_strange.loading_scan_inputs import ScanPoint
    from validate_mot_strange_loading_scans import write_json

    jobs = [
        (2026110503, disc, 0, power, (speed,))
        for disc, power in ((0, 10), (1, 1)) for speed in (0., .5, 1., 2., 3.)
    ]
    jobs.extend((2026110502, 0, 0, 1, (speed,)) for speed in (.25, 8., 15., 25.))
    prepared = {}
    for sample, disc, point_index, power, _ in jobs:
        key = sample, disc, point_index, power
        if key not in prepared:
            point = ScanPoint(trap_power_mw=power)
            prepared[key] = point, prepare_probe(point, sample, disc, point_index)
    results = []
    with ProcessPoolExecutor(max_workers=8, initializer=configure_worker) as pool:
        pending = {pool.submit(probe, job, *prepared[job[:4]]): job for job in jobs}
        for future in as_completed(pending):
            try:
                result = future.result()
            except Exception as error:
                result = {"job": pending[future], "error": repr(error)}
            results.append(result)
            write_json("retention_closure_probe.json", results)
            print(result["job"], result.get("elapsed_s"), result.get("error"), [
                (row["category"], row.get("off_core_pocket", {}).get("status"))
                for row in result.get("outcomes", ())
            ], flush=True)


def iterations():
    import numpy as np
    from pmot.mot_strange.loading_scan_inputs import ScanPoint
    from validate_mot_strange_loading_scans import write_json

    speed_sets = np.array_split(np.r_[np.arange(0., 80.25, .25), np.arange(85., 121., 5.)], 4)
    jobs = []
    for disc, power in ((0, 10), (1, 1)):
        point = ScanPoint(trap_power_mw=power)
        prepared = prepare_probe(point, 2026110503, disc, 0)
        jobs.extend((disc, point, speeds, prepared) for speeds in speed_sets)
    results = []
    with ProcessPoolExecutor(max_workers=8, initializer=configure_worker) as pool:
        for future in as_completed([pool.submit(audit_iterations, *job) for job in jobs]):
            result = future.result()
            results.append(result)
            write_json("retention_iteration_audit.json", results)
            failures = [
                (row["speed_m_per_s"], row["refinement"])
                for row in result["attempts"] if row.get("observation", {}).get("category", -1) < 0
            ]
            print(result["disc"], result["attempts"][0]["speed_m_per_s"], failures, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-audit", action="store_true")
    args = parser.parse_args()
    if args.iteration_audit:
        iterations()
    else:
        main()
