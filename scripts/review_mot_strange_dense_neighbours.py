"""Preserve fresh fixed-step proof for neighbours of resolved dense-mask edges."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from time_mot_strange_loading_rays import configure_worker


def review(index, ray):
    configure_worker()
    from audit_mot_strange_dense_pilot import BASE, FIXED, SPEEDS, check_ray
    from pmot.mot_strange.loading_scan_retention import agreed_category, observe
    from run_mot_strange_loading_pilot import ROOT, digest, now, write_json

    if Path(ray).name != ray:
        raise ValueError("Specify a local checkpoint filename")
    folder = BASE / f"configuration_{index:03d}"
    path = folder / ray
    with np.load(ROOT / "pilot" / folder.name / "runtime_inputs.npz", allow_pickle=False) as z:
        data = tuple(z[f"arr_{i}"] for i in range(12))
    for array in data:
        array.setflags(write=False)
    signature = digest(data)
    with np.load(ROOT / "pilot_launch_geometry.npz", allow_pickle=False) as z:
        positions, directions = z["positions"], z["directions"]
    with np.load(path, allow_pickle=False) as z:
        disc, point = int(z["disc"]), int(z["point"])
        r, direction, codes = z["position"], z["direction"], z["codes"]
    finding = check_ray((index, disc, point, signature, positions, directions, data[11][15]))
    report = dict(status="running", configuration=index, checkpoint_stage="dense_pilot",
                  ray=ray, prepared_sha256=signature,
                  original_checkpoint_sha256=sha256(path.read_bytes()).hexdigest(),
                  horizon_s=2., window_s=.1, nodes=[],
                  observer_source_sha256=sha256(Path("src/pmot/mot_strange/loading_scan_retention.py").read_bytes()).hexdigest())
    destination = folder / "fixed_neighbour_review" / f"{path.stem}_{os.getpid()}.json"
    for node in finding["missing_fixed_neighbours"]:
        row = dict(node=node, speed_m_per_s=float(SPEEDS[node]), original_category=int(codes[node]), attempts=[])
        report["nodes"].append(row)
        for pair in FIXED:
            row["fixed_pair_s"] = pair
            results = []
            for step in pair:
                attempt = dict(method="fixed", resolution=step)
                began = perf_counter()
                try:
                    result = observe(r, SPEEDS[node] * direction, data, 2., .1, 1., step)
                    results.append(result)
                    attempt.update(code=int(result[0]), elapsed_s=float(result[1]),
                                   position_m=result[2].tolist(), velocity_m_per_s=result[3].tolist(),
                                   first_core_s=float(result[4]), window_anchor_m=result[5].tolist(),
                                   window_metrics_m=[float(v) if np.isfinite(v) else None for v in result[6:]])
                except (ValueError, FloatingPointError) as error:
                    attempt["error"] = repr(error)
                attempt["worker_seconds"] = perf_counter() - began
                row["attempts"].append(attempt)
                write_json(destination, {**report, "updated_utc": now()})
            category = agreed_category(*results) if len(results) == 2 else None
            row["fixed_agreement"] = category
            write_json(destination, {**report, "updated_utc": now()})
            if category is not None:
                break
        print(f"Dense {index:03d} {ray}, node {node}: original={codes[node]}, fixed={row.get('fixed_agreement')}", flush=True)
    report["status"] = ("passed" if all(n.get("fixed_agreement") == n["original_category"] for n in report["nodes"])
                        else "requires_further_review")
    write_json(destination, {**report, "updated_utc": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    review(args.configuration, args.checkpoint)
