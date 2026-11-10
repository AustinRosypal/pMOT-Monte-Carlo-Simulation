"""Independently refine velocity and retention checks on saved positive pilot rays."""

import argparse
import os
from pathlib import Path
from time import perf_counter

from time_mot_strange_loading_rays import configure_worker


def check(index):
    configure_worker()
    import numpy as np

    from plan_mot_strange_loading_production import GRID, read_json
    from pmot.loading import VaporLoadingParameters
    from pmot.mot_strange.loading_scan_restoring import check_off_core_pocket
    from pmot.mot_strange.loading_scan_retention import CORE, OFF_CORE, agreed_category, observe, observe_grid
    from pmot.mot_strange.loading_scan_statistics import loading_by_ray
    from resume_mot_strange_loading_pilot import write_with_retry
    from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json

    manifest = read_json(ROOT / "pilot_manifest.json")
    vapor = VaporLoadingParameters(**manifest["vapor"])
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME
    label = f"configuration_{index:03d}"
    folder = cache / "pilot" / label
    with np.load(ROOT / "pilot" / label / "runtime_inputs.npz", allow_pickle=False) as source:
        data = tuple(source[f"arr_{i}"] for i in range(len(source.files)))
    for array in data:
        array.setflags(write=False)
    signature = digest(data)
    candidates = []
    for path in sorted(folder.glob("disc_*_point_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            if not int(z["unresolved"]) and not bool(z["near_tail"]):
                core = np.flatnonzero(z["codes"][1:len(GRID)] == CORE)
                if len(core):
                    candidates.append((len(core), path))
    if not candidates:
        raise ValueError(f"No resolved positive pilot ray for {label}")
    _, path = sorted(candidates)[len(candidates) // 2]
    with np.load(path, allow_pickle=False) as z:
        if z["prepared_sha256"].item() != signature:
            raise ValueError("Prepared inputs do not match the pilot checkpoint")
        r, direction, original, original_offcore = (
            z[key] for key in ("position", "direction", "codes", "offcore_verified")
        )
    velocity = np.r_[np.arange(0., 80.125, .125), np.arange(85., 121., 5.)]
    fine_size = len(GRID) * 2 - 1
    print(f"{label}: refining {path.name} on the 0.125 m/s grid", flush=True)
    start = perf_counter()
    codes, attempts = observe_grid(r, direction, velocity, data,
                                   horizon=manifest["horizon_s"], window=manifest["final_window_s"])
    verified = np.zeros(len(velocity), dtype=bool)
    witnesses = {}
    for node in np.flatnonzero(codes == OFF_CORE):
        record = next(record for (j, _, _), record in reversed(attempts.items())
                      if j == node and not isinstance(record, str) and record[0] == OFF_CORE)
        witnesses[int(node)] = check_off_core_pocket(record, data)
        verified[node] = witnesses[int(node)]["status"] == "stable_pocket_and_return_verified"
    unknown = int(np.count_nonzero(codes[1:] < 0))
    near_tail = bool(np.any((velocity >= 79.75) & ((codes == CORE) | verified)))
    common = np.r_[np.arange(0, fine_size, 2), np.arange(fine_size, len(velocity))]
    rates = {}
    for name, coarse_mask, fine_mask in (("core", original == CORE, codes == CORE),
                                         ("offcore", original_offcore, verified)):
        rates[name] = None if unknown or near_tail else {
            "coarse_per_m2": float(loading_by_ray(coarse_mask[None, None, :len(GRID)],
                                                 GRID, np.ones(1), vapor)[0, 0]),
            "fine_per_m2": float(loading_by_ray(fine_mask[None, None, :fine_size],
                                               velocity[:fine_size], np.ones(1), vapor)[0, 0]),
        }
    positive = np.flatnonzero((codes == CORE) & (velocity > 0.))
    extended = None
    if len(positive):
        node = positive[len(positive) // 2]
        v = velocity[node] * direction
        observations = [observe(r, v, data, 2.25, manifest["final_window_s"], 1., dt)
                        for dt in (5e-6, 2.5e-6)]
        extended = {"speed_m_per_s": float(velocity[node]), "horizon_s": 2.25,
                    "fixed_steps_s": [5e-6, 2.5e-6], "agreed_category": agreed_category(*observations),
                    "maximum_window_radius_m": [float(obs[7]) for obs in observations]}
    result = {
        "status": "diagnostic_only", "updated_utc": now(), "configuration": index,
        "source_ray": path.name, "prepared_sha256": signature, "worker_seconds": perf_counter() - start,
        "common_node_category_differences_m_per_s": velocity[common[codes[common] != original]].tolist(),
        "common_node_offcore_differences_m_per_s": velocity[common[verified[common] != original_offcore]].tolist(),
        "unknown_positive_nodes": unknown, "near_tail": near_tail, "quadrature_per_m2": rates,
        "extended_fixed_core_check": extended,
    }
    target = cache / "velocity_convergence" / label
    target.mkdir(parents=True, exist_ok=True)
    valid = [(key, val) for key, val in attempts.items() if not isinstance(val, str)]
    evidence = [np.r_[row[0], row[1], row[2], row[3], row[4], row[5], row[6:]] for _, row in valid]
    np.savez_compressed(target / "fine_grid.npz", velocities=velocity, codes=codes, offcore=verified,
                        evidence=evidence, keys=[key for key, _ in valid])
    write_json(target / "summary.json", result)
    write_with_retry(ROOT / "pilot_diagnostics" / f"velocity_convergence_{index:03d}.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int)
    check(parser.parse_args().configuration)
