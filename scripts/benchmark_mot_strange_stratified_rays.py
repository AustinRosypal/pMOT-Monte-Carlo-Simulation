"""Time an independent sampled-node replay against dense pilot checkpoints."""

import argparse
import os
from pathlib import Path
from time import perf_counter

from time_mot_strange_loading_rays import configure_worker


def compare_nodes(checkpoint, arrays, signature, strata, rng, manifest):
    import numpy as np

    from pmot.mot_strange.loading_scan_restoring import check_off_core_pocket
    from pmot.mot_strange.loading_scan_retention import OFF_CORE, observe_grid

    with np.load(checkpoint, allow_pickle=False) as z:
        if z["prepared_sha256"].item() != signature or int(z["unresolved"]) or bool(z["near_tail"]):
            raise ValueError(f"Dense comparison needs resolved, compatible evidence for {checkpoint}")
        r, direction = z["position"], z["direction"]
        speeds, dense_codes, dense_offcore = z["speeds"], z["codes"], z["offcore_verified"]
        dense_time = float(z["elapsed_s"])
    nodes = sorted({int(node) for ids, weights in strata
                    for node in rng.choice(ids, size=2, p=weights / weights.sum())})
    started = perf_counter()
    differences, outcomes = [], []
    for node in nodes:
        codes, attempts = observe_grid(
            r, direction, speeds[[node]], arrays, horizon=manifest["horizon_s"],
            window=manifest["final_window_s"],
        )
        code = int(codes[0])
        verified = False
        if code == OFF_CORE:
            witness = next(record for record in reversed(attempts.values())
                           if not isinstance(record, str) and record[0] == OFF_CORE)
            pocket = check_off_core_pocket(witness, arrays)
            verified = pocket["status"] == "stable_pocket_and_return_verified"
        actual = (code, verified)
        expected = (int(dense_codes[node]), bool(dense_offcore[node]))
        outcomes.append({"speed_m_per_s": float(speeds[node]), "category": code,
                         "verified_offcore": verified, "attempts": len(attempts)})
        if actual != expected:
            differences.append({"speed_m_per_s": float(speeds[node]),
                                "sampled": actual, "dense": expected})
    return {"ray": checkpoint.name, "sampled_elapsed_s": perf_counter() - started,
            "dense_elapsed_s": dense_time, "differences": differences, "outcomes": outcomes}


def benchmark(configuration, repeats):
    configure_worker()
    import numpy as np

    from diagnose_mot_strange_velocity_sampling import velocity_strata
    from plan_mot_strange_loading_production import read_json
    from pmot.loading import VaporLoadingParameters
    from pmot.mot_strange.loading_scan_retention import CORE, OFF_CORE, observe
    from resume_mot_strange_loading_pilot import write_with_retry
    from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json

    manifest = read_json(ROOT / "pilot_manifest.json")
    vapor = VaporLoadingParameters(**manifest["vapor"])
    strata = velocity_strata(vapor)
    label = f"configuration_{configuration:03d}"
    folder = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "pilot" / label
    published = ROOT / "pilot" / label
    with np.load(published / "runtime_inputs.npz", allow_pickle=False) as inputs:
        arrays = tuple(inputs[f"arr_{i}"] for i in range(len(inputs.files)))
    for array in arrays:
        array.setflags(write=False)
    signature = digest(arrays)
    source = []
    for file in sorted(folder.glob("disc_*_point_*.npz")):
        with np.load(file, allow_pickle=False) as z:
            if (z["prepared_sha256"].item() != signature or int(z["unresolved"])
                    or bool(z["near_tail"])):
                continue
            codes = z["codes"]
            source.append((file, float(z["elapsed_s"]), bool(np.any(codes[1:] == CORE)),
                           bool(np.any(codes[1:] == OFF_CORE))))
    if not source:
        raise RuntimeError("No compatible resolved pilot files")
    # Choose a fixed random sample, then add distinct slow and retained stress cases.
    rng = np.random.default_rng(2026110604)
    random_files = rng.choice(len(source), size=min(repeats, len(source)), replace=False)
    slow = sorted(range(len(source)), key=lambda i: source[i][1], reverse=True)[:2]
    core = [i for i, row in enumerate(source) if row[2]][:2]
    localized = [i for i, row in enumerate(source) if row[3]][:2]
    selected = list(dict.fromkeys([*map(int, random_files), *slow, *core, *localized]))
    output = folder.parent.parent / "pilot_diagnostics" / f"sampled_node_replay_{configuration:03d}.json"
    report = {"status": "running", "configuration": configuration, "input_sha256": signature,
              "rows": [], "expected": len(selected)}
    for index in selected:
        file, _, _, _ = source[index]
        if not report["rows"]:
            with np.load(file, allow_pickle=False) as z:
                observe(z["position"], 80 * z["direction"], arrays, manifest["horizon_s"],
                        manifest["final_window_s"], 1.)
        row = compare_nodes(file, arrays, signature, strata, rng, manifest)
        report["rows"].append(row)
        write_json(output, {**report, "updated_utc": now()})
        print(f"{file.name}: {len(row['outcomes'])} speeds, {row['sampled_elapsed_s']:.2f} s, "
              f"{len(row['differences'])} classification differences", flush=True)
    report["status"] = ("differences_found" if any(row["differences"] for row in report["rows"])
                        else "diagnostic_agreement")
    write_json(output, {**report, "updated_utc": now()})
    write_with_retry(
        ROOT / "pilot_diagnostics" / f"sampled_node_replay_{configuration:03d}_complete.json",
        {**report, "updated_utc": now()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int)
    parser.add_argument("--random-rays", type=int, default=8)
    args = parser.parse_args()
    if args.random_rays < 1:
        parser.error("--random-rays must be positive")
    benchmark(args.configuration, args.random_rays)
