"""Run finer independent observations on unresolved production speed nodes."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from time_mot_strange_loading_rays import configure_worker


def review(configuration, ray_name, fixed=False, finer_fixed=False, finer_adaptive=False,
           finest_fixed=False, tight_adaptive=False, very_tight_adaptive=False,
           ultra_tight_adaptive=False, dense_pilot=False, reuse_fixed_report=None):
    configure_worker()
    import numpy as np

    from mot_strange_sampled_design import FINE_GRID
    from mot_strange_pocket_recovery import check_pocket
    from mot_strange_review_overlays import (
        ADAPTIVE_PAIR, FINER_ADAPTIVE_PAIR, TIGHT_ADAPTIVE_PAIR,
        VERY_TIGHT_ADAPTIVE_PAIR, ULTRA_TIGHT_ADAPTIVE_PAIR,
        FINER_FIXED_PAIR_S, FINEST_FIXED_PAIR_S, FIXED_PAIR_S,
        record, verify_reused_attempts,
    )
    from pmot.mot_strange.loading_scan_retention import OFF_CORE, agreed_category, observe
    from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json
    from plan_mot_strange_loading_production import read_json

    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / (
        "pilot" if dense_pilot else "sampled_production")
    if dense_pilot:
        manifest = read_json(ROOT / "pilot_manifest.json")
        if manifest["seed"] != 2026110601:
            raise ValueError("Dense diagnostic pilot seed differs")
        protocol = {"horizon_s": manifest["horizon_s"],
                    "window_s": manifest["final_window_s"]}
    else:
        protocol = read_json(cache / "runtime_protocol.json")
    folder = cache / f"configuration_{configuration:03d}"
    if Path(ray_name).name != ray_name:
        raise ValueError("Specify a checkpoint filename")
    inputs = ((ROOT / "pilot" / folder.name) if dense_pilot else folder) / "runtime_inputs.npz"
    with np.load(inputs, allow_pickle=False) as z:
        data = tuple(z[f"arr_{i}"] for i in range(len(z.files)))
    for array in data:
        array.setflags(write=False)
    with np.load(folder / ray_name, allow_pickle=False) as z:
        if z["prepared_sha256"].item() != digest(data):
            raise ValueError("Unresolved ray prepared inputs differ")
        position, direction = z["position"], z["direction"]
        if dense_pilot:
            speed_grid = np.r_[np.arange(0., 80.25, .25), np.arange(85., 121., 5.)]
            np.testing.assert_array_equal(z["speeds"], speed_grid)
            unresolved = np.flatnonzero(z["codes"] < 0)
            with np.load(ROOT / "pilot_launch_geometry.npz", allow_pickle=False) as geometry:
                np.testing.assert_array_equal(position, geometry["positions"][int(z["disc"]), int(z["point"])])
                np.testing.assert_array_equal(direction, geometry["directions"][int(z["disc"])])
        else:
            speed_grid = FINE_GRID
            unresolved = np.unique(z["node_indices"][z["codes"] < 0])
    if not len(unresolved):
        raise ValueError("Checkpoint contains no unresolved speed")
    pairs = {"adaptive": FINER_ADAPTIVE_PAIR if finer_adaptive else ADAPTIVE_PAIR}
    if tight_adaptive:
        pairs["adaptive"] = TIGHT_ADAPTIVE_PAIR
    adaptive_observer = observe
    observer_proof = None
    if very_tight_adaptive or ultra_tight_adaptive:
        from mot_strange_review_observer import control_proof, make_observer

        pairs["adaptive"] = (ULTRA_TIGHT_ADAPTIVE_PAIR if ultra_tight_adaptive
                             else VERY_TIGHT_ADAPTIVE_PAIR)
        adaptive_observer = make_observer()
        observer_proof = control_proof()
    if fixed or finer_fixed:
        pairs["fixed"] = FINER_FIXED_PAIR_S if finer_fixed else FIXED_PAIR_S
    if finest_fixed:
        pairs["fixed"] = FINEST_FIXED_PAIR_S
    report = {"status": "independent_diagnostic", "configuration": configuration,
              "checkpoint_stage": "dense_pilot" if dense_pilot else "production",
              "ray": ray_name, "prepared_sha256": digest(data),
              "original_checkpoint_sha256": sha256((folder / ray_name).read_bytes()).hexdigest(),
              "nodes": []}
    path = folder / "numerical_review" / (Path(ray_name).stem + f"_{os.getpid()}.json")
    reusable = {}
    if reuse_fixed_report:
        sources = [reuse_fixed_report] if isinstance(reuse_fixed_report, str) else reuse_fixed_report
        for source_name in sources:
            if Path(source_name).name != source_name or "fixed" not in pairs:
                raise ValueError("Specify local original reports and a fixed refinement pair")
            content = (path.parent / source_name).read_bytes()
            old = json.loads(content)
            for key in ("configuration", "checkpoint_stage", "ray", "prepared_sha256",
                        "original_checkpoint_sha256"):
                if old.get(key) != report.get(key):
                    raise ValueError("Reusable fixed review has incompatible inputs")
            for old_node in old["nodes"]:
                node = old_node["node"]
                if node not in unresolved or old_node["speed_m_per_s"] != speed_grid[node]:
                    continue
                for old_attempt in old_node["attempts"]:
                    if (old_attempt.get("method") == "fixed" and "code" in old_attempt
                            and old_attempt["resolution"] in pairs["fixed"]
                            and "reused_from" not in old_attempt):
                        key = (int(node), old_attempt["resolution"])
                        if key in reusable:
                            raise ValueError("Specify only one original source for each fixed observation")
                        reusable[key] = {**old_attempt, "reused_from": {
                            "file": source_name, "sha256": sha256(content).hexdigest()}}
        if not reusable:
            raise ValueError("Source report contains no original requested fixed observation")
    for node in unresolved:
        speed = float(speed_grid[node])
        attempts = []
        row = {"speed_m_per_s": speed, "node": int(node), "attempts": attempts}
        row["adaptive_pair"] = pairs["adaptive"]
        if observer_proof is not None:
            row["adaptive_observer_proof"] = observer_proof
        if "fixed" in pairs:
            row["fixed_pair_s"] = pairs["fixed"]
        report["nodes"].append(row)
        outcomes = {}
        for method, resolution in ((method, step) for method, pair in pairs.items() for step in pair):
            attempt = {"method": method, "resolution": resolution}
            began = perf_counter()
            try:
                reused = reusable.get((int(node), resolution)) if method == "fixed" else None
                if reused is not None:
                    attempt = dict(reused)
                    verify_reused_attempts(path, {**report, "nodes": [{**row, "attempts": [attempt]}]})
                    result = record(attempt)
                else:
                    observer = adaptive_observer if method == "adaptive" else observe
                    result = observer(position, speed * direction, data, protocol["horizon_s"],
                                      protocol["window_s"], resolution if method == "adaptive" else 1.,
                                      resolution if method == "fixed" else 0.)
                outcomes[(method, resolution)] = result
                attempt.update({
                    "code": int(result[0]), "elapsed_s": float(result[1]),
                    "position_m": result[2].tolist(), "velocity_m_per_s": result[3].tolist(),
                    "first_core_s": float(result[4]), "window_anchor_m": result[5].tolist(),
                    "window_metrics_m": [float(x) if np.isfinite(x) else None for x in result[6:]],
                })
            except (ValueError, FloatingPointError) as error:
                attempt["error"] = repr(error)
            if "reused_from" not in attempt:
                attempt["worker_seconds"] = perf_counter() - began
            attempts.append(attempt)
            pair = pairs[method]
            if all((method, step) in outcomes for step in pair):
                first, second = (outcomes[(method, step)] for step in pair)
                row[f"{method}_agreement"] = agreed_category(first, second)
                if row[f"{method}_agreement"] == OFF_CORE:
                    row[f"{method}_pocket"] = check_pocket(second, data)
            write_json(path, {**report, "updated_utc": now()})
            print(f"{speed:g} m/s: {attempt}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--reuse-fixed-report", action="append",
                        help="Original local review filename; repeat for different fixed observations")
    parser.add_argument("--dense-pilot", action="store_true",
                        help="Review excluded dense diagnostic checkpoints on their frozen grid")
    fixed_options = parser.add_mutually_exclusive_group()
    fixed_options.add_argument("--fixed", action="store_true")
    fixed_options.add_argument("--finer-fixed", action="store_true")
    fixed_options.add_argument("--finest-fixed", action="store_true")
    adaptive_options = parser.add_mutually_exclusive_group()
    adaptive_options.add_argument("--finer-adaptive", action="store_true")
    adaptive_options.add_argument("--tight-adaptive", action="store_true")
    adaptive_options.add_argument("--very-tight-adaptive", action="store_true")
    adaptive_options.add_argument("--ultra-tight-adaptive", action="store_true")
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        review(args.configuration, checkpoint, args.fixed, args.finer_fixed, args.finer_adaptive,
               args.finest_fixed, args.tight_adaptive, args.very_tight_adaptive,
               args.ultra_tight_adaptive, args.dense_pilot, args.reuse_fixed_report)
