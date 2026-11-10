"""Audit the sampled planning pilot and freeze production counts."""

import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np

from diagnose_mot_strange_velocity_sampling import velocity_strata
from mot_strange_sampled_design import (
    FINE_GRID, PRODUCTION_LAUNCH_SEED, PRODUCTION_SPEED_SEED, contribution, draw_nodes,
)
from plan_mot_strange_loading_production import INFLATION, MAX_RAYS, TARGET, allocation, read_json
from pmot.loading import VaporLoadingParameters
from pmot.mot_strange.loading_scan_retention import CORE, OFF_CORE, OTHER, WALL
from resume_mot_strange_loading_pilot import write_with_retry
from run_mot_strange_loading_pilot import NAME, ROOT, digest, now


def audit_sampled_point(folder, index, geometry, strata, speed_seed):
    positions, directions, areas = geometry
    shape = positions.shape[:2]
    rates = {name: np.zeros(shape) for name in ("core", "offcore")}
    elapsed = np.zeros(shape)
    masses = np.array([weights.sum() for _, weights in strata])
    summary = read_json(folder / "summary.json")
    with np.load(folder / "runtime_inputs.npz", allow_pickle=False) as inputs:
        signature = digest(tuple(inputs[f"arr_{i}"] for i in range(len(inputs.files))))
    if summary["status"] != "checkpointed" or summary["prepared_sha256"] != signature:
        raise ValueError(f"Sampled pilot setting requires review: {folder}")
    for disc, point in np.ndindex(shape):
        path = folder / f"disc_{disc:04d}_point_{point:02d}.npz"
        with np.load(path, allow_pickle=False) as ray:
            nodes = draw_nodes(strata, speed_seed, index, disc, point)
            codes, verified = ray["codes"], ray["offcore_verified"]
            if (ray["prepared_sha256"].item() != signature
                    or not np.array_equal(ray["position"], positions[disc, point])
                    or not np.array_equal(ray["direction"], directions[disc])
                    or tuple(int(ray[key]) for key in ("configuration", "disc", "point")) != (index, disc, point)
                    or not np.array_equal(ray["node_indices"], nodes)
                    or codes.shape != nodes.shape or verified.shape != nodes.shape
                    or not np.isin(codes, (WALL, CORE, OFF_CORE, OTHER)).all()):
                raise ValueError(f"Invalid sampled pilot evidence in {path}")
            pockets = read_pockets(ray)
            if (set(pockets) != {str(key) for key in nodes[codes == OFF_CORE]}
                    or {key for key, item in pockets.items() if item["status"] == "stable_pocket_and_return_verified"}
                    != {str(key) for key in nodes[verified]}):
                raise ValueError(f"Off-core witness mismatch in {path}")
            rates["core"][disc, point] = contribution(codes == CORE, masses, areas[disc])
            rates["offcore"][disc, point] = contribution(verified, masses, areas[disc])
            elapsed[disc, point] = float(ray["elapsed_s"])
    if np.any(elapsed <= 0) or not np.isfinite(elapsed).all():
        raise ValueError(f"Invalid pilot compute time in {folder}")
    return rates, elapsed, signature


def read_pockets(ray):
    import json
    return json.loads(ray["pocket_json"].item())


def build_plan(cache):
    dense_path = ROOT / "pilot_manifest.json"
    dense = read_json(dense_path)
    sampled_path = cache / "manifest.json"
    sampled = read_json(sampled_path)
    if sampled["dense_manifest_sha256"] != sha256(dense_path.read_bytes()).hexdigest():
        raise ValueError("Sampled pilot physical inputs differ from the dense pilot")
    if (PRODUCTION_LAUNCH_SEED in (dense["seed"], sampled["launch_seed"])
            or PRODUCTION_SPEED_SEED == sampled["speed_seed"]):
        raise ValueError("Production randomness overlaps a pilot seed")
    recovery_path = cache / "recovery_revision.json"
    recovery = read_json(recovery_path) if recovery_path.exists() else None
    if recovery is not None:
        if (recovery["source_manifest_sha256"] != sha256(sampled_path.read_bytes()).hexdigest()
                or recovery["fallback_source_sha256"] !=
                sha256(Path("scripts/mot_strange_pocket_recovery.py").read_bytes()).hexdigest()):
            raise ValueError("Sampled-pilot search recovery provenance differs")
    for name, expected in dense["source_sha256"].items():
        if sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen physical source differs: {name}")
    strata = velocity_strata(VaporLoadingParameters(**dense["vapor"]))
    with np.load(cache / "launch_geometry.npz", allow_pickle=False) as launch:
        geometry = tuple(launch[key] for key in ("positions", "directions", "areas"))
    plans = {}
    for index in range(len(dense["points"])):
        folder = cache / f"configuration_{index:03d}"
        if not (folder / "summary.json").is_file():
            continue
        rates, elapsed, signature = audit_sampled_point(folder, index, geometry, strata,
                                                        sampled["speed_seed"])
        plans[str(index)] = {
            **allocation(rates, max_when_zero=False),
            "pilot_worker_seconds_per_ray": float(elapsed.mean()),
            "prepared_sha256": signature,
        }
    return {
        "status": "preview",
        "dense_manifest_sha256": sampled["dense_manifest_sha256"],
        "sampled_manifest_sha256": sha256(sampled_path.read_bytes()).hexdigest(),
        "sampled_pilot_recovery": recovery,
        "expected_settings": len(dense["points"]), "settings": plans,
        "production_launch_seed": PRODUCTION_LAUNCH_SEED,
        "production_speed_seed": PRODUCTION_SPEED_SEED,
        "velocity_grid_m_per_s": FINE_GRID.tolist(), "draws_per_stratum": 2,
        "target_relative_half_width": TARGET, "variance_inflation": INFLATION,
        "maximum_rays_per_setting": MAX_RAYS,
        "zero_pilot_policy": "Allocate by observed categories, with undefined relative width for zero observations.",
        "coarse_pilot_scope": "The 0.25 m/s planning pilot estimates cost and variance. Production rates use independent launches and the 0.125 m/s rule.",
    }


def validate_freeze(plan):
    if len(plan["settings"]) != plan["expected_settings"]:
        raise ValueError("Wait for the complete independent planning pilot")
    diagnostics = ROOT / "pilot_diagnostics"
    holdout = read_json(diagnostics / "sampled_node_holdout.json")
    if holdout["status"] != "diagnostic_agreement":
        raise ValueError("Held-out sampled-node comparison has unresolved differences")
    fine_replay = diagnostics / "fine_node_replay.json"
    if read_json(fine_replay)["status"] != "passed":
        raise ValueError("Fine-grid positive and boundary-node comparisons need review")
    convergence = []
    for index in (0, 1):
        path = diagnostics / f"velocity_convergence_{index:03d}.json"
        result = read_json(path)
        if (result["unknown_positive_nodes"] or result["near_tail"]
                or result["common_node_category_differences_m_per_s"]
                or result["common_node_offcore_differences_m_per_s"]
                or not result["extended_fixed_core_check"]
                or result["extended_fixed_core_check"]["agreed_category"] != CORE):
            raise ValueError(f"Velocity refinement or longer retention needs review: {path}")
        convergence.append(sha256(path.read_bytes()).hexdigest())
    return {**plan, "status": "frozen", "frozen_utc": now(),
            "refinement_check_sha256": convergence,
            "fine_replay_sha256": sha256(fine_replay.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_pilot"
    path = ROOT / ("sampled_production_plan.json" if args.freeze else "sampled_production_preview.json")
    if args.freeze and path.exists():
        raise FileExistsError(f"Production allocation is already frozen at {path}")
    plan = build_plan(cache)
    if args.freeze:
        plan = validate_freeze(plan)
    write_with_retry(path, {**plan, "updated_utc": now()})
    print(f"{len(plan['settings'])}/{plan['expected_settings']} pilot settings audited in {path}.")
