"""Reduce complete pilot samples and plan independent direction-disc counts."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

from pmot.loading import VaporLoadingParameters
from pmot.mot_strange.loading_scan_retention import CORE, OFF_CORE, OTHER, WALL
from pmot.mot_strange.loading_scan_statistics import loading_by_ray, loading_interval
from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json


GRID = np.arange(0., 80.25, .25)
VELOCITIES = np.r_[GRID, np.arange(85., 121., 5.)]
PRODUCTION_SEED = 2026110602
TARGET = .08
INFLATION = 1.25
MAX_RAYS = 12800


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reduce_configuration(root, index, positions, directions, areas, vapor):
    """Check coordinates, prepared inputs, classification evidence and tail support."""
    point = f"configuration_{index:03d}"
    with np.load(root / point / "runtime_inputs.npz", allow_pickle=False) as prepared:
        fingerprint = digest(tuple(prepared[f"arr_{i}"] for i in range(len(prepared.files))))
    files = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / root.name / point
    shape = (*positions.shape[:2], len(GRID))
    core = np.empty(shape, dtype=np.int8)
    offcore = np.empty_like(core)
    wall = np.empty_like(core)
    localized_only = np.empty_like(core)
    elapsed = np.empty(shape[:2])
    for disc, ray in np.ndindex(shape[:2]):
        path = files / f"disc_{disc:04d}_point_{ray:02d}.npz"
        with np.load(path, allow_pickle=False) as z:
            if (z["prepared_sha256"].item() != fingerprint
                    or tuple(int(z[k]) for k in ("configuration", "disc", "point")) != (index, disc, ray)
                    or not np.array_equal(z["position"], positions[disc, ray])
                    or not np.array_equal(z["direction"], directions[disc])
                    or not np.array_equal(z["speeds"], VELOCITIES)):
                raise ValueError(f"Launch inputs differ in {path.name}")
            codes, verified = z["codes"], z["offcore_verified"]
            if (codes.shape != VELOCITIES.shape or verified.shape != codes.shape
                    or not np.isin(codes[1:], (WALL, CORE, OFF_CORE, OTHER)).all()
                    or int(z["unresolved"]) != 0):
                raise ValueError(f"Unresolved or invalid positive-speed evidence in {path.name}")
            pockets = json.loads(z["pocket_json"].item())
            expected = {str(k) for k in np.flatnonzero(codes == OFF_CORE)}
            accepted = {str(k) for k in np.flatnonzero(verified)}
            if (set(pockets) != expected
                    or accepted != {key for key, witness in pockets.items()
                                    if witness["status"] == "stable_pocket_and_return_verified"}):
                raise ValueError(f"Off-core witness mismatch in {path.name}")
            if np.any((VELOCITIES >= 79.75) & ((codes == CORE) | verified)) or bool(z["near_tail"]):
                raise ValueError(f"Capture at the velocity cutoff requires extension in {path.name}")
            retained = codes[:len(GRID)]
            core[disc, ray] = retained == CORE
            offcore[disc, ray] = verified[:len(GRID)]
            wall[disc, ray] = retained == WALL
            localized_only[disc, ray] = (retained == OFF_CORE) & ~verified[:len(GRID)]
            elapsed[disc, ray] = float(z["elapsed_s"])
    rates = {name: loading_by_ray(mask, GRID, areas, vapor)
             for name, mask in (("core", core), ("offcore", offcore), ("wall", wall),
                                ("localized_only", localized_only))}
    if not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
        raise ValueError("Worker timing must be finite and positive")
    return rates, elapsed


def allocation(rates, *, max_when_zero=True):
    """Pick a single predeclared launch count for both retained categories."""
    observations = []
    for name in ("core", "offcore"):
        values = rates[name]
        mean = float(values.mean())
        within = float(values.var(axis=1, ddof=1).mean())
        between = max(0., float(values.mean(axis=1).var(ddof=1)) - within / values.shape[1])
        observations.append((name, mean, between, within))

    candidates = []
    for points in (16, 25, 32, 50):
        limit = MAX_RAYS // points
        for discs in sorted(set(range(128, limit + 1, 16)) | {limit}):
            predicted = {name: float(t.ppf(.975, discs - 1) *
                                      np.sqrt(INFLATION * (between + within / points) / discs) / mean)
                         if mean else None for name, mean, between, within in observations}
            candidates.append({
                "discs": discs, "points_per_disc": points, "rays": discs * points,
                "predicted_relative_half_width": predicted,
            })

    def worst(row):
        return max((x for x in row["predicted_relative_half_width"].values() if x is not None),
                   default=0.)

    zeros = [name for name, mean, _, _ in observations if not mean]
    acceptable = [row for row in candidates
                  if (not zeros or not max_when_zero) and worst(row) <= TARGET]
    if acceptable:
        selected = min(acceptable, key=lambda row: (row["rays"], -row["discs"]))
    elif zeros:
        selected = min(candidates, key=lambda row: (-row["rays"], worst(row), -row["discs"]))
    else:
        selected = min(candidates, key=lambda row: (worst(row), -row["discs"]))
    return {**selected, "both_predicted_targets_met": bool(acceptable and not zeros),
            "pilot_zero_categories": zeros}


def summarize(indices):
    manifest_path = ROOT / "pilot_manifest.json"
    manifest = read_json(manifest_path)
    vapor = VaporLoadingParameters(**manifest["vapor"])
    with np.load(ROOT / "pilot_launch_geometry.npz", allow_pickle=False) as archive:
        positions, directions, areas = [archive[k] for k in ("positions", "directions", "areas")]
    points = {}
    for index in indices:
        rates, elapsed = reduce_configuration(ROOT / "pilot", index, positions, directions, areas, vapor)
        points[index] = {
            "pilot_only": {k: loading_interval(v.mean(axis=1)) for k, v in rates.items()},
            "pilot_mean_worker_seconds_per_ray": float(elapsed.mean()),
            "pilot_median_worker_seconds_per_ray": float(np.median(elapsed)),
            "allocation_preview": allocation(rates),
        }
    return {
        "status": "pilot_diagnostic_pending_validation",
        "created_utc": now(), "pilot_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "pilot_seed": manifest["seed"], "production_seed": PRODUCTION_SEED,
        "target_relative_half_width": TARGET, "variance_inflation": INFLATION,
        "maximum_rays_per_setting": MAX_RAYS, "configurations": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", nargs="+", type=int)
    args = parser.parse_args()
    result = summarize(args.configuration)
    suffix = "_".join(map(str, args.configuration))
    output = ROOT / "pilot_diagnostics" / f"point_{suffix}.json"
    write_json(output, result)
    print(output)
