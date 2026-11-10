"""Compare randomized velocity quadrature to saved dense pilot masks."""

import argparse
import os
from pathlib import Path

import numpy as np

from plan_mot_strange_loading_production import GRID, read_json
from pmot.loading import VaporLoadingParameters
from pmot.mot_strange.loading_scan_retention import CORE
from run_mot_strange_loading_pilot import NAME, ROOT, now, write_json


STRATUM_EDGES_M_PER_S = np.array([0., 2., 4., 6., 8., 10., 12., 15., 18., 22.,
                                  26., 30., 35., 40., 50., 60., 70., 80.])


def velocity_strata(vapor):
    """Partition the existing positive-speed trapezoidal rule."""
    width = GRID[1] - GRID[0]
    weights = width * GRID**3 * np.exp(-GRID**2 / vapor.thermal_scale_m2_per_s2)
    weights[-1] *= .5
    weights *= vapor.loading_rate_prefactor
    strata = np.searchsorted(STRATUM_EDGES_M_PER_S[1:], GRID, side="left")
    return [(nodes, weights[nodes]) for k in range(len(STRATUM_EDGES_M_PER_S) - 1)
            if len(nodes := np.flatnonzero((strata == k) & (GRID > 0.)))]


def compare(config, repeats):
    manifest = read_json(ROOT / "pilot_manifest.json")
    vapor = VaporLoadingParameters(**manifest["vapor"])
    strata = velocity_strata(vapor)
    folder = (Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "pilot"
              / f"configuration_{config:03d}")
    with np.load(ROOT / "pilot_launch_geometry.npz", allow_pickle=False) as geometry:
        areas = geometry["areas"]
    shape = manifest["discs"], manifest["points_per_disc"], len(GRID)
    masks = {name: np.zeros(shape, dtype=bool) for name in ("core", "offcore")}
    for disc, point in np.ndindex(shape[:2]):
        path = folder / f"disc_{disc:04d}_point_{point:02d}.npz"
        with np.load(path, allow_pickle=False) as ray:
            if int(ray["unresolved"]) or bool(ray["near_tail"]):
                raise RuntimeError(f"Unresolved pilot evidence in {path.name}")
            masks["core"][disc, point] = ray["codes"][:len(GRID)] == CORE
            masks["offcore"][disc, point] = ray["offcore_verified"][:len(GRID)]

    rng = np.random.default_rng(2026110603)
    baseline = {name: np.zeros(shape[:2]) for name in masks}
    for nodes, weights in strata:
        for name, mask in masks.items():
            baseline[name] += mask[..., nodes] @ weights * areas[:, None]
    comparisons = []
    for draws in (1, 2, 4):
        totals = {name: np.zeros((repeats, *shape[:2])) for name in masks}
        for nodes, weights in strata:
            mass = float(weights.sum())
            selection = rng.choice(nodes, size=(repeats, *shape[:2], draws), p=weights / mass)
            discs, points = np.indices(shape[:2])
            for name, mask in masks.items():
                outcomes = mask[discs[None, ..., None], points[None, ..., None], selection]
                totals[name] += mass * outcomes.mean(axis=-1) * areas[None, :, None]
        stats = {}
        for name, values in totals.items():
            truth_by_disc = baseline[name].mean(axis=1)
            truth = float(truth_by_disc.mean())
            sampled_by_disc = values.mean(axis=2)
            error = sampled_by_disc.mean(axis=1) - truth
            base_variance = float(truth_by_disc.var(ddof=1))
            average_sampled_variance = float(sampled_by_disc.var(axis=1, ddof=1).mean())
            stats[name] = {
                "dense_pilot_mean_atoms_per_s": truth,
                "conditional_mean_error_atoms_per_s": float(error.mean()),
                "conditional_rms_relative_error": float(np.sqrt(np.mean(error**2)) / truth) if truth else None,
                "mean_disc_variance_inflation": average_sampled_variance / base_variance if base_variance else None,
            }
        comparisons.append({"draws_per_stratum": draws, "draws_per_ray": draws * len(strata),
                            "categories": stats})
    return {
        "status": "diagnostic_only", "created_utc": now(), "configuration": config,
        "repeats_on_same_dense_pilot": repeats, "stratum_edges_m_per_s": STRATUM_EDGES_M_PER_S.tolist(),
        "quadrature": "Trapezoidal nodal weights with speed cubed thermal factor; weighted draws with replacement",
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int)
    parser.add_argument("--repeats", type=int, default=512)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least two")
    result = compare(args.configuration, args.repeats)
    path = ROOT / "pilot_diagnostics" / f"velocity_sampling_{args.configuration:03d}.json"
    write_json(path, result)
    print(path)
