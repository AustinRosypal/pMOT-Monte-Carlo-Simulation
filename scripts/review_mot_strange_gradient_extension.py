"""Resolve extension unknowns with immutable adaptive and fixed-step evidence."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path
from time import perf_counter
import zipfile

os.environ["NUMBA_CACHE_DIR"] = str(
    Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / "strange-gradient-extension-review" / f"process_{os.getpid()}")

import numpy as np
import extend_mot_strange_gradient_loading as campaign


def review(path):
    from mot_strange_pocket_recovery import check_pocket
    from mot_strange_review_observer import control_proof, make_observer
    from mot_strange_review_overlays import (
        ADAPTIVE_PAIR, FINER_ADAPTIVE_PAIR, TIGHT_ADAPTIVE_PAIR, VERY_TIGHT_ADAPTIVE_PAIR,
        ULTRA_TIGHT_ADAPTIVE_PAIR, FIXED_PAIR_S, FINER_FIXED_PAIR_S, FINEST_FIXED_PAIR_S,
        audited_resolution, load_resolution,
    )
    from pmot.mot_strange.loading_scan_retention import OFF_CORE, agreed_category, observe

    document = campaign.read(campaign.OUTPUT / "manifest.json")
    with np.load(campaign.CACHE / path.parent.name / "runtime_inputs.npz") as z:
        data = tuple(z[f"arr_{i}"] for i in range(len(z.files)))
    for array in data:
        array.setflags(write=False)
    signature = campaign.digest(data)
    with np.load(path) as z:
        if z["prepared_sha256"].item() != signature:
            raise ValueError("Unknown checkpoint prepared inputs differ")
        index = int(z["configuration"])
        r, direction = z["position"], z["direction"]
        node_indices, codes = z["node_indices"], z["codes"]
        nodes = sorted(map(int, np.unique(node_indices[codes < 0])))
    try:
        found = load_resolution(path, node_indices, codes, signature)
    except ValueError:
        found = {}
    if set(found) == set(nodes):
        return {"file": str(path), "resolved": nodes, "reused_verified_review": True}
    report = {
        "status": "independent_diagnostic", "configuration": index,
        "checkpoint_stage": "production", "ray": path.name, "prepared_sha256": signature,
        "original_checkpoint_sha256": campaign.checksum(path), "nodes": [],
    }
    output = path.parent / "numerical_review" / f"{path.stem}_extension_{os.getpid()}.json"
    pairs = (
        (ADAPTIVE_PAIR, FIXED_PAIR_S), (FINER_ADAPTIVE_PAIR, FIXED_PAIR_S),
        (FINER_ADAPTIVE_PAIR, FINER_FIXED_PAIR_S), (TIGHT_ADAPTIVE_PAIR, FINER_FIXED_PAIR_S),
        (VERY_TIGHT_ADAPTIVE_PAIR, FINER_FIXED_PAIR_S), (ULTRA_TIGHT_ADAPTIVE_PAIR, FINER_FIXED_PAIR_S),
        (ULTRA_TIGHT_ADAPTIVE_PAIR, FINEST_FIXED_PAIR_S),
    )
    extended_observer = None
    for node in nodes:
        attempts, observations = [], {}
        row = {"node": node, "speed_m_per_s": node*.125, "attempts": attempts}
        report["nodes"].append(row)
        resolved = False
        for adaptive, fixed in pairs:
            row.update(adaptive_pair=adaptive, fixed_pair_s=fixed)
            extended = adaptive in (VERY_TIGHT_ADAPTIVE_PAIR, ULTRA_TIGHT_ADAPTIVE_PAIR)
            if extended:
                if extended_observer is None:
                    extended_observer = make_observer()
                row["adaptive_observer_proof"] = control_proof()
            else:
                row.pop("adaptive_observer_proof", None)
            for method, resolutions in (("adaptive", adaptive), ("fixed", fixed)):
                for resolution in resolutions:
                    key = method, resolution
                    if key in observations:
                        continue
                    began = perf_counter()
                    item = {"method": method, "resolution": resolution}
                    observer = extended_observer if method == "adaptive" and extended else observe
                    try:
                        result = observer(
                            r, direction*(node*.125), data, document["horizon_s"], document["window_s"],
                            resolution if method == "adaptive" else 1.,
                            resolution if method == "fixed" else 0.)
                        observations[key] = result
                        item.update({
                            "code": int(result[0]), "elapsed_s": float(result[1]),
                            "position_m": result[2].tolist(), "velocity_m_per_s": result[3].tolist(),
                            "first_core_s": float(result[4]), "window_anchor_m": result[5].tolist(),
                            "window_metrics_m": [float(x) if np.isfinite(x) else None for x in result[6:]],
                        })
                    except (ValueError, FloatingPointError) as error:
                        item["error"] = repr(error)
                    item["worker_seconds"] = perf_counter()-began
                    attempts.append(item)
                    report["updated_chicago"] = datetime.now(campaign.CHICAGO).isoformat()
                    campaign.write_json(output, report)
                    print(path.name, node, method, resolution, item.get("code", item.get("error")), flush=True)
                if all((method, resolution) in observations for resolution in resolutions):
                    pair = [observations[method, resolution] for resolution in resolutions]
                    row[f"{method}_agreement"] = agreed_category(*pair)
                    if row[f"{method}_agreement"] == OFF_CORE:
                        row[f"{method}_pocket"] = check_pocket(pair[-1], data)
            campaign.write_json(output, report)
            if audited_resolution(row, campaign.checksum(output)) is not None:
                resolved = True
                break
        if not resolved:
            raise ValueError(f"Full independent refinement ladder did not resolve {path.name} node {node}")
    report["status"] = "independent_diagnostic_complete"
    campaign.write_json(output, report)
    load_resolution(path, node_indices, codes, signature)
    return {"file": str(path), "resolved": nodes, "review": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=16)
    args = parser.parse_args()
    completion = campaign.read(campaign.OUTPUT / "production_completion.json")
    if len(completion) != 20 or sum(row["rays"] for row in completion.values()) != 256000:
        raise ValueError("Wait for all fixed production rays before the numerical review")
    if any(row["tail_hits"] for row in completion.values()):
        raise ValueError("Near-tail capture needs a separate velocity-support investigation")
    paths = []
    for index, row in completion.items():
        if not row["unresolved_nodes"]:
            continue
        folder = campaign.CACHE / "production" / f"configuration_{int(index):03d}"
        for path in folder.glob("disc_*_point_*.npz"):
            with zipfile.ZipFile(path) as archive:
                unresolved = int.from_bytes(archive.read("unresolved.npy")[-8:], "little", signed=True)
            if unresolved:
                paths.append(path)
    results = []
    if paths:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(review, path): path for path in paths}
            for future in as_completed(jobs):
                results.append(future.result())
                campaign.write_json(campaign.OUTPUT / "numerical_review_progress.json",
                                    {"finished": len(results), "expected": len(paths)})
    campaign.write_json(campaign.OUTPUT / "numerical_review_completion.json",
                        {"passed": True, "reviewed_rays": results, "raw_checkpoints_unchanged": True})
    print(f"Resolved {len(paths)} raw-unknown rays", flush=True)


if __name__ == "__main__":
    main()
