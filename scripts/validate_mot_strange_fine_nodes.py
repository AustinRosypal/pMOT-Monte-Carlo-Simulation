"""Replay positive and adjacent nodes from the saved 0.125 m/s diagnostics."""

import os
from pathlib import Path

from time_mot_strange_loading_rays import configure_worker


def main():
    configure_worker()
    import numpy as np

    from plan_mot_strange_loading_production import read_json
    from pmot.mot_strange.loading_scan_retention import CORE, OFF_CORE, observe
    from resume_mot_strange_loading_pilot import write_with_retry
    from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json
    from run_mot_strange_sampled_production import classify

    manifest = read_json(ROOT / "pilot_manifest.json")
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME
    report = {"status": "running", "settings": []}
    for index in (0, 1):
        label = f"configuration_{index:03d}"
        folder = cache / "velocity_convergence" / label
        summary = read_json(folder / "summary.json")
        checkpoint = cache / "pilot" / label / summary["source_ray"]
        with np.load(checkpoint, allow_pickle=False) as z:
            r, direction = z["position"], z["direction"]
        with np.load(ROOT / "pilot" / label / "runtime_inputs.npz", allow_pickle=False) as z:
            data = tuple(z[f"arr_{i}"] for i in range(len(z.files)))
        for array in data:
            array.setflags(write=False)
        if digest(data) != summary["prepared_sha256"]:
            raise ValueError(f"Fine-grid input fingerprint mismatch in {label}")
        with np.load(folder / "fine_grid.npz", allow_pickle=False) as z:
            speeds, expected, expected_offcore = z["velocities"], z["codes"], z["offcore"]
        retained = (expected == CORE) | (expected == OFF_CORE)
        transitions = np.flatnonzero(np.diff(retained.astype(int)))
        candidates = set(np.r_[transitions, transitions + 1].tolist())
        for label_code in (CORE, OFF_CORE):
            nodes = np.flatnonzero(expected == label_code)
            if len(nodes):
                candidates.add(int(nodes[len(nodes) // 2]))
        if not any(expected[n] == CORE and speeds[n] > 0 for n in candidates):
            raise ValueError(f"No positive core witness for {label}")
        observe(r, 80. * direction, data, manifest["horizon_s"], manifest["final_window_s"], 1.)
        rows = []
        for node in sorted(candidates):
            if not speeds[node]:
                continue
            codes, verified, _, _ = classify(r, direction, speeds[[node]], data,
                                             manifest["horizon_s"], manifest["final_window_s"])
            actual = int(codes[0]), bool(verified[0])
            reference = int(expected[node]), bool(expected_offcore[node])
            rows.append({"speed_m_per_s": float(speeds[node]), "sampled": actual,
                         "dense": reference, "agrees": actual == reference})
        report["settings"].append({"configuration": index, "source_ray": checkpoint.name, "rows": rows})
        write_json(cache / "fine_node_replay.json", {**report, "updated_utc": now()})
        print(f"{label}: {len(rows)} nodes, {sum(not r['agrees'] for r in rows)} disagreements", flush=True)
    report["status"] = ("passed" if all(r["agrees"] for point in report["settings"] for r in point["rows"])
                        else "requires_review")
    report["updated_utc"] = now()
    write_json(cache / "fine_node_replay.json", report)
    write_with_retry(ROOT / "pilot_diagnostics" / "fine_node_replay.json", report)


if __name__ == "__main__":
    main()
