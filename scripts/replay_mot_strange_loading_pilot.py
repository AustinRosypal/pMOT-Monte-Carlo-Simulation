"""Independently replay saved pilot nodes in a process with a private cache."""

import argparse
import json
import os
from pathlib import Path

from run_mot_strange_loading_pilot import NAME, ROOT, digest, now, write_json
from time_mot_strange_loading_rays import configure_worker


def replay(configuration, count, category=None):
    configure_worker()
    import numpy as np
    from pmot.mot_strange.loading_scan_retention import CORE, OFF_CORE, observe

    manifest = json.loads((ROOT / "pilot_manifest.json").read_text(encoding="utf-8"))
    name = f"configuration_{configuration:03d}"
    published = ROOT / "pilot" / name
    checkpoint_root = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "pilot" / name
    files = sorted(checkpoint_root.glob("disc_*_point_*.npz"))
    if category is not None:
        mask_key = "codes" if category == "core" else "offcore_verified"
        match = CORE if category == "core" else True
        selected = []
        for file in files:
            with np.load(file, allow_pickle=False) as saved:
                if (saved[mask_key][1:] == match).any():
                    selected.append(file)
            if len(selected) == count:
                break
        files = selected
    if not files:
        raise RuntimeError(f"No published checkpoint files for {name}")
    with np.load(published / "runtime_inputs.npz") as archive:
        data = tuple(archive[f"arr_{i}"] for i in range(len(archive.files)))
    for array in data:
        array.setflags(write=False)
    with np.load(ROOT / "pilot_launch_geometry.npz") as geometry:
        positions, directions = geometry["positions"], geometry["directions"]
    fingerprint = digest(data)
    report = {"status": "passed", "stage": "pilot_spot_check", "checked_utc": now(),
              "requested_category": category,
              "configuration": configuration, "input_sha256": fingerprint, "rays": []}
    for file in files[:count]:
        with np.load(file) as saved:
            if int(saved["configuration"]) != configuration or saved["prepared_sha256"].item() != fingerprint:
                raise RuntimeError(f"configuration or data fingerprint mismatch in {file}")
            disc, point = int(saved["disc"]), int(saved["point"])
            position, direction = saved["position"], saved["direction"]
            np.testing.assert_array_equal(position, positions[disc, point])
            np.testing.assert_array_equal(direction, directions[disc])
            speeds, codes = saved["speeds"], saved["codes"]
            indices = {0, len(speeds) // 2, len(speeds) - 1}
            for category in (CORE, OFF_CORE):
                matches = np.flatnonzero((codes == category) & (speeds > 0))
                if len(matches):
                    indices.add(int(matches[len(matches) // 2]))
            checked = []
            for key, expected in zip(saved["attempt_keys"], saved["attempts"], strict=True):
                index, method, resolution = int(key[0]), key[1], float(key[2])
                if index not in indices or expected[0] < 0:
                    continue
                record = observe(position, speeds[index] * direction, data,
                                 manifest["horizon_s"], manifest["final_window_s"],
                                 resolution if method == "adaptive" else 1.,
                                 resolution if method == "fixed" else 0.)
                actual = np.r_[record[0], record[1], record[2], record[3], record[4], record[5], record[6:]]
                np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)
                checked.append({"speed_m_per_s": float(speeds[index]), "method": str(method),
                                "resolution": resolution, "category": int(expected[0])})
            if not checked:
                raise RuntimeError(f"No definitive attempt replayed for {file}")
            report["rays"].append({"checkpoint": file.name, "attempts": checked})
    write_json(published / f"independent_spot_replay_{os.getpid()}.json", report)
    print(f"Replayed {sum(len(ray['attempts']) for ray in report['rays'])} saved attempts for {name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--category", choices=("core", "offcore"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    replay(args.configuration, args.count, args.category)
