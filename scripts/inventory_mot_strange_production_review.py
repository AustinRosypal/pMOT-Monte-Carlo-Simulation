"""List production draws that need independent numerical or tail review."""

import argparse
import os
from pathlib import Path

import numpy as np

from mot_strange_sampled_design import FINE_GRID
from run_mot_strange_loading_pilot import NAME, now, write_json


def inventory(configuration):
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_production"
    folder = cache / f"configuration_{configuration:03d}"
    files = sorted(folder.glob("disc_*_point_*.npz"))
    findings = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            unresolved, tail = int(z["unresolved"]), bool(z["tail_hit"])
            if not unresolved and not tail:
                continue
            codes, nodes = z["codes"], z["node_indices"]
            selected = np.unique(nodes[codes < 0])
            keys, records = z["attempt_keys"], z["attempts"]
            attempts = [
                {"speed_m_per_s": float(FINE_GRID[int(key[0])]), "method": str(key[1]),
                 "resolution": float(key[2]), "category": int(row[0]),
                 "elapsed_s": float(row[1]), "endpoint_m": row[2:5].tolist()}
                for key, row in zip(keys, records, strict=True) if int(key[0]) in selected
            ]
            findings.append({
                "ray": path.name, "unknown_speeds_m_per_s": FINE_GRID[selected].tolist(),
                "tail_hit": tail, "attempts": attempts,
            })
            print(path.name, FINE_GRID[selected].tolist(), f"tail={tail}", flush=True)
    path = folder / "review_queue.json"
    write_json(path, {"configuration": configuration, "checkpoint_count": len(files),
                      "updated_utc": now(), "findings": findings})
    print(f"{len(files)} checkpoints, {len(findings)} requiring review.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=int)
    inventory(parser.parse_args().configuration)
