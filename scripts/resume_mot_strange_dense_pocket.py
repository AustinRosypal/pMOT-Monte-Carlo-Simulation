"""Resume dense diagnostics with the guarded, existing out-of-cell recovery.

The original trajectory, checkpoint writer and pocket acceptance gates remain
unchanged. Only the worker's pocket-search entry point is wrapped. The bounded
fallback itself first reproduces and verifies the invalid out-of-cell probe.
"""

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

import run_mot_strange_loading_pilot as pilot
from resume_mot_strange_loading_pilot import write_with_retry


_initialize = pilot.initialize_worker
_run_ray = pilot.run_ray
CURRENT_TASK = None


def checked_pocket(observation, data):
    from mot_strange_pocket_recovery import check_pocket

    result = check_pocket(observation, data)
    if "equilibrium_search_fallback" in result:
        configuration, disc, point = CURRENT_TASK
        folder = pilot.STATE[-1] / "pocket_search_recovery"
        packed = np.r_[observation[0], observation[1], observation[2],
                       observation[3], observation[4], observation[5], observation[6:]]
        identity = sha256(packed.tobytes()).hexdigest()
        pilot.write_json(folder / f"disc_{disc:04d}_point_{point:02d}_{identity}.json", {
            "checked_utc": pilot.now(), "configuration": configuration,
            "disc": disc, "point": point, "pid": os.getpid(),
            "prepared_sha256": pilot.digest(data), "observation": packed.tolist(),
            "cell_half_length_m": float(data[11][15]), "pocket": result,
            "fallback_source_sha256": sha256(Path("scripts/mot_strange_pocket_recovery.py").read_bytes()).hexdigest(),
            "scope": "Original root failed outside the simulated cell; bounded fallback retained all original witness, residual, stability and return gates.",
        })
    return result


def initialize(*args):
    _initialize(*args)
    # Capture the original checker before installing our instrumentation.
    import mot_strange_pocket_recovery
    from pmot.mot_strange import loading_scan_restoring

    loading_scan_restoring.check_off_core_pocket = checked_pocket


def run_ray(task):
    global CURRENT_TASK
    CURRENT_TASK = task
    return _run_ray(task)


def main():
    failure = json.loads((pilot.ROOT / "pilot_failure.json").read_text())
    if failure["error"] not in (
        "ValueError('no unique ground steady state')",
        "RecursionError('maximum recursion depth exceeded')",
    ):
        raise ValueError("Saved dense failure does not match this guarded recovery")
    manifest = json.loads((pilot.ROOT / "pilot_manifest.json").read_text())
    for source, expected in manifest["source_sha256"].items():
        if sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen dense source changed: {source}")
    pilot.write_json = write_with_retry
    pilot.initialize_worker = initialize
    pilot.run_ray = run_ray
    pilot.main()


if __name__ == "__main__":
    main()
