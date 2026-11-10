"""Continue the sampled pilot with a bounded fallback for out-of-cell root probes."""

import argparse
from hashlib import sha256
import json
import msvcrt
import os
from pathlib import Path

import run_mot_strange_sampled_pilot as pilot
from run_mot_strange_loading_pilot import NAME, now, write_json


_initialize = pilot.initialize


def initialize(*args):
    _initialize(*args)
    from mot_strange_pocket_recovery import check_pocket
    from pmot.mot_strange import loading_scan_restoring
    loading_scan_restoring.check_off_core_pocket = check_pocket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_pilot"
    with (cache / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        failure = json.loads((cache / "failure.json").read_text(encoding="utf-8"))
        if failure["error"] != "ValueError('no unique ground steady state')":
            raise ValueError("Saved pilot failure does not match this recovery")
        revision_path = cache / "recovery_revision.json"
        if not revision_path.exists():
            write_json(revision_path, {
                "recorded_utc": now(), "source_manifest_sha256": sha256((cache / "manifest.json").read_bytes()).hexdigest(),
                "fallback_source_sha256": sha256(Path("scripts/mot_strange_pocket_recovery.py").read_bytes()).hexdigest(),
                "failure": failure,
                "scope": "Retry only failed off-core equilibrium probes outside the local neighborhood and cell. Apply the original witness gates to the bounded solution.",
            })
        pilot.initialize = initialize
        try:
            pilot.run_pilot(args.workers, cache)
        except Exception as error:
            write_json(cache / "resume_failure.json", {"updated_utc": now(), "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
