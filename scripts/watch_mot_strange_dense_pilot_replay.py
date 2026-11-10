"""Replay excluded dense diagnostic settings in fresh processes as they finish.

This records independent spot checks only. Numerical unknowns and tail support
still require separate review, and this script never publishes loading rates.
"""

import argparse
from hashlib import sha256
import json
import msvcrt
import os
from pathlib import Path
import subprocess
from time import sleep

from run_mot_strange_loading_pilot import NAME, ROOT, now, write_json


def completed_replay(folder, configuration):
    for path in folder.glob("independent_spot_replay_*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("status") == "passed"
                and report.get("configuration") == configuration
                and report.get("requested_category") is None
                and len(report.get("rays", [])) >= 2
                and all(ray.get("attempts") for ray in report["rays"])):
            return True
    return False


def monitor(cache, once=False):
    manifest = json.loads((ROOT / "pilot_manifest.json").read_text(encoding="utf-8"))
    total = len(manifest["points"])
    state = ROOT / "dense_pilot_replay_progress.json"
    while True:
        complete = []
        for index in range(total):
            folder = ROOT / "pilot" / f"configuration_{index:03d}"
            if completed_replay(folder, index):
                complete.append(index)
                continue
            summary = folder / "checkpoint_summary.json"
            if not summary.is_file():
                continue
            saved = json.loads(summary.read_text(encoding="utf-8"))
            if len(saved["rows"]) != manifest["discs"] * manifest["points_per_disc"]:
                raise ValueError(f"Incomplete diagnostic summary for {index}")
            for source, expected in manifest["source_sha256"].items():
                if sha256(Path(source).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Frozen diagnostic source changed: {source}")
            write_json(state, dict(status="replaying", active_configuration=index,
                                  replayed_settings=complete, expected_settings=total,
                                  updated_utc=now(), pid=os.getpid()))
            log = cache / f"independent_replay_{index:03d}.log"
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"\nFresh independent replay started {now()}\n")
                stream.flush()
                result = subprocess.run(
                    ["uv", "run", "python", "-u", "scripts/replay_mot_strange_loading_pilot.py",
                     str(index), "--count", "2"], stdout=stream, stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode or not completed_replay(folder, index):
                write_json(state, dict(status="failed", configuration=index, log=str(log),
                                      returncode=result.returncode, updated_utc=now()))
                raise RuntimeError(f"Independent diagnostic replay failed for {index}")
            complete.append(index)
        write_json(state, dict(status="spot_replays_complete" if len(complete) == total else "watching",
                              replayed_settings=complete, expected_settings=total,
                              numerical_mask_audit_separate=True, updated_utc=now()))
        if once or len(complete) == total:
            return
        sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "pilot"
    with (cache / "independent_replay_watch.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        monitor(cache, args.once)
