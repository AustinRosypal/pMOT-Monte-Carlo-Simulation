"""Resume frozen production with retries for Windows sharing violations."""

import argparse
import msvcrt
import os
from pathlib import Path
from time import sleep

import run_mot_strange_loading_pilot as shared
from run_mot_strange_sampled_production import run


_original = shared.write_json


def write_with_retry(path, value):
    for attempt in range(10):
        try:
            _original(path, value)
            return
        except PermissionError:
            if attempt < 9:
                sleep(min(.05 * 2**attempt, 2.))
            elif path.name == "progress.json" and value.get("status") == "running":
                _original(path.with_name("progress_mirror.json"), value)
                return
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / shared.NAME / "sampled_production"
    with (cache / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        shared.write_json = write_with_retry
        try:
            run(args.workers, cache)
        except Exception as error:
            write_with_retry(cache / "resume_failure.json", {"updated_utc": shared.now(), "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
