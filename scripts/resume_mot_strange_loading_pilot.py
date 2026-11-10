"""Resume the frozen strange-MOT pilot with Windows-safe progress publication."""

import os
from pathlib import Path
import sys
from time import sleep

import run_mot_strange_loading_pilot as pilot


_write_json = pilot.write_json


def write_with_retry(path, value):
    """Keep live progress available if the sync client holds the repo file open."""
    is_progress = path.name == "pilot_progress.json"
    if is_progress:
        local = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / pilot.NAME
        _write_json(local / path.name, value)
    for attempt in range(8):
        try:
            _write_json(path, value)
            return
        except PermissionError as error:
            if attempt < 7:
                sleep(min(.1 * 2**attempt, 2.))
                continue
            if not is_progress:
                raise
            print(f"Progress is available in the local cache while the sync folder is busy ({error}).",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    pilot.write_json = write_with_retry
    pilot.main()
