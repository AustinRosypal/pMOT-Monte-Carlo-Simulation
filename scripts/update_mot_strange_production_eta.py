"""Estimate production completion from complete-setting checkpoint throughput."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from plan_mot_strange_loading_production import read_json
from resume_mot_strange_loading_pilot import write_with_retry
from run_mot_strange_loading_pilot import NAME, ROOT, now


def launch_time():
    previous = sorted(ROOT.glob("production_launch_before_*.json"))
    path = previous[0] if previous else ROOT / "production_launch.json"
    return datetime.fromisoformat(read_json(path)["uv_started_utc"].replace("Z", "+00:00"))


def projection(cache, plan, current_launch):
    began = launch_time()
    changed_pool = current_launch.get("previous_workers", current_launch["workers"]) != current_launch["workers"]
    current_start = datetime.fromisoformat(current_launch["uv_started_utc"].replace("Z", "+00:00"))
    start = began
    observed = []
    completed = 0
    for key, point in sorted(plan["settings"].items(), key=lambda item: int(item[0])):
        index = int(key)
        folder = cache / f"configuration_{index:03d}"
        if not (folder / "summary.json").is_file():
            break
        summary = read_json(folder / "summary.json")
        if summary["rays"] != point["rays"]:
            raise ValueError(f"Checkpoint summary count differs for configuration {index}")
        end = datetime.fromtimestamp(max(p.stat().st_mtime for p in folder.glob("disc_*_point_*.npz")),
                                     timezone.utc)
        actual_wall_seconds = (end - start).total_seconds()
        if actual_wall_seconds <= 0:
            raise ValueError(f"Nonpositive wall interval for configuration {index}")
        predicted_worker_seconds = point["rays"] * point["pilot_worker_seconds_per_ray"]
        if not changed_pool or start >= current_start:
            observed.append({
                "configuration": index, "rays": point["rays"],
                "pilot_worker_seconds": predicted_worker_seconds, "actual_wall_seconds": actual_wall_seconds,
                "actual_worker_seconds": summary["worker_seconds"],
                "finished_utc": end.isoformat(),
            })
        completed += 1
        start = end
    if len(observed) < 2:
        return None
    remaining = sum(point["rays"] * point["pilot_worker_seconds_per_ray"]
                    for index, point in plan["settings"].items() if int(index) >= completed)
    ratio = sum(x["actual_wall_seconds"] for x in observed) / sum(x["pilot_worker_seconds"] for x in observed)
    rates = [x["actual_wall_seconds"] / x["pilot_worker_seconds"] for x in observed]
    estimates = {key: start + timedelta(seconds=remaining * rate)
                 for key, rate in (("central", ratio), ("observed_fast", min(rates)),
                                   ("observed_slow", max(rates)))}
    return {
        "status": "early_measured_projection" if len(observed) < 6 else "measured_projection",
        "updated_utc": now(), "original_start_utc": began.isoformat(),
        "completed_settings": completed, "total_settings": len(plan["settings"]),
        "timed_settings": len(observed), "workers": current_launch["workers"],
        "observed_settings": observed, "estimated_remaining_pilot_worker_hours": remaining / 3600,
        "production_finish_utc": {key: value.isoformat() for key, value in estimates.items()},
        "production_finish_central_time": {
            key: value.astimezone(ZoneInfo("America/Chicago")).isoformat() for key, value in estimates.items()
        },
        "method": "Convert remaining planning-pilot worker costs to wall time using completed production settings.",
        "sensitivity": "Fast and slow scenarios use the observed setting ratios. They are not probability bounds.",
        "audit_completion_not_included": True,
    }


if __name__ == "__main__":
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_production"
    result = projection(cache, read_json(ROOT / "sampled_production_plan.json"),
                        read_json(ROOT / "production_launch.json"))
    if result is None:
        print("Keep the provisional ETA until two settings finish entirely with the current worker count.")
    else:
        write_with_retry(ROOT / "production_eta_measured.json", result)
        print(json.dumps({key: result[key] for key in
                          ("status", "completed_settings", "timed_settings",
                           "production_finish_central_time")}, indent=2))
